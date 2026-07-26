"""
1D_a2a4 V2 Training Script — torchrun multi-GPU DDP

Three phases:
  phase1: UNet autoencoder pretraining on bvp_region_train.pt solutions
  phase2: Full V2PDEModel, warm-started from 1D_p best_model_copy
           (input_proj re-initialized; all other weights transferred)
  phase3: Full V2PDEModel with context, warm-started from Phase 2 best ckpt

Usage:
  torchrun --nproc_per_node=N train.py --phase phase1 --phase1_epochs 200
  torchrun --nproc_per_node=N train.py --phase phase2 --num_epochs 60
  torchrun --nproc_per_node=N train.py --phase phase3 --num_epochs 60 \
      --resume_from ./checkpoints/best_model

Data imbalance:
  WeightedEpochSampler with inverse-frequency class weights per k value.
"""
import os, sys, argparse, time, json, math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from model.v2_model import V2PDEModel
from model.unet1d_v6 import UNet1d
from model.pde_loss import get_pde_lambda
from data.dataset import (PDEContextDataset, EpochShuffleSampler,
                           WeightedEpochSampler, make_collate_fn)
from data.sequence_builder import SequenceBuilder


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase",            type=str,   default="phase2",
                   choices=["phase1", "phase2", "phase3"])
    p.add_argument("--phase1_epochs",    type=int,   default=200)
    p.add_argument("--batch_size",       type=int,   default=None)
    p.add_argument("--num_epochs",       type=int,   default=None)
    p.add_argument("--lr_projector",     type=float, default=None)
    p.add_argument("--lr_lora",          type=float, default=None)
    p.add_argument("--lambda_pde",       type=float, default=None)
    p.add_argument("--pde_use_gt_ref",   type=int,   default=None,
                   help="1=GT-referenced residual excess (subtract AE-floor); 0=raw residual")
    p.add_argument("--pde_warmup_start", type=int,   default=None)
    p.add_argument("--pde_warmup_end",   type=int,   default=None)
    p.add_argument("--grad_accum",       type=int,   default=None)
    p.add_argument("--target_repeat_per_epoch", type=int, default=None)
    p.add_argument("--no_context_prob",  type=float, default=None)
    p.add_argument("--val_no_context_prob", type=float, default=None)
    p.add_argument("--resume_from",      type=str,   default=None,
                   help="Resume from this checkpoint dir (phase2→phase3)")
    p.add_argument("--pretrained_1dp",   type=str,   default=None,
                   help="Override 1D_p ckpt path for phase2 warm-start")
    p.add_argument("--unet_checkpoint",  type=str,   default=None)
    p.add_argument("--no_warmstart",     type=int,   default=0,
                   help="phase2: 0=warm-start from 1D_p (default); 1=fresh init from "
                        "pretrained Qwen (cross-PDE warm-start ablation control)")
    p.add_argument("--output_dir",       type=str,   default=None)
    p.add_argument("--local_rank",       type=int,   default=-1)
    return p.parse_args()


# -----------------------------------------------------------------------
# DDP
# -----------------------------------------------------------------------

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) \
           or dist.get_rank() == 0


# -----------------------------------------------------------------------
# Phase 1: UNet pretraining
# -----------------------------------------------------------------------

def train_unet_phase(config: Config, num_epochs: int, local_rank: int):
    if is_rank0():
        print("=" * 70)
        print("Phase 1: UNet autoencoder pretraining")
        print("=" * 70)

    train_data = torch.load(config.train_data_path, weights_only=False)
    val_data   = torch.load(config.val_data_path,   weights_only=False)

    class FlatSolDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.samples = []
            for d in data:
                for s in d['solutions']:
                    self.samples.append(
                        torch.as_tensor(s, dtype=torch.float32)
                    )
        def __len__(self): return len(self.samples)
        def __getitem__(self, i): return self.samples[i]

    tr_ds = FlatSolDataset(train_data)
    va_ds = FlatSolDataset(val_data)

    if is_rank0():
        print(f"  Train solutions: {len(tr_ds):,}  |  Val: {len(va_ds):,}")

    tr_sampler = DistributedSampler(tr_ds, shuffle=True)
    va_sampler = DistributedSampler(va_ds, shuffle=False)
    tr_loader  = DataLoader(tr_ds, batch_size=64, sampler=tr_sampler,
                            num_workers=4, pin_memory=True)
    va_loader  = DataLoader(va_ds, batch_size=64, sampler=va_sampler,
                            num_workers=4, pin_memory=True)

    device = torch.device(f"cuda:{local_rank}")
    unet = UNet1d(layers=config.unet_channels, latent_dim=config.latent_dim,
                  solution_dim=config.solution_dim
                  ).to(device=device, dtype=torch.bfloat16)
    unet = DDP(unet, device_ids=[local_rank])

    opt       = optim.AdamW(unet.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val  = float("inf")

    for epoch in range(1, num_epochs + 1):
        tr_sampler.set_epoch(epoch)
        unet.train()
        tr_loss, n = 0.0, 0
        for x in tr_loader:
            x    = x.to(device, torch.bfloat16)
            z    = unet.module(x, "encode")
            x_hat = unet.module(z, "decode")
            loss  = F.mse_loss(x_hat.float(), x.float())
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item(); n += 1

        unet.eval()
        va_loss, nv = 0.0, 0
        with torch.no_grad():
            for x in va_loader:
                x    = x.to(device, torch.bfloat16)
                z    = unet.module(x, "encode")
                x_hat = unet.module(z, "decode")
                va_loss += F.mse_loss(x_hat.float(), x.float()).item(); nv += 1

        if is_rank0():
            va = va_loss / max(nv, 1)
            print(f"[UNet] epoch {epoch:03d} | train={tr_loss/max(n,1):.4e} "
                  f"| val={va:.4e}")
            if va < best_val:
                best_val = va
                os.makedirs(os.path.dirname(config.unet_checkpoint_path), exist_ok=True)
                torch.save(unet.module.state_dict(), config.unet_checkpoint_path)
                print(f"  -> best UNet saved ({va:.4e})")

    if is_rank0():
        print(f"Phase 1 done. best_val={best_val:.4e}")


# -----------------------------------------------------------------------
# Phase 2 / 3: Full V2 model training
# -----------------------------------------------------------------------

def train_phase(config: Config, num_epochs: int, local_rank: int,
                batch_size: int, grad_accum: int,
                phase: str = "phase2",
                resume_from: str = None,
                pretrained_1dp: str = None,
                no_warmstart: int = 0,
                no_context_prob: float = 0.0,
                val_no_context_prob: float = 0.0):

    if is_rank0():
        print("=" * 70)
        print(f"{phase.upper()}: V2PDEModel training (DDP)")
        print("=" * 70)

    # Load data
    train_data = torch.load(config.train_data_path, weights_only=False)
    val_data   = torch.load(config.val_data_path,   weights_only=False)

    # Build index tensors
    train_indices = torch.arange(len(train_data), dtype=torch.long)
    val_indices   = torch.arange(len(val_data),   dtype=torch.long)

    # ---- Build model ----
    if resume_from:
        if is_rank0():
            print(f"  Resuming from: {resume_from}")
        model = V2PDEModel.from_pretrained(resume_from, config)
    elif phase == "phase2" and not no_warmstart:
        ckpt_1dp = pretrained_1dp or config.pretrained_1dp_ckpt
        if is_rank0():
            print(f"  Warm-starting from 1D_p ckpt: {ckpt_1dp}")
        model = V2PDEModel.from_pretrained_1dp(ckpt_1dp, config)
    else:
        # no-warm-start phase2 (cross-PDE ablation control), or phase3 without resume_from:
        # fresh init = pretrained Qwen + fresh LoRA/projectors/heads + this problem's AE.
        if is_rank0():
            print("  Fresh init from pretrained Qwen (no warm-start)")
        model = V2PDEModel(config, unet_checkpoint=config.unet_checkpoint_path)

    dev = model.component_device

    # ---- SequenceBuilder ----
    seq_builder = SequenceBuilder(
        unet=model.unet, input_proj=model.input_proj,
        special_tok=model.special_tok,
        latent_dim=config.latent_dim, hidden_dim=config.qwen_hidden_dim,
        param_dim=config.param_dim,
        max_seq_len=config.max_seq_len, device=dev,
    )
    collate_fn = make_collate_fn(seq_builder, max_seq_len=config.max_seq_len)

    # ---- Datasets ----
    train_ds = PDEContextDataset(
        train_data, train_indices,
        max_context_p=config.max_context_p,
        seed=config.sampler_seed,
        no_context_prob=no_context_prob,
    )
    val_ds = PDEContextDataset(
        val_data, val_indices,
        max_context_p=config.max_context_p,
        seed=config.sampler_seed + 1,
        no_context_prob=val_no_context_prob,
    )

    # ---- Weighted sampler ----
    # NOTE: the corrected, zero-free dataset shifted every multiplicity DOWN by 1
    # (the trivial u==0 branch was removed), so k {2,4,5,6} -> {1,3,4,5}. The target
    # allocation below is the SAME proven scheme with keys shifted by -1:
    #   k=1 → 30% of total draws  (important, single-solution / boundary)   [was k=2]
    #   k=3 → 30% of total draws  (de-emphasise dominant class)            [was k=4]
    #   k=5 → 40% of total draws  (most multi-solution, hardest)           [was k=6]
    #   k=4 → exactly freq[4] draws (one full pass, the rare class)        [was k=5]
    k_counts_train = [len(d['solutions']) for d in train_data]
    freq = Counter(k_counts_train)
    N_train = len(train_data)
    if is_rank0():
        print(f"  k-distribution (train): {dict(sorted(freq.items()))}")

    # rare class k=4 gets one full traversal (no over/under-sampling)
    k_rare_draws = freq.get(4, 0)
    # Remaining draws distributed 30/30/40 among k=1,3,5
    remaining  = N_train - k_rare_draws
    TARGET_DRAWS = {
        1: int(remaining * 0.30),
        3: int(remaining * 0.30),
        4: k_rare_draws,
        5: remaining - int(remaining * 0.30) - int(remaining * 0.30),
    }
    # per-sample weight = target_draws[k] / freq[k]
    sample_weights = torch.tensor(
        [TARGET_DRAWS.get(k, 1) / freq[k] for k in k_counts_train],
        dtype=torch.float64,
    )
    if is_rank0():
        total_w = sample_weights.sum().item()
        for k in sorted(freq):
            draws = N_train * (TARGET_DRAWS.get(k, 1) / freq[k] * freq[k]) / total_w
            # exact draws ≈ TARGET_DRAWS[k] since weights normalised to N_train
            print(f"    k={k}: target_draws≈{TARGET_DRAWS.get(k,0):5d}  "
                  f"({TARGET_DRAWS.get(k,0)/N_train*100:.1f}%)  "
                  f"repeats/sample={TARGET_DRAWS.get(k,0)/freq[k]:.2f}x")

    epoch_draws = N_train // 2   # half the dataset per epoch
    if is_rank0():
        print(f"  epoch_draws={epoch_draws} (half of N_train={N_train})")
    train_sampler = WeightedEpochSampler(
        train_indices, weights=sample_weights,
        seed=config.sampler_seed,
        repeat_factor=config.target_repeat_per_epoch,
        num_samples=epoch_draws,
    )
    val_sampler = EpochShuffleSampler(val_indices, seed=config.sampler_seed + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=train_sampler, collate_fn=collate_fn,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              sampler=val_sampler,   collate_fn=collate_fn,
                              num_workers=0, pin_memory=False)

    # ---- Optimizers ----
    proj_params = (list(model.special_tok.parameters())
                   + list(model.input_proj.parameters())
                   + list(model.output_proj.parameters())
                   + list(model.dual_head.parameters()))
    lora_params = [p for p in model.qwen_model.parameters() if p.requires_grad]

    if is_rank0():
        print(f"  Projector params: {sum(p.numel() for p in proj_params):,}")
        print(f"  LoRA params:      {sum(p.numel() for p in lora_params):,}")

    opt_proj = optim.AdamW(proj_params, lr=config.lr_projector,
                           weight_decay=config.weight_decay)
    opt_lora = optim.AdamW(lora_params, lr=config.lr_lora,
                           weight_decay=config.weight_decay)

    sched_proj = optim.lr_scheduler.ReduceLROnPlateau(
        opt_proj, mode="min", factor=config.lr_plateau_factor,
        patience=config.lr_plateau_patience,
        min_lr=config.lr_projector * 0.01)
    sched_lora = optim.lr_scheduler.ReduceLROnPlateau(
        opt_lora, mode="min", factor=config.lr_plateau_factor,
        patience=config.lr_plateau_patience,
        min_lr=config.lr_lora * 0.01)

    writer = None
    if is_rank0():
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(config.log_dir, phase)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    best_val_mse = float("inf")
    es_best = float("inf"); es_wait = 0
    es_patience = int(os.environ.get("EARLY_STOP_PATIENCE", "15"))  # stop after N epochs w/o val improvement; 0 disables
    global_step  = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_ds.set_epoch(epoch)
        train_sampler.set_epoch(epoch)

        model.train(); model.unet.eval()
        r_total = r_mse = r_pde = r_ce = 0.0
        n_bat = 0
        opt_proj.zero_grad(); opt_lora.zero_grad()

        for bi, batch in enumerate(train_loader):
            out  = model(batch, current_epoch=epoch)
            loss = out["total_loss"] / grad_accum
            loss.backward()

            if (bi + 1) % grad_accum == 0:
                if dist.is_initialized():
                    ws = dist.get_world_size()
                    for param in proj_params + lora_params:
                        if param.grad is not None:
                            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                            param.grad.div_(ws)

                torch.nn.utils.clip_grad_norm_(proj_params, config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(lora_params, config.max_grad_norm)
                opt_proj.step(); opt_lora.step()
                opt_proj.zero_grad(); opt_lora.zero_grad()
                global_step += 1

                if is_rank0() and writer and global_step % 50 == 0:
                    writer.add_scalar("Loss/Total",  out["total_loss"].item(), global_step)
                    writer.add_scalar("Loss/MSE",    out["mse_loss"].item(),   global_step)
                    writer.add_scalar("Loss/PDE",    out["pde_loss"].item(),   global_step)
                    writer.add_scalar("Loss/CE",     out["ce_loss"].item(),    global_step)

            r_total += out["total_loss"].detach().item()
            r_mse   += out["mse_loss"].detach().item()
            r_pde   += out["pde_loss"].detach().item()
            r_ce    += out["ce_loss"].detach().item()
            n_bat   += 1

        # Validation
        model.eval()
        val_mse, nv = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch, current_epoch=epoch)
                val_mse += out["mse_loss"].item(); nv += 1
        val_mse /= max(nv, 1)

        if dist.is_initialized():
            vt = torch.tensor(val_mse, device=dev)
            dist.all_reduce(vt, op=dist.ReduceOp.AVG)
            val_mse = vt.item()

        prev_lrp = opt_proj.param_groups[0]["lr"]
        sched_proj.step(val_mse); sched_lora.step(val_mse)
        cur_lrp  = opt_proj.param_groups[0]["lr"]
        cur_lrl  = opt_lora.param_groups[0]["lr"]
        decayed  = cur_lrp < prev_lrp

        if is_rank0():
            n = max(n_bat, 1)
            print(
                f"[{phase}] Ep {epoch:3d}/{num_epochs} | "
                f"tot={r_total/n:.4e} mse={r_mse/n:.4e} "
                f"pde={r_pde/n:.4e} ce={r_ce/n:.4e} | "
                f"val_mse={val_mse:.4e} | "
                f"lr_p={cur_lrp:.2e} lr_l={cur_lrl:.2e}"
                + (" [LR↓]" if decayed else "") +
                f" | λ_pde={out['lambda_pde']:.3f} | {time.time()-t0:.1f}s"
            )
            if writer:
                writer.add_scalar("Loss/Train_MSE", r_mse/n,    epoch)
                writer.add_scalar("Loss/Val_MSE",   val_mse,    epoch)
                writer.add_scalar("LR/projector",   cur_lrp,    epoch)
                writer.add_scalar("LR/lora",        cur_lrl,    epoch)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                model.save_pretrained(os.path.join(config.output_dir, "best_model"))
                print(f"  -> Best model saved (val_mse={best_val_mse:.4e})")

            if epoch % 10 == 0:
                model.save_pretrained(os.path.join(config.output_dir, f"epoch_{epoch}"))

        # ---- Early stopping (ALL ranks; val_mse is all-reduce-synced above so es_wait
        #      is identical on every rank -> they break together, no DDP deadlock) ----
        if val_mse < es_best - 1e-9:
            es_best = val_mse; es_wait = 0
        else:
            es_wait += 1
        if es_patience and es_wait >= es_patience:
            if is_rank0():
                print(f"[early-stop] no val_mse improvement for {es_patience} epochs "
                      f"(best={es_best:.4e}); stopping at epoch {epoch}/{num_epochs}", flush=True)
            break

    if is_rank0():
        if writer:
            writer.close()
        summary = {
            "phase": phase, "num_epochs": num_epochs,
            "best_val_mse": best_val_mse,
            "batch_size": batch_size, "grad_accum": grad_accum,
            "lr_projector": config.lr_projector, "lr_lora": config.lr_lora,
        }
        with open(os.path.join(config.output_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"{phase} done. best_val_mse={best_val_mse:.4e}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args       = parse_args()
    local_rank = setup_ddp()
    config     = Config()

    # CLI overrides
    batch_size = args.batch_size   or config.batch_size
    num_epochs = args.num_epochs   or config.num_epochs
    grad_accum = args.grad_accum   or config.gradient_accumulation_steps
    if args.lr_projector     is not None: config.lr_projector     = args.lr_projector
    if args.lr_lora          is not None: config.lr_lora          = args.lr_lora
    if args.lambda_pde       is not None: config.lambda_pde       = args.lambda_pde
    if args.pde_use_gt_ref   is not None: config.pde_use_gt_ref   = bool(args.pde_use_gt_ref)
    if args.pde_warmup_start is not None: config.pde_warmup_start = args.pde_warmup_start
    if args.pde_warmup_end   is not None: config.pde_warmup_end   = args.pde_warmup_end
    if args.unet_checkpoint  is not None: config.unet_checkpoint_path = args.unet_checkpoint
    if args.output_dir       is not None:
        config.output_dir = args.output_dir
        os.makedirs(config.output_dir, exist_ok=True)
    if args.target_repeat_per_epoch is not None:
        config.target_repeat_per_epoch = max(1, args.target_repeat_per_epoch)

    no_ctx     = 0.0  if args.no_context_prob     is None else float(args.no_context_prob)
    val_no_ctx = no_ctx if args.val_no_context_prob is None else float(args.val_no_context_prob)
    no_ctx     = max(0.0, min(1.0, no_ctx))
    val_no_ctx = max(0.0, min(1.0, val_no_ctx))

    if is_rank0():
        ws = dist.get_world_size() if dist.is_initialized() else 1
        print("=" * 60)
        print("1D_a2a4 V2 PDE Model Training")
        print(f"  Phase:      {args.phase}")
        print(f"  World size: {ws}")
        print(f"  Batch/rank: {batch_size}  | grad_accum={grad_accum}")
        print(f"  Eff batch:  {batch_size * grad_accum * ws}")
        print(f"  Epochs:     {num_epochs}")
        print(f"  no_ctx:     train={no_ctx:.2f}  val={val_no_ctx:.2f}")
        print("=" * 60)

    if args.phase == "phase1":
        train_unet_phase(config, num_epochs=args.phase1_epochs,
                         local_rank=local_rank)

    elif args.phase == "phase2":
        if not os.path.exists(config.unet_checkpoint_path):
            raise FileNotFoundError(
                f"UNet checkpoint not found: {config.unet_checkpoint_path}\n"
                "Run phase1 first."
            )
        train_phase(config, num_epochs=num_epochs, local_rank=local_rank,
                    batch_size=batch_size, grad_accum=grad_accum,
                    phase="phase2",
                    resume_from=args.resume_from,
                    pretrained_1dp=args.pretrained_1dp,
                    no_warmstart=args.no_warmstart,
                    no_context_prob=no_ctx,
                    val_no_context_prob=val_no_ctx)

    elif args.phase == "phase3":
        resume = args.resume_from or os.path.join(config.output_dir, "best_model")
        if not os.path.exists(resume):
            raise FileNotFoundError(
                f"Phase 3 requires phase 2 checkpoint: {resume}\n"
                "Run phase2 first or supply --resume_from."
            )
        config.output_dir = os.path.join(
            os.path.dirname(config.output_dir), "phase3_checkpoints"
        )
        os.makedirs(config.output_dir, exist_ok=True)
        train_phase(config, num_epochs=num_epochs, local_rank=local_rank,
                    batch_size=batch_size, grad_accum=grad_accum,
                    phase="phase3",
                    resume_from=resume,
                    no_context_prob=no_ctx,
                    val_no_context_prob=val_no_ctx)

    cleanup_ddp()


if __name__ == "__main__":
    main()
