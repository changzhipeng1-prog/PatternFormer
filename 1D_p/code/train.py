"""
V2 Training Script — torchrun 8-GPU DDP

Usage:
    torchrun --nproc_per_node=8 --master_port=29500 train.py [OPTIONS]

Training strategy:
    - Phase 1 (optional, --phase phase1):
          UNet autoencoder pretraining on single solutions.
    - Phase 2 (default):
          Full V2PDEModel training with joint MSE + PDE(warmup) + CE loss.

Distributed setup:
    - torch.distributed + DistributedDataParallel (DDP)
    - Gradient accumulation across steps
    - Two optimizer groups:
          group A (lr_projector): SpecialTokenEmbeddings, InputProjector,
                                  OutputProjector, DualHead
          group B (lr_lora):     Qwen LoRA parameters
    - Both optimizers use CosineAnnealingLR scheduler

Sequence building:
    - SequenceBuilder is created once per process (model components shared).
    - DataLoader workers = 0 (SequenceBuilder uses GPU ops inside model,
      called from collate which runs in the main process).

Checkpointing:
    - Best model (lowest val MSE) → checkpoints/best_model/
    - Every 10 epochs → checkpoints/epoch_N/
    - Only rank-0 writes checkpoints and logs.
"""
import os
import sys
import argparse
import time
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from model.v2_model import V2PDEModel
from model.unet1d_v6 import UNet1d
from model.pde_loss import get_pde_lambda
from data.dataset import (PDEContextDataset, EpochShuffleSampler,
                          WeightedEpochSampler, make_collate_fn)
from data.sequence_builder import SequenceBuilder

# -----------------------------------------------------------------------
# Phase 1 uses pure reconstruction MSE (no Sobolev terms)
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="V2 PDE Model Training")
    p.add_argument("--phase",          type=str,   default="phase2",
                   choices=["phase1", "phase2", "both"])
    p.add_argument("--phase1_epochs",  type=int,   default=200,
                   help="UNet stage1 training epochs")
    p.add_argument("--batch_size",     type=int,   default=None)
    p.add_argument("--num_epochs",     type=int,   default=None)
    p.add_argument("--lr_projector",   type=float, default=None)
    p.add_argument("--lr_lora",        type=float, default=None)
    p.add_argument("--lambda_pde",     type=float, default=None)
    p.add_argument("--pde_use_gt_ref", type=int,   default=None,
                   help="1=GT-referenced residual excess (subtract AE-floor); 0=raw residual")
    p.add_argument("--pde_warmup_start", type=int, default=None)
    p.add_argument("--pde_warmup_end",   type=int, default=None)
    p.add_argument("--grad_accum",     type=int,   default=None)
    p.add_argument("--target_repeat_per_epoch", type=int, default=None,
                   help="Repeat each target p this many times per epoch")
    p.add_argument("--no_context_prob", type=float, default=None,
                   help="Probability of using empty context for each training sample (0~1)")
    p.add_argument("--val_no_context_prob", type=float, default=None,
                   help="Probability of using empty context for each validation sample (0~1)")
    p.add_argument("--logging_steps",  type=int,   default=None)
    p.add_argument("--unet_checkpoint", type=str,  default=None)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override checkpoint output directory")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Resume phase2 weights from checkpoint dir (e.g. checkpoints/best_model)")
    p.add_argument("--local_rank",     type=int,   default=-1,
                   help="Supplied automatically by torchrun")
    # ---- pretrained-vs-random backbone ablation ----
    p.add_argument("--random_init_backbone", type=int, default=None,
                   help="1 = frozen RANDOM Qwen-arch backbone + LoRA (ablation)")
    p.add_argument("--full_finetune_backbone", type=int, default=None,
                   help="1 = train ALL backbone params, no LoRA (from-scratch ablation)")
    p.add_argument("--scratch_hidden", type=int, default=None,
                   help=">0 shrinks backbone hidden_size (also sets qwen_hidden_dim)")
    p.add_argument("--scratch_layers", type=int, default=None,
                   help=">0 shrinks backbone num_hidden_layers")
    return p.parse_args()


# -----------------------------------------------------------------------
# DDP helpers
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

def train_unet_phase(config: Config, num_epochs: int = 20,
                     batch_size: int = 64, lr: float = 1e-3,
                     local_rank: int = 0):
    if is_rank0():
        print("=" * 70)
        print("Phase 1: UNet autoencoder pretraining")
        print("=" * 70)

    lookup = torch.load(config.data_lookup_path, weights_only=False)
    train_p_idx = torch.load(config.train_p_idx_path, weights_only=False)
    val_p_idx   = torch.load(config.val_p_idx_path,   weights_only=False)

    # Flat solution datasets
    from data.dataset import PDEContextDataset

    class FlatSolutionDataset(torch.utils.data.Dataset):
        def __init__(self, lookup, p_indices):
            self.samples = []
            sols = lookup["solutions_by_p"]
            for idx in (p_indices.tolist() if isinstance(p_indices, torch.Tensor)
                        else list(p_indices)):
                for k in range(sols[int(idx)].shape[0]):
                    self.samples.append(sols[int(idx)][k].float())

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            return self.samples[i]

    train_ds = FlatSolutionDataset(lookup, train_p_idx)
    val_ds   = FlatSolutionDataset(lookup, val_p_idx)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   shuffle=False)
    train_loader  = DataLoader(train_ds, batch_size=batch_size,
                               sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader    = DataLoader(val_ds, batch_size=batch_size,
                               sampler=val_sampler, num_workers=4, pin_memory=True)

    device = torch.device(f"cuda:{local_rank}")
    unet = UNet1d(
        layers=config.unet_channels,
        latent_dim=config.latent_dim,
        solution_dim=config.solution_dim,
    ).to(device=device, dtype=torch.bfloat16)
    unet = DDP(unet, device_ids=[local_rank])

    optimizer = optim.AdamW(unet.parameters(), lr=lr,
                            weight_decay=config.weight_decay)
    best_val = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_sampler.set_epoch(epoch)
        unet.train()
        tr_loss = 0.0
        n = 0
        for x in train_loader:
            x = x.to(device=device, dtype=torch.bfloat16)
            z    = unet.module(x, "encode")
            x_hat = unet.module(z, "decode")

            loss = F.mse_loss(x_hat.float(), x.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            n += 1

        unet.eval()
        va_loss = 0.0
        nv = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device=device, dtype=torch.bfloat16)
                z    = unet.module(x, "encode")
                x_hat = unet.module(z, "decode")

                loss = F.mse_loss(x_hat.float(), x.float())

                va_loss += loss.item()
                nv += 1

        tr = tr_loss / max(n, 1)
        va = va_loss / max(nv, 1)
        if is_rank0():
            print(f"[UNet] epoch {epoch:03d} | train_mse={tr:.4e} | val_mse={va:.4e}")
            if va < best_val:
                best_val = va
                os.makedirs(os.path.dirname(config.unet_checkpoint_path), exist_ok=True)
                torch.save(unet.module.state_dict(), config.unet_checkpoint_path)
                print(f"  -> best UNet saved ({va:.4e})")

    if is_rank0():
        print(f"UNet pretraining done.  best_val={best_val:.4e}")


# -----------------------------------------------------------------------
# Phase 2: Full V2 model training
# -----------------------------------------------------------------------

def train_phase2(config: Config, num_epochs: int, local_rank: int,
                 batch_size: int, grad_accum: int, resume_from: str = None,
                 no_context_prob: float = 0.0,
                 val_no_context_prob: float = 0.0):
    if is_rank0():
        print("=" * 70)
        print("Phase 2: V2PDEModel training (DDP)")
        print("=" * 70)

    # Load lookup and splits
    lookup      = torch.load(config.data_lookup_path, weights_only=False)
    train_p_idx = torch.load(config.train_p_idx_path, weights_only=False)
    val_p_idx   = torch.load(config.val_p_idx_path,   weights_only=False)

    # Build model (each rank builds its own; LoRA is loaded by each)
    if resume_from:
        if is_rank0():
            print(f"  Resuming phase2 weights from: {resume_from}")
        model = V2PDEModel.from_pretrained(resume_from, config)
    else:
        model = V2PDEModel(config, unet_checkpoint=config.unet_checkpoint_path)
    dev = model.component_device

    # NOTE: Qwen with device_map="auto" cannot be wrapped in DDP.
    # The small projector/head modules (special_tok, input_proj, output_proj,
    # dual_head) are on a single GPU and trained per-rank.  Without DDP sync
    # these ranks diverge slightly, but the effective learning rate is
    # multiplied by world_size — compensate by scaling lr_projector down if
    # training instability is observed.  A proper fix would use FSDP or
    # DeepSpeed ZeRO-3 for the full model.

    # SequenceBuilder (uses model's frozen modules; no gradient tracking needed)
    seq_builder = SequenceBuilder(
        unet=model.unet,
        input_proj=model.input_proj,
        special_tok=model.special_tok,
        latent_dim=config.latent_dim,
        hidden_dim=config.qwen_hidden_dim,
        max_seq_len=config.max_seq_len,
        device=dev,
    )
    collate_fn = make_collate_fn(seq_builder, max_seq_len=config.max_seq_len)

    # Datasets and samplers
    train_ds = PDEContextDataset(lookup, train_p_idx,
                                 max_context_p=config.max_context_p,
                                 seed=config.sampler_seed,
                                 no_context_prob=no_context_prob)
    val_ds   = PDEContextDataset(lookup, val_p_idx,
                                 max_context_p=config.max_context_p,
                                 seed=config.sampler_seed + 1,
                                 no_context_prob=val_no_context_prob)

    # Compute per-p weights: k_gt^power so multi-solution p values are over-sampled.
    solutions_by_p = lookup["solutions_by_p"]
    k_gt_train = torch.tensor(
        [solutions_by_p[int(i)].shape[0] for i in train_p_idx.tolist()],
        dtype=torch.float64,
    )
    train_weights = k_gt_train.pow(config.multisol_weight_power)
    if is_rank0():
        import collections
        kc = collections.Counter(k_gt_train.long().tolist())
        print(f"  Weighted sampler: k_gt distribution = {dict(sorted(kc.items()))}")
        print(f"  multisol_weight_power = {config.multisol_weight_power}")

    train_sampler = WeightedEpochSampler(
        train_p_idx,
        weights=train_weights,
        seed=config.sampler_seed,
        repeat_factor=config.target_repeat_per_epoch,
    )
    val_sampler = EpochShuffleSampler(
        val_p_idx, seed=config.sampler_seed + 1, repeat_factor=1
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    if is_rank0():
        world = dist.get_world_size() if dist.is_initialized() else 1
        steps_per_rank = len(train_loader)
        global_samples_per_epoch = len(train_p_idx) * config.target_repeat_per_epoch
        print(
            f"  target_repeat_per_epoch: {config.target_repeat_per_epoch} "
            f"(global target samples/epoch={global_samples_per_epoch})"
        )
        print(f"  train steps/rank/epoch: {steps_per_rank} | world_size={world}")
        print(f"  no_context_prob (train): {no_context_prob:.2f}")
        print(f"  no_context_prob (val):   {val_no_context_prob:.2f}")

    # Optimizers
    projector_params = (
        list(model.special_tok.parameters())
        + list(model.input_proj.parameters())
        + list(model.output_proj.parameters())
        + list(model.dual_head.parameters())
    )
    lora_params = [p for p in model.qwen_model.parameters() if p.requires_grad]

    if is_rank0():
        print(f"  Projector params : {sum(p.numel() for p in projector_params):,}")
        print(f"  LoRA params      : {sum(p.numel() for p in lora_params):,}")
        print(f"  UNet params      : {sum(p.numel() for p in model.unet.parameters()):,}  (frozen)")

    opt_proj = optim.AdamW(projector_params, lr=config.lr_projector,
                           weight_decay=config.weight_decay)
    opt_lora = optim.AdamW(lora_params,      lr=config.lr_lora,
                           weight_decay=config.weight_decay)

    # ReduceLROnPlateau: automatically decays LR when val_mse stops improving.
    # patience=lr_plateau_patience epochs, factor=lr_plateau_factor, min_lr=1% of init.
    sched_proj = optim.lr_scheduler.ReduceLROnPlateau(
        opt_proj, mode="min",
        factor=config.lr_plateau_factor,
        patience=config.lr_plateau_patience,
        min_lr=config.lr_projector * 0.01,
    )
    sched_lora = optim.lr_scheduler.ReduceLROnPlateau(
        opt_lora, mode="min",
        factor=config.lr_plateau_factor,
        patience=config.lr_plateau_patience,
        min_lr=config.lr_lora * 0.01,
    )

    # TensorBoard (rank-0 only)
    writer = None
    if is_rank0():
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(config.log_dir, "phase2_v2")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    best_val_mse = float("inf")
    es_best = float("inf"); es_wait = 0
    es_patience = int(os.environ.get("EARLY_STOP_PATIENCE", "15"))  # stop after N epochs w/o val improvement; 0 disables
    global_step  = 0

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_ds.set_epoch(epoch)
        train_sampler.set_epoch(epoch)

        # ---- Train ----
        model.train()
        model.unet.eval()   # UNet always eval (frozen)

        run_total = run_mse = run_pde = run_ce = 0.0
        n_batches = 0

        opt_proj.zero_grad()
        opt_lora.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            out = model(batch, current_epoch=epoch)
            loss = out["total_loss"] / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                # Synchronise gradients across ranks before stepping.
                # Each rank processed a different shard of data; averaging their
                # gradients gives the correct data-parallel gradient estimate.
                if dist.is_initialized():
                    world_size = dist.get_world_size()
                    all_params = projector_params + lora_params
                    for param in all_params:
                        if param.grad is not None:
                            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                            param.grad.div_(world_size)

                torch.nn.utils.clip_grad_norm_(projector_params, config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(lora_params,      config.max_grad_norm)
                opt_proj.step()
                opt_lora.step()
                opt_proj.zero_grad()
                opt_lora.zero_grad()
                global_step += 1

                if is_rank0() and writer and global_step % config.logging_steps == 0:
                    writer.add_scalar("Loss/Total_Step",  out["total_loss"].item(), global_step)
                    writer.add_scalar("Loss/MSE_Step",    out["mse_loss"].item(),   global_step)
                    writer.add_scalar("Loss/PDE_Step",    out["pde_loss"].item(),   global_step)
                    writer.add_scalar("Loss/CE_Step",     out["ce_loss"].item(),    global_step)
                    writer.add_scalar("PDE/Lambda",       out["lambda_pde"],        global_step)

            run_total += out["total_loss"].detach().item()
            run_mse   += out["mse_loss"].detach().item()
            run_pde   += out["pde_loss"].detach().item()
            run_ce    += out["ce_loss"].detach().item()
            n_batches += 1

        n = max(n_batches, 1)
        avg_total = run_total / n
        avg_mse   = run_mse   / n
        avg_pde   = run_pde   / n
        avg_ce    = run_ce    / n

        # ---- Validate ----
        model.eval()
        val_mse = 0.0
        nv = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch, current_epoch=epoch)
                val_mse += out["mse_loss"].item()
                nv += 1
        val_mse /= max(nv, 1)

        # Sync val_mse across ranks
        if dist.is_initialized():
            val_tensor = torch.tensor(val_mse, device=dev)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            val_mse = val_tensor.item()

        # ReduceLROnPlateau: pass validation MSE so scheduler can detect plateau.
        prev_lr_proj = opt_proj.param_groups[0]["lr"]
        prev_lr_lora = opt_lora.param_groups[0]["lr"]
        sched_proj.step(val_mse)
        sched_lora.step(val_mse)
        cur_lr_proj = opt_proj.param_groups[0]["lr"]
        cur_lr_lora = opt_lora.param_groups[0]["lr"]
        lr_decayed = (cur_lr_proj < prev_lr_proj) or (cur_lr_lora < prev_lr_lora)

        epoch_time = time.time() - epoch_start

        if is_rank0():
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"total={avg_total:.4e}  mse={avg_mse:.4e}  "
                f"pde={avg_pde:.4e}  ce={avg_ce:.4e} | "
                f"val_mse={val_mse:.4e} | "
                f"lr_proj={cur_lr_proj:.2e}  lr_lora={cur_lr_lora:.2e}"
                + (" [LR↓]" if lr_decayed else "") +
                f" | lambda_pde={out['lambda_pde']:.4f} | time={epoch_time:.1f}s"
            )
            if writer:
                writer.add_scalar("LR/projector", cur_lr_proj, epoch)
                writer.add_scalar("LR/lora",      cur_lr_lora, epoch)
            if writer:
                writer.add_scalar("Loss/Train_Total", avg_total, epoch)
                writer.add_scalar("Loss/Train_MSE",   avg_mse,   epoch)
                writer.add_scalar("Loss/Train_PDE",   avg_pde,   epoch)
                writer.add_scalar("Loss/Train_CE",    avg_ce,    epoch)
                writer.add_scalar("Loss/Val_MSE",     val_mse,   epoch)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_dir = os.path.join(config.output_dir, "best_model")
                model.save_pretrained(best_dir)
                print(f"  -> Best model saved (val_mse={best_val_mse:.4e})")

            if epoch % 10 == 0:
                save_dir = os.path.join(config.output_dir, f"epoch_{epoch}")
                model.save_pretrained(save_dir)

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

    if is_rank0() and writer:
        writer.close()
        summary = {
            "version":          "v2_dual_head",
            "num_epochs":       num_epochs,
            "best_val_mse":     best_val_mse,
            "config": {
                "batch_size":       batch_size,
                "grad_accum":       grad_accum,
                "lr_projector":     config.lr_projector,
                "lr_lora":          config.lr_lora,
                "lambda_pde":       config.lambda_pde,
                "lambda_ce":        config.lambda_ce,
                "latent_dim":       config.latent_dim,
                "max_seq_len":      config.max_seq_len,
            },
        }
        with open(os.path.join(config.output_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Training complete.  best_val_mse={best_val_mse:.4e}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args    = parse_args()
    local_rank = setup_ddp()

    config = Config()

    # Override from CLI
    batch_size  = args.batch_size     or config.batch_size
    num_epochs  = args.num_epochs     or config.num_epochs
    grad_accum  = args.grad_accum     or config.gradient_accumulation_steps
    if args.lr_projector     is not None: config.lr_projector     = args.lr_projector
    if args.lr_lora          is not None: config.lr_lora          = args.lr_lora
    if args.lambda_pde       is not None: config.lambda_pde       = args.lambda_pde
    if args.pde_use_gt_ref   is not None: config.pde_use_gt_ref   = bool(args.pde_use_gt_ref)
    if args.pde_warmup_start is not None: config.pde_warmup_start = args.pde_warmup_start
    if args.pde_warmup_end   is not None: config.pde_warmup_end   = args.pde_warmup_end
    if args.logging_steps    is not None: config.logging_steps    = args.logging_steps
    if args.unet_checkpoint  is not None: config.unet_checkpoint_path = args.unet_checkpoint
    if args.target_repeat_per_epoch is not None:
        config.target_repeat_per_epoch = max(1, int(args.target_repeat_per_epoch))
    if args.output_dir is not None:
        config.output_dir = args.output_dir
        os.makedirs(config.output_dir, exist_ok=True)
    # ---- backbone ablation flags ----
    if args.random_init_backbone   is not None: config.random_init_backbone   = bool(args.random_init_backbone)
    if args.full_finetune_backbone is not None: config.full_finetune_backbone = bool(args.full_finetune_backbone)
    if args.scratch_layers is not None and args.scratch_layers > 0:
        config.scratch_layers = args.scratch_layers
    if args.scratch_hidden is not None and args.scratch_hidden > 0:
        config.scratch_hidden = args.scratch_hidden
        config.qwen_hidden_dim = args.scratch_hidden   # keep projectors/heads in sync

    no_context_prob = 0.0 if args.no_context_prob is None else float(args.no_context_prob)
    no_context_prob = max(0.0, min(1.0, no_context_prob))
    if args.val_no_context_prob is None:
        val_no_context_prob = no_context_prob
    else:
        val_no_context_prob = float(args.val_no_context_prob)
    val_no_context_prob = max(0.0, min(1.0, val_no_context_prob))

    if is_rank0():
        print("=" * 60)
        print("V2 PDE Model Training")
        print("=" * 60)
        if dist.is_initialized():
            print(f"  World size:  {dist.get_world_size()}")
        print(f"  Phase:       {args.phase}")
        print(f"  Batch size:  {batch_size}  (per rank)")
        print(f"  Grad accum:  {grad_accum}")
        print(f"  Effective B: {batch_size * grad_accum * max(dist.get_world_size() if dist.is_initialized() else 1, 1)}")
        print(f"  Epochs:      {num_epochs}")
        print(f"  target_repeat_per_epoch: {config.target_repeat_per_epoch}")
        print(f"  no_context_prob: {no_context_prob:.2f}")
        print(f"  val_no_context_prob: {val_no_context_prob:.2f}")
        print(f"  lr_proj:     {config.lr_projector}")
        print(f"  lr_lora:     {config.lr_lora}")
        print(f"  lambda_pde:  {config.lambda_pde}")
        print(f"  latent_dim:  {config.latent_dim}")
        print(f"  max_seq_len: {config.max_seq_len}")

    if args.phase in ("phase1", "both"):
        train_unet_phase(config, num_epochs=args.phase1_epochs, batch_size=64,
                         lr=1e-3, local_rank=local_rank)

    if args.phase in ("phase2", "both"):
        if not os.path.exists(config.unet_checkpoint_path):
            raise FileNotFoundError(
                f"UNet checkpoint not found: {config.unet_checkpoint_path}\n"
                "Run phase1 first or supply --unet_checkpoint."
            )
        train_phase2(config, num_epochs=num_epochs, local_rank=local_rank,
                     batch_size=batch_size, grad_accum=grad_accum,
                     resume_from=args.resume_from,
                     no_context_prob=no_context_prob,
                     val_no_context_prob=val_no_context_prob)

    cleanup_ddp()


if __name__ == "__main__":
    main()
