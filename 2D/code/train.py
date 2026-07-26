"""
V3 Training Script — torchrun 8-GPU DDP
========================================

Phase 2 only: full V3PDEModel training (autoencoder pre-trained separately
via train_unet.py + run_train_unet.sh).

Usage:
    torchrun --nproc_per_node=8 --master_port=29500 train.py [OPTIONS]

Training strategy:
    - Joint MSE + PDE + CE loss
    - Two optimizer groups:
          group A (lr_projector): SpecialTokenEmbeddings, InputProjector,
                                  OutputProjector, DualHead
          group B (lr_lora):     Qwen LoRA parameters
    - ReduceLROnPlateau schedulers on validation MSE
    - Gradient accumulation + gradient sync across DDP ranks

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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from model.v3_model import V3PDEModel
from data.dataset import (PDEContextDataset, EpochShuffleSampler,
                           WeightedEpochSampler, make_collate_fn)
from data.sequence_builder import SequenceBuilder


# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="V3 PDE Model Training (phase2 only)")
    p.add_argument("--batch_size",     type=int,   default=None)
    p.add_argument("--num_epochs",     type=int,   default=None)
    p.add_argument("--lr_projector",   type=float, default=None)
    p.add_argument("--lr_lora",        type=float, default=None)
    p.add_argument("--lambda_mse",     type=float, default=None)
    p.add_argument("--lambda_pde",     type=float, default=None)
    p.add_argument("--pde_use_gt_ref", type=int, default=None,
                   help="1: penalize residual EXCESS over the AE-decoded GT reference; 0: raw residual")
    p.add_argument("--grad_accum",     type=int,   default=None)
    p.add_argument("--target_repeat_per_epoch", type=int, default=None)
    p.add_argument("--no_context_prob", type=float, default=None)
    p.add_argument("--val_no_context_prob", type=float, default=None)
    p.add_argument("--logging_steps",  type=int,   default=None)
    p.add_argument("--unet_checkpoint", type=str,  default=None,
                   help="Path to autoencoder2d.pt")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--resume_from", type=str, default=None,
                   help="Resume from checkpoint dir (e.g. checkpoints/best_model)")
    p.add_argument("--init_from_v2", type=str, default=None,
                   help="Init LoRA + projectors from v2 checkpoint dir "
                        "(e.g. checkpoints/v2_init). Ignored if --resume_from is set.")
    p.add_argument("--local_rank",     type=int,   default=-1)
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
# Phase 2: Full V3 model training
# -----------------------------------------------------------------------

def train_phase2(config: Config, num_epochs: int, local_rank: int,
                 batch_size: int, grad_accum: int, resume_from: str = None,
                 init_from_v2: str = None,
                 no_context_prob: float = 0.0,
                 val_no_context_prob: float = 0.0):
    if is_rank0():
        print("=" * 70)
        print("V3PDEModel training — 2D PDE (DDP)")
        print("=" * 70)

    # Load lookup and splits
    lookup      = torch.load(config.data_lookup_path, weights_only=False)
    train_p_idx = torch.load(config.train_p_idx_path, weights_only=False)
    val_p_idx   = torch.load(config.val_p_idx_path,   weights_only=False)

    # Extract FEM mesh data stored in the dataset
    mesh_coord      = lookup["coord"].float()        # [145, 2]
    mesh_elem       = lookup["elem"].long()          # [256, 3]
    mesh_free_nodes = lookup["free_nodes"].long()    # [n_free]

    # Build model (pass mesh so it can init canonicalize + PDE data)
    if resume_from:
        if is_rank0():
            print(f"  Resuming weights from: {resume_from}")
        model = V3PDEModel.from_pretrained(resume_from, config,
                                           mesh_coord=mesh_coord,
                                           mesh_elem=mesh_elem,
                                           mesh_free_nodes=mesh_free_nodes)
    else:
        model = V3PDEModel(config, unet_checkpoint=config.unet_checkpoint_path,
                           mesh_coord=mesh_coord,
                           mesh_elem=mesh_elem,
                           mesh_free_nodes=mesh_free_nodes)

        # --- Optionally initialize from v2 checkpoint ---
        if init_from_v2 and os.path.isdir(init_from_v2):
            if is_rank0():
                print(f"  [init_from_v2] Loading v2 weights from: {init_from_v2}")
            from peft import set_peft_model_state_dict
            import safetensors.torch as sf_torch

            # 1. LoRA adapter weights
            adapter_path = os.path.join(init_from_v2, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                adapter_state = sf_torch.load_file(adapter_path, device="cpu")
                incompatible = set_peft_model_state_dict(model.qwen_model, adapter_state)
                if is_rank0():
                    print(f"  [init_from_v2] LoRA loaded. Incompatible keys: {incompatible}")
            else:
                if is_rank0():
                    print(f"  [init_from_v2] WARNING: adapter_model.safetensors not found!")

            # 2. Projectors, dual head, special tokens
            dev_cpu = torch.device("cpu")
            for fname, module in [
                ("input_projector.pt",  model.input_proj),
                ("output_projector.pt", model.output_proj),
                ("dual_head.pt",        model.dual_head),
                ("special_tokens.pt",   model.special_tok),
            ]:
                fpath = os.path.join(init_from_v2, fname)
                if os.path.exists(fpath):
                    state = torch.load(fpath, map_location=dev_cpu, weights_only=True)
                    module.load_state_dict(state)
                    if is_rank0():
                        print(f"  [init_from_v2] {fname} loaded.")
                else:
                    if is_rank0():
                        print(f"  [init_from_v2] WARNING: {fname} not found, skipping.")
        elif init_from_v2:
            if is_rank0():
                print(f"  [init_from_v2] WARNING: directory not found: {init_from_v2}")

    dev = model.component_device

    # SequenceBuilder (uses frozen autoencoder; no gradient needed)
    seq_builder = SequenceBuilder(
        unet=model.autoencoder,
        input_proj=model.input_proj,
        special_tok=model.special_tok,
        latent_dim=config.latent_dim,
        hidden_dim=config.qwen_hidden_dim,
        max_seq_len=config.max_seq_len,
        device=dev,
        p_scale=config.p_input_scale,   # normalize p to ~[-0.09, 1.0]
    )
    collate_fn = make_collate_fn(
        seq_builder,
        max_seq_len=config.max_seq_len,
        solution_dim=config.solution_dim,
    )

    # Datasets
    train_ds = PDEContextDataset(lookup, train_p_idx,
                                 max_context_p=config.max_context_p,
                                 seed=config.sampler_seed,
                                 no_context_prob=no_context_prob)
    val_ds   = PDEContextDataset(lookup, val_p_idx,
                                 max_context_p=config.max_context_p,
                                 seed=config.sampler_seed + 1,
                                 no_context_prob=val_no_context_prob)

    # Weighted sampler: over-sample p values with more solutions
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
        print(f"  target_repeat_per_epoch: {config.target_repeat_per_epoch} "
              f"(global samples/epoch={global_samples_per_epoch})")
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
        ae_params = sum(p.numel() for p in model.autoencoder.parameters())
        print(f"  Autoencoder params: {ae_params:,}  (frozen)")

    opt_proj = optim.AdamW(projector_params, lr=config.lr_projector,
                           weight_decay=config.weight_decay)
    opt_lora = optim.AdamW(lora_params,      lr=config.lr_lora,
                           weight_decay=config.weight_decay)

    sched_proj = optim.lr_scheduler.ReduceLROnPlateau(
        opt_proj, mode="min",
        factor=config.lr_plateau_factor,
        patience=config.lr_plateau_patience,
        min_lr=config.lr_projector * config.lr_min_ratio,
    )
    sched_lora = optim.lr_scheduler.ReduceLROnPlateau(
        opt_lora, mode="min",
        factor=config.lr_plateau_factor,
        patience=config.lr_plateau_patience,
        min_lr=config.lr_lora * config.lr_min_ratio,
    )

    # TensorBoard (rank-0 only)
    writer = None
    if is_rank0():
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(config.log_dir, "phase2_v3")
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
        model.autoencoder.eval()   # autoencoder always eval (frozen)

        run_total = run_mse = run_pde = run_ce = 0.0
        n_batches = 0

        opt_proj.zero_grad()
        opt_lora.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            out  = model(batch, current_epoch=epoch)
            loss = out["total_loss"] / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                # Sync gradients across DDP ranks
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
                    writer.add_scalar("Loss/Total_Step", out["total_loss"].item(), global_step)
                    writer.add_scalar("Loss/MSE_Step",   out["mse_loss"].item(),   global_step)
                    writer.add_scalar("Loss/CE_Step",    out["ce_loss"].item(),    global_step)

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

        prev_lr_proj = opt_proj.param_groups[0]["lr"]
        prev_lr_lora = opt_lora.param_groups[0]["lr"]
        sched_proj.step(val_mse)
        sched_lora.step(val_mse)
        cur_lr_proj = opt_proj.param_groups[0]["lr"]
        cur_lr_lora = opt_lora.param_groups[0]["lr"]
        lr_decayed  = (cur_lr_proj < prev_lr_proj) or (cur_lr_lora < prev_lr_lora)

        epoch_time = time.time() - epoch_start

        if is_rank0():
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"total={avg_total:.4e}  mse={avg_mse:.4e}  pde={avg_pde:.4e}  ce={avg_ce:.4e} | "
                f"val_mse={val_mse:.4e} | "
                f"lr_proj={cur_lr_proj:.2e}  lr_lora={cur_lr_lora:.2e}"
                + (" [LR↓]" if lr_decayed else "") +
                f" | time={epoch_time:.1f}s",
                flush=True,
            )
            if writer:
                writer.add_scalar("Loss/Train_Total", avg_total, epoch)
                writer.add_scalar("Loss/Train_MSE",   avg_mse,   epoch)
                writer.add_scalar("Loss/Train_PDE",   avg_pde,   epoch)
                writer.add_scalar("Loss/Train_CE",    avg_ce,    epoch)
                writer.add_scalar("Loss/Val_MSE",     val_mse,   epoch)
                writer.add_scalar("LR/projector",     cur_lr_proj, epoch)
                writer.add_scalar("LR/lora",          cur_lr_lora, epoch)

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
            "version":      "v3_dual_head_2d",
            "num_epochs":   num_epochs,
            "best_val_mse": best_val_mse,
            "config": {
                "solution_dim":   config.solution_dim,
                "latent_dim":     config.latent_dim,
                "batch_size":     batch_size,
                "grad_accum":     grad_accum,
                "lr_projector":   config.lr_projector,
                "lr_lora":        config.lr_lora,
                "lambda_pde":     config.lambda_pde,
                "pde_use_gt_ref": getattr(config, "pde_use_gt_ref", True),
                "lambda_ce":      config.lambda_ce,
                "max_seq_len":    config.max_seq_len,
            },
        }
        with open(os.path.join(config.output_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Training complete.  best_val_mse={best_val_mse:.4e}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args       = parse_args()
    local_rank = setup_ddp()
    config     = Config()

    # Override from CLI
    batch_size = args.batch_size or config.batch_size
    num_epochs = args.num_epochs or config.num_epochs
    grad_accum = args.grad_accum or config.gradient_accumulation_steps
    if args.lr_projector  is not None: config.lr_projector  = args.lr_projector
    if args.lr_lora       is not None: config.lr_lora       = args.lr_lora
    if args.lambda_mse    is not None: config.lambda_mse    = args.lambda_mse
    if args.lambda_pde    is not None: config.lambda_pde    = args.lambda_pde
    if args.pde_use_gt_ref is not None: config.pde_use_gt_ref = bool(args.pde_use_gt_ref)
    if args.logging_steps is not None: config.logging_steps = args.logging_steps
    if args.unet_checkpoint is not None: config.unet_checkpoint_path = args.unet_checkpoint
    if args.target_repeat_per_epoch is not None:
        config.target_repeat_per_epoch = max(1, int(args.target_repeat_per_epoch))
    if args.output_dir is not None:
        config.output_dir = args.output_dir
        os.makedirs(config.output_dir, exist_ok=True)

    no_context_prob = 0.0 if args.no_context_prob is None else float(args.no_context_prob)
    no_context_prob = max(0.0, min(1.0, no_context_prob))
    if args.val_no_context_prob is None:
        val_no_context_prob = no_context_prob
    else:
        val_no_context_prob = float(args.val_no_context_prob)
    val_no_context_prob = max(0.0, min(1.0, val_no_context_prob))

    if is_rank0():
        print("=" * 60)
        print("V3 PDE Model Training — 2D Bifurcation (145-node FEM)")
        print("=" * 60)
        if dist.is_initialized():
            print(f"  World size:  {dist.get_world_size()}")
        print(f"  Batch size:  {batch_size}  (per rank)")
        print(f"  Grad accum:  {grad_accum}")
        eff_b = batch_size * grad_accum * max(dist.get_world_size() if dist.is_initialized() else 1, 1)
        print(f"  Effective B: {eff_b}")
        print(f"  Epochs:      {num_epochs}")
        print(f"  solution_dim:{config.solution_dim}")
        print(f"  latent_dim:  {config.latent_dim}")
        print(f"  max_seq_len: {config.max_seq_len}")
        print(f"  lr_proj:     {config.lr_projector}")
        print(f"  lr_lora:     {config.lr_lora}")
        print(f"  lambda_pde:  {config.lambda_pde}  (GT-ref residual excess={config.pde_use_gt_ref}, warmup {config.pde_warmup_start}→{config.pde_warmup_end})")
        print(f"  no_context_prob: {no_context_prob:.2f}")

    if not os.path.exists(config.unet_checkpoint_path):
        raise FileNotFoundError(
            f"SolutionAutoencoder2D checkpoint not found: {config.unet_checkpoint_path}\n"
            "Run 'sbatch run_train_unet.sh' first (Phase 1)."
        )

    train_phase2(config, num_epochs=num_epochs, local_rank=local_rank,
                 batch_size=batch_size, grad_accum=grad_accum,
                 resume_from=args.resume_from,
                 init_from_v2=args.init_from_v2,
                 no_context_prob=no_context_prob,
                 val_no_context_prob=val_no_context_prob)

    cleanup_ddp()


if __name__ == "__main__":
    main()
