"""
train_unet.py — Phase 1: SolutionAutoencoder2D pretraining
===========================================================
Trains the MLP autoencoder on all 11198 solutions from p_solutions_lookup_2d.pt.

Split (by s-value, stratified):
    train 70% / val 15% / test 15%

Loss: MSE reconstruction  ||decode(encode(u)) - u||^2

Saves:
    checkpoints/best_model/autoencoder2d.pt   ← best val loss
    data/train_p_idx.pt / val_p_idx.pt / test_p_idx.pt
    data/split_info.pt                        ← split metadata

Usage (single GPU):
    python train_unet.py

Usage (SLURM, single GPU):
    sbatch run_train_unet.sh
"""

import os, sys, argparse, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))
from model.autoencoder2d import SolutionAutoencoder2D

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'p_solutions_lookup_2d.pt')
CKPT_DIR   = './checkpoints/best_model'
DATA_DIR   = './data'
LOG_DIR    = './logs'

SOLUTION_DIM = 145
LATENT_DIM   = 256
HIDDEN_DIM   = 512

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO  = 0.15  (remainder)

BATCH_SIZE   = 256
NUM_EPOCHS   = 500
LR_INIT      = 1e-3
LR_MIN       = 1e-5
WEIGHT_DECAY = 1e-4
LOG_EVERY    = 10     # epochs


# ── Dataset ───────────────────────────────────────────────────────────────────
class SolutionDataset(Dataset):
    """
    Flat dataset: each item is a single [145] solution vector.
    Loaded in float32 for training (original float64 → converted).
    """
    def __init__(self, solutions: torch.Tensor):
        # solutions: [N, 145]  float32
        self.data = solutions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]   # [145]


def load_and_split(seed=42):
    """
    Load dataset, split by s-value (70/15/15 stratified),
    return flat solution tensors for each split.
    """
    print(f"Loading dataset from:\n  {DATA_PATH}")
    ds = torch.load(DATA_PATH, weights_only=False)

    p_values      = ds['p_values']           # [N_s] float32
    solutions_by_p = ds['solutions_by_p']    # list of [K_i, 145] float64

    N_s = len(p_values)
    print(f"  {N_s} s-values, "
          f"{sum(s.shape[0] for s in solutions_by_p)} total solutions")

    # Stratified split: keep proportions of 2-sol vs 10-sol s-values
    rng = np.random.default_rng(seed)
    idx_all = np.arange(N_s)

    n_train = int(N_s * TRAIN_RATIO)
    n_val   = int(N_s * VAL_RATIO)
    n_test  = N_s - n_train - n_val

    # Shuffle and split
    perm = rng.permutation(N_s)
    train_idx = torch.from_numpy(perm[:n_train].copy()).long()
    val_idx   = torch.from_numpy(perm[n_train:n_train+n_val].copy()).long()
    test_idx  = torch.from_numpy(perm[n_train+n_val:].copy()).long()

    print(f"  Split: train={len(train_idx)} / val={len(val_idx)} / test={len(test_idx)} s-values")

    # Save split indices
    os.makedirs(DATA_DIR, exist_ok=True)
    torch.save(train_idx, os.path.join(DATA_DIR, 'train_p_idx.pt'))
    torch.save(val_idx,   os.path.join(DATA_DIR, 'val_p_idx.pt'))
    torch.save(test_idx,  os.path.join(DATA_DIR, 'test_p_idx.pt'))
    torch.save({
        'train_p_idx': train_idx, 'val_p_idx': val_idx, 'test_p_idx': test_idx,
        'p_values': p_values, 'split_seed': seed,
        'train_ratio': TRAIN_RATIO, 'val_ratio': VAL_RATIO,
    }, os.path.join(DATA_DIR, 'split_info.pt'))
    print(f"  Saved split indices to {DATA_DIR}/")

    # Collect flat solution tensors per split (float32 for training)
    def collect(idx_tensor):
        parts = []
        for i in idx_tensor.tolist():
            sols = solutions_by_p[i].float()   # [K, 145] float64 → float32
            parts.append(sols)
        return torch.cat(parts, dim=0)         # [N_sols, 145]

    train_sols = collect(train_idx)
    val_sols   = collect(val_idx)
    test_sols  = collect(test_idx)

    print(f"  Solution counts: train={len(train_sols)} / "
          f"val={len(val_sols)} / test={len(test_sols)}")
    return train_sols, val_sols, test_sols


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Phase 1: SolutionAutoencoder2D Pretraining")
    print(f"Device : {device}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    train_sols, val_sols, test_sols = load_and_split(seed=args.seed)

    train_ds = SolutionDataset(train_sols)
    val_ds   = SolutionDataset(val_sols)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4,
                              pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4,
                              pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = SolutionAutoencoder2D(
        solution_dim=SOLUTION_DIM,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model  : SolutionAutoencoder2D  ({total_params:,} params)")
    print(f"  solution_dim={SOLUTION_DIM}, latent_dim={LATENT_DIM}, hidden={HIDDEN_DIM}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=LR_MIN)

    # ── Training ──────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    log_path = os.path.join(LOG_DIR, 'unet_train.jsonl')
    log_f = open(log_path, 'w')

    print(f"\n{'Epoch':>6}  {'LR':>10}  {'TrainMSE':>12}  {'ValMSE':>12}  "
          f"{'BestVal':>12}  {'Time':>6}")
    print("-"*65)

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            u = batch.to(device)                  # [B, 145]
            z = model(u, 'encode')                # [B, latent]
            u_rec = model(z, 'decode')            # [B, 145]
            loss = nn.functional.mse_loss(u_rec, u)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(u)

        train_loss /= len(train_ds)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                u = batch.to(device)
                z = model(u, 'encode')
                u_rec = model(z, 'decode')
                val_loss += nn.functional.mse_loss(u_rec, u).item() * len(u)
        val_loss /= len(val_ds)

        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # Save best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(CKPT_DIR, 'autoencoder2d.pt'))

        # Log
        record = {
            'epoch': epoch, 'lr': lr_now,
            'train_mse': train_loss, 'val_mse': val_loss,
            'best_val_mse': best_val_loss, 'time_s': elapsed,
        }
        log_f.write(json.dumps(record) + '\n')
        log_f.flush()

        if epoch % LOG_EVERY == 0 or epoch == 1 or is_best:
            best_mark = ' ★' if is_best else ''
            print(f"{epoch:>6}  {lr_now:>10.2e}  {train_loss:>12.6f}  "
                  f"{val_loss:>12.6f}  {best_val_loss:>12.6f}  "
                  f"{elapsed:>5.1f}s{best_mark}", flush=True)

    log_f.close()

    # ── Final evaluation on test set ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("Final evaluation on test set...")
    model.load_state_dict(
        torch.load(os.path.join(CKPT_DIR, 'autoencoder2d.pt'),
                   map_location=device, weights_only=True))
    model.eval()
    test_ds     = SolutionDataset(test_sols)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)
    test_loss = 0.0
    max_err   = 0.0
    with torch.no_grad():
        for batch in test_loader:
            u = batch.to(device)
            z = model(u, 'encode')
            u_rec = model(z, 'decode')
            test_loss += nn.functional.mse_loss(u_rec, u).item() * len(u)
            max_err = max(max_err,
                          (u_rec - u).abs().max().item())
    test_loss /= len(test_ds)

    print(f"  Test MSE     : {test_loss:.6f}")
    print(f"  Test MaxErr  : {max_err:.6f}")
    print(f"  Best val MSE : {best_val_loss:.6f}")
    print(f"\nCheckpoint: {CKPT_DIR}/autoencoder2d.pt")
    print(f"Log       : {log_path}")
    print("Done.")


# ── Entry point ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--num_epochs', type=int,   default=NUM_EPOCHS)
    p.add_argument('--batch_size', type=int,   default=BATCH_SIZE)
    p.add_argument('--lr',         type=float, default=LR_INIT)
    p.add_argument('--seed',       type=int,   default=42)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
