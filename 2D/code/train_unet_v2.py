"""
train_unet_v2.py — Phase 1 v2: SolutionAutoencoder2D (upgraded)
================================================================
Upgraded architecture vs v1:
    latent_dim   : 256 → 512   (wider bottleneck)
    hidden_dim   : 512 → 1024  (wider MLP)
    depth        : 3 blocks → 4 blocks per encoder/decoder

Saves to checkpoints/autoencoder2d_v2.pt (does NOT overwrite v1).

Usage:
    python train_unet_v2.py
    sbatch run_train_unet_v2.sh
"""

import os, sys, argparse, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model.autoencoder2d import _MLPBlock

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'p_solutions_lookup_2d.pt')
CKPT_DIR     = './checkpoints/best_model'
CKPT_NAME    = 'autoencoder2d_v2.pt'
DATA_DIR     = './data'
LOG_DIR      = './logs'

SOLUTION_DIM = 145
LATENT_DIM   = 512    # 256 → 512
HIDDEN_DIM   = 1024   # 512 → 1024
NUM_BLOCKS   = 4      # 3  → 4  per encoder/decoder

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15

BATCH_SIZE   = 256
NUM_EPOCHS   = 500
LR_INIT      = 1e-3
LR_MIN       = 1e-5
WEIGHT_DECAY = 1e-4
LOG_EVERY    = 10


# ── Upgraded autoencoder ──────────────────────────────────────────────────────
class SolutionAutoencoder2D_v2(nn.Module):
    """
    Deeper, wider MLP autoencoder.
        Encoder: 145 → 1024 × 4 → 512
        Decoder: 512 → 1024 × 4 → 145
    """

    def __init__(self):
        super().__init__()
        self.solution_dim = SOLUTION_DIM
        self.latent_dim   = LATENT_DIM
        self.hidden_dim   = HIDDEN_DIM

        enc_blocks = [_MLPBlock(SOLUTION_DIM, HIDDEN_DIM)]
        for _ in range(NUM_BLOCKS - 1):
            enc_blocks.append(_MLPBlock(HIDDEN_DIM, HIDDEN_DIM))
        enc_blocks += [nn.Linear(HIDDEN_DIM, LATENT_DIM), nn.LayerNorm(LATENT_DIM)]
        self.encoder = nn.Sequential(*enc_blocks)

        dec_blocks = [_MLPBlock(LATENT_DIM, HIDDEN_DIM)]
        for _ in range(NUM_BLOCKS - 1):
            dec_blocks.append(_MLPBlock(HIDDEN_DIM, HIDDEN_DIM))
        dec_blocks.append(nn.Linear(HIDDEN_DIM, SOLUTION_DIM))
        self.decoder = nn.Sequential(*dec_blocks)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'encode':
            return self._encode(x)
        elif mode == 'decode':
            return self._decode(x)
        raise ValueError(f"Unknown mode '{mode}'")

    def _encode(self, x):
        seq = x.dim() == 3
        if seq:
            B, T, D = x.shape
            x = x.reshape(B * T, D)
        z = self.encoder(x)
        if seq:
            z = z.view(B, T, self.latent_dim)
        return z

    def _decode(self, z):
        seq = z.dim() == 3
        if seq:
            B, T, L = z.shape
            z = z.reshape(B * T, L)
        x = self.decoder(z)
        if seq:
            x = x.view(B, T, self.solution_dim)
        return x


# ── Dataset ───────────────────────────────────────────────────────────────────
class SolutionDataset(Dataset):
    def __init__(self, solutions: torch.Tensor):
        self.data = solutions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_and_split(seed=42):
    print(f"Loading dataset from:\n  {DATA_PATH}")
    ds = torch.load(DATA_PATH, weights_only=False)

    p_values       = ds['p_values']
    solutions_by_p = ds['solutions_by_p']
    N_s = len(p_values)
    print(f"  {N_s} s-values, "
          f"{sum(s.shape[0] for s in solutions_by_p)} total solutions")

    rng = np.random.default_rng(seed)
    n_train = int(N_s * TRAIN_RATIO)
    n_val   = int(N_s * VAL_RATIO)
    perm    = rng.permutation(N_s)

    # Reuse existing split indices if present so AE v2 uses same split as v1
    train_idx_path = os.path.join(DATA_DIR, 'train_p_idx.pt')
    val_idx_path   = os.path.join(DATA_DIR, 'val_p_idx.pt')
    test_idx_path  = os.path.join(DATA_DIR, 'test_p_idx.pt')
    if os.path.exists(train_idx_path):
        train_idx = torch.load(train_idx_path, weights_only=True)
        val_idx   = torch.load(val_idx_path,   weights_only=True)
        test_idx  = torch.load(test_idx_path,  weights_only=True)
        print("  Reusing existing data split indices from ./data/")
    else:
        train_idx = torch.from_numpy(perm[:n_train].copy()).long()
        val_idx   = torch.from_numpy(perm[n_train:n_train+n_val].copy()).long()
        test_idx  = torch.from_numpy(perm[n_train+n_val:].copy()).long()
        os.makedirs(DATA_DIR, exist_ok=True)
        torch.save(train_idx, train_idx_path)
        torch.save(val_idx,   val_idx_path)
        torch.save(test_idx,  test_idx_path)

    print(f"  Split: train={len(train_idx)} / val={len(val_idx)} / test={len(test_idx)} s-values")

    def collect(idx_tensor):
        parts = [solutions_by_p[i].float() for i in idx_tensor.tolist()]
        return torch.cat(parts, dim=0)

    train_sols = collect(train_idx)
    val_sols   = collect(val_idx)
    test_sols  = collect(test_idx)
    print(f"  Solution counts: train={len(train_sols)} / val={len(val_sols)} / test={len(test_sols)}")
    return train_sols, val_sols, test_sols


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"SolutionAutoencoder2D v2 — Upgraded Architecture")
    print(f"  solution_dim={SOLUTION_DIM}, latent_dim={LATENT_DIM}, "
          f"hidden_dim={HIDDEN_DIM}, blocks={NUM_BLOCKS}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    train_sols, val_sols, test_sols = load_and_split(seed=args.seed)

    train_ds = SolutionDataset(train_sols)
    val_ds   = SolutionDataset(val_sols)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4,
                              pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = SolutionAutoencoder2D_v2().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}\n")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=LR_MIN)

    best_val_loss = float('inf')
    ckpt_path = os.path.join(CKPT_DIR, CKPT_NAME)
    log_path  = os.path.join(LOG_DIR, 'unet_v2_train.jsonl')
    log_f = open(log_path, 'w')

    print(f"{'Epoch':>6}  {'LR':>10}  {'TrainMSE':>12}  {'ValMSE':>12}  "
          f"{'BestVal':>12}  {'Time':>6}")
    print("-"*65)

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for batch in train_loader:
            u = batch.to(device)
            z = model(u, 'encode')
            u_rec = model(z, 'decode')
            loss = nn.functional.mse_loss(u_rec, u)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(u)
        train_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                u = batch.to(device)
                z = model(u, 'encode')
                u_rec = model(z, 'decode')
                val_loss += nn.functional.mse_loss(u_rec, u).item() * len(u)
        val_loss /= len(val_ds)

        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

        log_f.write(json.dumps({
            'epoch': epoch, 'lr': lr_now,
            'train_mse': train_loss, 'val_mse': val_loss,
            'best_val_mse': best_val_loss, 'time_s': elapsed,
        }) + '\n')
        log_f.flush()

        if epoch % LOG_EVERY == 0 or epoch == 1 or is_best:
            best_mark = ' ★' if is_best else ''
            print(f"{epoch:>6}  {lr_now:>10.2e}  {train_loss:>12.6f}  "
                  f"{val_loss:>12.6f}  {best_val_loss:>12.6f}  "
                  f"{elapsed:>5.1f}s{best_mark}", flush=True)

    log_f.close()

    # Final test evaluation
    print(f"\n{'='*60}")
    print("Final evaluation on test set...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
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
            max_err = max(max_err, (u_rec - u).abs().max().item())
    test_loss /= len(test_ds)

    print(f"  Test MSE     : {test_loss:.6f}")
    print(f"  Test MaxErr  : {max_err:.6f}")
    print(f"  Best val MSE : {best_val_loss:.6f}")
    print(f"\nCheckpoint saved: {ckpt_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--num_epochs', type=int,   default=NUM_EPOCHS)
    p.add_argument('--batch_size', type=int,   default=BATCH_SIZE)
    p.add_argument('--lr',         type=float, default=LR_INIT)
    p.add_argument('--seed',       type=int,   default=42)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
