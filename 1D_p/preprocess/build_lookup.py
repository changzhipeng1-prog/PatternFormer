"""Preprocessing for 1D_p: branch continuation outputs -> training lookup + splits.

Consumes the per-branch arrays written by data_gen/cbmfem/run_branch.py:
    branch_{b}_p_new.npy   [T_b]        parameter p along branch b
    branch_{b}_S_new.npy   [1024, T_b]  solution u(x) at each p on branch b

Produces (matching qwen_pde/config.py expectations):
    p_solutions_lookup.pt  dict{p_values [P], solutions_by_p list[ Tensor[K_i,1024] ], ...}
    p_solutions_meta.pt    dict of dataset statistics
    train_p_idx.pt / val_p_idx.pt / test_p_idx.pt   LongTensor split indices

Recipe (recorded in p_solutions_meta.pt of the shipped dataset):
    p_round_decimals=3, drop_zero_solution=True, zero_sol_threshold=1e-4,
    train/val/test = 0.70/0.15/0.15, seed=42.

NOTE: the canonical splits used for the paper are the shipped *_p_idx.pt files.
This script reproduces the same recipe end-to-end; rerun it to regenerate the
lookup from raw branch outputs.

Usage:
    python build_lookup.py --branch_dir ../data_gen/cbmfem --out_dir ../data
"""
import os
import argparse
import numpy as np
import torch


def build(branch_dir, out_dir, num_branches=8, solution_dim=1024,
          p_round_decimals=3, drop_zero_solution=True, zero_sol_threshold=1e-4,
          dedup_l2=1e-3, train_ratio=0.70, val_ratio=0.15, seed=42):
    # ---- 1. load all branches, collect (rounded_p, solution) pairs ----
    by_p = {}
    loaded = []
    for b in range(num_branches):
        pf = os.path.join(branch_dir, f"branch_{b}_p_new.npy")
        sf = os.path.join(branch_dir, f"branch_{b}_S_new.npy")
        if not (os.path.exists(pf) and os.path.exists(sf)):
            continue
        loaded.append(b)
        p_arr = np.load(pf).reshape(-1).astype(np.float64)
        S = np.load(sf)
        if S.shape[0] == solution_dim:        # [1024, T] -> [T, 1024]
            S = S.T
        for p, u in zip(p_arr, S.astype(np.float64)):
            pr = round(float(p), p_round_decimals)
            if drop_zero_solution and np.linalg.norm(u) < zero_sol_threshold:
                continue
            bucket = by_p.setdefault(pr, [])
            if dedup_l2 and any(np.linalg.norm(u - v) < dedup_l2 for v in bucket):
                continue
            bucket.append(u)

    # ---- 2. assemble sorted lookup ----
    p_values = sorted(by_p.keys())
    solutions_by_p = [torch.tensor(np.stack(by_p[p]), dtype=torch.float32) for p in p_values]
    ks = [s.shape[0] for s in solutions_by_p]
    lookup = {
        "format_version": 1,
        "p_values": torch.tensor(p_values, dtype=torch.float32),
        "solutions_by_p": solutions_by_p,
    }

    # ---- 3. train/val/test split over p-indices ----
    P = len(p_values)
    perm = np.random.RandomState(seed).permutation(P)
    n_tr = int(train_ratio * P)
    n_va = int(val_ratio * P)
    train_idx = torch.tensor(perm[:n_tr], dtype=torch.long)
    val_idx = torch.tensor(perm[n_tr:n_tr + n_va], dtype=torch.long)
    test_idx = torch.tensor(perm[n_tr + n_va:], dtype=torch.long)

    meta = {
        "loaded_branches": loaded, "num_unique_p": P,
        "p_min": float(min(p_values)) if P else None,
        "p_max": float(max(p_values)) if P else None,
        "num_solutions_total": int(sum(ks)),
        "num_solutions_per_p_min": int(min(ks)) if ks else 0,
        "num_solutions_per_p_max": int(max(ks)) if ks else 0,
        "num_solutions_per_p_mean": float(np.mean(ks)) if ks else 0.0,
        "train_ratio": train_ratio, "val_ratio": val_ratio,
        "test_ratio": round(1 - train_ratio - val_ratio, 6),
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "drop_zero_solution": drop_zero_solution, "zero_sol_threshold": zero_sol_threshold,
        "seed": seed, "p_round_decimals": p_round_decimals,
    }

    os.makedirs(out_dir, exist_ok=True)
    torch.save(lookup, os.path.join(out_dir, "p_solutions_lookup.pt"))
    torch.save(meta, os.path.join(out_dir, "p_solutions_meta.pt"))
    torch.save(train_idx, os.path.join(out_dir, "train_p_idx.pt"))
    torch.save(val_idx, os.path.join(out_dir, "val_p_idx.pt"))
    torch.save(test_idx, os.path.join(out_dir, "test_p_idx.pt"))
    print(f"built lookup: {P} unique p, {sum(ks)} solutions; "
          f"split {len(train_idx)}/{len(val_idx)}/{len(test_idx)} -> {out_dir}")
    return lookup, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch_dir", default="../data_gen/cbmfem")
    ap.add_argument("--out_dir", default="../data")
    ap.add_argument("--num_branches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.branch_dir, args.out_dir, num_branches=args.num_branches, seed=args.seed)


if __name__ == "__main__":
    main()
