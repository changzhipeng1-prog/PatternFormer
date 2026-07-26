"""Preprocessing for a2a4: region dataset -> train/val/test splits.

Consumes the curated region dataset produced by the data generator
(data_gen/generate_fullspace_gpu.py -> simplify_dataset.py -> region selection):

    bvp_region.pt   list[ {"params": (a4, a2), "solutions": [arr1024, ...]} ]

Produces (matching qwen_pde/config.py expectations):
    bvp_region_train.pt / bvp_region_val.pt / bvp_region_test.pt

Recipe: random 80/10/10 split, seed=42.

NOTE: the canonical splits used for the paper are the shipped bvp_region_*.pt
files. This script reproduces the same recipe.

Usage:
    python split_dataset.py --region ../data/bvp_region.pt --out_dir ../data
"""
import os
import argparse
import numpy as np
import torch


def split(region_path, out_dir, train_ratio=0.80, val_ratio=0.10, seed=42):
    data = torch.load(region_path, map_location="cpu", weights_only=False)
    N = len(data)
    perm = np.random.RandomState(seed).permutation(N)
    n_tr = int(train_ratio * N)
    n_va = int(val_ratio * N)
    splits = {
        "train": perm[:n_tr],
        "val": perm[n_tr:n_tr + n_va],
        "test": perm[n_tr + n_va:],
    }
    os.makedirs(out_dir, exist_ok=True)
    for name, idx in splits.items():
        subset = [data[int(i)] for i in idx]
        torch.save(subset, os.path.join(out_dir, f"bvp_region_{name}.pt"))
    print(f"split {N} samples -> "
          f"train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])} "
          f"-> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="../data/bvp_region.pt")
    ap.add_argument("--out_dir", default="../data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    split(args.region, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
