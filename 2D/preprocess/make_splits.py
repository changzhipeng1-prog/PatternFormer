"""Preprocessing for 2D: filtered lookup -> train/val/test split indices.

Consumes the D4-filtered lookup produced by data_gen/filter_d4_dataset.py:

    p_solutions_lookup_2d_filtered.pt
        dict{ p_values [N_s], solutions_by_p list[ Tensor[K_i,145] ],
              coord [145,2], elem [256,3], free_nodes [113], dbc [32] }

Produces (matching qwen_pde/config.py expectations):
    train_p_idx.pt / val_p_idx.pt / test_p_idx.pt   LongTensor over s-indices
    split_info.pt                                    dict of split metadata

Recipe (identical to qwen_pde/train_unet.py:load_and_split): shuffle the N_s
s-values with numpy default_rng(seed) and split 70/15/15.

NOTE: the canonical splits used for the paper are the shipped *_p_idx.pt files.
This script reproduces the same recipe.

Usage:
    python make_splits.py --lookup ../data/p_solutions_lookup_2d_filtered.pt --out_dir ../data
"""
import os
import argparse
import numpy as np
import torch


def make_splits(lookup_path, out_dir, train_ratio=0.70, val_ratio=0.15, seed=42):
    ds = torch.load(lookup_path, weights_only=False)
    p_values = ds["p_values"]
    N_s = len(p_values)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_s)
    n_train = int(N_s * train_ratio)
    n_val = int(N_s * val_ratio)
    train_idx = torch.from_numpy(perm[:n_train].copy()).long()
    val_idx = torch.from_numpy(perm[n_train:n_train + n_val].copy()).long()
    test_idx = torch.from_numpy(perm[n_train + n_val:].copy()).long()

    os.makedirs(out_dir, exist_ok=True)
    torch.save(train_idx, os.path.join(out_dir, "train_p_idx.pt"))
    torch.save(val_idx, os.path.join(out_dir, "val_p_idx.pt"))
    torch.save(test_idx, os.path.join(out_dir, "test_p_idx.pt"))
    torch.save({
        "train_p_idx": train_idx, "val_p_idx": val_idx, "test_p_idx": test_idx,
        "p_values": p_values, "split_seed": seed,
        "train_ratio": train_ratio, "val_ratio": val_ratio,
    }, os.path.join(out_dir, "split_info.pt"))
    print(f"split {N_s} s-values -> "
          f"train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", default="../data/p_solutions_lookup_2d_filtered.pt")
    ap.add_argument("--out_dir", default="../data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    make_splits(args.lookup, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
