#!/usr/bin/env python3
"""
Simplify bvp_complete.pt:
  - Keep only k in {2, 4, 7, 8}
  - For k=8 cells: drop the solution with the largest residual → k becomes 7
  - Result: k in {2, 4, 7}  (three-class dataset)

Residual (same formula as newton_on_device):
  Lu[0]    = 2*u[0] - 2*u[1]
  Lu[i]    = 2*u[i] - u[i-1] - u[i+1]
  Lu[N-1]  = 2*u[N-1] - u[N-2]
  Fu = Lu/h + h*(a4*u^4 + a2*u^2),   h = 1/N,  residual = ||Fu||_2
"""

import os
import numpy as np
import torch
from collections import Counter

IN_PATH  = "./data/bvp_complete.pt"
OUT_PATH = "./data/bvp_simplified.pt"
N = 1024
H = 1.0 / N

KEEP_K = {2, 4, 7, 8}


def residual_norm(u, a4, a2):
    """Compute ||Fu||_2 for a single solution array u of shape (N,)."""
    Lu = np.empty(N, dtype=np.float64)
    Lu[0]    = 2*u[0] - 2*u[1]
    Lu[1:-1] = 2*u[1:-1] - u[:-2] - u[2:]
    Lu[-1]   = 2*u[-1]  - u[-2]
    Fu = Lu / H + H * (a4 * u**4 + a2 * u**2)
    return np.linalg.norm(Fu)


def process(raw):
    out = []
    k8_residuals = []   # collect residuals of dropped solutions for diagnostics

    for d in raw:
        a4, a2 = float(d['params'][0]), float(d['params'][1])
        sols = d['solutions']
        k = len(sols)

        if k not in KEEP_K:
            continue

        if k == 8:
            res = [residual_norm(s, a4, a2) for s in sols]
            worst = int(np.argmax(res))
            k8_residuals.append((max(res), sorted(res)))
            kept = [s for i, s in enumerate(sols) if i != worst]
            out.append({'params': d['params'], 'solutions': kept})
        else:
            out.append(d)

    return out, k8_residuals


def main():
    print(f"Loading {IN_PATH} …", flush=True)
    raw = torch.load(IN_PATH, weights_only=False)
    print(f"  {len(raw):,} total records", flush=True)

    k_dist_in = Counter(len(d['solutions']) for d in raw)
    print(f"  Input k-distribution: {dict(sorted(k_dist_in.items()))}", flush=True)

    print("Processing …", flush=True)
    out, k8_res = process(raw)

    k_dist_out = Counter(len(d['solutions']) for d in out)
    print(f"  Output k-distribution: {dict(sorted(k_dist_out.items()))}", flush=True)
    print(f"  Total records kept: {len(out):,}", flush=True)

    if k8_res:
        dropped_res = [r[0] for r in k8_res]
        second_res  = [r[1][-2] for r in k8_res]  # second-worst
        print(f"\n  k=8 → k=7 (dropped worst solution):")
        print(f"    Dropped residual  — min={min(dropped_res):.2e}  "
              f"max={max(dropped_res):.2e}  mean={np.mean(dropped_res):.2e}")
        print(f"    2nd-worst residual— min={min(second_res):.2e}  "
              f"max={max(second_res):.2e}  mean={np.mean(second_res):.2e}")
        ratio = np.array(dropped_res) / np.array(second_res)
        print(f"    Ratio dropped/2nd — min={ratio.min():.2f}  "
              f"max={ratio.max():.2f}  mean={ratio.mean():.2f}", flush=True)

    print(f"\nSaving to {OUT_PATH} …", flush=True)
    torch.save(out, OUT_PATH)
    size_gb = os.path.getsize(OUT_PATH) / 1e9
    print(f"Done. File size: {size_gb:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
