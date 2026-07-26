"""a2a4 coverage, stepped continuation seeded from the MODEL (BFS, reference-matched).

This is the honest analog of the reference solver: the ground-truth extension grid
(grid_gt_continuation.py) seeds the in-box cells from the dataset and then propagates
OUTWARD by BFS Newton continuation from every solved neighbour (8-connectivity).  Here
we run the IDENTICAL propagation but seed the in-box cells from the MODEL's own raw
output (grid_direct.pt) instead of the dataset.  Because the model's in-box predictions
already cover the in-box solution set, a multi-directional continuation reaches the same
cells the GT does -- including the fold-born branches at the a4->0 edge that a single
straight-line march approaches from the wrong direction and misses.

Saves a2a4_cov_M3_bfs.pt (same schema as a2a4_cov_M3.pt) and prints the coverage table.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from model.newton_fdm import newton_refine, residual_l2

N = 1024
H = 1.0 / N
COVER, DEDUP, REFINE_OK = 1e-2, 1e-3, 1e-6
A2_LO, A2_HI, A4_LO, A4_HI = -14.934, -2.0, 0.301, 0.798
MAXIT = 80


def rl2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def in_box(a2, a4):
    return (A2_LO <= a2 <= A2_HI) and (A4_LO <= a4 <= A4_HI)


def refine_one(u, a4, a2):
    ru, ok, _ = newton_refine(np.asarray(u, np.float64).reshape(-1), a4, a2,
                              h=H, max_iter=MAXIT, abs_tol=1e-9)
    ru = np.asarray(ru, np.float64).reshape(-1)
    return ru if (ok and residual_l2(ru, a4, a2, H) < REFINE_OK and np.linalg.norm(ru) > 1e-6) else None


def dedup(sols):
    out = []
    for u in sols:
        if u is not None and not any(rl2(u, v) < DEDUP for v in out):
            out.append(u)
    return out


def main():
    G = torch.load(os.path.join(HERE, "grid_direct.pt"), map_location="cpu", weights_only=False)
    GT = torch.load(os.path.join(HERE, "grid_gt_cont.pt"), map_location="cpu", weights_only=False)
    A2 = np.array(G["a2"], float); A4 = np.array(G["a4"], float)

    # ---- seed the in-box cells from the MODEL's raw output, refined at the cell ----
    solved = {}
    for i, a2 in enumerate(A2):
        for j, a4 in enumerate(A4):
            if not in_box(float(a2), float(a4)):
                continue
            seeds = G["gen"][(float(a2), float(a4))].numpy().astype(np.float64)
            if seeds.ndim == 1:
                seeds = seeds[None]
            sols = dedup([refine_one(s, float(a4), float(a2)) for s in seeds])
            if sols:
                solved[(i, j)] = sols
    print(f"in-box cells seeded from model: {len(solved)}  k={[len(v) for v in solved.values()]}", flush=True)

    # ---- BFS Newton continuation outward, identical to the reference (grid_gt_continuation) ----
    nbr = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for rnd in range(40):
        newly = 0
        for i, a2 in enumerate(A2):
            for j, a4 in enumerate(A4):
                if (i, j) in solved:
                    continue
                inits = []
                for di, dj in nbr:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < len(A2) and 0 <= nj < len(A4) and (ni, nj) in solved:
                        inits.extend(solved[(ni, nj)])
                if not inits:
                    continue
                sols = dedup([refine_one(s, float(a4), float(a2)) for s in inits])
                if sols:
                    solved[(i, j)] = sols
                    newly += 1
        print(f"  BFS round {rnd}: +{newly} (total {len(solved)}/{len(A2)*len(A4)})", flush=True)
        if newly == 0:
            break

    # ---- score coverage against the GT continuation grid ----
    cov = np.full((len(A4), len(A2)), np.nan)
    rows = []
    for i, a2 in enumerate(A2):
        for j, a4 in enumerate(A4):
            gt = np.asarray(GT["gt"][(float(a2), float(a4))], np.float64)
            k = gt.shape[0]
            if k == 0:
                continue
            distinct = solved.get((i, j), [])
            c = sum(1 for g in gt if any(rl2(d, g) < COVER for d in distinct))
            cov[j, i] = c / k
            rows.append((j, i, k, c, in_box(float(a2), float(a4))))

    torch.save({"a2": A2.tolist(), "a4": A4.tolist(), "cov": cov,
                "box": [A2_LO, A2_HI, A4_LO, A4_HI]},
               os.path.join(HERE, "a2a4_cov_M3_bfs.pt"))

    m = ~np.isnan(cov)
    inb = np.zeros_like(cov, bool)
    for j, i, k, c, ib in rows:
        inb[j, i] = ib
    ood = m & (~inb)
    nfull = int((np.abs(cov[m] - 1.0) < 1e-9).sum())
    print("\n===== a2a4 coverage  M3-BFS (model-seeded reference continuation) =====")
    print(f"ALL cells n={m.sum():3d} | mean cov={np.nanmean(cov[m]):.3f}")
    print(f"OOD cells n={ood.sum():3d} | mean cov={np.nanmean(cov[ood]):.3f}")
    print(f"fully-covered cells: {nfull}/{m.sum()}  ({m.sum()-nfull} below 1.0)")
    bad = [(round(float(A2[i]), 1), round(float(A4[j]), 3), float(cov[j, i]))
           for j in range(cov.shape[0]) for i in range(cov.shape[1])
           if m[j, i] and cov[j, i] < 1.0 - 1e-9]
    print("cells still < 1.0:", bad)
    print("saved a2a4_cov_M3_bfs.pt")


if __name__ == "__main__":
    main()
