"""a2a4 coverage, METHOD 3 (stepped continuation), analogous to the 1D p=50 march.

For each target (a2,a4): start from the model's RAW output at the nearest in-training
cell, refine it there, then march along a straight line to the target in small steps,
Newton-refining and carrying the solution set forward at every step.  At the target,
dedup and score coverage.  This should recover branches a single jump (M2) misses.

Saves a2a4_cov_M3.pt and prints an M1/M2/M3 table.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from model.newton_fdm import newton_refine, residual_l2

N = 1024
H = 1.0 / N
COVER, DEDUP, REFINE_OK = 1e-2, 1e-3, 1e-6
A2_LO, A2_HI, A4_LO, A4_HI = -14.934, -2.0, 0.301, 0.798
DA2_STEP, DA4_STEP = 1.0, 0.0625   # sub-step sizes for the march


def rl2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def dedup(sols):
    out = []
    for u in sols:
        if u is not None and not any(rl2(u, v) < DEDUP for v in out):
            out.append(u)
    return out


def refine_one(u, a4, a2):
    ru, ok, _ = newton_refine(u, a4, a2, h=H, max_iter=80, abs_tol=1e-9)
    ru = np.asarray(ru, np.float64).reshape(-1)
    return ru if (ok and residual_l2(ru, a4, a2, H) < REFINE_OK and np.linalg.norm(ru) > 1e-6) else None


def main():
    G = torch.load(os.path.join(HERE, "grid_direct.pt"), map_location="cpu", weights_only=False)
    GT = torch.load(os.path.join(HERE, "grid_gt_cont.pt"), map_location="cpu", weights_only=False)
    M1d = torch.load(os.path.join(HERE, "a2a4_cov_M1.pt"), map_location="cpu", weights_only=False)
    M2d = torch.load(os.path.join(HERE, "a2a4_cov_M2.pt"), map_location="cpu", weights_only=False)
    A2 = np.array(G["a2"]); A4 = np.array(G["a4"])
    a2r = A2.max() - A2.min(); a4r = A4.max() - A4.min()
    inbox = [(float(a2), float(a4)) for a2 in A2 for a4 in A4
             if A2_LO <= a2 <= A2_HI and A4_LO <= a4 <= A4_HI]

    def nearest_inbox(a2, a4):
        return min(inbox, key=lambda b: ((a2 - b[0]) / a2r) ** 2 + ((a4 - b[1]) / a4r) ** 2)

    def march(seeds, a2_0, a4_0, a2_t, a4_t):
        nsub = max(int(round(abs(a2_t - a2_0) / DA2_STEP)),
                   int(round(abs(a4_t - a4_0) / DA4_STEP)), 1)
        cur = dedup([refine_one(u, a4_0, a2_0) for u in seeds])
        for t in range(1, nsub + 1):
            if not cur:
                break
            a2 = a2_0 + (a2_t - a2_0) * t / nsub
            a4 = a4_0 + (a4_t - a4_0) * t / nsub
            cur = dedup([refine_one(u, a4, a2) for u in cur])
        return cur

    covM3 = np.full((len(A4), len(A2)), np.nan)
    rows = []
    for ia, a2 in enumerate(A2):
        for jb, a4 in enumerate(A4):
            gt = np.asarray(GT["gt"][(float(a2), float(a4))], np.float64)
            k = gt.shape[0]
            if k == 0:
                continue
            n2, n4 = nearest_inbox(float(a2), float(a4))
            seeds = G["gen"][(n2, n4)].numpy().astype(np.float64)
            distinct = march(seeds, n2, n4, float(a2), float(a4))
            c = sum(1 for g in gt if any(rl2(d, g) < COVER for d in distinct))
            covM3[jb, ia] = c / k
            in_box = (A2_LO <= a2 <= A2_HI and A4_LO <= a4 <= A4_HI)
            rows.append((jb, ia, k, c, in_box))
        print(f"  a2={a2:+.2f} done ({ia+1}/{len(A2)})", flush=True)

    torch.save({"a2": A2.tolist(), "a4": A4.tolist(), "cov": covM3,
                "box": [A2_LO, A2_HI, A4_LO, A4_HI]},
               os.path.join(HERE, "a2a4_cov_M3.pt"))

    cov1 = np.array(M1d["cov"]); cov2 = np.array(M2d["cov"]); cov3 = covM3
    m = ~np.isnan(cov3)
    inb = np.zeros_like(cov3, bool)
    for jb, ia, k, c, ib in rows:
        inb[jb, ia] = ib
    ood = m & (~inb)
    print("\n===== a2a4 coverage  M1 / M2 / M3 (stepped continuation) =====")
    print(f"ALL cells   n={m.sum():3d} | M1={np.nanmean(cov1[m]):.3f}  M2={np.nanmean(cov2[m]):.3f}  M3={np.nanmean(cov3[m]):.3f}")
    print(f"OOD cells   n={ood.sum():3d} | M1={np.nanmean(cov1[ood]):.3f}  M2={np.nanmean(cov2[ood]):.3f}  M3={np.nanmean(cov3[ood]):.3f}")
    nfull = int((np.abs(cov3[m] - 1.0) < 1e-9).sum())
    print(f"M3 fully-covered cells: {nfull}/{m.sum()}   ({m.sum()-nfull} below 1.0)")
    bad = [(round(float(A2[ia]),1), round(float(A4[jb]),3), float(cov3[jb,ia]))
           for jb in range(cov3.shape[0]) for ia in range(cov3.shape[1])
           if m[jb,ia] and cov3[jb,ia] < 1.0 - 1e-9]
    print("M3 cells still < 1.0:", bad)
    print("saved a2a4_cov_M3.pt")


if __name__ == "__main__":
    main()
