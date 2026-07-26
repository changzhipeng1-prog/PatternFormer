"""a2a4 extrapolation coverage, METHOD 2 (seed Newton from the nearest in-training cell).

For each target (a2,a4) we take the model's RAW output (before refine) at the nearest
in-training-rectangle grid cell and use those fields as Newton initial guesses AT the
target parameter, then dedup and score coverage.  Inside the box the nearest in-box cell
is the cell itself, so M2 == M1 there; the routes only differ outside the box.

Recomputes M1 (refine the cell's own output) under identical settings for a fair compare.
Saves a2a4_cov_M2.pt and prints an M1-vs-M2 table.
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


def coverage(seeds, a2, a4, gt):
    distinct = dedup([refine_one(u, float(a4), float(a2)) for u in seeds])
    c = sum(1 for g in gt if any(rl2(d, g) < COVER for d in distinct))
    return c, len(gt)


def main():
    G = torch.load(os.path.join(HERE, "grid_direct.pt"), map_location="cpu", weights_only=False)
    GT = torch.load(os.path.join(HERE, "grid_gt_cont.pt"), map_location="cpu", weights_only=False)
    A2 = np.array(G["a2"]); A4 = np.array(G["a4"])
    a2r = A2.max() - A2.min(); a4r = A4.max() - A4.min()
    inbox = [(float(a2), float(a4)) for a2 in A2 for a4 in A4
             if A2_LO <= a2 <= A2_HI and A4_LO <= a4 <= A4_HI]

    def nearest_inbox(a2, a4):
        return min(inbox, key=lambda b: ((a2 - b[0]) / a2r) ** 2 + ((a4 - b[1]) / a4r) ** 2)

    covM1 = np.full((len(A4), len(A2)), np.nan)
    covM2 = np.full((len(A4), len(A2)), np.nan)
    rows = []
    for ia, a2 in enumerate(A2):
        for jb, a4 in enumerate(A4):
            gt = np.asarray(GT["gt"][(float(a2), float(a4))], np.float64)
            k = gt.shape[0]
            if k == 0:
                continue
            genM1 = G["gen"][(float(a2), float(a4))].numpy().astype(np.float64)
            c1, _ = coverage(genM1, a2, a4, gt)
            n2, n4 = nearest_inbox(float(a2), float(a4))
            genM2 = G["gen"][(n2, n4)].numpy().astype(np.float64)
            c2, _ = coverage(genM2, a2, a4, gt)
            covM1[jb, ia] = c1 / k; covM2[jb, ia] = c2 / k
            in_box = (A2_LO <= a2 <= A2_HI and A4_LO <= a4 <= A4_HI)
            rows.append((float(a2), float(a4), k, c1, c2, in_box))
        print(f"  a2={a2:+.2f} done ({ia+1}/{len(A2)})", flush=True)

    torch.save({"a2": A2.tolist(), "a4": A4.tolist(), "cov": covM2,
                "box": [A2_LO, A2_HI, A4_LO, A4_HI]},
               os.path.join(HERE, "a2a4_cov_M2.pt"))

    # ---- M1 vs M2 summary ----
    rows = np.array([(r[2], r[3], r[4], r[5]) for r in rows], float)  # k, c1, c2, inbox
    cov1 = rows[:, 1] / rows[:, 0]; cov2 = rows[:, 2] / rows[:, 0]; inb = rows[:, 3].astype(bool)
    def report(mask, name):
        m1, m2 = cov1[mask].mean(), cov2[mask].mean()
        better = int((cov2[mask] > cov1[mask] + 1e-9).sum())
        worse = int((cov2[mask] < cov1[mask] - 1e-9).sum())
        print(f"{name:16s} n={mask.sum():3d} | mean cov  M1={m1:.3f}  M2={m2:.3f} "
              f"| M2>M1:{better}  M2<M1:{worse}  tie:{mask.sum()-better-worse}")
    print("\n===== a2a4 coverage: M1 (direct) vs M2 (seed from nearest in-box) =====")
    report(np.ones(len(inb), bool), "ALL cells")
    report(inb, "in-training")
    report(~inb, "OOD (outside)")
    print("saved a2a4_cov_M2.pt")


if __name__ == "__main__":
    main()
