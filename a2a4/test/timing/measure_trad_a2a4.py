"""Timing of the TRADITIONAL classical solver to obtain ALL coexisting solutions
FROM SCRATCH at selected a2a4 test parameters (a4,a2), with NO learned warm start.

a2a4 PDE:  -u'' + a4*u^4 + a2*u^2 = 0,  u'(0)=0, u(1)=0,  N=1024 second-order FDM
(identical residual convention to the data generator, model.newton_fdm).

Unlike 1D_p (which uses an expensive multigrid continuation to N=4096), the a2a4
data generator is single-resolution N=1024.  So the classical cost of getting EVERY
branch at a brand-new parameter is dominated by the SEARCH: you do not know how many
branches exist nor where their basins are, so you must launch damped-Newton from a
battery of structured initial guesses and collect all distinct converged non-zero
roots.  This is the cost the learned forward pass replaces with a single shot.

Battery (parameter-agnostic, does NOT use GT):
  half-cosine modes  phi_j(x)=cos((2j-1) pi x /2)   (each satisfies u'(0)=0,u(1)=0)
  x amplitudes A in [0.05, 10]   x both signs       -> N_INIT guesses
For each (k in {1,3,5}) we select the first N_PER test params for which the battery
recovers ALL k ground-truth branches (full coverage), so the timing is a fair
apples-to-apples "find every branch" comparison against ours.

Output -> timing/trad_a2a4_timing.csv  (one row per selected target).
"""
import os
import sys
import csv
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(TEST, "..", "code"))
from model.newton_fdm import newton_refine, residual_l2

import torch

GEN = os.path.join(TEST, "generated_solutions.pt")
CSV = os.path.join(HERE, "trad_a2a4_timing.csv")

N = 1024
H = 1.0 / N
X = np.linspace(0.0, 1.0, N)
NEWTON_TOL = 1e-9          # ||F|| < 1e-9 (data-generator convergence)
NEWTON_MAX = 60
DEDUP = 1e-3               # distinct-root threshold (matches correct_dataset_fdm)
COVER_TOL = 1e-2          # GT branch counted "found" if some root is within this rel-L2
K_TARGETS = [1, 3, 5]
N_PER = 2                  # full-coverage params to time per k
MAX_SCAN = 50              # cap params scanned per k while seeking full coverage
A_MAX = 10.0
N_MODES = 6


def make_battery():
    """Parameter-agnostic structured initial guesses (denser at low amplitude)."""
    A = np.concatenate([np.linspace(0.05, 1.5, 18), np.linspace(1.6, A_MAX, 30)])
    inits = []
    for j in range(1, N_MODES + 1):
        phi = np.cos((2 * j - 1) * np.pi * X / 2.0)
        for a in A:
            inits.append(a * phi)
            inits.append(-a * phi)
    return inits


BATTERY = make_battery()


def multistart(a4, a2):
    """Run damped-Newton from every battery init; collect distinct non-zero roots.
    Returns (solutions, total_newton_iters, wall_seconds)."""
    sols, iters = [], 0
    t0 = time.perf_counter()
    for u0 in BATTERY:
        ru, ok, it = newton_refine(u0, a4, a2, h=H, max_iter=NEWTON_MAX, abs_tol=NEWTON_TOL)
        iters += it
        if ok and np.linalg.norm(ru) > 1e-6 and residual_l2(ru, a4, a2, H) < 1e-6:
            if not any(np.linalg.norm(ru - v) < DEDUP for v in sols):
                sols.append(ru)
    return sols, iters, time.perf_counter() - t0


def nonzero_gt(rec):
    gt = rec["gt"].numpy().astype(np.float64)
    return gt[np.linalg.norm(gt, axis=1) > 1e-6]


def coverage(sols, gt):
    return sum(1 for g in gt
               if any(np.linalg.norm(s - g) / np.linalg.norm(g) < COVER_TOL for s in sols))


def main():
    data = torch.load(GEN, map_location="cpu", weights_only=False)
    by_k = {k: [] for k in K_TARGETS}
    for r in data:
        k = nonzero_gt(r).shape[0]
        if k in by_k:
            by_k[k].append(r)
    print(f"battery = {len(BATTERY)} inits  ({N_MODES} modes x "
          f"{len(BATTERY)//(2*N_MODES)} amplitudes x 2 signs)", flush=True)

    rows = []
    for k in K_TARGETS:
        picked = 0
        for rec in by_k[k][:MAX_SCAN]:
            a4, a2 = float(rec["params"][0]), float(rec["params"][1])
            gt = nonzero_gt(rec)
            sols, iters, t = multistart(a4, a2)
            cov = coverage(sols, gt)
            full = (cov == k)
            print(f"  k={k} (a4={a4:.3f},a2={a2:.3f})  cov={cov}/{k}  distinct={len(sols)}  "
                  f"iters={iters}  t={t:.2f}s  {'[USE]' if full else '[skip: partial]'}",
                  flush=True)
            if full:
                rows.append({"a4": a4, "a2": a2, "k": k, "n_init": len(BATTERY),
                             "n_found": len(sols), "coverage": cov,
                             "total_time_s": t, "total_newton_iters": iters})
                picked += 1
                if picked >= N_PER:
                    break

    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["a4", "a2", "k", "n_init", "n_found",
                                          "coverage", "total_time_s", "total_newton_iters"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {CSV}", flush=True)
    print("\n===== TRADITIONAL (classical multi-start) SUMMARY =====")
    for r in rows:
        print(f"  k={r['k']}  (a4={r['a4']:.3f},a2={r['a2']:.3f})  "
              f"{r['total_time_s']:.2f}s   {r['n_found']} roots found "
              f"({r['total_newton_iters']} Newton iters over {r['n_init']} inits)")


if __name__ == "__main__":
    main()
