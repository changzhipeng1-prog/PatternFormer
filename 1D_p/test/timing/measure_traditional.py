"""Timing of the TRADITIONAL multi-scale solver (the existing data-generation
program) to obtain ALL coexisting solutions FROM SCRATCH at p=18 and p=7,
at three resolutions N=1024/2048/4096.

Traditional pipeline (identical to data generation; paper/1D_p/data_gen/cbmfem):
  Initial_guess3 (polynomial-root predictor)  -> coarse initial guesses
  TF5 multigrid V-cycles                        -> coarse solve from scratch
  coarse-to-fine continuation via Filtering    -> refine to target grid
                                                   (gg levels: gg=10->1024,
                                                    11->2048, 12->4096)
NO prior solution and NO learned warm start are used -- this is the cost of
finding every branch with the classical method alone.

For each (p, resolution) we record: wall-clock time, #solutions found,
final grid size, and the initial-guess vs continuation time split.
Output -> timing/trad_timing.csv  (one row per (p, resolution)).
"""
import os
import sys
import csv
import time
import numpy as np
from numpy import linalg as LA

HERE = os.path.dirname(os.path.abspath(__file__))
CBMFEM = os.path.join(HERE, "..", "..", "data_gen", "cbmfem")
sys.path.insert(0, CBMFEM)

from Filtering import Filtering
from Polynomial3 import Initial_guess3
from Sexample1 import faSh1
from WB1 import TF5

CSV = os.path.join(HERE, "trad_timing.csv")
TARGET_PS = [18, 7]
GG_BY_N = [(1024, 10), (2048, 11), (4096, 12)]


def solve_all_solutions(p, gg, return_timing=True):
    """From-scratch multi-scale solve of ALL coexisting solutions at parameter p,
    refined through gg coarse-to-fine levels (final grid N = 2**gg).
    Faithful to eq57/run_p18_timed_newton.solve_p18_all_solutions, p-generalized."""
    start = time.perf_counter()
    t_ig_start = start

    x = np.zeros((0, 0))
    s1 = Initial_guess3(x, p).copy()
    s2 = np.zeros((1, 1))
    goal = np.zeros((1, 1))
    f, _ = faSh1(s2, p)

    # coarse 1D solve from polynomial roots + multigrid TF5
    for iii in range(len(s1[0, :])):
        x = s1[:, iii:iii + 1].copy()
        for _ in range(3000):
            x = TF5(x, 1, p, 1, 5, goal)
            if LA.norm(f(x), 2) < 1e-9:
                break
        ncon = 0
        for jjjj in range(len(s2[0, :])):
            if abs(x[:, 0:1] - s2[:, jjjj:jjjj + 1]).max() < 1e-5:
                ncon = 1
        if ncon == 0 and LA.norm(f(x), 2) < 1e-9:
            s2 = np.append(s2, x[:, 0:1], axis=1)
    t_ig_end = time.perf_counter()

    erase = []
    for jjjj in range(len(s2[0, :])):
        if LA.norm(f(s2[:, jjjj:jjjj + 1]), 2) > 1e-6:
            erase.append(jjjj)
    if erase:
        s2 = np.delete(s2, erase, 1)

    t_filt_start = time.perf_counter()
    for k in range(gg):
        g = Initial_guess3(s2, p)
        f, df = faSh1(g[:, 0:1], p)
        s2 = Filtering(s2, 4, f, df, p, 1).copy()
    t_filt_end = time.perf_counter()

    timing = {
        "initial_guess_seconds": float(t_ig_end - t_ig_start),
        "continuation_seconds": float(t_filt_end - t_filt_start),
        "total_seconds": float(t_filt_end - start),
        "grid_size": int(s2.shape[0]),
    }
    return s2, timing


def dedup_real_solutions(s2, atol_imag=1e-7, atol_dup=1e-7):
    real_cols = []
    for i in range(s2.shape[1]):
        col = s2[:, i:i + 1]
        if np.max(np.abs(col.imag)) < atol_imag:
            x = col.real.copy()
            if not any(LA.norm(x - y, 2) < atol_dup for y in real_cols):
                real_cols.append(x)
    return real_cols


def main():
    rows = []
    for p in TARGET_PS:
        print(f"\n===== TRADITIONAL multi-scale solve, p={p} =====", flush=True)
        for N, gg in GG_BY_N:
            print(f"  -- N={N} (gg={gg}) ...", flush=True)
            s2, t = solve_all_solutions(p, gg)
            nfound = len(dedup_real_solutions(s2))
            rows.append({"p": p, "resolution": t["grid_size"], "gg": gg,
                         "n_found": nfound,
                         "total_seconds": t["total_seconds"],
                         "initial_guess_seconds": t["initial_guess_seconds"],
                         "continuation_seconds": t["continuation_seconds"]})
            print(f"     N={t['grid_size']}  found={nfound}  "
                  f"total={t['total_seconds']:.2f}s  "
                  f"(init={t['initial_guess_seconds']:.2f}s, "
                  f"continuation={t['continuation_seconds']:.2f}s)", flush=True)

    with open(CSV, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=["p", "resolution", "gg", "n_found",
                                             "total_seconds", "initial_guess_seconds",
                                             "continuation_seconds"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {CSV}", flush=True)

    print("\n============  TRADITIONAL TIMING SUMMARY  ============")
    for p in TARGET_PS:
        print(f"\np = {p}")
        for r in [r for r in rows if r["p"] == p]:
            print(f"  N={r['resolution']:>4}  found={r['n_found']}  "
                  f"total={r['total_seconds']:8.2f} s")


if __name__ == "__main__":
    main()
