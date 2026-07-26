"""Count the Newton iterations the TRADITIONAL multi-scale solver needs to obtain
ALL coexisting solutions from scratch at p=18 and p=7, for N=1024/2048/4096.

We wrap np.linalg.inv with a counter. In this cbmfem code base inv() is called
ONLY at a Newton step (the Jacobian dF is built with np.identity, never inv;
the coarse TF5 pre-solve uses k=1 gradient steps, no inv). So the inv-count is
exactly the number of Newton (Jacobian-solve) iterations performed.

We record TWO files:
  timing/trad_steps.csv         -- aggregate, one row per (p, resolution):
      newton_iters_total / newton_iters_finest_level / per_level_iters / n_found
  timing/trad_steps_detail.csv  -- GRANULAR, one row PER individual Newton solve:
      p, target_resolution, level, grid_size, solve_idx, newton_iters
The detail file is captured by wrapping Filtering's Newton() and reading the
iteration count it appends to its result (Newton.py returns np.append(x, ct)).

Comparable quantity to OUR method (which Newton-refines only at the target grid):
  finest-level iters / n_found  ~  per-branch Newton steps.
"""
import os
import sys
import csv
import numpy as np
from numpy import linalg as LA

HERE = os.path.dirname(os.path.abspath(__file__))
CBMFEM = os.path.join(HERE, "..", "..", "data_gen", "cbmfem")
sys.path.insert(0, CBMFEM)

# ---- install Newton-iteration counter (count every Jacobian inverse/solve) ----
_real_inv = np.linalg.inv
INV = {"n": 0}


def _counting_inv(a):
    INV["n"] += 1
    return _real_inv(a)


np.linalg.inv = _counting_inv

import Newton as _Newton_mod             # noqa: E402  (import AFTER patching inv)
from Filtering import Filtering          # noqa: E402
from Polynomial3 import Initial_guess3   # noqa: E402
from Sexample1 import faSh1              # noqa: E402
from WB1 import TF5                      # noqa: E402

# ---- record the iteration count of EVERY individual Newton solve ----
# Filtering does `from Newton import Newton` INSIDE the function (re-imported each
# call), so we patch the attribute on the Newton module itself. Newton.py returns
# np.append(x, ct), ct = its internal iteration counter -> we log ct per solve,
# tagged with the current (p, target-N, level, grid-size).
_real_Newton = _Newton_mod.Newton
DETAIL = []                 # list of dict rows, one per Newton solve
CUR = {"p": None, "N": None, "level": None, "grid": None}


def _recording_newton(f, df, x, tol, Ma):
    out = _real_Newton(f, df, x, tol, Ma)
    if out is not None:
        try:
            ct = int(round(float(np.real(out[-1, 0]))))
        except Exception:
            ct = -1
        DETAIL.append({"p": CUR["p"], "target_resolution": CUR["N"],
                       "level": CUR["level"], "grid_size": int(x.shape[0]),
                       "solve_idx": len(DETAIL), "newton_iters": ct})
    return out


_Newton_mod.Newton = _recording_newton

CSV = os.path.join(HERE, "trad_steps.csv")
CSV_DETAIL = os.path.join(HERE, "trad_steps_detail.csv")
TARGET_PS = [18, 7]
GG_BY_N = [(1024, 10), (2048, 11), (4096, 12)]


def solve_count_iters(p, gg):
    """From-scratch multi-scale solve; return (s2, info) with per-level Newton-iter
    counts. Identical numerics to measure_traditional.solve_all_solutions."""
    INV["n"] = 0
    x = np.zeros((0, 0))
    s1 = Initial_guess3(x, p).copy()
    s2 = np.zeros((1, 1))
    goal = np.zeros((1, 1))
    f, _ = faSh1(s2, p)

    # coarse 1D solve (TF5 k=1 multigrid; uses gradient steps, no inv)
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
    coarse_iters = INV["n"]

    erase = []
    for jjjj in range(len(s2[0, :])):
        if LA.norm(f(s2[:, jjjj:jjjj + 1]), 2) > 1e-6:
            erase.append(jjjj)
    if erase:
        s2 = np.delete(s2, erase, 1)

    level_iters = []
    for k in range(gg):
        g = Initial_guess3(s2, p)
        f, df = faSh1(g[:, 0:1], p)
        before = INV["n"]
        CUR["level"] = k                       # tag detail rows with this level
        s2 = Filtering(s2, 4, f, df, p, 1).copy()
        level_iters.append(INV["n"] - before)

    info = {
        "grid_size": int(s2.shape[0]),
        "coarse_iters": int(coarse_iters),
        "level_iters": level_iters,
        "newton_iters_total": int(INV["n"]),
        "newton_iters_finest": int(level_iters[-1]) if level_iters else 0,
    }
    return s2, info


def dedup_real_solutions(s2, atol_imag=1e-7, atol_dup=1e-7):
    cols = []
    for i in range(s2.shape[1]):
        col = s2[:, i:i + 1]
        if np.max(np.abs(col.imag)) < atol_imag:
            xr = col.real.copy()
            if not any(LA.norm(xr - y, 2) < atol_dup for y in cols):
                cols.append(xr)
    return cols


def main():
    rows = []
    for p in TARGET_PS:
        print(f"\n===== TRADITIONAL Newton-iteration count, p={p} =====", flush=True)
        for N, gg in GG_BY_N:
            print(f"  -- N={N} (gg={gg}) ...", flush=True)
            CUR["p"], CUR["N"] = p, N           # tag detail rows with (p, target N)
            s2, info = solve_count_iters(p, gg)
            nfound = len(dedup_real_solutions(s2))
            per_sol = info["newton_iters_finest"] / max(nfound, 1)
            rows.append({"p": p, "resolution": info["grid_size"], "gg": gg,
                         "n_found": nfound,
                         "newton_iters_total": info["newton_iters_total"],
                         "newton_iters_finest_level": info["newton_iters_finest"],
                         "finest_iters_per_solution": round(per_sol, 2),
                         "coarse_presolve_iters": info["coarse_iters"],
                         "per_level_iters": "|".join(str(x) for x in info["level_iters"])})
            print(f"     N={info['grid_size']}  found={nfound}  "
                  f"total_newton_iters={info['newton_iters_total']}  "
                  f"finest_level={info['newton_iters_finest']} "
                  f"(~{per_sol:.1f}/solution)", flush=True)

    with open(CSV, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=["p", "resolution", "gg", "n_found",
                                             "newton_iters_total",
                                             "newton_iters_finest_level",
                                             "finest_iters_per_solution",
                                             "coarse_presolve_iters",
                                             "per_level_iters"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {CSV}", flush=True)

    # ---- granular: one row per individual Newton solve ----
    with open(CSV_DETAIL, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=["p", "target_resolution", "level",
                                             "grid_size", "solve_idx", "newton_iters"])
        w.writeheader()
        for d in DETAIL:
            w.writerow(d)
    print(f"wrote {len(DETAIL)} per-solve rows -> {CSV_DETAIL}", flush=True)

    print("\n========  TRADITIONAL NEWTON-ITERATION SUMMARY  ========")
    for p in TARGET_PS:
        print(f"\np = {p}")
        for r in [r for r in rows if r["p"] == p]:
            print(f"  N={r['resolution']:>4}  found={r['n_found']}  "
                  f"total_iters={r['newton_iters_total']:>6}  "
                  f"finest_level={r['newton_iters_finest_level']:>5}  "
                  f"(~{r['finest_iters_per_solution']:.1f}/sol)")


if __name__ == "__main__":
    main()
