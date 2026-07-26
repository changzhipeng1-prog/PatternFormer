"""Timing of OUR method to obtain the true solutions at p=7 (k=3) and p=18 (k=7),
then refine to higher resolutions by linear up-interpolation + Newton.

Pipeline timed per parameter p:
  (1) Qwen forward  : model.generate(...) -> candidate solutions at native 1024
  (2) refine @1024  : cbmfem damped-Newton on the 1024 candidates  -> true 1024 solutions
  (3) ->2048        : linear-interpolate the refined 1024 solutions to 2048, Newton refine
  (4) ->4096        : linear-interpolate the refined 2048 solutions to 4096, Newton refine

Each higher resolution is warm-started from the previous (cascade mesh refinement).
Per stage we record wall-clock time, Newton steps, final residual ||F||, converged.

Output -> timing/ours_timing.csv  (one row per (p, resolution, solution-branch))
        + per-p stage summary printed to stdout.
"""
import os
import sys
import csv
import time
import numpy as np
import torch
import scipy.optimize as sciopt

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(TEST, "..", "code"))
from config import Config
from model.v2_model import V2PDEModel
from model.traditional_refine import refine_with_newton_trace, _resample_1d_linear

C = Config()
CKPT = os.path.join(TEST, "..", "best_ckpt", "best_model")
GEN = os.path.join(TEST, "generated_solutions.pt")
CSV = os.path.join(HERE, "ours_timing.csv")

TARGET_PS = [7.0, 18.0]
RESOLUTIONS = [1024, 2048, 4096]
NEWTON_TOL = 1e-9
NEWTON_MAX = 50
FWD_WARMUP = 1
FWD_REPEAT = 5


def rel_l2(pred, gt):
    return float(np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-12))


def match_best_k(preds, gt):
    cost = np.linalg.norm(preds[:, None, :] - gt[None, :, :], axis=2)
    r, c = sciopt.linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def newton_at(u_np, p, n_work):
    """Refine u_np (any length) at resolution n_work: linear-resample then Newton.
    Returns (refined_np[n_work], steps, residual, converged, t_seconds)."""
    t0 = time.perf_counter()
    ru, ok, _, trace = refine_with_newton_trace(
        torch.tensor(u_np, dtype=torch.float64), p,
        tol=NEWTON_TOL, max_iter=NEWTON_MAX,
        working_grid_size=n_work, return_working_grid=True)
    t = time.perf_counter() - t0
    out = ru.detach().cpu().numpy().astype(np.float64).reshape(-1)
    steps = int(trace[-1]["iter"]) if len(trace) else NEWTON_MAX
    resid = float(trace[-1]["residual"]) if len(trace) else float("inf")
    return out, steps, resid, bool(ok), t


def main():
    cuda = torch.cuda.is_available()
    print(f"Loading {CKPT} ...  cuda={cuda}", flush=True)
    model = V2PDEModel.from_pretrained(CKPT, C)
    model.eval()

    data = torch.load(GEN, map_location="cpu", weights_only=False)
    rows = []

    for tp in TARGET_PS:
        rec = min(data, key=lambda r: abs(float(r["p"]) - tp))
        p = float(rec["p"])
        gt = rec["gt"].numpy().astype(np.float64)
        k = gt.shape[0]
        print(f"\n===== p={p:.4f}  (k={k}) =====", flush=True)

        # ---------- (1) Qwen forward timing ----------
        def fwd():
            with torch.no_grad():
                preds = model.generate([], [], target_p_val=p,
                                       max_solutions=C.max_solutions_per_p)
            if cuda:
                torch.cuda.synchronize()
            return preds
        for _ in range(FWD_WARMUP):
            fwd()
        fwd_times = []
        for _ in range(FWD_REPEAT):
            t0 = time.perf_counter()
            preds = fwd()
            fwd_times.append(time.perf_counter() - t0)
        fwd_times = np.array(fwd_times)
        gen = (np.stack([pp.cpu().double().numpy().reshape(-1) for pp in preds])
               if len(preds) else np.zeros((0, C.solution_dim)))
        t_fwd_mean, t_fwd_std = float(fwd_times.mean()), float(fwd_times.std())
        print(f"  [forward]  {gen.shape[0]} candidates  "
              f"t = {t_fwd_mean*1e3:.1f} +/- {t_fwd_std*1e3:.1f} ms "
              f"(n={FWD_REPEAT})", flush=True)

        # match the k closest candidates to GT (strategy B: best-k)
        pairs = match_best_k(gen, gt)               # [(pred_i, gt_j), ...] length k
        cur = [gen[pi].copy() for pi, gj in pairs]  # current solutions, start at 1024
        gj_of = [gj for pi, gj in pairs]

        # ---------- (2-4) cascade refine over resolutions ----------
        for R in RESOLUTIONS:
            stage_t, nxt = 0.0, []
            for b, (u, gj) in enumerate(zip(cur, gj_of)):
                ru, steps, resid, ok, t = newton_at(u, p, R)
                stage_t += t
                # rel-L2 vs GT (GT only at 1024; compare on common 1024 grid)
                ru_1024 = _resample_1d_linear(ru, 1024) if R != 1024 else ru
                rl2 = rel_l2(ru_1024, gt[gj])
                rows.append({"p": p, "k": k, "resolution": R, "branch": b,
                             "time_s": t, "newton_steps": steps,
                             "residual": resid, "converged": int(ok),
                             "rel_l2_vs_gt1024": rl2})
                nxt.append(ru)
            cur = nxt  # warm-start next resolution from this refined solution
            avg_steps = np.mean([r["newton_steps"] for r in rows
                                 if r["p"] == p and r["resolution"] == R])
            print(f"  [refine @{R:>4}]  k={k} branches  "
                  f"t = {stage_t*1e3:.1f} ms  (avg {avg_steps:.1f} Newton steps)",
                  flush=True)

        # carry forward-time as separate rows (resolution=0 marker)
        rows.append({"p": p, "k": k, "resolution": 0, "branch": -1,
                     "time_s": t_fwd_mean, "newton_steps": -1,
                     "residual": -1, "converged": -1, "rel_l2_vs_gt1024": -1})

    # ---------- write CSV ----------
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["p", "k", "resolution", "branch",
                                          "time_s", "newton_steps", "residual",
                                          "converged", "rel_l2_vs_gt1024"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {CSV}", flush=True)

    # ---------- per-p stage summary ----------
    print("\n================  OUR-METHOD TIMING SUMMARY  ================")
    for tp in TARGET_PS:
        rs = [r for r in rows if abs(r["p"] - tp) < 0.5]
        if not rs:
            continue
        p = rs[0]["p"]; k = rs[0]["k"]
        t_fwd = next(r["time_s"] for r in rs if r["resolution"] == 0)
        print(f"\np = {p:.3f}  (k={k} coexisting solutions)")
        print(f"  forward (all {k}+ candidates) : {t_fwd*1e3:8.1f} ms")
        cum = t_fwd
        for R in RESOLUTIONS:
            sr = [r for r in rs if r["resolution"] == R]
            tt = sum(r["time_s"] for r in sr)
            st = np.mean([r["newton_steps"] for r in sr])
            mr = np.median([r["residual"] for r in sr])
            cum += tt
            print(f"  refine @{R:>4} (k branches)     : {tt*1e3:8.1f} ms   "
                  f"avg {st:.1f} steps   median ||F||={mr:.1e}   "
                  f"[cumulative {cum*1e3:8.1f} ms]")


if __name__ == "__main__":
    main()
