"""Timing of OUR method to obtain ALL coexisting a2a4 solutions at the same target
parameters used by measure_trad_a2a4.py.

Pipeline timed per parameter (a4,a2):
  (1) Qwen forward : model.generate(...) -> all candidate branches at native N=1024
  (2) refine @1024 : FDM damped-Newton (model.newton_fdm) on the matched-k candidates
                     -> FDM-exact solutions (||F|| < 1e-9)

a2a4 is single-resolution (N=1024) -- there is no multi-grid cascade (unlike 1D_p),
so the cost is one forward pass + k cheap Newton refines.  Forward time is the mean
of FWD_REPEAT timed runs after a warm-up.

Reads the target list from timing/trad_a2a4_timing.csv (same params).
Output -> timing/ours_a2a4_timing.csv (one row per target).
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
from model.newton_fdm import newton_refine, residual_l2

C = Config()
CKPT = os.path.join(TEST, "..", "best_ckpt", "best_model")
GEN = os.path.join(TEST, "generated_solutions.pt")
TRAD = os.path.join(HERE, "trad_a2a4_timing.csv")
CSV = os.path.join(HERE, "ours_a2a4_timing.csv")

N = 1024
H = 1.0 / N
NEWTON_MAX = 60
NEWTON_TOL = 1e-9
FWD_WARMUP = 1
FWD_REPEAT = 5


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def match_best_k(preds, gt):
    cost = np.linalg.norm(preds[:, None, :] - gt[None, :, :], axis=2)
    r, c = sciopt.linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def nonzero_gt(rec):
    gt = rec["gt"].numpy().astype(np.float64)
    return gt[np.linalg.norm(gt, axis=1) > 1e-6]


def main():
    cuda = torch.cuda.is_available()
    print(f"Loading {CKPT} ...  cuda={cuda}", flush=True)
    model = V2PDEModel.from_pretrained(CKPT, C)
    model.eval()

    data = torch.load(GEN, map_location="cpu", weights_only=False)
    def find_rec(a4, a2):
        return min(data, key=lambda r: abs(float(r["params"][0]) - a4) + abs(float(r["params"][1]) - a2))

    targets = list(csv.DictReader(open(TRAD)))
    rows = []
    for tr in targets:
        a4, a2, k = float(tr["a4"]), float(tr["a2"]), int(tr["k"])
        rec = find_rec(a4, a2)
        gt = nonzero_gt(rec)
        print(f"\n===== (a4={a4:.3f},a2={a2:.3f})  k={k} =====", flush=True)

        # ---------- (1) Qwen forward timing ----------
        def fwd():
            with torch.no_grad():
                preds = model.generate([], [], target_params=(a4, a2),
                                       max_solutions=C.max_solutions_per_p)
            if cuda:
                torch.cuda.synchronize()
            return preds
        for _ in range(FWD_WARMUP):
            fwd()
        fwd_t = []
        for _ in range(FWD_REPEAT):
            t0 = time.perf_counter()
            preds = fwd()
            fwd_t.append(time.perf_counter() - t0)
        fwd_t = np.array(fwd_t)
        gen = (np.stack([pp.cpu().double().numpy().reshape(-1) for pp in preds])
               if len(preds) else np.zeros((0, N)))
        t_fwd = float(fwd_t.mean())
        print(f"  [forward]  {gen.shape[0]} candidates  "
              f"t={t_fwd*1e3:.1f} +/- {fwd_t.std()*1e3:.1f} ms (n={FWD_REPEAT})", flush=True)

        # ---------- (2) refine the matched-k candidates @1024 ----------
        pairs = match_best_k(gen, gt) if gen.shape[0] and gt.shape[0] else []
        t_ref, steps_tot, resids, rl2s = 0.0, 0, [], []
        for pi, gj in pairs:
            t0 = time.perf_counter()
            ru, ok, it = newton_refine(gen[pi], a4, a2, h=H, max_iter=NEWTON_MAX, abs_tol=NEWTON_TOL)
            t_ref += time.perf_counter() - t0
            steps_tot += it
            resids.append(residual_l2(ru, a4, a2, H))
            rl2s.append(rel_l2(ru, gt[gj]))
        total = t_fwd + t_ref
        print(f"  [refine @1024]  k={len(pairs)} branches  t={t_ref*1e3:.1f} ms  "
              f"steps={steps_tot}  median||F||={np.median(resids):.1e}  "
              f"median rel-L2={np.median(rl2s):.1e}", flush=True)
        print(f"  [total] forward+refine = {total*1e3:.1f} ms", flush=True)

        rows.append({"a4": a4, "a2": a2, "k": k, "n_found": len(pairs),
                     "forward_s": f"{t_fwd:.5f}", "refine_s": f"{t_ref:.5f}",
                     "total_s": f"{total:.5f}", "newton_steps_total": steps_tot})

    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["a4", "a2", "k", "n_found", "forward_s",
                                          "refine_s", "total_s", "newton_steps_total"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {CSV}", flush=True)


if __name__ == "__main__":
    main()
