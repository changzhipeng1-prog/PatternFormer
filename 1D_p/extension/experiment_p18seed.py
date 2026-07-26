"""Experiment: seed Newton at the extrapolated p with the IN-DISTRIBUTION p=18 Qwen
output (instead of the model's direct out-of-distribution output at that p).

Idea: the model is reliable in-distribution (p=18, covers all 7 branches); use those
as Newton initial guesses at p>18.  Compare coverage of:
   (A) direct  : refine the model's own output AT p   (current extension pipeline)
   (B) p18-seed: refine the model's p=18 output at p  (this experiment)
against the continued GT at p.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from config import Config
from model.v2_model import V2PDEModel
from model.traditional_refine import refine_with_newton_trace

C = Config()
CKPT = os.path.join(HERE, "..", "best_ckpt", "best_model")
GT = os.path.join(HERE, "gt_extension.pt")
DIRECT = os.path.join(HERE, "generated_extension.pt")
COVER, DEDUP, REFINE_OK = 1e-2, 1e-3, 1e-6


def rl2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def refine_set(outs, p):
    """refine a set of seeds at p -> distinct genuine solutions."""
    sols = []
    for u in outs:
        ru, ok, _, tr = refine_with_newton_trace(torch.tensor(u), float(p), tol=1e-9, max_iter=30)
        res = float(tr[-1]["residual"]) if tr else float("inf")
        if ok and res < REFINE_OK:
            sols.append(np.asarray(ru.detach().cpu(), np.float64).reshape(-1))
    distinct = []
    for u in sols:
        if not any(rl2(u, v) < DEDUP for v in distinct):
            distinct.append(u)
    return distinct


def coverage(distinct, gt):
    return sum(1 for g in gt if any(rl2(d, g) < COVER for d in distinct))


def main():
    g = torch.load(GT, map_location="cpu", weights_only=False)
    direct_data = torch.load(DIRECT, map_location="cpu", weights_only=False)
    p_targets, gt_by_p = g["p_targets"], g["gt"]
    direct_by_p = {float(r["p"]): r["generated"].numpy().astype(np.float64) for r in direct_data}

    print(f"Loading {CKPT} ...", flush=True)
    model = V2PDEModel.from_pretrained(CKPT, C)
    model.eval()

    # in-distribution p=18 outputs (the seeds)
    with torch.no_grad():
        preds18 = model.generate([], [], target_p_val=18.0, max_solutions=C.max_solutions_per_p)
    gen18 = np.stack([pp.cpu().float().numpy().reshape(-1) for pp in preds18])
    print(f"p=18 Qwen output: {gen18.shape[0]} candidates "
          f"(in-distribution; covers {coverage(refine_set(gen18, 18.0), gt_by_p[p_targets[0]])}+ branches)\n",
          flush=True)

    print(f"{'p':>6} {'n_gt':>4} | {'direct cov':>10} | {'p18-seed cov':>12}")
    print("-" * 42)
    rows = []
    for p in p_targets:
        gt = gt_by_p[p]
        cov_direct = coverage(refine_set(direct_by_p[p], p), gt)
        cov_seed = coverage(refine_set(gen18, p), gt)
        rows.append((p, gt.shape[0], cov_direct, cov_seed))
        print(f"{p:>6.2f} {gt.shape[0]:>4} | {cov_direct:>10} | {cov_seed:>12}")

    torch.save({"rows": rows, "gen18": gen18}, os.path.join(HERE, "p18seed_result.pt"))
    print("\nsaved p18seed_result.pt")


if __name__ == "__main__":
    main()
