"""1D_p test statistics. STRATEGY B ONLY.

Validity rule (strict): for each GT solution, ONLY the single closest Qwen output
is a valid prediction; all other (extra) outputs are invalid and are ignored --
they are NOT post-processed. For each valid (output, GT) pair we record:
    p, direct rel-L2, post-processing rel-L2 (cbmfem Newton), Newton steps, converged.

All values are written to test/stats.csv (one row per valid solution) so that
re-plotting never needs to recompute. Output -> test/stats.csv (+ test/stats.pt).
"""
import os
import sys
import numpy as np
import torch
import scipy.optimize as sciopt
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from model.traditional_refine import refine_with_newton_trace

GEN = os.environ.get("GEN_OUT", os.path.join(HERE, "generated_solutions.pt"))
OUT = os.environ.get("STATS_OUT", os.path.join(HERE, "stats.pt"))
CSV = os.environ.get("STATS_CSV", os.path.join(HERE, "stats.csv"))
NEWTON_TOL = 1e-9
NEWTON_MAX = 30


def rel_l2(pred, gt):
    return float(np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-12))


def match(preds, gt):
    cost = np.linalg.norm(preds[:, None, :] - gt[None, :, :], axis=2)
    r, c = sciopt.linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def process(rec):
    gt = rec["gt"].numpy().astype(np.float64)
    gen = rec["generated"].numpy().astype(np.float64)
    p = float(rec["p"])
    if gen.shape[0] == 0:
        return None
    out = {"p": [], "direct": [], "post": [], "resid": [], "steps": [], "conv": []}
    for pi, gj in match(gen, gt):                 # best-k matched pairs (closest output per GT)
        d = rel_l2(gen[pi], gt[gj])
        ru, ok, _, trace = refine_with_newton_trace(torch.tensor(gen[pi]), p,
                                                    tol=NEWTON_TOL, max_iter=NEWTON_MAX)
        rr = rel_l2(ru.detach().cpu().numpy().astype(np.float64).reshape(-1), gt[gj])
        # post-processing residual: cbmfem FEM ||F(u_refined)|| (data-gen convention)
        resid = float(trace[-1]["residual"]) if len(trace) else float("inf")
        st = int(trace[-1]["iter"]) if (ok and len(trace)) else (NEWTON_MAX + 1)
        out["p"].append(p); out["direct"].append(d); out["post"].append(rr)
        out["resid"].append(resid); out["steps"].append(st); out["conv"].append(bool(ok))
    return out


def main():
    data = torch.load(GEN, map_location="cpu", weights_only=False)
    ncpu = int(os.environ.get("STAT_NCPU", "16"))
    print(f"params={len(data)}  ncpu={ncpu}", flush=True)
    with Pool(ncpu) as pool:
        recs = [r for r in pool.map(process, data, chunksize=8) if r is not None]

    agg = {k: np.concatenate([np.array(r[k], float) for r in recs if len(r[k])])
           for k in ["p", "direct", "post", "resid", "steps"]}
    agg["conv"] = np.concatenate([np.array(r["conv"], bool) for r in recs if len(r["conv"])])
    agg["n_params"] = len(recs)
    agg["n_solutions"] = int(agg["direct"].size)
    torch.save(agg, OUT)

    # ---- CSV: one row per valid solution (source of truth for plotting) ----
    import csv
    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "direct_rel_l2", "post_rel_l2", "post_residual", "newton_steps", "converged"])
        for pp, d, po, rs, st, cv in zip(agg["p"], agg["direct"], agg["post"],
                                         agg["resid"], agg["steps"], agg["conv"]):
            w.writerow([f"{pp:.6f}", f"{d:.6e}", f"{po:.6e}", f"{rs:.6e}", int(st), int(bool(cv))])
    print(f"wrote {agg['n_solutions']} rows -> {CSV}")

    def s(x):
        a = np.asarray(x, float)
        return f"median={np.median(a):.3e} mean={a.mean():.3e} p90={np.percentile(a,90):.3e}"
    print("DIRECT rel-L2 :", s(agg["direct"]))
    print("POST   rel-L2 :", s(agg["post"]))
    print("POST   steps  :", s(agg["steps"][agg["conv"]]),
          f"  converged {agg['conv'].mean()*100:.1f}%")
    print(f"solutions={agg['n_solutions']}  saved -> {OUT}")


if __name__ == "__main__":
    main()
