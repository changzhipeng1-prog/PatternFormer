"""2D test statistics. STRATEGY B ONLY (identical policy to 1D_p / a2a4).

Validity rule (strict): for each GT solution, ONLY the single closest Qwen output
is a valid prediction; all other (extra) outputs are invalid and ignored -- they
are NOT post-processed. For each valid (output, GT) pair we record:
    s, k, direct rel-L2, post rel-L2 (FEM Newton), post residual ||F_free||, steps, converged.

PDE: -Delta u - u^2 = -s sin(pi x) sin(pi y) on the ell=3 FEM mesh (145 nodes).
Post-proc == data-gen FEM operator (free nodes only).
All values -> test/stats.csv (one row per valid solution) + test/stats.pt.
"""
import os
import sys
import numpy as np
import torch
import scipy.optimize as sciopt
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from model.newton_refine_2d import newton_refine_2d

# GEN_OUT/STATS_OUT/STATS_CSV override so an ablation checkpoint can be scored
# without clobbering the shipped warm-start baseline artifacts (defaults = ship paths).
GEN = os.environ.get("GEN_OUT", os.path.join(HERE, "generated_solutions.pt"))
OUT = os.environ.get("STATS_OUT", os.path.join(HERE, "stats.pt"))
CSV = os.environ.get("STATS_CSV", os.path.join(HERE, "stats.csv"))
NEWTON_TOL = 1e-9
NEWTON_MAX = 30

_DATA = torch.load(GEN, map_location="cpu", weights_only=False)
COORD = _DATA["coord"].float()
ELEM = _DATA["elem"].long()
FREE = _DATA["free_nodes"].long()


def rel_l2(pred, gt):
    return float(np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-12))


def match(preds, gt):
    cost = np.linalg.norm(preds[:, None, :] - gt[None, :, :], axis=2)
    r, c = sciopt.linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def process(rec):
    gt = rec["gt"].numpy().astype(np.float64)        # [k,145]
    gen = rec["generated"].numpy().astype(np.float64)  # [n,145]
    s = float(rec["s"])
    if gen.shape[0] == 0:
        return None
    out = {"s": [], "k": [], "direct": [], "post": [],
           "resid": [], "steps": [], "conv": []}
    k = gt.shape[0]
    for pi, gj in match(gen, gt):
        d = rel_l2(gen[pi], gt[gj])
        pu = torch.tensor(gen[pi], dtype=torch.float32)
        ur, ok, nit, hist = newton_refine_2d(pu, s, COORD, ELEM, FREE,
                                             tol=NEWTON_TOL, max_iter=NEWTON_MAX)
        ur = ur.detach().cpu().numpy().astype(np.float64).reshape(-1)
        rr = rel_l2(ur, gt[gj])
        resid = float(hist[-1]) if len(hist) else float("inf")
        out["s"].append(s); out["k"].append(k)
        out["direct"].append(d); out["post"].append(rr); out["resid"].append(resid)
        out["steps"].append(int(nit)); out["conv"].append(bool(ok))
    return out


def main():
    data = _DATA["results"]
    ncpu = int(os.environ.get("STAT_NCPU", "16"))
    print(f"params={len(data)}  ncpu={ncpu}", flush=True)
    with Pool(ncpu) as pool:
        recs = [r for r in pool.map(process, data, chunksize=4) if r is not None]

    keys = ["s", "k", "direct", "post", "resid", "steps"]
    agg = {kk: np.concatenate([np.array(r[kk], float) for r in recs if len(r[kk])]) for kk in keys}
    agg["conv"] = np.concatenate([np.array(r["conv"], bool) for r in recs if len(r["conv"])])
    agg["n_params"] = len(recs)
    agg["n_solutions"] = int(agg["direct"].size)
    torch.save(agg, OUT)

    import csv
    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s", "k", "direct_rel_l2", "post_rel_l2",
                    "post_residual", "newton_steps", "converged"])
        for s, k, d, po, rs, st, cv in zip(agg["s"], agg["k"], agg["direct"],
                                           agg["post"], agg["resid"], agg["steps"], agg["conv"]):
            w.writerow([f"{s:.6f}", int(k), f"{d:.6e}", f"{po:.6e}",
                        f"{rs:.6e}", int(st), int(bool(cv))])
    print(f"wrote {agg['n_solutions']} rows -> {CSV}")

    def ss(x):
        a = np.asarray(x, float)
        return f"median={np.median(a):.3e} mean={a.mean():.3e} p90={np.percentile(a,90):.3e}"
    print("DIRECT rel-L2 :", ss(agg["direct"]))
    print("POST   rel-L2 :", ss(agg["post"]))
    print("POST   steps  :", ss(agg["steps"][agg["conv"]]), f"  converged {agg['conv'].mean()*100:.1f}%")
    print(f"solutions={agg['n_solutions']}  saved -> {OUT}")


if __name__ == "__main__":
    main()
