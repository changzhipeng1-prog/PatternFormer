"""a2a4 test statistics. STRATEGY B ONLY (identical policy to 1D_p).

NOTE: ~24% of a2a4 GT branches are the TRIVIAL zero solution u==0 (||gt||==0), for
which a relative L2 error is undefined (division by zero). Per the agreed policy we
EXCLUDE the trivial u==0 branch from the GT set and report statistics ONLY over the
non-trivial coexisting solutions. The multiplicity k below is the NON-ZERO branch
count (n_zero excluded per param is recorded for transparency).

Validity rule (strict): for each (non-zero) GT solution, ONLY the single closest Qwen
output is a valid prediction; all other outputs are invalid and ignored -- they are
NOT post-processed. For each valid (output, GT) pair we record:
    a4, a2, k, direct rel-L2, post rel-L2 (FDM Newton), post residual ||F||, steps, converged.

PDE: -u'' + a4 u^4 + a2 u^2 = 0.  Post-proc == data-gen FDM operator.
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
from model.newton_fdm import newton_refine, residual_l2

# GEN_OUT/STATS_OUT/STATS_CSV override the I/O paths so an ablation checkpoint can be
# scored without clobbering the shipped warm-start baseline artifacts (default = ship paths).
GEN = os.environ.get("GEN_OUT", os.path.join(HERE, "generated_solutions.pt"))
OUT = os.environ.get("STATS_OUT", os.path.join(HERE, "stats.pt"))
CSV = os.environ.get("STATS_CSV", os.path.join(HERE, "stats.csv"))
NEWTON_TOL = 1e-6          # relative fallback (unused when ABS_TOL set)
NEWTON_ABS = 1e-9          # ABSOLUTE ||F|| target == data-gen criterion (float64 floor ~1e-11)
NEWTON_MAX = 50
ZERO_NORM = 1e-6           # GT branch with ||gt|| < this is the trivial u==0 -> excluded


def rel_l2(pred, gt):
    return float(np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-12))


def match(preds, gt):
    cost = np.linalg.norm(preds[:, None, :] - gt[None, :, :], axis=2)
    r, c = sciopt.linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def process(rec):
    gt_all = rec["gt"].numpy().astype(np.float64)
    gen = rec["generated"].numpy().astype(np.float64)
    a4, a2 = float(rec["params"][0]), float(rec["params"][1])
    if gen.shape[0] == 0:
        return None
    # exclude the trivial u==0 branch (||gt||~0): rel-L2 is undefined there
    nz = np.linalg.norm(gt_all, axis=1) > ZERO_NORM
    n_zero = int((~nz).sum())
    gt = gt_all[nz]
    if gt.shape[0] == 0:
        return None
    out = {"a4": [], "a2": [], "k": [], "n_zero": [], "direct": [], "post": [],
           "resid": [], "steps": [], "conv": []}
    k = gt.shape[0]                       # non-zero branch multiplicity
    for pi, gj in match(gen, gt):
        d = rel_l2(gen[pi], gt[gj])
        N = gen[pi].shape[0]
        h = 1.0 / N
        ur, ok, nit = newton_refine(gen[pi], a4, a2, h=h, tol=NEWTON_TOL,
                                    max_iter=NEWTON_MAX, abs_tol=NEWTON_ABS)
        ur = np.asarray(ur, dtype=np.float64).reshape(-1)
        rr = rel_l2(ur, gt[gj])
        resid = residual_l2(ur, a4, a2, h)
        out["a4"].append(a4); out["a2"].append(a2); out["k"].append(k)
        out["n_zero"].append(n_zero)
        out["direct"].append(d); out["post"].append(rr); out["resid"].append(resid)
        out["steps"].append(int(nit)); out["conv"].append(bool(ok))
    return out


def main():
    data = torch.load(GEN, map_location="cpu", weights_only=False)
    ncpu = int(os.environ.get("STAT_NCPU", "16"))
    print(f"params={len(data)}  ncpu={ncpu}", flush=True)
    with Pool(ncpu) as pool:
        recs = [r for r in pool.map(process, data, chunksize=8) if r is not None]

    keys = ["a4", "a2", "k", "n_zero", "direct", "post", "resid", "steps"]
    agg = {kk: np.concatenate([np.array(r[kk], float) for r in recs if len(r[kk])]) for kk in keys}
    agg["conv"] = np.concatenate([np.array(r["conv"], bool) for r in recs if len(r["conv"])])
    agg["n_params"] = len(recs)
    agg["n_solutions"] = int(agg["direct"].size)
    torch.save(agg, OUT)

    import csv
    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a4", "a2", "k", "n_zero_excluded", "direct_rel_l2", "post_rel_l2",
                    "post_residual", "newton_steps", "converged"])
        for a4, a2, k, nz, d, po, rs, st, cv in zip(agg["a4"], agg["a2"], agg["k"],
                                                    agg["n_zero"], agg["direct"], agg["post"],
                                                    agg["resid"], agg["steps"], agg["conv"]):
            w.writerow([f"{a4:.6f}", f"{a2:.6f}", int(k), int(nz), f"{d:.6e}", f"{po:.6e}",
                        f"{rs:.6e}", int(st), int(bool(cv))])
    print(f"wrote {agg['n_solutions']} rows (non-zero branches only) -> {CSV}")
    print(f"k (non-zero multiplicity) values present: {sorted(set(agg['k'].astype(int).tolist()))}")

    def s(x):
        a = np.asarray(x, float)
        return f"median={np.median(a):.3e} mean={a.mean():.3e} p90={np.percentile(a,90):.3e}"
    print("DIRECT rel-L2 :", s(agg["direct"]))
    print("POST   rel-L2 :", s(agg["post"]))
    print("POST   steps  :", s(agg["steps"][agg["conv"]]), f"  converged {agg['conv'].mean()*100:.1f}%")
    print(f"solutions={agg['n_solutions']}  saved -> {OUT}")


if __name__ == "__main__":
    main()
