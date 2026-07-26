"""1D_p EXTRAPOLATION ground truth.

The training/test data stops at p=18.  Here we obtain the TRUE coexisting
solutions at EXTRAPOLATED parameters p in {18.01, 18.1, 18.5, 19, 20} by the
traditional method requested: take the known solution branches at the boundary
p=18 (data_gen/cbmfem/initial_S2.npy, 8 columns; col 0 is the trivial u==0 which
is dropped) and CONTINUE each branch outward in p with damped-Newton parameter
continuation (the same FDM operator faSh1 that generated the training GT).

A branch that folds / stops converging before reaching a target p genuinely has
no real solution there -> it is simply absent from the GT at that p (this is how
the true coexisting-solution count at the extrapolated parameter is obtained).

Output -> extension/gt_extension.pt :
    {"p_targets": [...], "gt": {p: ndarray [k,1024]}, "provenance": {p: [branch_id...]}}
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
# faSh1 (FDM residual/Jacobian) + the damped-Newton used by post-processing
sys.path.insert(0, os.path.join(HERE, "..", "data_gen", "cbmfem"))
from Sexample1 import faSh1
from model.traditional_refine import _damped_newton_solve

import torch

P_BOUNDARY = 18.0
P_TARGETS = [18.01, 18.05, 18.1, 18.5, 19.0]
SEED_NPY = os.path.join(HERE, "..", "data_gen", "cbmfem", "initial_S2.npy")
OUT = os.path.join(HERE, "gt_extension.pt")

NEWTON_TOL = 1e-9           # damped-Newton stops here (the 1024-dim FDM floor is ~1e-11)
NEWTON_MAX = 80
ACCEPT_RESID = 1e-7         # accept a converged solution by its FINAL residual (robust to
                            # line-search stalling just above an over-tight tol)
MAX_DIFF = 0.25             # ||u_new - u_old||_inf jump guard (run_branch.py uses 0.2)
ZERO_NORM = 1e-2            # drop trivial / collapsed branches
DEDUP_RELL2 = 1e-3         # merge branches that have coalesced


def solve_at(u_seed, p):
    """One Newton solve of F(u;p)=0 from a seed.  Returns (u_real[1024] or None, reason).
    Acceptance is by the final residual (not just the solver's ok flag), since the damped
    line search can stall a hair above an over-tight tol while ||F|| is already ~1e-11."""
    x0 = np.asarray(u_seed, np.float64).reshape(-1, 1).astype(np.complex128)
    F, dF = faSh1(x0, p)
    x, ok, reason = _damped_newton_solve(F, dF, x0.copy(), tol=NEWTON_TOL, max_iter=NEWTON_MAX)
    r = float(np.linalg.norm(F(x)))
    if (not np.isfinite(r)) or (r > ACCEPT_RESID):
        return None, f"{reason}:resid={r:.2e}"
    if np.linalg.norm(np.imag(x[:, 0])) > 1e-6:
        return None, "imag_too_large"
    return np.real(x[:, 0]).astype(np.float64), "ok"


def march(u_seed, p_from, p_to, dp0=0.02):
    """Adaptive parameter continuation from p_from to p_to.  Returns u at p_to or None."""
    p, u, dp = p_from, np.asarray(u_seed, np.float64).copy(), dp0
    while p < p_to - 1e-12:
        step = min(dp, p_to - p)
        pn = p + step
        un, _ = solve_at(u, pn)
        if (un is None) or (np.linalg.norm(un - u, np.inf) > MAX_DIFF):
            dp *= 0.5
            if dp < 1e-5:
                return None                       # fold / branch ends before p_to
            continue
        u, p = un, pn
        dp = min(dp0, dp * 1.5)                    # grow step back after a success
    return u


def main():
    S = np.load(SEED_NPY)                          # [1024, 8] complex
    seeds = {b: np.real(S[:, b]).astype(np.float64) for b in range(S.shape[1])
             if np.linalg.norm(np.real(S[:, b])) > ZERO_NORM}   # drop trivial col 0
    print(f"non-trivial boundary branches at p={P_BOUNDARY}: {sorted(seeds)}", flush=True)

    targets = sorted(P_TARGETS)
    gt = {p: [] for p in targets}
    prov = {p: [] for p in targets}

    for b, u0 in seeds.items():
        u, p_cur = u0.copy(), P_BOUNDARY
        for p in targets:
            u_t = march(u, p_cur, p)
            if u_t is None:
                print(f"  branch {b}: folds before p={p}  -> absent from p>={p}", flush=True)
                break
            gt[p].append(u_t); prov[p].append(b)
            u, p_cur = u_t, p                       # advance the seed
        else:
            print(f"  branch {b}: survives all targets up to p={targets[-1]}", flush=True)

    # de-duplicate coalesced branches per target
    out_gt, out_prov = {}, {}
    for p in targets:
        arr = gt[p]
        keep, keep_b = [], []
        for u, b in zip(arr, prov[p]):
            if any(np.linalg.norm(u - v) / (np.linalg.norm(v) + 1e-12) < DEDUP_RELL2 for v in keep):
                continue
            keep.append(u); keep_b.append(b)
        out_gt[p] = np.stack(keep) if keep else np.zeros((0, S.shape[0]))
        out_prov[p] = keep_b
        print(f"  p={p}: GT branches = {len(keep)}  (from boundary branches {keep_b})", flush=True)

    torch.save({"p_targets": targets, "gt": out_gt, "provenance": out_prov,
                "boundary": P_BOUNDARY}, OUT)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
