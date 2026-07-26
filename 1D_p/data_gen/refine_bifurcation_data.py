"""Find the MISSED branches (Newton refinement fails at the fold) for panel C.

A converged Newton refine lands exactly on its GT branch, so we do NOT need to refine
the ~99% that clearly converge; we only refine the fold-window candidates whose DIRECT
prediction is far off, and keep the ones whose refine still fails (post rel-L2 > 1e-2).

Saved -> data_gen/bif_missed.pt   (list of {p, branch})
Run:  python refine_bifurcation_data.py     (fast: ~120 refines)
"""
import os, sys, numpy as np, torch
import scipy.optimize as sciopt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
from model.traditional_refine import refine_with_newton_trace

GEN = os.path.join(ROOT, "test", "generated_solutions.pt")
LK = os.path.join(ROOT, "data", "p_solutions_lookup.pt")
OUT = os.path.join(HERE, "bif_missed.pt")
WINDOWS = [(15.9, 16.3), (17.6, 18.0)]    # the two fold windows (genuine misses live only here)
DIRECT_CAND = 0.2                          # only refine in-window matches with direct rel-L2 above this
POST_FAIL = 1e-2                           # refine still above this == a missed branch
NEWTON_TOL, NEWTON_MAX = 1e-9, 30

_bid = torch.load(LK, weights_only=False)["branch_ids_by_p"]


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def in_window(p):
    return any(lo <= p < hi for lo, hi in WINDOWS)


def main():
    g = torch.load(GEN, map_location="cpu", weights_only=False)
    missed, refined = [], 0
    for rec in g:
        p = float(rec["p"])
        if not in_window(p):
            continue
        gen = rec["generated"]
        if gen is None or len(gen) == 0:
            continue
        gt = rec["gt"].numpy().astype(np.float64)
        gen = gen.numpy().astype(np.float64)
        bids = _bid[int(rec["idx"])]
        bids = bids.numpy() if torch.is_tensor(bids) else np.asarray(bids)
        if len(bids) != gt.shape[0]:
            continue
        cost = np.linalg.norm(gen[:, None, :] - gt[None, :, :], axis=2)
        r, c = sciopt.linear_sum_assignment(cost)
        for gi, gj in zip(r, c):
            if rel_l2(gen[gi], gt[gj]) <= DIRECT_CAND:
                continue                                  # good direct -> refines fine
            refined += 1
            ru, ok, _, _ = refine_with_newton_trace(torch.tensor(gen[gi]), p,
                                                    tol=NEWTON_TOL, max_iter=NEWTON_MAX)
            ru = ru.detach().cpu().numpy().astype(np.float64).reshape(-1)
            if (not ok) or rel_l2(ru, gt[gj]) > POST_FAIL:
                missed.append({"p": p, "branch": int(bids[gj])})
    print(f"refined {refined} window candidates; missed branches (Newton fails) = {len(missed)}")
    torch.save(missed, OUT)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
