"""Far extrapolation p=20, 25:  how many branches EXIST, and coverage by
   (A) model DIRECT output at p,
   (B) p18-seed SINGLE Newton jump (one solve at p from the model's p=18 output),
   (C) p18-seed STEPPED continuation (march the model's p=18 output 18 -> p).

GT = continuation of the 7 true p=18 branches (initial_S2) to p (survivors only).
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "data_gen", "cbmfem"))
from gen_gt_extension import solve_at, march, ZERO_NORM, DEDUP_RELL2
from model.traditional_refine import refine_with_newton_trace

TARGETS = [20.0, 25.0]
COVER, DEDUP, REFINE_OK = 1e-2, 1e-3, 1e-6
SEED_NPY = os.path.join(HERE, "..", "data_gen", "cbmfem", "initial_S2.npy")


def rl2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def dedup(sols):
    out = []
    for u in sols:
        if not any(rl2(u, v) < DEDUP for v in out):
            out.append(u)
    return out


def coverage(distinct, gt):
    return sum(1 for g in gt if any(rl2(d, g) < COVER for d in distinct))


def gt_at(p, true18):
    """Continue the true p=18 branches to p; return surviving branches."""
    sols = []
    for u0 in true18:
        u = march(u0, 18.0, p, dp0=0.1)
        if u is not None:
            sols.append(u)
    return dedup(sols)


def refine_all(outs, p):
    """Single Newton refine of a set of seeds at p -> distinct genuine solutions."""
    sols = []
    for u in outs:
        ru, ok, _, tr = refine_with_newton_trace(torch.tensor(u), float(p), tol=1e-9, max_iter=30)
        res = float(tr[-1]["residual"]) if tr else float("inf")
        if ok and res < REFINE_OK:
            sols.append(np.asarray(ru.detach().cpu(), np.float64).reshape(-1))
    return dedup(sols)


def continue_all(outs, p):
    """Stepped continuation of each seed from 18 -> p (model-seeded continuation)."""
    sols = []
    for u in outs:
        u_t = march(np.asarray(u, np.float64), 18.0, p, dp0=0.1)
        if u_t is not None:
            sols.append(u_t)
    return dedup(sols)


def main():
    S = np.load(SEED_NPY)
    true18 = [np.real(S[:, b]).astype(np.float64) for b in range(S.shape[1])
              if np.linalg.norm(np.real(S[:, b])) > ZERO_NORM]          # 7 true branches
    gen18 = torch.load(os.path.join(HERE, "p18seed_result.pt"),
                       map_location="cpu", weights_only=False)["gen18"]   # model p=18 output
    direct = torch.load(os.path.join(HERE, "far_direct.pt"), map_location="cpu", weights_only=False)

    print(f"{'p':>5} | {'n_exist':>7} | {'direct':>6} | {'p18-seed 1-jump':>15} | {'p18-seed stepped':>16}")
    print("-" * 62)
    rows = []
    for p in TARGETS:
        gt = gt_at(p, true18)
        cov_direct = coverage(refine_all(direct[float(p)].numpy().astype(np.float64), p), gt)
        cov_jump = coverage(refine_all(gen18, p), gt)
        cov_step = coverage(continue_all(gen18, p), gt)
        rows.append((p, len(gt), cov_direct, cov_jump, cov_step))
        print(f"{p:>5.0f} | {len(gt):>7} | {cov_direct:>6} | {cov_jump:>15} | {cov_step:>16}")
    torch.save(rows, os.path.join(HERE, "far_p_result.pt"))
    print("\nsaved far_p_result.pt")


if __name__ == "__main__":
    main()
