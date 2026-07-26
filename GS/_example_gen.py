"""Unified single-param example generator (one TEST param, fixed seed).
Produces, from ONE reproducible run, BOTH figures' data so they are consistent:
  (a) the full solution set of det U sigma=0.1 x PASSES  -> Fig 1 (max-distinct)
  (b) per-sigma cumulative-distinct curves               -> noise curve
By construction Fig1's solution count == noise-curve(sigma=0.1) endpoint ==
the grid 'total' protocol (det + 5 noise passes at sigma=0.1).

  python _example_gen.py --setup L1 --ti 576 --ckpt L1/ckpt/epoch_60
  python _example_gen.py --setup L2 --ti 595 --ckpt L2/ckpt/epoch_60 --L2 --maxiter 30000 --early_iter 0
"""
import os, sys, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
COVER = 0.15        # D4 rel-L2 threshold for dedup / GT-match
SEED = 1234         # fixed -> reproducible counts
SIGMAS = [0.05, 0.1, 0.2]
FIG_SIGMA = 0.1     # which sigma's union is the Fig1 solution set (grid/headline protocol)
PASSES = 5          # det + 5 noise passes (== grid 'total' protocol)


def d4_set(A):
    g = [torch.rot90(A, k, (-2, -1)) for k in range(4)]
    At = A.transpose(-1, -2)
    g += [torch.rot90(At, k, (-2, -1)) for k in range(4)]
    return torch.stack(g)


def d4_rel_l2(a, b):
    gb = d4_set(b)
    num = (a[None] - gb).flatten(2).norm(dim=2)
    den = b.flatten(1).norm(dim=1).clamp_min(1e-8)
    return (num / den).mean(1).min().item()


def main(a):
    sys.path.insert(0, os.path.join(HERE, a.setup, "code"))
    from config import Config
    from model.v3_model import GSPDEModel
    from gs_torch import GSOperator
    from eval_ord import solve_track

    C = Config()
    lookup = torch.load(C.data_lookup_path, weights_only=False)
    ns = torch.load(C.norm_stats_path)
    params = lookup["p_values"]; sols = lookup["solutions_by_p"]
    model = GSPDEModel.from_pretrained(a.ckpt, C, local_rank=0,
                                       p_mean=ns["p_mean"], p_std=ns["p_std"]); model.eval()
    op = GSOperator(C.op_path, device="cuda", dtype=torch.float64)
    if a.L2:
        op.DA /= 4.0; op.DS /= 4.0

    ti = a.ti
    rho, mu = float(params[ti, 0]), float(params[ti, 1])
    true = sols[ti]; Kt = true.shape[0]

    def refine_into(fields, kept, is_gt):
        o = solve_track(op, rho, mu, fields[:, 0], fields[:, 1], tol=1e-9,
                        maxiter=a.maxiter, early_iter=a.early_iter, early_tol=0.05)
        for k in np.where(o["conv"])[0]:
            f = torch.tensor(np.stack([o["A"][k], o["S"][k]]), dtype=torch.float32)
            if all(d4_rel_l2(f, g) > COVER for g in kept):
                kept.append(f)
                ds = [d4_rel_l2(f, true[j]) for j in range(Kt)]
                is_gt.append(bool(min(ds) < COVER))
        return kept, is_gt

    # shared deterministic pass (no noise -> fixed regardless of seed, but seed anyway)
    torch.manual_seed(SEED)
    det_kept, det_gt = refine_into(model.generate_fixed_k(params[ti], 24), [], [])
    det0 = len(det_kept)

    curves = {}; fig_sols = None; fig_gt = None
    for si, sg in enumerate(SIGMAS):
        kept, is_gt = list(det_kept), list(det_gt)
        cum = [len(kept)]
        for p in range(PASSES):
            torch.manual_seed(SEED + 1000 * (si + 1) + p)   # fixed per (sigma, pass)
            kept, is_gt = refine_into(model.generate_fixed_k(params[ti], 24, noise_std=sg), kept, is_gt)
            cum.append(len(kept))
        curves[str(sg)] = cum
        if abs(sg - FIG_SIGMA) < 1e-9:
            fig_sols = torch.stack(kept) if kept else torch.zeros(0, 2, 128, 128)
            fig_gt = torch.tensor(is_gt)
        print(f"  sigma={sg}: cum={cum}", flush=True)

    out = os.path.join(HERE, a.setup, "results", f"fig1_{a.setup}_ti{ti}_noise.pt")
    torch.save({"sols": fig_sols, "is_gt": fig_gt, "ti": ti, "param": (rho, mu),
                "gt_count": Kt, "noise": FIG_SIGMA, "passes": PASSES,
                "sigmas": SIGMAS, "curves": curves, "det0": det0}, out)
    print(f"{a.setup} ti={ti} ({rho:.4f},{mu:.4f}) Kt={Kt} det0={det0} "
          f"sigma{FIG_SIGMA}-union={curves[str(FIG_SIGMA)][-1]} "
          f"(GT={int(fig_gt.sum())}/beyond={int((~fig_gt.bool()).sum())}) -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--setup", required=True, choices=["L1", "L2"])
    p.add_argument("--ti", type=int, required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--L2", action="store_true")
    p.add_argument("--maxiter", type=int, default=8000)
    p.add_argument("--early_iter", type=int, default=3000)
    main(p.parse_args())
