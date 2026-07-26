"""Noise-sweep curve data: for each TEST param and each sigma, record the cumulative
#distinct after the det pass and each added noise pass (union, D4-dedup).
x = # forward passes (1 = det only; +1 per noise pass); curves = different sigma.
Sharded with --idx_start/--idx_end. -> per-param {sigma: [c0,c1,...,cP]}.

  python eval_noise_curve.py --code ../code --ckpt ../ckpt/epoch_60 --sigmas 0.05,0.1,0.2 --passes 6 --out ...
  (L=2: add --L2 --maxiter 30000 --early_iter 0)
"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
COVER = 0.15


def main(a):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    sys.path.insert(0, os.path.abspath(a.code))
    import torch
    from config import Config
    from model.v3_model import GSPDEModel
    from gs_torch import GSOperator
    from eval_ord import solve_track, d4_rel_l2
    C = Config()
    lookup = torch.load(C.data_lookup_path, weights_only=False); ns = torch.load(C.norm_stats_path)
    params = lookup["p_values"]; sols = lookup["solutions_by_p"]
    op = GSOperator(C.op_path, device="cuda", dtype=torch.float64)
    if a.L2:
        op.DA /= 4.0; op.DS /= 4.0
    model = GSPDEModel.from_pretrained(os.path.abspath(a.ckpt), C, local_rank=0,
                                       p_mean=ns["p_mean"], p_std=ns["p_std"]); model.eval()
    sigmas = [float(s) for s in a.sigmas.split(",")]
    base = torch.load(C.test_p_idx_path).tolist()
    hi = a.idx_end if a.idx_end >= 0 else len(base)
    targets = base[a.idx_start:hi]

    def refine_dedup(pool, kept):
        o = solve_track(op, float(params_ti[0]), float(params_ti[1]), pool[:, 0], pool[:, 1],
                        tol=1e-9, maxiter=a.maxiter, early_iter=a.early_iter, early_tol=0.05)
        for k in np.where(o["conv"])[0]:
            f = torch.tensor(np.stack([o["A"][k], o["S"][k]]), dtype=torch.float32)
            if all(d4_rel_l2(f, g) > COVER for g in kept):
                kept.append(f)
        return kept

    out = {}
    for n, ti in enumerate(targets):
        params_ti = (float(params[ti, 0]), float(params[ti, 1]))
        det = refine_dedup(model.generate_fixed_k(params[ti], 24), [])   # det pass (shared)
        rec = {}
        for sg in sigmas:
            kept = list(det); cum = [len(kept)]
            for _ in range(a.passes):
                kept = refine_dedup(model.generate_fixed_k(params[ti], 24, noise_std=sg), kept)
                cum.append(len(kept))
            rec[str(sg)] = cum
        out[int(ti)] = rec
        if n % 10 == 0:
            print(f"  {n}/{len(targets)} p{ti}: det={len(det)} " +
                  " ".join(f"s{sg}->{rec[str(sg)][-1]}" for sg in sigmas), flush=True)
    json.dump({"sigmas": sigmas, "passes": a.passes, "recs": out}, open(a.out, "w"))
    print(f"DONE [{a.idx_start}:{hi}] n={len(out)} -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--code", default="../code"); p.add_argument("--ckpt", default="../ckpt/epoch_60")
    p.add_argument("--sigmas", default="0.05,0.1,0.2"); p.add_argument("--passes", type=int, default=6)
    p.add_argument("--idx_start", type=int, default=0); p.add_argument("--idx_end", type=int, default=-1)
    p.add_argument("--maxiter", type=int, default=8000); p.add_argument("--early_iter", type=int, default=3000)
    p.add_argument("--L2", action="store_true"); p.add_argument("--out", required=True)
    main(p.parse_args())
