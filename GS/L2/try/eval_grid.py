"""Per-param solution counts for the 2x3 distribution figure + random-vs-ours.
For each parameter: generate K initial guesses from one SOURCE, FDM-refine, D4-dedup,
and record total / matched(in-dataset) / beyond(new) distinct solutions.

SOURCES (--method):
  random : degree<=4 random polynomial fields (classical cold start, no continuation)
  stop   : the STOP model, fixed-K=24            (--code ../baseline_stop --ckpt ../ckpt/stop_base)
  det    : ordered model, fixed-K=24             (--code ../code --ckpt ../ckpt/epoch_60)
  noise  : ordered model, det + N noise passes   (--code ../code --ckpt ../ckpt/epoch_60 --passes 5)

Sharded with --split {train,test,all} --idx_start --idx_end (run many GPUs in parallel).
"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
COVER = 0.15


def main(a):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    sys.path.insert(0, os.path.abspath(a.code))
    sys.path.insert(0, os.path.join(HERE, "..", "data_gen"))   # gs_continuation.poly_perturb
    import torch, scipy.io as sio
    from config import Config
    from gs_torch import GSOperator
    from eval_ord import solve_track, d4_rel_l2
    from gs_continuation import poly_perturb
    DEV = torch.device("cuda:0")
    C = Config()
    lookup = torch.load(C.data_lookup_path, weights_only=False); ns = torch.load(C.norm_stats_path)
    params = lookup["p_values"]; sols = lookup["solutions_by_p"]
    op = GSOperator(C.op_path, device="cuda", dtype=torch.float64)
    if a.L2:
        op.DA /= 4.0; op.DS /= 4.0
    model = None
    if a.method in ("stop", "det", "noise"):
        from model.v3_model import GSPDEModel
        model = GSPDEModel.from_pretrained(os.path.abspath(a.ckpt), C, local_rank=0,
                                           p_mean=ns["p_mean"], p_std=ns["p_std"]); model.eval()
    xg = op.Tx.new_tensor(sio.loadmat(C.op_path)["x"].ravel())
    gen_t = torch.Generator(device="cuda").manual_seed(a.seed)

    tr = torch.load(C.train_p_idx_path).tolist(); te = torch.load(C.test_p_idx_path).tolist()
    base = {"train": tr, "test": te, "all": list(range(len(params)))}[a.split]
    hi = a.idx_end if a.idx_end >= 0 else len(base)
    targets = base[a.idx_start:hi]

    def pools_for(ti):
        if a.method == "random":
            A0 = poly_perturb(xg, a.rscale, a.M, gen_t, "cuda", op.dtype).cpu().numpy()
            S0 = poly_perturb(xg, a.rscale, a.M, gen_t, "cuda", op.dtype).cpu().numpy()
            return [np.stack([A0, S0], 1)]
        if a.method == "noise":
            return [model.generate_fixed_k(params[ti], a.M)] + \
                   [model.generate_fixed_k(params[ti], a.M, noise_std=a.noise) for _ in range(a.passes)]
        return [model.generate_fixed_k(params[ti], a.M)]      # stop / det

    recs = []
    for n, ti in enumerate(targets):
        rho, mu = float(params[ti, 0]), float(params[ti, 1]); true = sols[ti]; Kt = true.shape[0]
        kept = []; matched = 0
        for pool in pools_for(ti):
            o = solve_track(op, rho, mu, pool[:, 0], pool[:, 1], tol=1e-9,
                            maxiter=a.maxiter, early_iter=a.early_iter, early_tol=0.05)
            for k in np.where(o["conv"])[0]:
                f = torch.tensor(np.stack([o["A"][k], o["S"][k]]), dtype=torch.float32)
                if all(d4_rel_l2(f, g) > COVER for g in kept):
                    kept.append(f)
                    if min(d4_rel_l2(f, true[j]) for j in range(Kt)) < COVER:
                        matched += 1
        total = len(kept)
        recs.append(dict(ti=int(ti), rho=rho, mu=mu, Kt=int(Kt),
                         total=total, matched=matched, beyond=total - matched))
        if n % 20 == 0:
            print(f"  {a.method} {n}/{len(targets)} p{ti} Kt={Kt} total={total} matched={matched} beyond={total-matched}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"method": a.method, "split": a.split, "recs": recs}, open(a.out, "w"), indent=2)
    tot = np.array([r["total"] for r in recs])
    print(f"DONE {a.method}/{a.split}[{a.idx_start}:{hi}] n={len(recs)} mean total={tot.mean():.2f} -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["random", "stop", "det", "noise"])
    p.add_argument("--code", default="../code"); p.add_argument("--ckpt", default="../ckpt/epoch_60")
    p.add_argument("--split", default="all", choices=["train", "test", "all"])
    p.add_argument("--idx_start", type=int, default=0); p.add_argument("--idx_end", type=int, default=-1)
    p.add_argument("--M", type=int, default=24); p.add_argument("--passes", type=int, default=5)
    p.add_argument("--noise", type=float, default=0.1); p.add_argument("--rscale", type=float, default=0.5)
    p.add_argument("--maxiter", type=int, default=8000); p.add_argument("--early_iter", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--L2", action="store_true")
    p.add_argument("--out", required=True)
    main(p.parse_args())
