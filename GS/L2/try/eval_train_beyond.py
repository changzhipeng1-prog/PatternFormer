"""Appendix experiment: at TRAINING-set Gray-Scott parameters, run LLM-MSO with the
deterministic pass + a few noise-augmented passes, refine and D4-dedup, and check
whether the model still produces solutions BEYOND the training data (is_gt=False).

For each chosen training param we save the full distinct solution set with a GT-matched
flag (fig1-style), so two examples can be drawn Fig.3B-style into the appendix.

  python eval_train_beyond.py --code ../code --ckpt ../ckpt/epoch_60 \
      --sigma 0.1 --passes 5 --n_params 8 --out results/train_beyond.pt
"""
import os, sys, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COVER = 0.15   # D4-rel-L2 dedup / GT-match threshold (paper protocol)


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
    op.DA /= 4.0; op.DS /= 4.0   # L2 (four-fold-smaller) coefficients
    model = GSPDEModel.from_pretrained(os.path.abspath(a.ckpt), C, local_rank=0,
                                       p_mean=ns["p_mean"], p_std=ns["p_std"]); model.eval()

    train = torch.load(C.train_p_idx_path).tolist()
    # GT (D4-reduced) count per training param; pick the highest-multiplicity ones for rich galleries
    gtc = {ti: (0 if sols[ti] is None else len(sols[ti])) for ti in train}
    chosen = sorted(train, key=lambda t: -gtc[t])[:a.n_params]
    print(f"chosen {len(chosen)} training params (by GT count): "
          + ", ".join(f"ti{t}(K={gtc[t]})" for t in chosen), flush=True)

    def gt_fields(ti):
        s = sols[ti]
        if s is None: return []
        s = s if torch.is_tensor(s) else torch.as_tensor(np.asarray(s))
        return [s[j].float() for j in range(s.shape[0])]

    def refine_dedup(pool, kept, p0, p1):
        o = solve_track(op, float(p0), float(p1), pool[:, 0], pool[:, 1],
                        tol=1e-9, maxiter=a.maxiter, early_iter=a.early_iter, early_tol=0.05)
        for k in np.where(o["conv"])[0]:
            f = torch.tensor(np.stack([o["A"][k], o["S"][k]]), dtype=torch.float32)
            if all(d4_rel_l2(f, g) > COVER for g in kept):
                kept.append(f)
        return kept

    out = {}
    for n, ti in enumerate(chosen):
        p0, p1 = float(params[ti, 0]), float(params[ti, 1])
        kept = refine_dedup(model.generate_fixed_k(params[ti], 24), [], p0, p1)            # det pass
        for _ in range(a.passes):                                                           # noise passes
            kept = refine_dedup(model.generate_fixed_k(params[ti], 24, noise_std=a.sigma), kept, p0, p1)
        gts = gt_fields(ti)
        is_gt = [bool(any(d4_rel_l2(f, g) <= COVER for g in gts)) for f in kept]            # matched to training data?
        sol_stack = torch.stack(kept) if kept else torch.zeros(0, 2, 128, 128)
        out[int(ti)] = {"sols": sol_stack, "is_gt": torch.tensor(is_gt), "param": (p0, p1),
                        "gt_count": gtc[ti]}
        nmatch = int(sum(is_gt)); nnew = len(kept) - nmatch
        print(f"  [{n+1}/{len(chosen)}] ti{ti} p=({p0:.4f},{p1:.4f}) GT={gtc[ti]}: "
              f"total={len(kept)}  matched={nmatch}  NEW(beyond-data)={nnew}", flush=True)

    torch.save(out, a.out)
    tot_new = sum(int((~r["is_gt"].bool()).sum()) for r in out.values())
    print(f"\nDONE -> {a.out}  ({len(out)} params, {tot_new} total beyond-data solutions)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--code", default="../code"); p.add_argument("--ckpt", default="../ckpt/epoch_60")
    p.add_argument("--sigma", type=float, default=0.1); p.add_argument("--passes", type=int, default=5)
    p.add_argument("--n_params", type=int, default=8)
    p.add_argument("--maxiter", type=int, default=30000); p.add_argument("--early_iter", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "results", "train_beyond.pt"))
    main(p.parse_args())
