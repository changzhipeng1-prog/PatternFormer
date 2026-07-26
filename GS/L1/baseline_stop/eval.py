"""Evaluate the trained generator on OOD test parameters (no-context, fixed-K).

B1: uses model.generate_fixed_k (force K, ignore STOP) -- the production path.
The old STOP-honoring model.generate() under-generated (count collapse).

For each test param: generate the solution set, compare to ground truth:
  - K_pred vs K_true (does the model emit the right NUMBER of solutions?)
  - D4-aware rel_l2 of each generated solution to its nearest true solution
  - coverage: fraction of true solutions matched within a tolerance
Visualizes true vs generated for a few params.
"""
import argparse, os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import Config
from model.v3_model import GSPDEModel


def d4_set(A):
    """A:[2,128,128] -> [8,2,128,128] dihedral transforms."""
    g = [torch.rot90(A, k, (-2, -1)) for k in range(4)]
    At = A.transpose(-1, -2)
    g += [torch.rot90(At, k, (-2, -1)) for k in range(4)]
    return torch.stack(g)


def d4_rel_l2(a, b):
    """min over D4 of per-channel rel_l2(a, g(b)); a,b:[2,128,128]."""
    gb = d4_set(b)                                  # [8,2,128,128]
    num = (a[None] - gb).flatten(2).norm(dim=2)     # [8,2]
    den = b.flatten(1).norm(dim=1).clamp_min(1e-8)  # [2]
    return (num / den).mean(1).min().item()


def main(a):
    C = Config()
    lookup = torch.load(a.lookup or C.data_lookup_path, weights_only=False)
    test_idx = torch.load(a.test_idx or C.test_p_idx_path).tolist()
    ns = torch.load(a.norm_stats or C.norm_stats_path)
    model = GSPDEModel.from_pretrained(a.ckpt, C, local_rank=0,
                                       p_mean=ns["p_mean"], p_std=ns["p_std"])
    model.eval()
    params = lookup["p_values"]; sols = lookup["solutions_by_p"]
    train_idx = torch.load(a.train_idx or C.train_p_idx_path).tolist()

    # context library = TRAIN params (the "known solved" set). In local mode a
    # target draws its nearest train params (normalized (rho,mu) distance); this
    # matches deployment and is ~4x more relevant than random context.
    mode = a.context_mode
    tr_p = params[torch.as_tensor(train_idx, dtype=torch.long)].float()
    p_scale = tr_p.std(0).clamp_min(1e-8)
    def pick_ctx(ti, rng):
        if mode == "local":
            d = ((tr_p - params[ti].float()) / p_scale).pow(2).sum(1).sqrt()
            order = [train_idx[i] for i in torch.argsort(d).tolist() if train_idx[i] != ti]
            return order[:C.max_context_p]
        pool = [x for x in train_idx if x != ti]
        return rng.choice(pool, size=min(C.max_context_p, len(pool)), replace=False).tolist()

    rng = np.random.RandomState(0)
    rows = []           # (target_idx, K_true, K_pred, mean_rel, coverage)
    viz = []
    n_eval = min(a.n_params, len(test_idx))
    targets = test_idx[:n_eval]
    for ti in targets:
        gen = [torch.as_tensor(x, dtype=torch.float32)
               for x in model.generate_fixed_k(params[ti], a.max_gen)]  # B1: fixed-K, ignore STOP
        true = sols[ti]                                                          # [K,2,128,128]
        Kt, Kp = true.shape[0], len(gen)
        # greedy D4-aware match each generated -> nearest true
        rels = []
        matched_true = set()
        for g in gen:
            ds = [d4_rel_l2(g, true[j]) for j in range(Kt)]
            j = int(np.argmin(ds)); rels.append(ds[j])
            if ds[j] < 0.15:
                matched_true.add(j)
        mean_rel = float(np.mean(rels)) if rels else float("nan")
        cov = len(matched_true) / Kt
        rows.append((ti, Kt, Kp, mean_rel, cov))
        if len(viz) < 6:
            viz.append((ti, true, gen))
        print(f"p{ti} (ρ={params[ti,0]:.3f},μ={params[ti,1]:.3f}): "
              f"K_true={Kt} K_gen={Kp} mean_relL2={mean_rel:.3f} coverage={cov:.2f}")

    # in-distribution (train param) count check: is the K-collapse OOD-only?
    tr_targets = train_idx[:12]
    tr_kt, tr_kp = [], []
    for ti in tr_targets:
        gen = model.generate_fixed_k(params[ti], a.max_gen)
        tr_kt.append(sols[ti].shape[0]); tr_kp.append(len(gen))
    print(f"\n[in-distribution train params] K_true mean {np.mean(tr_kt):.1f}  "
          f"K_gen mean {np.mean(tr_kp):.1f}  (per: true={tr_kt} gen={tr_kp})")

    arr = np.array([(r[1], r[2], r[3], r[4]) for r in rows], float)
    print(f"\n=== over {len(rows)} test params ===")
    print(f"K_true mean {arr[:,0].mean():.1f}  K_gen mean {arr[:,1].mean():.1f}  "
          f"|ΔK| mean {np.abs(arr[:,0]-arr[:,1]).mean():.1f}")
    print(f"gen rel_l2 mean {np.nanmean(arr[:,2]):.3f}  median {np.nanmedian(arr[:,2]):.3f}")
    print(f"coverage mean {arr[:,3].mean():.2f}")

    # visualize true (top) vs gen (bottom) for a few params
    for vi, (ti, true, gen) in enumerate(viz):
        ncol = max(true.shape[0], len(gen), 1)
        fig, ax = plt.subplots(2, ncol, figsize=(1.3 * ncol, 3.0), squeeze=False)
        for j in range(ncol):
            for r in range(2): ax[r, j].axis("off")
            if j < true.shape[0]:
                ax[0, j].imshow(true[j, 0].numpy(), cmap="viridis", vmin=0, vmax=0.6)
            if j < len(gen):
                ax[1, j].imshow(gen[j][0].float().numpy(), cmap="viridis", vmin=0, vmax=0.6)
        ax[0, 0].set_ylabel("TRUE", fontsize=9); ax[1, 0].set_ylabel("GEN", fontsize=9)
        fig.suptitle(f"ρ={params[ti,0]:.3f} μ={params[ti,1]:.3f}  K_true={true.shape[0]} K_gen={len(gen)}",
                     fontsize=10)
        fig.tight_layout(); fig.savefig(f"{os.path.dirname(__file__)}/outputs/eval_p{ti}.png", dpi=110)
    print(f"wrote {len(viz)} comparison figs to outputs/eval_p*.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/tmp/eval_ckpt")
    p.add_argument("--n_params", type=int, default=40)
    p.add_argument("--max_gen", type=int, default=24)
    # data-path overrides (e.g. evaluate a model trained on the *_big dataset)
    p.add_argument("--lookup", type=str, default=None)
    p.add_argument("--train_idx", type=str, default=None)
    p.add_argument("--test_idx", type=str, default=None)
    p.add_argument("--norm_stats", type=str, default=None)
    p.add_argument("--context_mode", type=str, default="random", choices=["random", "local"])
    main(p.parse_args())
