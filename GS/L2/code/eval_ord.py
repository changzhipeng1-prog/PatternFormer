"""Eval for qwen_ord (ordered, NO-STOP, fixed-K) — run FROM the project dir (cwd has config.py).

Per test param:
  1. generate_fixed_k(p, K) -> K initial-guess fields (deterministic ordered free-run).
  2. residual stats of the RAW guesses (before refine).
  3. FDM quasi-Newton refine (lenient) -> converged steady states.
  4. D4-dedup -> # DISTINCT solutions + GT coverage.

Reports det_distinct/param + coverage + init-residual. Compares to baseline (L1: 2.53 / 4.27).
Run (L1, from qwen_ord/):     python eval_ord.py --ckpt ./checkpoints_final/best_model --out ./eval/ord_eval.json
Run (L2, from qwen_ord_L2/):  python eval_ord.py --L2 --baseline_det 2.0 --ckpt ./checkpoints_final/best_model --out ./eval/ord_eval.json
"""
import os, sys, argparse, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# self-contained: config.py, model/, gs_torch.py are all bundled in this dir (code/)
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np, torch
from config import Config
from model.v3_model import GSPDEModel
from gs_torch import GSOperator

DEV = torch.device("cuda:0")


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


def _residual_maxabs(op, rho, mu, A, S):
    dt = op.dtype
    A = torch.as_tensor(A, dtype=dt, device=DEV); S = torch.as_tensor(S, dtype=dt, device=DEV)
    B = A.shape[0]
    rho = torch.as_tensor(rho, dtype=dt, device=DEV).expand(B).reshape(B, 1, 1)
    mu = torch.as_tensor(mu, dtype=dt, device=DEV).expand(B).reshape(B, 1, 1)
    rA = op.DA * op.lap(A) + (-S * A * A + (mu + rho) * A)
    rS = op.DS * op.lap(S) + (S * A * A - rho * (1.0 - S))
    return torch.maximum(rA.abs().amax((1, 2)), rS.abs().amax((1, 2))).cpu().numpy()


def solve_track(op, rho, mu, A0, S0, tol, maxiter, early_iter, early_tol, stepsize=0.1):
    dt = op.dtype
    A = torch.as_tensor(A0, dtype=dt, device=DEV).clone(); B, n, _ = A.shape
    rho = torch.as_tensor(rho, dtype=dt, device=DEV).expand(B).reshape(B, 1, 1)
    mu = torch.as_tensor(mu, dtype=dt, device=DEV).expand(B).reshape(B, 1, 1)
    S = torch.as_tensor(S0, dtype=dt, device=DEV).clone()
    active = torch.ones(B, dtype=torch.bool, device=DEV)
    crit = torch.full((B,), float("inf"), dtype=dt, device=DEV)
    DA, DS, Lam = op.DA, op.DS, op.Lam
    for it in range(1, maxiter + 1):
        rA = DA * op.lap(A) + (-S * A * A + (mu + rho) * A)
        rS = DS * op.lap(S) + (S * A * A - rho * (1.0 - S))
        c = torch.maximum(rA.abs().amax((1, 2)), rS.abs().amax((1, 2)))
        crit = torch.where(active, c, crit)
        diverged = (~torch.isfinite(c)) | (c > 1000.0)
        active = active & ~diverged & ~(c < tol)
        if it == early_iter:
            active = active & ~(c > early_tol)
        if not active.any():
            break
        m1U = (mu + rho) - 2*A*S; m2V = rho + A*A; m1V = -A*A; m2U = 2*A*S
        disc = torch.sqrt(torch.clamp((m1U-m2V)**2 + 4*m1V*m2U, min=0.0)); half = (m1U+m2V)/2
        beta = ((half+disc/2).amax((1, 2)) + (half-disc/2).amin((1, 2))).reshape(B, 1, 1)/2
        dA = torch.nan_to_num(op.inv_shift(rA, DA*Lam+beta)); dS = torch.nan_to_num(op.inv_shift(rS, DS*Lam+beta))
        mm = active.reshape(B, 1, 1)
        A = torch.where(mm, A - stepsize*dA, A); S = torch.where(mm, S - stepsize*dS, S)
    conv = torch.isfinite(crit) & (crit < tol)
    rngA = A.reshape(B, -1).amax(1) - A.reshape(B, -1).amin(1)
    return dict(A=A.cpu().numpy(), S=S.cpu().numpy(),
                conv=(conv & (rngA > 0.05)).cpu().numpy())


def eval_all(model, op, params, sols, targets, K, maxiter, early_iter, early_tol):
    recs = []; init_res_all = []
    for n, ti in enumerate(targets):
        rho, mu = float(params[ti, 0]), float(params[ti, 1])
        fields = model.generate_fixed_k(params[ti], K)              # [K,2,128,128]
        ir = _residual_maxabs(op, rho, mu, fields[:, 0], fields[:, 1]); init_res_all.append(ir)
        o = solve_track(op, rho, mu, fields[:, 0], fields[:, 1], tol=1e-9,
                        maxiter=maxiter, early_iter=early_iter, early_tol=early_tol)
        true = sols[ti]; Kt = true.shape[0]
        kept = []; matched = set()
        for k in np.where(o["conv"])[0]:
            f = torch.tensor(np.stack([o["A"][k], o["S"][k]]), dtype=torch.float32)
            if all(d4_rel_l2(f, g) > 0.15 for g in kept):
                kept.append(f)
                ds = [d4_rel_l2(f, true[j]) for j in range(Kt)]; j = int(np.argmin(ds))
                if ds[j] < 0.15:
                    matched.add(j)
        recs.append(dict(ti=int(ti), Kt=int(Kt), conv=int(o["conv"].sum()),
                         distinct=len(kept), cov=len(matched) / Kt))
        if n % 10 == 0:
            print(f"  {n}/{len(targets)} p{ti} Kt={Kt} conv={int(o['conv'].sum())} "
                  f"distinct={len(kept)} cov={len(matched)/Kt:.2f}", flush=True)
    ir = np.concatenate(init_res_all)
    dist = np.array([r["distinct"] for r in recs], float)
    cov = np.array([r["cov"] for r in recs], float)
    kt = np.array([r["Kt"] for r in recs], float)
    return dict(n_params=len(recs), K=K,
                det_distinct_mean=float(dist.mean()), coverage_mean=float(cov.mean()),
                Ktrue_mean=float(kt.mean()),
                init_residual={"mean": float(ir.mean()), "median": float(np.median(ir)),
                               "p90": float(np.percentile(ir, 90)), "max": float(ir.max())},
                recs=recs)


def main(a):
    C = Config()
    if a.data_lookup: C.data_lookup_path = a.data_lookup
    if a.test_idx: C.test_p_idx_path = a.test_idx
    if a.norm_stats: C.norm_stats_path = a.norm_stats
    lookup = torch.load(C.data_lookup_path, weights_only=False)
    test_idx = torch.load(C.test_p_idx_path).tolist()
    ns = torch.load(C.norm_stats_path)
    params = lookup["p_values"]; sols = lookup["solutions_by_p"]
    model = GSPDEModel.from_pretrained(a.ckpt, C, local_rank=0,
                                       p_mean=ns["p_mean"], p_std=ns["p_std"]); model.eval()
    op = GSOperator(C.op_path, device="cuda", dtype=torch.float64)
    if a.L2:
        op.DA /= 4.0; op.DS /= 4.0

    base = list(range(len(params))) if a.all_params else list(test_idx)
    hi = a.idx_end if a.idx_end >= 0 else len(base)
    targets = base[a.idx_start:hi]
    if not a.all_params and a.n > 0:
        targets = test_idx[:a.n]
    K = a.K if a.K > 0 else C.fixed_k
    print(f"=== qwen_ord eval ckpt={a.ckpt} K={K} n={len(targets)} L2={a.L2} "
          f"(FDM maxiter={a.maxiter} early={a.early_iter}/{a.early_tol}) ===", flush=True)
    res = eval_all(model, op, params, sols, targets, K, a.maxiter, a.early_iter, a.early_tol)
    res["baseline_det"] = a.baseline_det
    print(f"\n=== SUMMARY === det_distinct={res['det_distinct_mean']:.2f} "
          f"cov={res['coverage_mean']:.3f} Ktrue={res['Ktrue_mean']:.2f} "
          f"init_res(med)={res['init_residual']['median']:.2e}  baseline_det={a.baseline_det}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="./checkpoints_final/best_model")
    p.add_argument("--K", type=int, default=0, help="0 -> config.fixed_k (24)")
    p.add_argument("--n", type=int, default=-1, help="# test params (-1 = all)")
    p.add_argument("--maxiter", type=int, default=8000)
    p.add_argument("--early_iter", type=int, default=3000)
    p.add_argument("--early_tol", type=float, default=0.05)
    p.add_argument("--L2", action="store_true", help="L=2: divide op DA/DS by 4")
    p.add_argument("--all_params", action="store_true", help="eval ALL params (train+val+test)")
    p.add_argument("--idx_start", type=int, default=0)
    p.add_argument("--idx_end", type=int, default=-1)
    p.add_argument("--data_lookup", type=str, default=None)
    p.add_argument("--test_idx", type=str, default=None)
    p.add_argument("--norm_stats", type=str, default=None)
    p.add_argument("--baseline_det", type=float, default=2.53)
    p.add_argument("--out", default="./eval/ord_eval.json")
    main(p.parse_args())
