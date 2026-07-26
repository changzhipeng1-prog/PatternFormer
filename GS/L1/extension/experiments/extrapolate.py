"""Extrapolation WITHOUT training: push the existing model to produce solutions for
parameters that currently yield zero, and for off-grid (rho,mu) with no GT at all.

Two modes (--mode):
  rescue : the in-grid params that the headline full-param eval scored 0 distinct
           (read from eval/full_chunks/*.json). They HAVE GT (Kt>=1) but the det
           single pass found nothing. Goal: how many we can rescue to >=1 solution.
  grid   : a (rho,mu) grid that EXTENDS beyond / between the training grid -> no GT.
           Goal: does the model emit fields that FDM can refine to a real steady state?

Levers (all inference-only, no retraining):
  - noise multipass UNION: N independent noised forward passes, fresh eps each, union of distinct.
  - sigma sweep: try several noise_std, union across all of them.
  - relaxed residual gate: a solution "counts" if FDM max-abs residual < --tol (default 1e-6,
    vs the strict headline 1e-9) and the field is non-trivial (range(A) > 0.05).

Run from qwen_ord/ (or qwen_ord_L2/ with --L2) with the torch124 python.
  CUDA_VISIBLE_DEVICES=0 <py> experiments/extrapolate.py --mode rescue --out experiments/extrap_rescue_L1.json
"""
import os, sys, json, glob, argparse
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
sys.path.insert(0, ROOT)
import numpy as np, torch
from config import Config
from model.v3_model import GSPDEModel
from gs_torch import GSOperator
DEV = torch.device("cuda:0")


def d4_rel_l2(a, b):
    g = [torch.rot90(b, k, (-2, -1)) for k in range(4)]
    bt = b.transpose(-1, -2); g += [torch.rot90(bt, k, (-2, -1)) for k in range(4)]
    gb = torch.stack(g)
    return ((a[None]-gb).flatten(2).norm(dim=2)/b.flatten(1).norm(dim=1).clamp_min(1e-8)).mean(1).min().item()


def solve_track(op, rho, mu, A0, S0, maxiter, early_iter, tol, early_tol=0.3, ftol=1e-9, stepsize=0.1):
    dt = op.dtype
    A = torch.as_tensor(A0, dtype=dt, device=DEV).clone(); B, n, _ = A.shape
    rho = torch.as_tensor(rho, dtype=dt, device=DEV).expand(B).reshape(B, 1, 1)
    mu = torch.as_tensor(mu, dtype=dt, device=DEV).expand(B).reshape(B, 1, 1)
    S = torch.as_tensor(S0, dtype=dt, device=DEV).clone()
    active = torch.ones(B, dtype=torch.bool, device=DEV); crit = torch.full((B,), float("inf"), dtype=dt, device=DEV)
    DA, DS, Lam = op.DA, op.DS, op.Lam
    for it in range(1, maxiter+1):
        rA = DA*op.lap(A) + (-S*A*A + (mu+rho)*A); rS = DS*op.lap(S) + (S*A*A - rho*(1.0-S))
        c = torch.maximum(rA.abs().amax((1, 2)), rS.abs().amax((1, 2))); crit = torch.where(active, c, crit)
        active = active & ~((~torch.isfinite(c)) | (c > 1000.0)) & ~(c < ftol)
        if early_iter and it == early_iter: active = active & ~(c > early_tol)
        if not active.any(): break
        m1U = (mu+rho)-2*A*S; m2V = rho+A*A; m1V = -A*A; m2U = 2*A*S
        disc = torch.sqrt(torch.clamp((m1U-m2V)**2 + 4*m1V*m2U, min=0.0)); half = (m1U+m2V)/2
        beta = ((half+disc/2).amax((1, 2)) + (half-disc/2).amin((1, 2))).reshape(B, 1, 1)/2
        dA = torch.nan_to_num(op.inv_shift(rA, DA*Lam+beta)); dS = torch.nan_to_num(op.inv_shift(rS, DS*Lam+beta))
        mm = active.reshape(B, 1, 1); A = torch.where(mm, A-stepsize*dA, A); S = torch.where(mm, S-stepsize*dS, S)
    conv = torch.isfinite(crit) & (crit < tol); rngA = A.reshape(B, -1).amax(1) - A.reshape(B, -1).amin(1)
    keep = conv & (rngA > 0.05)
    return A.cpu().numpy(), S.cpu().numpy(), keep.cpu().numpy(), crit.cpu().numpy()


def union_generate(m, op, p, K, sigmas, N, maxiter, early_iter, tol):
    """N noised passes per sigma; union of D4-distinct converged solutions. Returns kept fields + residuals."""
    rho, mu = float(p[0]), float(p[1]); kept = []; kept_res = []
    for sg in sigmas:
        for _ in range(N):
            f = m.generate_fixed_k(p, K, noise_std=sg)
            A, S, keep, crit = solve_track(op, rho, mu, f[:, 0], f[:, 1], maxiter, early_iter, tol)
            for k in np.where(keep)[0]:
                fk = torch.tensor(np.stack([A[k], S[k]]), dtype=torch.float32)
                if all(d4_rel_l2(fk, g) > 0.15 for g in kept):
                    kept.append(fk); kept_res.append(float(crit[k]))
    return kept, kept_res


@torch.no_grad()
def main(a):
    C = Config()
    lookup = torch.load(C.data_lookup_path, weights_only=False); ns = torch.load(C.norm_stats_path)
    params = lookup["p_values"]; sols = lookup["solutions_by_p"]
    m = GSPDEModel.from_pretrained(a.ckpt, C, local_rank=0, p_mean=ns["p_mean"], p_std=ns["p_std"]); m.eval()
    op = GSOperator(C.op_path, device="cuda", dtype=torch.float64)
    if a.L2: op.DA /= 4.0; op.DS /= 4.0
    K = C.fixed_k; sigmas = [float(s) for s in a.sigmas.split(",")]

    targets = []   # list of dicts: {p, Kt, ti or None, gt}
    if a.mode == "rescue":
        recs = []
        for f in sorted(glob.glob("eval/full_chunks/chunk_*.json")):
            recs += json.load(open(f))["recs"]
        zero = sorted([r for r in recs if r["distinct"] == 0], key=lambda r: -r["Kt"])
        if a.n > 0: zero = zero[:a.n]
        for r in zero:
            ti = r["ti"]; targets.append({"p": params[ti], "Kt": int(sols[ti].shape[0]), "ti": ti, "gt": sols[ti]})
    else:  # grid extrapolation: extend beyond the training grid edges (no GT)
        rho = params[:, 0].numpy(); mu = params[:, 1].numpy()
        r0, r1, u0, u1 = rho.min(), rho.max(), mu.min(), mu.max()
        dr, du = 0.25*(r1-r0), 0.25*(u1-u0)              # extend 25% past each edge
        rs = np.linspace(r0-dr, r1+dr, a.grid); us = np.linspace(u0-du, u1+du, a.grid)
        for rr in rs:
            for uu in us:
                inside = (r0 <= rr <= r1) and (u0 <= uu <= u1)
                targets.append({"p": torch.tensor([rr, uu], dtype=torch.float32), "Kt": 0,
                                "ti": None, "gt": None, "extrap": not inside})
    print(f"=== EXTRAP mode={a.mode} | {len(targets)} targets | sigmas={sigmas} N={a.N} "
          f"tol={a.tol} FDM {a.maxiter}/{a.early_iter} L2={a.L2} ===", flush=True)

    rows = []
    for i, t in enumerate(targets):
        kept, res = union_generate(m, op, t["p"], K, sigmas, a.N, a.maxiter, a.early_iter, a.tol)
        ncov = 0
        if t["gt"] is not None and len(kept):
            Kt = t["gt"].shape[0]; matched = set()
            for fk in kept:
                ds = [d4_rel_l2(fk, t["gt"][j]) for j in range(Kt)]; j = int(np.argmin(ds))
                if ds[j] < 0.15: matched.add(j)
            ncov = len(matched)
        r = {"ti": t["ti"], "rho": float(t["p"][0]), "mu": float(t["p"][1]), "Kt": t["Kt"],
             "n_solutions": len(kept), "n_cover_gt": ncov,
             "min_residual": (min(res) if res else None), "extrap": t.get("extrap", False)}
        rows.append(r)
        tag = "X" if r["extrap"] else " "
        print(f"  {i+1}/{len(targets)} {tag} rho={r['rho']:.4f} mu={r['mu']:.4f} Kt={r['Kt']:2d} "
              f"-> {r['n_solutions']} sols (cov {r['n_cover_gt']})", flush=True)

    n = len(rows); got = [r for r in rows if r["n_solutions"] >= 1]
    summary = {"mode": a.mode, "n_targets": n, "n_with_solution": len(got),
               "rescue_rate": len(got)/n if n else 0.0,
               "mean_solutions": float(np.mean([r["n_solutions"] for r in rows])) if n else 0.0,
               "sigmas": sigmas, "N": a.N, "tol": a.tol}
    if a.mode == "grid":
        ex = [r for r in rows if r["extrap"]]; exgot = [r for r in ex if r["n_solutions"] >= 1]
        summary["n_extrap"] = len(ex); summary["n_extrap_with_solution"] = len(exgot)
        summary["extrap_rate"] = len(exgot)/len(ex) if ex else 0.0
    print("\n=== SUMMARY ===")
    for k, v in summary.items(): print(f"  {k}: {v}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="./checkpoints_final/epoch_60")
    p.add_argument("--mode", choices=["rescue", "grid"], default="rescue")
    p.add_argument("--n", type=int, default=0, help="rescue: cap # zero-distinct params (0=all)")
    p.add_argument("--grid", type=int, default=15, help="grid: NxN (rho,mu) grid")
    p.add_argument("--sigmas", default="0.1,0.2,0.3")
    p.add_argument("--N", type=int, default=4, help="noised passes per sigma")
    p.add_argument("--tol", type=float, default=1e-6, help="relaxed residual gate (headline=1e-9)")
    p.add_argument("--maxiter", type=int, default=15000)
    p.add_argument("--early_iter", type=int, default=4000)
    p.add_argument("--L2", action="store_true")
    p.add_argument("--out", default="./experiments/extrap_rescue_L1.json")
    main(p.parse_args())
