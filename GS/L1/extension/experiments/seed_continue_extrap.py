"""GS parameter extrapolation via the *second* route (seed-from-boundary), the
analogue of the Track-A "seed + Newton/continuation" method, which has NOT been
tried on Gray--Scott before (all prior GS OOD here is direct-style generation).

For each OOD target p* = (rho*, mu*) just outside the training band, with the
nearest in-band frontier parameter p_b used as the seed:

  reference : continue the TRUE in-band solution set (lookup sols at p_b) to p*
              with the GS quasi-Newton solver  -> the solutions that actually exist at p*
  M1 direct : model.generate_fixed_k(p*) -> refine at p*            (the published route)
  M2 jump   : take the model's in-band solutions at p_b, single solve at p*
  M3 cont   : take the model's in-band solutions at p_b, stepped continuation to p*

Metric = coverage of the reference set (D4 rel-L2 < 0.15), per method.
Everything deterministic (no noise) for an apples-to-apples comparison.

Run from GS/L1/code (so eval/full_chunks etc. resolve), torch124 python:
  CUDA_VISIBLE_DEVICES=0 python ../extension/experiments/seed_continue_extrap.py \
      --ckpt ../ckpt/epoch_60 --out ../extension/experiments/seed_continue_L1.json
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # for `import extrapolate`
import numpy as np, torch
# reuse the solver + dedup metric + model/op imports from the existing script
from extrapolate import solve_track, d4_rel_l2, DEV
from config import Config
from model.v3_model import GSPDEModel
from gs_torch import GSOperator


def dedup(fields, thr=0.15):
    out = []
    for f in fields:
        if all(d4_rel_l2(f, g) > thr for g in out):
            out.append(f)
    return out


def clean_at(op, rho, mu, batch, maxiter, early_iter, tol):
    """Solve the GS system at (rho,mu) seeded by `batch` [B,2,n,n]; return list of
    converged non-trivial fields as [2,n,n] tensors."""
    if batch.shape[0] == 0:
        return []
    A, S, keep, crit = solve_track(op, rho, mu, batch[:, 0], batch[:, 1],
                                   maxiter, early_iter, tol)
    return [torch.tensor(np.stack([A[k], S[k]]), dtype=torch.float32)
            for k in np.where(keep)[0]]


def continue_set(op, p0, p1, seeds, nsteps, maxiter, early_iter, tol):
    """Warm-started continuation of `seeds` (list of [2,n,n]) from p0 to p1 in
    `nsteps` linear steps; branches that diverge are dropped. Returns deduped list."""
    cur = list(seeds)
    rhos = np.linspace(p0[0], p1[0], nsteps + 1)[1:]
    mus = np.linspace(p0[1], p1[1], nsteps + 1)[1:]
    for rr, uu in zip(rhos, mus):
        if not cur:
            break
        cur = clean_at(op, float(rr), float(uu), torch.stack(cur),
                       maxiter, early_iter, tol)
    return dedup(cur)


def coverage(distinct, ref, thr=0.15):
    if not ref:
        return 0
    cov = 0
    for g in ref:
        if any(d4_rel_l2(d, g) < thr for d in distinct):
            cov += 1
    return cov


@torch.no_grad()
def main(a):
    C = Config()
    lk = torch.load(C.data_lookup_path, weights_only=False)
    ns = torch.load(C.norm_stats_path)
    params = lk["p_values"].numpy()
    sols = lk["solutions_by_p"]
    m = GSPDEModel.from_pretrained(a.ckpt, C, local_rank=0,
                                   p_mean=ns["p_mean"], p_std=ns["p_std"]); m.eval()
    op = GSOperator(C.op_path, device="cuda", dtype=torch.float64)
    if a.L2:
        op.DA /= 4.0; op.DS /= 4.0
    K = C.fixed_k
    MI, EI, TOL = a.maxiter, a.early_iter, a.tol

    rho, mu = params[:, 0], params[:, 1]
    r_lo, r_hi = rho.min(), rho.max()
    mu_lo, mu_hi = mu.min(), mu.max()
    # outward push direction in (rho,mu); the GS pattern band runs along (+rho,-mu),
    # so its outer ends are reached by stepping diagonally, not by +rho at fixed mu.
    DIRS = {"diag": (1.0, -0.27), "rho": (1.0, 0.0), "mu": (0.0, -1.0)}
    dvec = DIRS[a.dir]

    # --- frontier seed params: per mu-band, the in-band param farthest along +dvec ---
    edges = np.linspace(mu_lo, mu_hi, a.n_seeds + 1)
    proj = rho * dvec[0] + mu * dvec[1]
    seeds_idx = []
    for j in range(a.n_seeds):
        sel = np.where((mu >= edges[j]) & (mu <= edges[j + 1]))[0]
        if len(sel) == 0:
            continue
        seeds_idx.append(int(sel[np.argmax(proj[sel])]))
    seeds_idx = sorted(set(seeds_idx))
    dists = [float(x) for x in a.drho.split(",")]
    if a.smoke:
        seeds_idx = seeds_idx[:3]; dists = dists[:1]

    print(f"=== GS seed-from-boundary OOD | {len(seeds_idx)} seeds x {len(dists)} dist "
          f"| dir={a.dir}{dvec} | step={a.cont_step} | tol={TOL} FDM {MI}/{EI} L2={a.L2} ===", flush=True)

    rows = []
    for bi in seeds_idx:
        p_b = (float(rho[bi]), float(mu[bi]))
        true_b = [sols[bi][k].float() for k in range(sols[bi].shape[0])]
        # model's in-band solution set at the seed param (deterministic single pass)
        gen_b = m.generate_fixed_k(torch.tensor(p_b, dtype=torch.float32), K, noise_std=0.0)
        gen_b = torch.as_tensor(gen_b, dtype=torch.float32).reshape(-1, 2, 128, 128)
        model_b = dedup(clean_at(op, p_b[0], p_b[1], gen_b, MI, EI, TOL))
        cov_inband = coverage(model_b, dedup(true_b))
        print(f"\n[seed bi={bi}] p_b=({p_b[0]:.4f},{p_b[1]:.4f})  "
              f"true={len(true_b)}  model in-band distinct={len(model_b)} (cov {cov_inband})", flush=True)

        for dr in dists:
            p_t = (p_b[0] + dr * dvec[0], p_b[1] + dr * dvec[1])
            ncont = max(1, int(round(dr / a.cont_step)))
            # what truly exists at p* (continuation of the true in-band set); GS has no
            # exhaustive GT, so this is only a "do patterns persist here" reference.
            ref = continue_set(op, p_b, p_t, dedup(true_b), ncont, MI, EI, TOL)
            # direct generation at p* (the published OOD route)
            gen_t = m.generate_fixed_k(torch.tensor(p_t, dtype=torch.float32), K, noise_std=0.0)
            gen_t = torch.as_tensor(gen_t, dtype=torch.float32).reshape(-1, 2, 128, 128)
            direct = dedup(clean_at(op, p_t[0], p_t[1], gen_t, MI, EI, TOL))
            # seed from the model's in-band solutions: single solve at p*, and stepped
            seed1 = continue_set(op, p_b, p_t, model_b, 1, MI, EI, TOL)
            seedN = continue_set(op, p_b, p_t, model_b, ncont, MI, EI, TOL)
            r = {"bi": bi, "rho_b": p_b[0], "mu_b": p_b[1], "dr": dr,
                 "rho_t": p_t[0], "mu_t": p_t[1], "n_ref": len(ref),
                 "n_direct": len(direct), "n_seed1": len(seed1), "n_seedN": len(seedN),
                 "cov_direct": coverage(direct, ref), "cov_seed1": coverage(seed1, ref),
                 "cov_seedN": coverage(seedN, ref)}
            rows.append(r)
            print(f"   dr={dr:.4f} -> p*=({p_t[0]:.4f},{p_t[1]:.4f})  exist={len(ref):2d} | "
                  f"#sols  direct={len(direct):2d} seed-1step={len(seed1):2d} seed-stepped={len(seedN):2d}",
                  flush=True)

    # summary: GS has no exhaustive GT, so the headline metric is the mean number of
    # distinct non-trivial solutions found per OOD target (direct vs seed-from-boundary).
    def mean(rs, key):
        return float(np.mean([r[key] for r in rs])) if rs else 0.0
    def rate(rs, key):
        return float(np.mean([r[key] >= 1 for r in rs])) if rs else 0.0
    summary = {"n_cases": len(rows), "dir": a.dir,
               "mean_direct": mean(rows, "n_direct"), "mean_seed1": mean(rows, "n_seed1"),
               "mean_seedN": mean(rows, "n_seedN"),
               "rate_direct": rate(rows, "n_direct"), "rate_seed1": rate(rows, "n_seed1"),
               "rate_seedN": rate(rows, "n_seedN")}
    by_d = {}
    for dr in sorted(set(r["dr"] for r in rows)):
        rs = [r for r in rows if r["dr"] == dr]
        by_d[f"{dr:.4f}"] = {"n": len(rs), "direct": mean(rs, "n_direct"),
                             "seed1": mean(rs, "n_seed1"), "seedN": mean(rs, "n_seedN")}
    summary["by_distance"] = by_d
    print("\n=== SUMMARY (mean # distinct non-trivial solutions per OOD target) ===")
    print(f"  cases={summary['n_cases']}  direct={summary['mean_direct']:.2f}  "
          f"seed-1step={summary['mean_seed1']:.2f}  seed-stepped={summary['mean_seedN']:.2f}")
    print(f"  frac targets with >=1 sol:  direct={summary['rate_direct']:.2f}  "
          f"seed-1step={summary['rate_seed1']:.2f}  seed-stepped={summary['rate_seedN']:.2f}")
    for d, v in by_d.items():
        print(f"   dr={d}: n={v['n']}  direct={v['direct']:.2f}  seed-1step={v['seed1']:.2f}  "
              f"seed-stepped={v['seedN']:.2f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="../ckpt/epoch_60")
    p.add_argument("--n_seeds", type=int, default=8, help="# frontier seed params (mu-bands)")
    p.add_argument("--dir", choices=["diag", "rho", "mu"], default="diag", help="outward push direction")
    p.add_argument("--drho", default="0.004,0.008,0.012", help="OOD push magnitudes")
    p.add_argument("--cont_step", type=float, default=0.0015, help="continuation step size")
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--maxiter", type=int, default=8000)
    p.add_argument("--early_iter", type=int, default=3000)
    p.add_argument("--L2", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default="../extension/experiments/seed_continue_L1.json")
    main(p.parse_args())
