"""Test-time latent optimization along the solution BAND (no training): can the BASELINE
model (epoch_60) extrapolate solutions past the GT box if we polish its latent guesses by
gradient descent on the physics residual?

Along the band line mu = a*rho + b, for each (rho,mu) we:
  1. generate K=24 latent guesses with the model (one noised pass),
  2. optimize each 256-d latent z to minimize the FDM residual of decode(z), in 3 modes:
       model_only      : no optimization (baseline generate -> FDM)
       latentopt_pure  : minimize residual ONLY  -> tests the "does it collapse to trivial?" hypothesis
       latentopt_nt    : minimize residual + non-triviality hinge (range(A) >= floor)
  3. decode -> FDM quasi-Newton refine -> D4-dedup -> # distinct converged solutions.
Also records the mean A-range of the optimized fields (low range == bleached/trivial).

Inference only; loads the saved baseline ckpt, never trains. Run from qwen_ord/.
Writes experiments/latent_opt_extrap.json + experiments/latent_opt_extrap.png.
"""
import os, sys, json, argparse
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
sys.path.insert(0, ROOT)
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import Config
from model.v3_model import GSPDEModel
from gs_torch import GSOperator
DEV = torch.device("cuda:0"); BF = torch.bfloat16


def d4_rel_l2(a, b):
    g = [torch.rot90(b, k, (-2, -1)) for k in range(4)]
    bt = b.transpose(-1, -2); g += [torch.rot90(bt, k, (-2, -1)) for k in range(4)]
    gb = torch.stack(g)
    return ((a[None]-gb).flatten(2).norm(dim=2)/b.flatten(1).norm(dim=1).clamp_min(1e-8)).mean(1).min().item()


def solve_track(op, rho, mu, A0, S0, maxiter=12000, early_iter=3000, tol=1e-6, early_tol=0.3, ftol=1e-9, stepsize=0.1):
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
    return A.cpu().numpy(), S.cpu().numpy(), (conv & (rngA > 0.05)).cpu().numpy()


def distinct_count(op, rho, mu, fields, maxiter, early_iter):
    A, S, conv = solve_track(op, rho, mu, fields[:, 0], fields[:, 1], maxiter, early_iter)
    kept = []
    for k in np.where(conv)[0]:
        fk = torch.tensor(np.stack([A[k], S[k]]), dtype=torch.float32)
        if all(d4_rel_l2(fk, g) > 0.15 for g in kept): kept.append(fk)
    return len(kept)


def latent_opt(m, z0, rho, mu, mode, nstep, lr, lam_nt, rfloor):
    """Optimize the 24 latents to minimize physics residual (+/- non-triviality). Returns decoded fields (np)."""
    z = z0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    K = z.shape[0]
    rr = torch.full((K,), rho, device=DEV); uu = torch.full((K,), mu, device=DEV)
    for _ in range(nstep):
        dec = m.autoencoder(z.to(BF), "decode").float()                 # [K,2,128,128]
        res, _ = m.pde_res(dec, rr, uu, floor=0.0)                      # raw mean residual (pure descent)
        loss = res
        if mode == "nt":
            A = dec[:, 0]; rng = A.flatten(1).amax(1) - A.flatten(1).amin(1)
            loss = res + lam_nt * torch.relu(rfloor - rng).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        dec = m.autoencoder(z.to(BF), "decode").float()
    return dec.cpu().numpy()


@torch.no_grad()
def gen_init(m, rho, mu, K, sigma):
    fields = m.generate_fixed_k(np.array([rho, mu], dtype=np.float32), K, noise_std=sigma)  # np [K,2,128,128]
    f = torch.tensor(np.asarray(fields), dtype=torch.float32, device=DEV)
    z = m.autoencoder(f.to(BF), "encode").float()
    return f.cpu().numpy(), z


def main(a):
    C = Config(); ns = torch.load(C.norm_stats_path)
    m = GSPDEModel.from_pretrained(a.ckpt, C, local_rank=0, p_mean=ns["p_mean"], p_std=ns["p_std"]); m.eval()
    op = GSOperator(C.op_path, device="cuda", dtype=torch.float64)
    if a.L2: op.DA /= 4.0; op.DS /= 4.0
    K = C.fixed_k
    # band line + box, from the v2 data builder artifact (fallback: refit)
    bnd = np.load("data_extrap2/band.npz"); aa, bb = float(bnd["a"]), float(bnd["b"])
    lk = torch.load(C.data_lookup_path, weights_only=False); P = lk["p_values"].numpy()
    r0, r1, u0, u1 = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
    dr = 0.25*(r1-r0); ue_lo, ue_hi = u0-0.25*(u1-u0), u1+0.25*(u1-u0)
    rhos = np.linspace(r0-dr, r1+dr, a.npts)
    print(f"band mu={aa:.4f}*rho+{bb:.4f}; box rho[{r0:.4f},{r1:.4f}]; {a.npts} pts along diagonal", flush=True)

    modes = ["model_only", "latentopt_pure", "latentopt_nt"]
    rows = []
    for i, rho in enumerate(rhos):
        mu = float(np.clip(aa*rho + bb, ue_lo, ue_hi))
        inside = bool(r0 <= rho <= r1)
        f0, z0 = gen_init(m, float(rho), mu, K, a.sigma)
        rec = {"rho": float(rho), "mu": mu, "inside": inside}
        for mode in modes:
            if mode == "model_only":
                fields = f0
            else:
                fields = latent_opt(m, z0, float(rho), mu, "nt" if mode.endswith("nt") else "pure",
                                    a.nstep, a.lr, a.lam_nt, a.rfloor)
            rec[mode] = distinct_count(op, float(rho), mu, fields, a.maxiter, a.early_iter)
            rec[mode+"_range"] = float(np.mean(fields[:, 0].reshape(K, -1).max(1) - fields[:, 0].reshape(K, -1).min(1)))
        rows.append(rec)
        print(f"  {i+1}/{a.npts} rho={rho:.4f} mu={mu:.4f} {'IN ' if inside else 'EXT'} | "
              f"model {rec['model_only']} | pure {rec['latentopt_pure']} (rng {rec['latentopt_pure_range']:.2f}) | "
              f"nt {rec['latentopt_nt']} (rng {rec['latentopt_nt_range']:.2f})", flush=True)

    json.dump({"band": [aa, bb], "box": [float(r0), float(r1)], "rows": rows,
               "cfg": vars(a)}, open(a.out, "w"), indent=2)
    # ---- plot ----
    rr = np.array([r["rho"] for r in rows])
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for mode, col in [("model_only", "gray"), ("latentopt_pure", "tab:red"), ("latentopt_nt", "tab:green")]:
        ax.plot(rr, [r[mode] for r in rows], "o-", color=col, label=mode)
        ax2.plot(rr, [r[mode+"_range"] for r in rows], "o-", color=col, label=mode)
    for x in (r0, r1):
        ax.axvline(x, color="k", ls="--", lw=1); ax2.axvline(x, color="k", ls="--", lw=1)
    ax.axvspan(r0, r1, color="k", alpha=0.05)
    ax.set_ylabel("# distinct converged solutions"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Test-time latent optimization along the band (baseline ckpt, no training)\n"
                 "shaded = GT box; outside = extrapolation", fontsize=11)
    ax2.axhline(0.30, color="0.6", ls=":", lw=1); ax2.set_ylabel("mean A-range of fields\n(low = trivial/bleached)")
    ax2.set_xlabel(r"$\rho$ along band $\mu=a\rho+b$"); ax2.legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig("experiments/latent_opt_extrap.png", dpi=130, bbox_inches="tight"); plt.close()
    print(f"wrote {a.out} and experiments/latent_opt_extrap.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="./checkpoints_final/epoch_60")
    p.add_argument("--npts", type=int, default=25)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--nstep", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lam_nt", type=float, default=0.3)
    p.add_argument("--rfloor", type=float, default=0.35)
    p.add_argument("--maxiter", type=int, default=12000)
    p.add_argument("--early_iter", type=int, default=3000)
    p.add_argument("--L2", action="store_true")
    p.add_argument("--out", default="./experiments/latent_opt_extrap.json")
    main(p.parse_args())
