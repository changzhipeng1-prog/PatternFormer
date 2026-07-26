"""Visualize the off-grid extrapolation: (1) a heatmap of #solutions over the extended
(rho,mu) grid with the training-grid box + the four corners marked, and (2) the actual
generated/converged fields at the BOTTOM-LEFT and TOP-RIGHT corners (both 0-solution),
with the BOTTOM-RIGHT corner (band continuation, has solutions) shown for contrast.

Reads experiments/extrap_grid_L1.json for the heatmap; re-generates only the corner
points (cheap) to render their fields. Run from qwen_ord/ with torch124 python.
Writes experiments/grid_extrap_heatmap.png and experiments/corners_fields.png.
"""
import os, sys, json
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
sys.path.insert(0, ROOT)
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from config import Config
from model.v3_model import GSPDEModel
from gs_torch import GSOperator
DEV = torch.device("cuda:0")


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


# ---------- 1) heatmap from the grid json ----------
d = json.load(open("experiments/extrap_grid_L1.json")); rows = d["rows"]
rho = sorted(set(round(r["rho"], 6) for r in rows)); mu = sorted(set(round(r["mu"], 6) for r in rows))
ri = {v: i for i, v in enumerate(rho)}; ui = {v: i for i, v in enumerate(mu)}
H = np.full((len(mu), len(rho)), np.nan)
for r in rows:
    H[ui[round(r["mu"], 6)], ri[round(r["rho"], 6)]] = r["n_solutions"]
# training-grid extent (the "inside" region)
ins = [r for r in rows if not r["extrap"]]
r0 = min(r["rho"] for r in ins); r1 = max(r["rho"] for r in ins)
u0 = min(r["mu"] for r in ins); u1 = max(r["mu"] for r in ins)

fig, ax = plt.subplots(figsize=(8.2, 6.2))
im = ax.pcolormesh(np.array(rho), np.array(mu), H, cmap="viridis", shading="nearest")
ax.add_patch(Rectangle((r0, u0), r1-r0, u1-u0, fill=False, ec="white", lw=2.2, ls="--"))
ax.text(r0, u1, " training grid", color="white", va="bottom", ha="left", fontsize=10, weight="bold")
corners = {"BOTTOM-LEFT": (rho[0], mu[0]), "TOP-RIGHT": (rho[-1], mu[-1]),
           "TOP-LEFT": (rho[0], mu[-1]), "BOTTOM-RIGHT": (rho[-1], mu[0])}
for name, (rr, uu) in corners.items():
    n = int(H[ui[round(uu, 6)], ri[round(rr, 6)]])
    ax.scatter([rr], [uu], s=160, marker="s", facecolors="none",
               edgecolors=("orange" if n > 0 else "red"), linewidths=2.5)
    ax.annotate(f"{name}\n{n} sols", (rr, uu), color=("orange" if n > 0 else "red"),
                fontsize=8.5, weight="bold", ha="center",
                va=("bottom" if uu < (u0+u1)/2 else "top"))
ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$\mu$")
ax.set_title("Off-grid extrapolation (no training): # distinct solutions over extended $(\\rho,\\mu)$\n"
             "solutions live on a diagonal BAND; off-band corners are empty", fontsize=11)
plt.colorbar(im, ax=ax, label="# distinct converged solutions")
plt.tight_layout(); plt.savefig("experiments/grid_extrap_heatmap.png", dpi=130, bbox_inches="tight"); plt.close()
print("wrote experiments/grid_extrap_heatmap.png", flush=True)

# ---------- 2) corner fields ----------
C = Config(); ns = torch.load(C.norm_stats_path)
m = GSPDEModel.from_pretrained("./checkpoints_final/epoch_60", C, local_rank=0,
                               p_mean=ns["p_mean"], p_std=ns["p_std"]); m.eval()
op = GSOperator(C.op_path, device="cuda", dtype=torch.float64)
NCOL = 6

@torch.no_grad()
def corner(rr, uu, sigmas=(0.2, 0.3), N=2):
    p = torch.tensor([rr, uu], dtype=torch.float32)
    raw = m.generate_fixed_k(p, C.fixed_k, noise_std=sigmas[0])   # raw generated candidates
    kept = []
    for sg in sigmas:
        for _ in range(N):
            f = m.generate_fixed_k(p, C.fixed_k, noise_std=sg)
            A, S, keep = solve_track(op, rr, uu, f[:, 0], f[:, 1])
            for k in np.where(keep)[0]:
                fk = torch.tensor(np.stack([A[k], S[k]]), dtype=torch.float32)
                if all(d4_rel_l2(fk, g) > 0.15 for g in kept): kept.append(fk)
    return raw, kept

picks = [("BOTTOM-LEFT (off-band)", rho[0], mu[0]),
         ("TOP-RIGHT (off-band)", rho[-1], mu[-1]),
         ("BOTTOM-RIGHT (band continuation)", rho[-1], mu[0])]
fig, axes = plt.subplots(len(picks), NCOL, figsize=(NCOL*1.7, len(picks)*1.7), squeeze=False)
for i, (name, rr, uu) in enumerate(picks):
    raw, kept = corner(rr, uu)
    show = ([(k[0].numpy(), "converged") for k in kept] if kept
            else [(raw[j, 0], "raw (non-conv)") for j in range(NCOL)])
    tag = f"{len(kept)} solutions" if kept else "0 solutions"
    for j in range(NCOL):
        a = axes[i][j]; a.set_xticks([]); a.set_yticks([])
        if j < len(show):
            img, kind = show[j]; a.imshow(img, cmap="viridis")
            ec = "orange" if kind == "converged" else "red"
            for sp in a.spines.values(): sp.set_edgecolor(ec); sp.set_linewidth(2.6)
            if i == 0: a.set_title(kind if j == 0 else "", fontsize=8)
        else:
            for sp in a.spines.values(): sp.set_visible(False)
    axes[i][0].set_ylabel(f"{name}\n$\\rho$={rr:.4f} $\\mu$={uu:.4f}\n[{tag}]",
                          fontsize=8, rotation=0, ha="right", va="center", labelpad=46)
    print(f"{name}: rho={rr:.4f} mu={uu:.4f} -> {len(kept)} converged solutions", flush=True)
fig.suptitle("Corner extrapolation: generated fields at the off-grid corners\n"
             "orange = FDM-converged solution; red = raw generated candidate that does NOT converge",
             fontsize=10.5)
plt.tight_layout(rect=[0.06, 0, 1, 0.93])
plt.savefig("experiments/corners_fields.png", dpi=130, bbox_inches="tight"); plt.close()
print("wrote experiments/corners_fields.png", flush=True)
