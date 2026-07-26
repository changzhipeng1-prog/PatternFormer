"""Regenerate and save the actual solution FIELDS at the off-grid extrapolation points
that produced non-trivial solutions (for marking them on the atlas).
Reads experiments/extrap_grid_L1.json (extrap rows with n_solutions>=1), regenerates each,
saves the first converged A-field. Writes experiments/extrap_fields.npz.
Run from qwen_ord/ with torch124 python.
"""
import os, sys, json
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
sys.path.insert(0, ROOT)
import numpy as np, torch
from config import Config
from model.v3_model import GSPDEModel
from gs_torch import GSOperator
DEV = torch.device("cuda:0")


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


@torch.no_grad()
def main():
    C = Config(); C.norm_stats_path = "./data_expanded/norm_stats_L2.pt"; ns = torch.load(C.norm_stats_path)
    grid = json.load(open("experiments/extrap_grid_L2.json"))["rows"]
    pts = [(r["rho"], r["mu"]) for r in grid if r["extrap"] and r["n_solutions"] >= 1]
    print("extrap points with solution:", pts, flush=True)
    m = GSPDEModel.from_pretrained("./checkpoints_final/epoch_60", C, local_rank=0,
                                   p_mean=ns["p_mean"], p_std=ns["p_std"]); m.eval()
    op = GSOperator(C.op_path, device="cuda", dtype=torch.float64); op.DA /= 4.0; op.DS /= 4.0  # L2
    rhos, mus, fields = [], [], []
    for rr, uu in pts:
        got = None
        for sg in (0.2, 0.3):
            f = m.generate_fixed_k(torch.tensor([rr, uu], dtype=torch.float32), C.fixed_k, noise_std=sg)
            A, S, keep = solve_track(op, rr, uu, f[:, 0], f[:, 1])
            idx = np.where(keep)[0]
            if len(idx): got = A[idx[0]]; break
        if got is not None:
            rhos.append(rr); mus.append(uu); fields.append(got)
            print(f"  saved field at rho={rr:.4f} mu={uu:.4f}", flush=True)
    np.savez("experiments/extrap_fields.npz", rho=np.array(rhos), mu=np.array(mus), A=np.array(fields))
    print(f"wrote experiments/extrap_fields.npz ({len(fields)} fields)")


if __name__ == "__main__":
    main()
