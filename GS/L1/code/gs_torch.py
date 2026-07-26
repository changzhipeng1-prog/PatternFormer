"""Batched torch port of the collaborator's quasi-Newton FDM (Np=1) solver
for steady-state 2D Gray-Scott.

Solves, per candidate (broadcast over a batch dim B):
    DA*Lap(A) - S*A^2 + (mu+rho)*A      = 0
    DS*Lap(S) + S*A^2 - rho*(1-S)       = 0
on the Neumann unit-square operator exported from MATLAB (op_n128.mat).

The whole iteration is batched matmul + elementwise, so hundreds of candidate
solves run together on one A100. fp64 throughout (residual target 1e-9).
"""
import os
import numpy as np
import scipy.io as sio
import torch


class GSOperator:
    """Holds Tx, TxInv, Lambda2D for the tensor-product eigen-solver."""

    def __init__(self, mat_path, device="cuda", dtype=torch.float64):
        m = sio.loadmat(mat_path)
        self.device = device
        self.dtype = dtype
        self.n = int(m["n"].item())
        self.L = float(m["L"].item())
        self.DA = float(m["DA"].item())
        self.DS = float(m["DS"].item())
        Tx = torch.as_tensor(m["Tx"], dtype=dtype, device=device)
        TxInv = torch.as_tensor(m["TxInv"], dtype=dtype, device=device)
        eigx = torch.as_tensor(m["eigx"].ravel(), dtype=dtype, device=device)
        self.Tx = Tx
        self.TxT = Tx.T.contiguous()
        self.TxInv = TxInv
        self.TxInvT = TxInv.T.contiguous()
        # Lambda2D[i,j] = eigx[i] + eigx[j]  (square symmetric: eigy = eigx)
        self.Lam = eigx[:, None] + eigx[None, :]

    def _to_eig(self, X):
        # TxInv @ X @ TxInv^T
        return self.TxInv @ X @ self.TxInvT

    def _from_eig(self, X):
        # Tx @ X @ Tx^T
        return self.Tx @ X @ self.TxT

    def lap(self, X):
        return self._from_eig(self._to_eig(X) * self.Lam)

    def inv_shift(self, R, scale):
        # [D*Lam + beta]^{-1} R  in eigenbasis;  scale broadcast (B,n,n) or (n,n)
        return self._from_eig(self._to_eig(R) / scale)


def recover_S_from_A(op, A, rho, mu):
    """Linear recovery of S from A (matches example3_2D.m lines 103-118)."""
    t = op.DA * op.lap(A) + ((mu + rho) * A - rho)
    return op.inv_shift(t, -op.DS * op.Lam - rho)


def solve_batch(op, rho, mu, A0, S0=None, stepsize=0.1, tol=1e-9,
                maxiter=30000, maxtol=1000.0, early_iter=1500, early_tol=1e-2,
                verbose=False):
    """Batched quasi-Newton solve.

    rho, mu : scalar or (B,) tensors/arrays
    A0      : (B,n,n) initial activator
    S0      : (B,n,n) or None (recover linearly)
    Returns dict: A,S (B,n,n), res (B,), iters (int), converged (B,) bool.
    """
    dev, dt = op.device, op.dtype
    A = torch.as_tensor(A0, dtype=dt, device=dev).clone()
    B, n, _ = A.shape
    rho = torch.as_tensor(rho, dtype=dt, device=dev).expand(B).reshape(B, 1, 1)
    mu = torch.as_tensor(mu, dtype=dt, device=dev).expand(B).reshape(B, 1, 1)

    if S0 is None:
        S = recover_S_from_A(op, A, rho, mu)
    else:
        S = torch.as_tensor(S0, dtype=dt, device=dev).clone()

    active = torch.ones(B, dtype=torch.bool, device=dev)
    crit = torch.full((B,), float("inf"), dtype=dt, device=dev)
    DA, DS, Lam = op.DA, op.DS, op.Lam

    it = 0
    for it in range(1, maxiter + 1):
        rA = DA * op.lap(A) + (-S * A * A + (mu + rho) * A)
        rS = DS * op.lap(S) + (S * A * A - rho * (1.0 - S))
        cA = rA.abs().amax(dim=(1, 2))
        cS = rS.abs().amax(dim=(1, 2))
        c = torch.maximum(cA, cS)
        # update crit only for active candidates
        crit = torch.where(active, c, crit)

        diverged = (~torch.isfinite(c)) | (c > maxtol)
        converged = c < tol
        active = active & ~diverged & ~converged
        if it == early_iter:   # early-kill slow seeds (unlikely to converge)
            active = active & ~(c > early_tol)
        if verbose and (it % 200 == 0 or not active.any()):
            print(f"  it {it}: active={int(active.sum())} "
                  f"crit[min,med,max]={float(c[torch.isfinite(c)].min()):.2e},"
                  f"{float(c[torch.isfinite(c)].median()):.2e},"
                  f"{float(c[torch.isfinite(c)].max()):.2e}")
        if not active.any():
            break

        # beta from local 2x2 reaction-Jacobian spectral bounds (per candidate)
        m1U = (mu + rho) - 2.0 * A * S
        m2V = rho + A * A
        m1V = -A * A
        m2U = 2.0 * A * S
        z = (m1U - m2V) ** 2 + 4.0 * m1V * m2U
        disc = torch.sqrt(torch.clamp(z, min=0.0))   # real part of complex sqrt
        half = (m1U + m2V) / 2.0
        maax = half + disc / 2.0
        miin = half - disc / 2.0
        beta = (maax.amax(dim=(1, 2)) + miin.amin(dim=(1, 2))) / 2.0  # (B,)
        beta = beta.reshape(B, 1, 1)

        dA = op.inv_shift(rA, DA * Lam + beta)
        dS = op.inv_shift(rS, DS * Lam + beta)
        dA = torch.nan_to_num(dA)
        dS = torch.nan_to_num(dS)
        m = active.reshape(B, 1, 1)
        A = torch.where(m, A - stepsize * dA, A)
        S = torch.where(m, S - stepsize * dS, S)

    converged = torch.isfinite(crit) & (crit < tol)
    return dict(A=A, S=S, res=crit, iters=it, converged=converged)


if __name__ == "__main__":
    # ---- validation against MATLAB on the 8 base seeds ----
    HERE = os.path.dirname(os.path.abspath(__file__))
    op = GSOperator(f"{HERE}/op_n128.mat", device="cuda")
    A0 = np.transpose(sio.loadmat(f"{HERE}/validate_seeds.mat")["A0_init"], (2, 0, 1))
    ref = sio.loadmat(f"{HERE}/matlab_ref.mat")
    A_ref = np.transpose(ref["A_out"], (2, 0, 1))
    S_ref = np.transpose(ref["S_out"], (2, 0, 1))

    out = solve_batch(op, 0.04, 0.065, A0, tol=1e-9, verbose=True)
    A = out["A"].cpu().numpy(); S = out["S"].cpu().numpy()
    print(f"\niters={out['iters']}  converged={out['converged'].sum().item()}/{len(A0)}")
    print("res:", np.array2string(out["res"].cpu().numpy(), precision=2))
    dA = np.abs(A - A_ref).reshape(len(A0), -1).max(1)
    dS = np.abs(S - S_ref).reshape(len(A0), -1).max(1)
    print("max|A_torch - A_matlab| per seed:", np.array2string(dA, precision=2))
    print("max|S_torch - S_matlab| per seed:", np.array2string(dS, precision=2))
    print(f"OVERALL  A:{dA.max():.2e}  S:{dS.max():.2e}")
