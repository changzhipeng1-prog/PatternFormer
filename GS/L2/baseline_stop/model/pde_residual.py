"""FDM Gray-Scott steady-state residual as a physics-informed loss term.

Uses the same tensor-product Neumann Laplacian operator (op_n128.mat) the data
generator solved with, so a true steady state has residual ~0. For a decoded
prediction the residual measures how far it is from satisfying the PDE — a
ground-truth-free regularizer that holds at any (rho,mu), incl. OOD.

    rA = DA*Lap(A) - S*A^2 + (mu+rho)*A
    rS = DS*Lap(S) + S*A^2 - rho*(1-S)
    loss = mean(rA^2 + rS^2)
"""
import scipy.io as sio
import torch
import torch.nn as nn


class GSResidual(nn.Module):
    """Floor-hinged FDM Gray-Scott residual.

    The raw residual has its GLOBAL minimum at the TRIVIAL homogeneous state
    (A=0, S=1 -> residual ~1e-12), which is ~8 orders BELOW the best a real
    pattern can reach once it passes through the (lossy) autoencoder
    (AE round-trip residual ~1.7e-4, max ~4.8e-4). Penalizing the raw residual
    therefore exerts a constant pull toward the bleached/trivial solution we
    explicitly removed from the data, and the Laplacian amplifies the AE's tiny
    high-frequency errors (DA*lambda_max ~ 16x).

    Fix: penalize only residual ABOVE a floor, relu(res2 - floor). A correctly
    decoded real solution (~5e-4) and the trivial state (~1e-12) both sit BELOW
    the floor => zero penalty => no incentive to bleach. Only genuinely
    unphysical/blurry predictions (res2 ~ 6e-2) are penalized, and only down to
    the floor. This turns the term from a hard constraint into a safe polish.
    """
    def __init__(self, op_path: str, DA: float, DS: float, device,
                 dtype=torch.float32, floor: float = 1e-3):
        super().__init__()
        m = sio.loadmat(op_path)
        Tx = torch.as_tensor(m["Tx"], dtype=dtype)
        TxInv = torch.as_tensor(m["TxInv"], dtype=dtype)
        eigx = torch.as_tensor(m["eigx"].ravel(), dtype=dtype)
        self.register_buffer("Tx", Tx.to(device))
        self.register_buffer("TxT", Tx.t().contiguous().to(device))
        self.register_buffer("TxInv", TxInv.to(device))
        self.register_buffer("TxInvT", TxInv.t().contiguous().to(device))
        self.register_buffer("Lam", (eigx[:, None] + eigx[None, :]).to(device))  # [n,n]
        self.DA = DA; self.DS = DS
        self.floor = float(floor)   # ~2x the AE round-trip residual ceiling

    def lap(self, X):
        # X: [N,n,n] ; returns the tensor-product Neumann Laplacian (matches solver)
        t = self.TxInv @ X @ self.TxInvT
        t = t * self.Lam
        return self.Tx @ t @ self.TxT

    def forward(self, pred, rho, mu):
        """pred: [N,2,128,128] physical (A,S). rho,mu: [N]. -> hinged loss + raw residual."""
        A = pred[:, 0].float()
        S = pred[:, 1].float()
        rho = rho.float().view(-1, 1, 1)
        mu = mu.float().view(-1, 1, 1)
        rA = self.DA * self.lap(A) - S * A * A + (mu + rho) * A
        rS = self.DS * self.lap(S) + S * A * A - rho * (1.0 - S)
        res2 = rA * rA + rS * rS                      # [N,n,n] pointwise squared residual
        raw = res2.mean()
        hinged = torch.relu(res2 - self.floor).mean()  # only penalize ABOVE the AE floor
        return hinged, raw.detach()


def get_pde_lambda(epoch, start, end, target):
    if epoch < start:
        return 0.0
    if epoch >= end:
        return target
    return target * (epoch - start) / max(1, end - start)
