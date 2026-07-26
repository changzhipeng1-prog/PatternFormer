"""
PDE 残差 Loss — h²-scaled + warmup 策略 (V2)

PDE:  -u'' + u²(u² - p) = 0,  x ∈ [0,1]
BC:   u'(0) = 0  (Neumann),  u(1) = 0  (Dirichlet)

Scaling trick (v4):
  Both sides multiplied by h², yielding:
    R_i = -(u_{i+1} - 2u_i + u_{i-1}) + h² · u²(u² - p) = 0
  Laplacian term O(ε), nonlinear term O(h²·ε), PDE loss ~ O(1e-2).
  Keeps gradient magnitudes compatible with MSE ~ O(1e-4).

V2 note: u shape is now [B, 1024] (one solution per forward call from DualHead),
         NOT [B, T, 1024] as in V1. Both shapes are supported via dynamic dispatch.
"""
import torch


def compute_pde_loss(u: torch.Tensor, p: torch.Tensor, dx: float) -> torch.Tensor:
    """
    h²-scaled PDE residual loss.

    Args:
        u: predicted solution, shape [B, 1024] or [B, T, 1024]
        p: corresponding p values, shape [B] or [B, T]
        dx: grid spacing (1/1023)

    Returns:
        pde_loss: scalar tensor
    """
    u = u.float()
    p = p.float()
    h2 = dx ** 2

    if u.dim() == 2:
        # [B, 1024] — V2 single-solution mode
        p_expand = p.unsqueeze(-1)           # [B, 1]
        u_left   = u[:, :-2]                 # [B, 1022]
        u_center = u[:, 1:-1]
        u_right  = u[:, 2:]

        laplacian    = -(u_right - 2 * u_center + u_left)
        nonlinear    = h2 * u_center ** 2 * (u_center ** 2 - p_expand)
        interior_res = laplacian + nonlinear

        lap_left     = -(2 * u[:, 1] - 2 * u[:, 0])
        nl_left      = h2 * u[:, 0] ** 2 * (u[:, 0] ** 2 - p)
        bc_left_res  = lap_left + nl_left

        bc_right     = u[:, -1]

    elif u.dim() == 3:
        # [B, T, 1024] — sequence mode (backward-compatible with V1)
        p_expand = p.unsqueeze(-1)           # [B, T, 1]
        u_left   = u[:, :, :-2]
        u_center = u[:, :, 1:-1]
        u_right  = u[:, :, 2:]

        laplacian    = -(u_right - 2 * u_center + u_left)
        nonlinear    = h2 * u_center ** 2 * (u_center ** 2 - p_expand)
        interior_res = laplacian + nonlinear

        lap_left     = -(2 * u[:, :, 1] - 2 * u[:, :, 0])
        nl_left      = h2 * u[:, :, 0] ** 2 * (u[:, :, 0] ** 2 - p_expand.squeeze(-1))
        bc_left_res  = lap_left + nl_left

        bc_right     = u[:, :, -1]
    else:
        raise ValueError(f"u must be 2D or 3D, got {u.dim()}D")

    pde_loss = (
        interior_res.pow(2).mean()
        + bc_left_res.pow(2).mean()
        + bc_right.pow(2).mean()
    )
    return pde_loss


def get_pde_lambda(epoch: int, warmup_start: int = 5, warmup_end: int = 15,
                   target_lambda: float = 0.05) -> float:
    """
    Linear warmup scheduler for PDE loss weight.

    - Epoch 1 … warmup_start:        lambda = 0      (pure MSE)
    - Epoch warmup_start+1 … warmup_end: linearly ramp 0 → target_lambda
    - Epoch > warmup_end:            lambda = target_lambda
    """
    if epoch <= warmup_start:
        return 0.0
    elif epoch <= warmup_end:
        progress = (epoch - warmup_start) / (warmup_end - warmup_start)
        return target_lambda * progress
    else:
        return target_lambda
