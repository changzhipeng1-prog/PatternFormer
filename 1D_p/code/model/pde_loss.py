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


def residual_components(u: torch.Tensor, p: torch.Tensor, dx: float):
    """Return the raw (signed) h²-scaled residual components.

    interior_res : [B, 1022] (or [B, T, 1022]) ; bc_left_res, bc_right : [B] (or [B, T])
    These are the same quantities the original loss squared+meaned; exposing them
    lets us reference a per-sample baseline (the AE-decoded GT residual).
    """
    u = u.float()
    p = p.float()
    h2 = dx ** 2

    if u.dim() == 2:
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

    return interior_res, bc_left_res, bc_right


def compute_pde_loss(u: torch.Tensor, p: torch.Tensor, dx: float,
                     ref_u: torch.Tensor = None) -> torch.Tensor:
    """h²-scaled PDE residual loss.

    Args:
        u: predicted solution, [B, 1024] or [B, T, 1024]
        p: corresponding p values, [B] or [B, T]
        dx: grid spacing (1/1023)
        ref_u: optional AE-decoded GROUND-TRUTH solution (same shape as u). When
            given, the loss penalizes only the per-element EXCESS of the prediction's
            squared residual over the reference's squared residual:
                mean(relu(R²(u) - R²(ref_u))).
            The frozen autoencoder cannot represent the GT exactly, so even a CORRECT
            prediction has a nonzero FDM residual ("AE floor"); subtracting the
            per-sample reference removes that floor so a correct branch incurs ~0
            physics penalty (no conflict with the multi-branch MSE), and only slots
            LESS physical than the AE itself can represent are pulled toward the
            true Newton basin.  ref_u is treated as a constant (detached).

    Returns:
        pde_loss: scalar tensor
    """
    i_p, bl_p, br_p = residual_components(u, p, dx)

    if ref_u is None:
        return (i_p.pow(2).mean() + bl_p.pow(2).mean() + br_p.pow(2).mean())

    with torch.no_grad():
        i_r, bl_r, br_r = residual_components(ref_u, p, dx)

    relu = torch.relu
    return (
        relu(i_p.pow(2)  - i_r.pow(2)).mean()
        + relu(bl_p.pow(2) - bl_r.pow(2)).mean()
        + relu(br_p.pow(2) - br_r.pow(2)).mean()
    )


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
