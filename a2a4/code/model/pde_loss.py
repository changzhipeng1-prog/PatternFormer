"""
PDE Residual Loss for: -u'' + a4*u^4 + a2*u^2 = 0
BC: u'(0)=0 (Neumann), u(1)=0 (Dirichlet), N=1024 grid points, h=1/N

Same finite-difference scheme as the BVP Newton solver:
  Lu[0]    = 2*u[0] - 2*u[1]          (ghost point from Neumann BC)
  Lu[i]    = 2*u[i] - u[i-1] - u[i+1]
  Lu[-1]   = 2*u[-1] - u[-2]          (next point u[N]=0 from Dirichlet)
  Fu = Lu/h + h*(a4*u^4 + a2*u^2)

Loss = mean(Fu^2) — keeps gradient magnitudes compatible with MSE.
"""
import torch


def residual_components(u: torch.Tensor, a4: torch.Tensor, a2: torch.Tensor,
                        dx: float):
    """Return the raw (signed) h²-scaled residual components.

    SCALING is unified with the 1D_p reference (h²·R_phys = -(2nd diff) + h²·f) so
    the two 1D experiments share the same physics-loss magnitude and the same
    lambda_pde.  The BC DISCRETIZATION, however, is kept problem/data-specific:
    a2a4's GT solver enforces the Dirichlet u(1)=0 via a GHOST node (u[N]=0), so its
    GT satisfies 2u[-1]-u[-2]≈0 (verified empirically) rather than u[-1]=0.  We
    therefore evaluate the right boundary as the ghost-node FDM residual
    (2u[-1]-u[-2]) + h²·f, NOT 1D_p's u[-1] form — BC treatment must match the data
    generator, it is not something to blindly unify across problems.  Only the
    nonlinearity differs from 1D_p: f(u) = a4·u⁴ + a2·u² (vs 1D_p's u²(u²-p)).

    NOTE: the previous a2a4 form used h-scaling (Lu/h + h·f) = h·R_phys, which is
    1/h ≈ 1024× larger per residual (≈1e6× in squared loss); that is purely a
    scaling-convention difference (NOT a larger AE error) and is exactly why the
    same lambda_pde=0.05 over-dominated here.  Scaling now matched to 1D_p's h²-form.

    interior_res : [B, 1022] ; bc_left_res, bc_right : [B]
    """
    u  = u.float()
    a4 = a4.float().unsqueeze(-1)   # [B, 1]
    a2 = a2.float().unsqueeze(-1)   # [B, 1]
    h2 = dx ** 2

    u_left   = u[:, :-2]
    u_center = u[:, 1:-1]
    u_right  = u[:, 2:]

    laplacian    = -(u_right - 2 * u_center + u_left)            # = -(2nd diff)
    nonlinear    = h2 * (a4 * u_center ** 4 + a2 * u_center ** 2)
    interior_res = laplacian + nonlinear

    lap_left     = -(2 * u[:, 1] - 2 * u[:, 0])                  # Neumann u'(0)=0
    nl_left      = h2 * (a4.squeeze(-1) * u[:, 0] ** 4 + a2.squeeze(-1) * u[:, 0] ** 2)
    bc_left_res  = lap_left + nl_left

    # Right BC: Dirichlet u(1)=0 via ghost node u[N]=0 (matches a2a4 GT data, which
    # satisfies 2u[-1]-u[-2]≈0, NOT u[-1]=0). h²-scaled, same form as the interior.
    lap_right    = 2 * u[:, -1] - u[:, -2]                       # = -(2nd diff) at N-1, ghost u[N]=0
    nl_right     = h2 * (a4.squeeze(-1) * u[:, -1] ** 4 + a2.squeeze(-1) * u[:, -1] ** 2)
    bc_right     = lap_right + nl_right

    return interior_res, bc_left_res, bc_right


def compute_pde_loss(u: torch.Tensor, a4: torch.Tensor, a2: torch.Tensor,
                     dx: float, ref_u: torch.Tensor = None) -> torch.Tensor:
    """PDE residual loss.

    Args:
        u:  predicted solutions [B, 1024]
        a4, a2: [B] parameter values
        dx: grid spacing h = 1/N
        ref_u: optional AE-decoded GROUND-TRUTH solution (same shape as u). When
            given, penalize only the per-element EXCESS of the prediction's squared
            residual over the reference's: mean(relu(R^2(u) - R^2(ref_u))). The frozen
            AE cannot represent the GT exactly, so even a correct prediction has a
            nonzero FDM residual ("AE floor", ~1e-2 in this h²-scaled convention,
            same scale as 1D_p); subtracting the per-sample reference removes that
            floor so a correct branch incurs ~0 penalty. ref_u is detached.

    Returns:
        scalar PDE residual loss
    """
    i_p, b0_p, bN_p = residual_components(u, a4, a2, dx)

    if ref_u is None:
        return i_p.pow(2).mean() + b0_p.pow(2).mean() + bN_p.pow(2).mean()

    with torch.no_grad():
        i_r, b0_r, bN_r = residual_components(ref_u, a4, a2, dx)

    relu = torch.relu
    return (
        relu(i_p.pow(2)  - i_r.pow(2)).mean()
        + relu(b0_p.pow(2) - b0_r.pow(2)).mean()
        + relu(bN_p.pow(2) - bN_r.pow(2)).mean()
    )


def get_pde_lambda(epoch: int, warmup_start: int = 5, warmup_end: int = 15,
                   target_lambda: float = 0.05) -> float:
    if epoch <= warmup_start:
        return 0.0
    elif epoch <= warmup_end:
        progress = (epoch - warmup_start) / (warmup_end - warmup_start)
        return target_lambda * progress
    else:
        return target_lambda
