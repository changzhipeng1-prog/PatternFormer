"""
pde_loss_2d.py  —  FEM residual loss for the 2D PDE (V3)

PDE:   -Δu - u² = -s·sin(πx)sin(πy),   u = 0 on ∂Ω,   Ω = [0,1]²
Mesh:  ell=3 triangulation (145 nodes, 256 triangles)

FEM weak form  →  nonlinear system:
    F(u) = K·u  −  ∫(u² − s·f)·φ_i dΩ  =  0     (free nodes only)

where
    K   = stiffness matrix  [n×n]
    f_i = s·sin(π·x_i)·sin(π·y_i)   (nodal RHS)
    φ_i = Lagrange basis functions
    integration by 16-point Gauss quadrature on each triangle

Analogous to v2's pde_loss.py (finite-difference 1D), but using the FEM
stiffness matrix and Gauss integration precomputed from coord/elem.

API
---
build_pde_data(coord, elem, free_nodes) -> PDE2DData
compute_pde_loss_2d(pred_u, p_vals, pde_data) -> scalar tensor
get_pde_lambda(epoch, warmup_start, warmup_end, target_lambda) -> float
"""
import math
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


# ── 16-point Gauss quadrature on the reference triangle ─────────────────────
# Same table used in generate_dataset_2d.py
_GP = np.array([
    [9.703785126946112e-3, 8.602401356562194e-1],
    [4.612207990645205e-2, 8.602401356562194e-1],
    [9.363778443732850e-2, 8.602401356562194e-1],
    [1.300560792168344e-1, 8.602401356562194e-1],
    [2.891208422438901e-2, 5.835904323689168e-1],
    [1.374191041345744e-1, 5.835904323689168e-1],
    [2.789904634965088e-1, 5.835904323689168e-1],
    [3.874974834066942e-1, 5.835904323689168e-1],
    [5.021012321136977e-2, 2.768430136381238e-1],
    [2.386486597314429e-1, 2.768430136381238e-1],
    [4.845083266304333e-1, 2.768430136381238e-1],
    [6.729468631505064e-1, 2.768430136381238e-1],
    [6.546699455501446e-2, 5.710419611451768e-2],
    [3.111645522443570e-1, 5.710419611451768e-2],
    [6.317312516411253e-1, 5.710419611451768e-2],
    [8.774288093304679e-1, 5.710419611451768e-2],
], dtype=np.float64)
_GW = np.array([
    5.423225910525254e-3, 1.016725956447879e-2,
    1.016725956447879e-2, 5.423225910525254e-3,
    2.258404928236993e-2, 4.233972452174629e-2,
    4.233972452174629e-2, 2.258404928236993e-2,
    3.538806789808595e-2, 6.634421610704973e-2,
    6.634421610704973e-2, 3.538806789808595e-2,
    2.356836819338233e-2, 4.418508852236173e-2,
    4.418508852236173e-2, 2.356836819338233e-2,
], dtype=np.float64)
# Barycentric coordinates of Gauss points: [16, 3]
_PHI = np.stack([_GP[:, 0], _GP[:, 1], 1 - _GP[:, 0] - _GP[:, 1]], axis=1)


@dataclass
class PDE2DData:
    """
    Precomputed FEM tensors needed for the 2D PDE residual loss.
    All tensors are stored as float32 on CPU (moved to GPU in compute_pde_loss_2d).
    """
    stiff_A:   torch.Tensor   # [145, 145] dense stiffness matrix (symmetric)
    v0:        torch.Tensor   # [256] long  — element vertex 0 indices
    v1:        torch.Tensor   # [256] long  — element vertex 1 indices
    v2:        torch.Tensor   # [256] long  — element vertex 2 indices
    wt:        torch.Tensor   # [256, 16]  integration weights (detB * GW / 2)
    PHI:       torch.Tensor   # [16, 3]    Gauss basis values
    b_nodes:   torch.Tensor   # [145]      sin(πx)*sin(πy) at each node
    free_nodes: torch.Tensor  # [n_free]  long  free DOF indices
    n_nodes:   int            # 145


def build_pde_data(
    coord: torch.Tensor,   # [145, 2] float (x,y)
    elem:  torch.Tensor,   # [256, 3] long  vertex indices
    free_nodes: torch.Tensor,  # [n_free] long
) -> PDE2DData:
    """
    Precompute all FEM tensors from the mesh.  Call once at model init.

    Args:
        coord:      node coordinates from dataset ('coord' key)
        elem:       triangle connectivity from dataset ('elem' key)
        free_nodes: free DOF indices from dataset ('free_nodes' key)

    Returns:
        PDE2DData instance (all float32 CPU tensors)
    """
    c   = coord.double().numpy()   # [145, 2]
    el  = elem.long().numpy()      # [256, 3]

    n_nodes = c.shape[0]
    n_elem  = el.shape[0]

    # ── Stiffness matrix ───────────────────────────────────────────────────
    v0_, v1_, v2_ = el[:, 0], el[:, 1], el[:, 2]
    B00 = c[v0_, 0] - c[v2_, 0];  B01 = c[v1_, 0] - c[v2_, 0]
    B10 = c[v0_, 1] - c[v2_, 1];  B11 = c[v1_, 1] - c[v2_, 1]
    detB = np.abs(B00 * B11 - B01 * B10)               # [256]

    Bi00 =  B11 / detB;  Bi01 = -B01 / detB
    Bi10 = -B10 / detB;  Bi11 =  B00 / detB

    gpx = np.stack([Bi00, Bi10, -Bi00 - Bi10], axis=1)  # [256, 3]
    gpy = np.stack([Bi01, Bi11, -Bi01 - Bi11], axis=1)  # [256, 3]

    # Local stiffness: Aloc[e, i, j] = (detB/2) * (gpx_i*gpx_j + gpy_i*gpy_j)
    Aloc = (detB / 2)[:, None, None] * (
        gpx[:, :, None] * gpx[:, None, :]
        + gpy[:, :, None] * gpy[:, None, :]
    )                                                    # [256, 3, 3]

    # Assemble into dense matrix (145 nodes is small enough)
    stiff_np = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    verts = np.stack([v0_, v1_, v2_], axis=1)            # [256, 3]
    for e in range(n_elem):
        for i in range(3):
            for j in range(3):
                stiff_np[verts[e, i], verts[e, j]] += Aloc[e, i, j]

    # ── Gauss integration weights: detB * GW  →  [256, 16] ──────────────
    # Note: stiffness matrix uses detB/2 (= triangle area) because K_ij is
    # an exact formula (grad · grad × area).  The nonlinear Gauss quadrature
    # uses detB (no /2): ∫_elem f dΩ = ∫_ref f × |J| dξ = detB × ∫_ref f dξ,
    # and the GW weights already sum to 0.5 (area of the reference triangle).
    GW_t  = _GW.astype(np.float64)                      # [16]
    wt_np = detB[:, None] * GW_t[None, :]                # [256, 16]

    # ── sin(πx)sin(πy) at each node ───────────────────────────────────────
    b_np = np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1])  # [145]

    return PDE2DData(
        stiff_A   = torch.from_numpy(stiff_np).float(),
        v0        = torch.from_numpy(v0_.astype(np.int64)),
        v1        = torch.from_numpy(v1_.astype(np.int64)),
        v2        = torch.from_numpy(v2_.astype(np.int64)),
        wt        = torch.from_numpy(wt_np).float(),
        PHI       = torch.from_numpy(_PHI.astype(np.float64)).float(),
        b_nodes   = torch.from_numpy(b_np).float(),
        free_nodes= free_nodes.long(),
        n_nodes   = n_nodes,
    )


def residual_sq_2d(
    u:        torch.Tensor,    # [N_sol, 145]  decoded solutions (float)
    p_vals:   torch.Tensor,    # [N_sol]       s values
    pde_data: PDE2DData,
) -> torch.Tensor:
    """Per-free-node squared FEM residual  R²  →  [N_sol, n_free].

        F = K·u − ∫(u² − s·f)·φ dΩ      (weak form, free nodes only)

    Differentiable w.r.t. u.  This is the raw physical quantity; how it is turned
    into a scalar loss is decided by compute_pde_loss_2d below.
    """
    dev   = u.device
    dtype = u.dtype
    N     = u.shape[0]

    A   = pde_data.stiff_A.to(dev, dtype)      # [145, 145]
    v0  = pde_data.v0.to(dev)                  # [256]
    v1  = pde_data.v1.to(dev)
    v2  = pde_data.v2.to(dev)
    wt  = pde_data.wt.to(dev, dtype)           # [256, 16]
    PHI = pde_data.PHI.to(dev, dtype)          # [16, 3]
    b_n = pde_data.b_nodes.to(dev, dtype)      # [145]
    fn  = pde_data.free_nodes.to(dev)          # [n_free]

    stiff_term = u @ A                         # K·u  [N, 145]

    u_verts = torch.stack([u[:, v0], u[:, v1], u[:, v2]], dim=-1)
    u_gauss = u_verts @ PHI.t()                # [N, 256, 16]
    u_gauss_sq = u_gauss ** 2

    sb_nodes  = p_vals.unsqueeze(1) * b_n.unsqueeze(0)   # [N, 145]
    sb_verts  = torch.stack([sb_nodes[:, v0], sb_nodes[:, v1], sb_nodes[:, v2]], dim=-1)
    sb_gauss  = sb_verts @ PHI.t()             # [N, 256, 16]

    integrand = wt.unsqueeze(0) * (u_gauss_sq - sb_gauss)   # [N, 256, 16]
    ce = integrand @ PHI                       # [N, 256, 3]

    coeff = torch.zeros(N, pde_data.n_nodes, dtype=dtype, device=dev)
    coeff.scatter_add_(1, v0.unsqueeze(0).expand(N, -1), ce[:, :, 0])
    coeff.scatter_add_(1, v1.unsqueeze(0).expand(N, -1), ce[:, :, 1])
    coeff.scatter_add_(1, v2.unsqueeze(0).expand(N, -1), ce[:, :, 2])

    R = stiff_term - coeff                     # [N, 145]
    return R[:, fn].pow(2)                      # [N, n_free]


def compute_pde_loss_2d(
    pred_u:   torch.Tensor,    # [N_sol, 145]  predicted decoded solutions (float32)
    p_vals:   torch.Tensor,    # [N_sol]        s values for each solution (float32)
    pde_data: PDE2DData,
    ref_u:    torch.Tensor = None,   # [N_sol, 145]  AE-decoded GROUND TRUTH for the same slots
) -> torch.Tensor:
    """
    FEM physics-residual loss for the multi-branch generator.

    GT-REFERENCED RESIDUAL EXCESS  (ref_u given)
    --------------------------------------------
    A perfectly correct prediction still has a non-zero residual once it passes
    through the frozen autoencoder: the decoder's reconstruction error δu, amplified
    by the stiffness/Laplacian operator, pins per-node R² at ~1e-3 (median) — ~9
    orders ABOVE the FEM-discretization residual of the exact solution (~1e-12).
    Penalizing the RAW residual would therefore tax correct solutions at a level
    comparable to / above the MSE term (~3e-4) and over-smooth them, which is why
    the naive PDE loss fought the ordered multi-branch MSE.

    Fix: do not penalize residual in absolute terms; penalize only how much the
    prediction's residual EXCEEDS the residual that the AE-decoded ground truth
    reaches for the SAME (s, branch).  ref_u = D(E(u_gt)) is the per-sample, per-s
    achievable reference, so

        loss = mean( relu( R²(pred_u) − R²(ref_u) ) ).

    When pred_u == ref_u (correct branch) the excess is 0 → no gradient → no
    conflict with the MSE term.  Only slots that are LESS physical than the AE can
    represent (blurry / hallucinated branches) are pulled down — toward the correct
    Newton basin, where the post-processing Newton refinement then snaps them to
    machine precision.  ref_u is treated as a constant (detached upstream).

    With ref_u=None this falls back to the plain mean squared residual.

    Returns:
        scalar PDE loss tensor (differentiable w.r.t. pred_u)
    """
    r2_pred = residual_sq_2d(pred_u, p_vals, pde_data)     # [N, n_free]
    if ref_u is None:
        return r2_pred.mean()
    with torch.no_grad():
        r2_ref = residual_sq_2d(ref_u, p_vals, pde_data)   # per-sample reference
    return torch.relu(r2_pred - r2_ref).mean()             # penalize only the excess


def get_pde_lambda(
    epoch: int,
    warmup_start: int = 10,
    warmup_end:   int = 25,
    target_lambda: float = 0.05,
) -> float:
    """
    Linear warmup scheduler for PDE loss weight.  Same as v2.

    - epoch <= warmup_start:          lambda = 0
    - warmup_start < epoch <= warmup_end: linearly 0 → target_lambda
    - epoch > warmup_end:             lambda = target_lambda
    """
    if epoch <= warmup_start:
        return 0.0
    elif epoch <= warmup_end:
        progress = (epoch - warmup_start) / (warmup_end - warmup_start)
        return target_lambda * progress
    else:
        return target_lambda
