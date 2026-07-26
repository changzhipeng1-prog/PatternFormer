"""
newton_refine_2d.py — Newton refinement for 2D FEM PDE solutions

PDE:  -Δu - u² = -s·sin(πx)sin(πy),  u=0 on ∂Ω,  Ω=[0,1]²

Given a Qwen-predicted solution û (145-dim), runs damped Newton iterations
on the FEM residual F(u)=0 restricted to free nodes, using û as initial guess.

API
---
newton_refine_2d(pred_u, s, coord, elem, free_nodes, tol, max_iter)
    -> refined_u [145], converged (bool), n_iters (int), residual_history (list)

refine_solution_list(pred_list, s, coord, elem, free_nodes, ...)
    -> list of refined tensors, stats dict
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from typing import List, Tuple, Dict

# 16-point Gauss quadrature (same table as pde_loss_2d.py / generate_dataset_2d.py)
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
_PHI = np.stack([_GP[:, 0], _GP[:, 1], 1 - _GP[:, 0] - _GP[:, 1]], axis=1)  # [16, 3]


def _build_stiffness(coord: np.ndarray, elem: np.ndarray) -> sp.csr_matrix:
    """Assemble global stiffness matrix K [n_nodes, n_nodes]."""
    n = len(coord)
    v0, v1, v2 = elem[:, 0], elem[:, 1], elem[:, 2]
    B00 = coord[v0, 0] - coord[v2, 0];  B01 = coord[v1, 0] - coord[v2, 0]
    B10 = coord[v0, 1] - coord[v2, 1];  B11 = coord[v1, 1] - coord[v2, 1]
    detB = np.abs(B00 * B11 - B01 * B10)
    Bi00 =  B11 / detB;  Bi01 = -B01 / detB
    Bi10 = -B10 / detB;  Bi11 =  B00 / detB
    # Local gradient matrices
    gpx = np.stack([Bi00, Bi10, -Bi00 - Bi10], axis=1)  # [n_elem, 3]
    gpy = np.stack([Bi01, Bi11, -Bi01 - Bi11], axis=1)
    Aloc = (detB / 2)[:, None, None] * (
        gpx[:, :, None] * gpx[:, None, :] + gpy[:, :, None] * gpy[:, None, :]
    )   # [n_elem, 3, 3]
    verts = np.stack([v0, v1, v2], axis=1)
    R = np.repeat(verts, 3, axis=1).ravel()
    C = np.tile(verts, (1, 3)).ravel()
    return sp.csr_matrix((Aloc.ravel(), (R, C)), shape=(n, n))


def _assemble_F_J(
    coord: np.ndarray, elem: np.ndarray,
    u: np.ndarray, s: float,
    A: sp.csr_matrix, detB: np.ndarray,
) -> Tuple[np.ndarray, sp.csr_matrix]:
    """
    Assemble residual F(u) and Jacobian J(u) for the 2D FEM system.

    F(u)_i = Σ_j K_ij u_j - ∫_Ω (u² - s·f) φ_i dΩ
    J(u)_ij = K_ij - ∫_Ω 2u · φ_i · φ_j dΩ
    """
    n = len(coord)
    v0, v1, v2 = elem[:, 0], elem[:, 1], elem[:, 2]

    # RHS: s·sin(πx)·sin(πy) at nodes
    b = s * np.sin(np.pi * coord[:, 0]) * np.sin(np.pi * coord[:, 1])

    # Gather element values
    ue = np.stack([u[v0], u[v1], u[v2]], axis=1)   # [n_elem, 3]
    fe = np.stack([b[v0], b[v1], b[v2]], axis=1)   # [n_elem, 3]

    # Evaluate at Gauss points
    uh = ue @ _PHI.T   # [n_elem, 16]
    fh = fe @ _PHI.T   # [n_elem, 16]

    wt = detB[:, None] * _GW[None, :]   # [n_elem, 16]  (includes |det J| = detB/2 * 2)

    # ── Residual ───────────────────────────────────────────────────────────
    intgd = wt * (uh ** 2 - fh)                         # [n_elem, 16]
    ce = (intgd[:, :, None] * _PHI[None, :, :]).sum(axis=1)  # [n_elem, 3]
    coeff = np.zeros(n)
    np.add.at(coeff, v0, ce[:, 0])
    np.add.at(coeff, v1, ce[:, 1])
    np.add.at(coeff, v2, ce[:, 2])
    F = A @ u - coeff

    # ── Jacobian nonlinear part: -2 ∫ u·φᵢ·φⱼ dΩ ─────────────────────────
    ji  = wt * uh                                        # [n_elem, 16]
    je  = -2 * np.einsum('eg,gk,gi->eki', ji, _PHI, _PHI)  # [n_elem, 3, 3]
    verts = np.stack([v0, v1, v2], axis=1)
    R = np.repeat(verts, 3, axis=1).ravel()
    C = np.tile(verts, (1, 3)).ravel()
    J = A + sp.csr_matrix((je.ravel(), (R, C)), shape=(n, n))

    return F, J


def newton_refine_2d(
    pred_u: torch.Tensor,      # [145] float32 — Qwen decoded solution
    s: float,                  # parameter value
    coord: torch.Tensor,       # [145, 2] float
    elem:  torch.Tensor,       # [256, 3] long
    free_nodes: torch.Tensor,  # [113] long
    tol: float = 1e-9,
    max_iter: int = 30,
    alpha_min: float = 1e-4,
    backtrack_beta: float = 0.5,
    diverge_factor: float = 1e3,  # reject if ||u_refined - u_pred|| > factor * ||u_pred||
) -> Tuple[torch.Tensor, bool, int, List[float]]:
    """
    Damped Newton refinement on FEM residual F(u)=0 (free nodes only).

    Returns:
        refined_u  [145] float32  — refined solution (falls back to pred_u if failed)
        converged  bool
        n_iters    int
        res_history list[float]   — ||F_free||_2 per iteration (for diagnostics)
    """
    c_np  = coord.double().numpy()
    el_np = elem.long().numpy()
    fn    = free_nodes.long().numpy()

    # Precompute stiffness and detB once
    v0_, v1_, v2_ = el_np[:, 0], el_np[:, 1], el_np[:, 2]
    B00 = c_np[v0_, 0] - c_np[v2_, 0];  B01 = c_np[v1_, 0] - c_np[v2_, 0]
    B10 = c_np[v0_, 1] - c_np[v2_, 1];  B11 = c_np[v1_, 1] - c_np[v2_, 1]
    detB = np.abs(B00 * B11 - B01 * B10)   # [256]

    A = _build_stiffness(c_np, el_np)

    u = pred_u.detach().cpu().double().numpy().copy()   # [145] float64
    u_init_norm = float(np.linalg.norm(u))

    res_history = []

    for it in range(max_iter):
        F, J = _assemble_F_J(c_np, el_np, u, s, A, detB)
        F_free = F[fn]
        r_norm = float(np.linalg.norm(F_free))
        res_history.append(r_norm)

        if not np.isfinite(r_norm):
            return pred_u, False, it, res_history

        if r_norm < tol:
            refined = torch.from_numpy(u.astype(np.float32))
            return refined, True, it + 1, res_history

        # Solve J_free · δu = -F_free
        J_free = J[np.ix_(fn, fn)].tocsc()
        try:
            du_free = spla.spsolve(J_free, -F_free)
        except Exception:
            return pred_u, False, it, res_history

        if not np.isfinite(du_free).all():
            return pred_u, False, it, res_history

        # Damped step with backtracking
        alpha = 1.0
        accepted = False
        while alpha >= alpha_min:
            u_try = u.copy()
            u_try[fn] += alpha * du_free
            F_try, _ = _assemble_F_J(c_np, el_np, u_try, s, A, detB)
            r_try = float(np.linalg.norm(F_try[fn]))
            if np.isfinite(r_try) and r_try < r_norm:
                u = u_try
                accepted = True
                break
            alpha *= backtrack_beta

        if not accepted:
            return pred_u, False, it, res_history

    # Final residual check
    F, _ = _assemble_F_J(c_np, el_np, u, s, A, detB)
    r_final = float(np.linalg.norm(F[fn]))
    res_history.append(r_final)

    if not np.isfinite(r_final) or r_final >= tol:
        return pred_u, False, max_iter, res_history

    # Sanity check: don't accept if solution drifted to a completely different branch
    drift = float(np.linalg.norm(u - pred_u.detach().cpu().double().numpy()))
    if u_init_norm > 1e-8 and drift > diverge_factor * u_init_norm:
        return pred_u, False, max_iter, res_history

    refined = torch.from_numpy(u.astype(np.float32))
    return refined, True, max_iter, res_history


def refine_solution_list(
    pred_list: List[torch.Tensor],
    s: float,
    coord: torch.Tensor,
    elem: torch.Tensor,
    free_nodes: torch.Tensor,
    tol: float = 1e-9,
    max_iter: int = 30,
) -> Tuple[List[torch.Tensor], Dict]:
    """
    Refine a list of predicted solutions; failed items fall back to original.

    Returns:
        refined_list  List[Tensor]
        stats         dict  {total, success, fallback, mean_iters, mean_final_res}
    """
    refined_list = []
    n_success = 0
    iter_list = []
    res_list  = []

    for u in pred_list:
        u_ref, ok, n_it, hist = newton_refine_2d(
            u, s, coord, elem, free_nodes, tol=tol, max_iter=max_iter
        )
        refined_list.append(u_ref)
        if ok:
            n_success += 1
            iter_list.append(n_it)
            res_list.append(hist[-1] if hist else float("nan"))
        else:
            iter_list.append(n_it)
            res_list.append(hist[-1] if hist else float("nan"))

    stats = {
        "total":         len(pred_list),
        "success":       n_success,
        "fallback":      len(pred_list) - n_success,
        "mean_iters":    float(np.mean(iter_list)) if iter_list else 0.0,
        "mean_final_res": float(np.nanmean(res_list)) if res_list else float("nan"),
    }
    return refined_list, stats
