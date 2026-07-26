import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch


# Self-contained default: the cbmfem solver shipped in this repo at
#   paper/1D_p/data_gen/cbmfem  (this file lives at paper/1D_p/code/model/).
# The post-processing Newton solver is THE SAME code that generated the GT data.
_DEFAULT_TRAD_SOLVER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data_gen", "cbmfem")
)
# Override before importing this module to point elsewhere (same faSh1/Newton API):
#   export BBBB_PDE_NEWTON_DIR=/path/to/cbmfem
_TRAD_SOLVER_DIR = os.environ.get("BBBB_PDE_NEWTON_DIR", _DEFAULT_TRAD_SOLVER_DIR)
if _TRAD_SOLVER_DIR not in sys.path:
    sys.path.insert(0, _TRAD_SOLVER_DIR)

try:
    from Sexample1 import faSh1  # type: ignore
    from Newton import Newton  # type: ignore
    _HAS_TRAD_SOLVER = True
except Exception:
    _HAS_TRAD_SOLVER = False


def _to_complex_col(u: torch.Tensor) -> np.ndarray:
    # Keep maximum available precision when moving to numpy; avoid unnecessary
    # float32 round-off before Newton residual evaluation.
    arr = u.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)
    return arr.astype(np.complex128)


def _safe_tensor_from_np(x: np.ndarray, ref: torch.Tensor) -> torch.Tensor:
    x_real = np.real(x).reshape(-1)
    out = torch.from_numpy(x_real).to(device=ref.device, dtype=ref.dtype)
    return out


def _resample_1d_linear(x: np.ndarray, out_len: int) -> np.ndarray:
    """
    Linear resampling on [0, 1] for 1D column vector.

    Args:
        x: [N] real-valued array
        out_len: target length >= 2
    """
    in_len = int(x.shape[0])
    if in_len == out_len:
        return x.copy()
    if in_len < 2 or out_len < 2:
        raise ValueError("resample length must be >= 2")

    x_old = np.linspace(0.0, 1.0, in_len, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, out_len, dtype=np.float64)
    y_new = np.interp(x_new, x_old, x.astype(np.float64))
    return y_new


def _damped_newton_solve(
    F,
    dF,
    x0: np.ndarray,
    tol: float,
    max_iter: int,
    alpha_min: float = 1e-4,
    backtrack_beta: float = 0.5,
) -> Tuple[np.ndarray, bool, str]:
    """
    Damped Newton with backtracking line search on residual norm.
    This is more conservative than full-step Newton and helps avoid
    jumping across nearby branch basins.
    """
    x = x0.copy()
    for _ in range(max_iter):
        r = F(x)
        if not np.isfinite(np.real(r)).all():
            return x, False, "non_finite_residual"
        r_norm = float(np.linalg.norm(r, ord=2))
        if r_norm < tol:
            return x, True, "ok"

        J = dF(x)
        if not np.isfinite(np.real(J)).all():
            return x, False, "non_finite_jacobian"

        try:
            dx = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            # Least-squares fallback for near-singular Jacobian.
            dx = np.linalg.lstsq(J, -r, rcond=None)[0]

        alpha = 1.0
        accepted = False
        while alpha >= alpha_min:
            x_try = x + alpha * dx
            r_try = F(x_try)
            if np.isfinite(np.real(r_try)).all():
                r_try_norm = float(np.linalg.norm(r_try, ord=2))
                if r_try_norm < r_norm:
                    x = x_try
                    accepted = True
                    break
            alpha *= backtrack_beta

        if not accepted:
            return x, False, "line_search_failed"

    # Final convergence check after exhausting iterations.
    r = F(x)
    if np.isfinite(np.real(r)).all() and float(np.linalg.norm(r, ord=2)) < tol:
        return x, True, "ok"
    return x, False, "max_iter"


def _damped_newton_solve_trace(
    F,
    dF,
    x0: np.ndarray,
    tol: float,
    max_iter: int,
    alpha_min: float = 1e-4,
    backtrack_beta: float = 0.5,
):
    """
    Same as _damped_newton_solve but returns per-iteration residual trace.

    Returns:
        x, ok, reason, trace
        where trace is a list of dict:
          {"iter": int, "residual": float, "accepted_alpha": float}
        iter=0 is the initial point before any Newton update.
    """
    x = x0.copy()
    trace = []

    for it in range(max_iter):
        r = F(x)
        if not np.isfinite(np.real(r)).all():
            trace.append({"iter": it, "residual": float("nan"), "accepted_alpha": 0.0})
            return x, False, "non_finite_residual", trace

        r_norm = float(np.linalg.norm(r, ord=2))
        trace.append({"iter": it, "residual": r_norm, "accepted_alpha": 0.0})

        if r_norm < tol:
            return x, True, "ok", trace

        J = dF(x)
        if not np.isfinite(np.real(J)).all():
            return x, False, "non_finite_jacobian", trace

        try:
            dx = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(J, -r, rcond=None)[0]

        alpha = 1.0
        accepted = False
        while alpha >= alpha_min:
            x_try = x + alpha * dx
            r_try = F(x_try)
            if np.isfinite(np.real(r_try)).all():
                r_try_norm = float(np.linalg.norm(r_try, ord=2))
                if r_try_norm < r_norm:
                    x = x_try
                    trace[-1]["accepted_alpha"] = float(alpha)
                    accepted = True
                    break
            alpha *= backtrack_beta

        if not accepted:
            return x, False, "line_search_failed", trace

    r = F(x)
    if np.isfinite(np.real(r)).all():
        r_norm = float(np.linalg.norm(r, ord=2))
        trace.append({"iter": max_iter, "residual": r_norm, "accepted_alpha": 0.0})
        if r_norm < tol:
            return x, True, "ok", trace
    else:
        trace.append({"iter": max_iter, "residual": float("nan"), "accepted_alpha": 0.0})
    return x, False, "max_iter", trace


def refine_with_newton(
    pred_u: torch.Tensor,
    target_p: float,
    tol: float = 1e-9,
    max_iter: int = 30,
    working_grid_size: int = None,
    return_working_grid: bool = True,
) -> Tuple[torch.Tensor, bool, str]:
    """
    Refine one predicted solution with traditional Newton solver.

    Returns:
        refined_u, success, reason
        - if working_grid_size is set and return_working_grid=True:
            refined_u has length working_grid_size
        - otherwise:
            refined_u has same length as pred_u
    """
    if not _HAS_TRAD_SOLVER:
        return pred_u, False, "solver_unavailable"

    try:
        x0_real = pred_u.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
        n_in = int(x0_real.shape[0])
        if n_in < 3:
            return pred_u, False, "invalid_length"

        n_work = int(working_grid_size) if working_grid_size is not None else n_in
        if n_work < 3:
            return pred_u, False, "invalid_working_grid"
        if n_work != n_in:
            x0_real = _resample_1d_linear(x0_real, out_len=n_work)

        x0 = x0_real.reshape(-1, 1).astype(np.complex128)
        n = x0.shape[0]

        F, dF = faSh1(x0, target_p)
        x_ref_col, ok, reason = _damped_newton_solve(
            F, dF, x0.copy(), tol=tol, max_iter=max_iter
        )
        if not ok:
            return pred_u, False, reason

        if x_ref_col.shape[0] < n:
            return pred_u, False, "newton_output_short"
        x_ref = x_ref_col[:n, 0]
        if not np.isfinite(np.real(x_ref)).all():
            return pred_u, False, "non_finite"

        imag_norm = float(np.linalg.norm(np.imag(x_ref)))
        if imag_norm > 1e-3:
            return pred_u, False, "imag_too_large"

        x_ref_real = np.real(x_ref).reshape(-1).astype(np.float64)
        # Important: Newton may converge in complex space, but taking real-part
        # projection can still break the real-valued PDE residual significantly.
        # Re-check the residual on projected real solution before accepting.
        x_ref_real_col = x_ref_real.reshape(-1, 1).astype(np.complex128)
        real_proj_res = float(np.linalg.norm(F(x_ref_real_col), ord=2))
        real_proj_res_tol = max(1e-6, tol * 1e3)
        if (not np.isfinite(real_proj_res)) or (real_proj_res > real_proj_res_tol):
            return pred_u, False, "real_projection_residual_too_large"

        if (n_work != n_in) and (not return_working_grid):
            x_ref_real = _resample_1d_linear(x_ref_real, out_len=n_in)
        refined = torch.from_numpy(x_ref_real).to(device=pred_u.device, dtype=torch.float64)
        return refined, True, "ok"
    except Exception:
        return pred_u, False, "exception"


def refine_with_newton_trace(
    pred_u: torch.Tensor,
    target_p: float,
    tol: float = 1e-9,
    max_iter: int = 30,
    working_grid_size: int = None,
    return_working_grid: bool = True,
):
    """
    Refine one predicted solution and return Newton residual trace.

    Returns:
        refined_u, success, reason, trace
    """
    if not _HAS_TRAD_SOLVER:
        return pred_u, False, "solver_unavailable", []

    try:
        x0_real = pred_u.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
        n_in = int(x0_real.shape[0])
        if n_in < 3:
            return pred_u, False, "invalid_length", []

        n_work = int(working_grid_size) if working_grid_size is not None else n_in
        if n_work < 3:
            return pred_u, False, "invalid_working_grid", []
        if n_work != n_in:
            x0_real = _resample_1d_linear(x0_real, out_len=n_work)

        x0 = x0_real.reshape(-1, 1).astype(np.complex128)
        n = x0.shape[0]

        F, dF = faSh1(x0, target_p)
        x_ref_col, ok, reason, trace = _damped_newton_solve_trace(
            F, dF, x0.copy(), tol=tol, max_iter=max_iter
        )
        if not ok:
            return pred_u, False, reason, trace

        if x_ref_col.shape[0] < n:
            return pred_u, False, "newton_output_short", trace
        x_ref = x_ref_col[:n, 0]
        if not np.isfinite(np.real(x_ref)).all():
            return pred_u, False, "non_finite", trace

        imag_norm = float(np.linalg.norm(np.imag(x_ref)))
        if imag_norm > 1e-3:
            return pred_u, False, "imag_too_large", trace

        x_ref_real = np.real(x_ref).reshape(-1).astype(np.float64)
        # Same acceptance criterion as refine_with_newton():
        # validate residual after real-part projection.
        x_ref_real_col = x_ref_real.reshape(-1, 1).astype(np.complex128)
        real_proj_res = float(np.linalg.norm(F(x_ref_real_col), ord=2))
        real_proj_res_tol = max(1e-6, tol * 1e3)
        if (not np.isfinite(real_proj_res)) or (real_proj_res > real_proj_res_tol):
            return pred_u, False, "real_projection_residual_too_large", trace

        if (n_work != n_in) and (not return_working_grid):
            x_ref_real = _resample_1d_linear(x_ref_real, out_len=n_in)
        refined = torch.from_numpy(x_ref_real).to(device=pred_u.device, dtype=torch.float64)
        return refined, True, "ok", trace
    except Exception:
        return pred_u, False, "exception", []


def refine_prediction_list_with_newton(
    pred_list: List[torch.Tensor],
    target_p: float,
    tol: float = 1e-9,
    max_iter: int = 30,
    working_grid_size: int = None,
    return_working_grid: bool = True,
) -> Tuple[List[torch.Tensor], Dict[str, int]]:
    """
    Refine a list of predicted solutions. Failed items auto-fallback.
    """
    stats: Dict[str, int] = {"total": 0, "success": 0, "fallback": 0}
    refined_list: List[torch.Tensor] = []

    for u in pred_list:
        stats["total"] += 1
        u_ref, ok, _ = refine_with_newton(
            u,
            target_p=target_p,
            tol=tol,
            max_iter=max_iter,
            working_grid_size=working_grid_size,
            return_working_grid=return_working_grid,
        )
        if ok:
            stats["success"] += 1
        else:
            stats["fallback"] += 1
        refined_list.append(u_ref)

    return refined_list, stats


def compute_newton_residual_l2(u: torch.Tensor, target_p: float) -> float:
    """
    Residual norm based on the exact operator used by traditional Newton:
        ||F(u)||_2, where F comes from Sexample1.faSh1.
    """
    if not _HAS_TRAD_SOLVER:
        return float("nan")
    try:
        x = _to_complex_col(u)
        F, _ = faSh1(x, target_p)
        r = F(x)
        return float(np.linalg.norm(r, ord=2))
    except Exception:
        return float("nan")

