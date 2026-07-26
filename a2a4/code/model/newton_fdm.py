"""Post-processing FDM damped-Newton solver for the a2a4 problem.

PDE:  -u'' + a4*u^4 + a2*u^2 = 0  on [0,1],  Neumann u'(0)=0,  Dirichlet u(1)=0.

The residual convention is IDENTICAL to the data generator
(data_gen/generate_fullspace_gpu.py):

    Fu = Lu/h + h*f,
    Lu = 2*u_i - u_{i-1} - u_{i+1}   (positive discrete Laplacian),
    f  = a4*u^4 + a2*u^2,
    Neumann left:        row 0   -> 2*u0 - 2*u1
    Dirichlet-via-ghost: row N-1 -> 2*u_{N-1} - u_{N-2}   (ghost u[N]=0)

Because this is exactly the discrete root the ground-truth solutions satisfy,
the Newton solver here is BOTH the post-processing refiner AND the same operator
used to generate the dataset (post-proc == data-gen).

This module is self-contained (numpy + scipy only); pass the grid size N and
step h=1/N explicitly so it does not depend on the training Config.
"""
import numpy as np
import scipy.linalg as la

DEFAULT_N = 1024


def fd_F(u, a4, a2, h):
    """Residual vector in the data-generation convention (see module docstring)."""
    u = np.asarray(u, dtype=np.float64)
    N = len(u)
    F = np.zeros(N)
    f = a4 * u ** 4 + a2 * u ** 2
    F[0] = (2 * u[0] - 2 * u[1]) / h + h * f[0]
    F[1:-1] = (2 * u[1:-1] - u[:-2] - u[2:]) / h + h * f[1:-1]
    F[-1] = (2 * u[-1] - u[-2]) / h + h * f[-1]
    return F


def residual_l2(u, a4, a2, h):
    """||F(u)||_2 in the data-gen / post-proc norm."""
    return float(np.linalg.norm(fd_F(u, a4, a2, h)))


def newton_refine(u0, a4, a2, h=None, tol=1e-6, max_iter=30, abs_tol=None):
    """Damped Newton refinement of an initial guess u0.

    Convergence:
      * if abs_tol is given  -> ABSOLUTE: stop when ||F|| < abs_tol. This matches
        the data generator (generate_fullspace_gpu.py uses ||F|| < 1e-9). In
        float64 the residual floor of the exact discrete solution is ~1e-11 (the
        1/h factor does NOT prevent it; a ~5e-3 floor only appears when the input
        is stored in float32, which the model output / dataset are).
      * else                 -> RELATIVE: stop when ||F|| < tol * ||F(u0)||
        (scale-free; the original default, kept for backward compatibility).

    Returns (refined_u, converged: bool, n_iters: int).
    """
    u = np.asarray(u0, dtype=np.float64).copy()
    N = len(u)
    if h is None:
        h = 1.0 / N
    r0 = float(np.linalg.norm(fd_F(u, a4, a2, h))) + 1e-30
    thresh = abs_tol if abs_tol is not None else tol * r0
    for it in range(max_iter):
        F = fd_F(u, a4, a2, h)
        r = float(np.linalg.norm(F))
        if r < thresh:
            return u, True, it
        # Tridiagonal Jacobian matching Fu = Lu/h + h*f:
        #   diag  = 2/h + h*f'(u),  f' = 4*a4*u^3 + 2*a2*u
        #   super = -1/h interior, -2/h at the Neumann row, (0 at last row)
        #   sub   = -1/h interior (incl. the right ghost row)
        d = 2.0 / h + h * (4 * a4 * u ** 3 + 2 * a2 * u)
        sup = np.full(N - 1, -1.0 / h); sup[0] = -2.0 / h
        sub = np.full(N - 1, -1.0 / h)
        ab = np.zeros((3, N)); ab[0, 1:] = sup; ab[1] = d; ab[2, :-1] = sub
        try:
            du = la.solve_banded((1, 1), ab, -F)
        except Exception:
            return u, False, it
        if not np.isfinite(du).all():
            return u, False, it
        alpha = 1.0
        for _ in range(10):
            ut = u + alpha * du
            if np.linalg.norm(fd_F(ut, a4, a2, h)) < r:
                u = ut
                break
            alpha *= 0.5
    r = float(np.linalg.norm(fd_F(u, a4, a2, h)))
    return u, r < thresh, max_iter
