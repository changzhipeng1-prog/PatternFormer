"""canonicalize.py — stable ordering for the solution set at one parameter.

Sort the K coexisting solutions by ascending mean of the A channel. Gives a
deterministic teacher-forcing order. (Note: our data is NOT D4-reduced, so
rotated/reflected copies can share a mean and tie — a future refinement is to
D4-canonicalize each solution before sorting.)
"""
import torch


def canonicalize_solutions(solutions: torch.Tensor) -> torch.Tensor:
    """solutions: [K, C, H, W]  ->  reordered [K, C, H, W] by ascending mean(A)."""
    K = solutions.shape[0]
    if K <= 1:
        return solutions
    key = solutions[:, 0].float().flatten(1).mean(dim=1)   # mean of A channel
    order = torch.argsort(key)
    return solutions[order]


def canonicalize_batch(solutions_list: list) -> list:
    return [canonicalize_solutions(s) for s in solutions_list]
