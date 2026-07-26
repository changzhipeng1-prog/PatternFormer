"""
canonicalize.py — Ordering for 2D FEM solutions (145 nodes, D4-filtered dataset)

After D4 filtering (filter_d4_dataset.py), each D4 rotation orbit is
represented by exactly one canonical solution. Within a given s-value the
remaining solutions have DISTINCT integral values, so a simple ascending
sort by ∫u dΩ ≈ mean(u) gives a stable, unique ordering.

This replaces the earlier (u_center, x_asym) ordering, which was needed
to match the raw dataset's internal order but is no longer necessary once
D4 duplicates are removed.

API:
    canonicalize_solutions(solutions) → reorder [K, 145] tensor
    canonicalize_batch(solutions_list)
"""
import torch


def canonicalize_solutions(solutions: torch.Tensor) -> torch.Tensor:
    """
    Sort solutions at the same s-value by ascending mean(u).

    Args:
        solutions: [K, 145] float tensor

    Returns:
        [K, 145] reordered tensor (ascending integral value)
    """
    K = solutions.shape[0]
    if K <= 1:
        return solutions
    means = solutions.float().mean(dim=1)   # [K]
    order = torch.argsort(means)
    return solutions[order]


def canonicalize_batch(solutions_list: list) -> list:
    """Apply canonicalize_solutions to each element of a list."""
    return [canonicalize_solutions(s) for s in solutions_list]
