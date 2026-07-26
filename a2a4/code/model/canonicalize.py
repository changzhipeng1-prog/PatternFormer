"""
Physical canonicalization: order multiple solutions at the same p value.

Strategy:
    Sort solutions ascending by their spatial integral (proxy: sum over grid points,
    equivalent to L1 norm divided by N, and proportional to the trapezoidal-rule
    integral over [0,1]).

Rationale:
    At a fixed p, the 1-8 branches produce distinct solution profiles.
    Without a canonical order, the autoregressive model must account for
    all K! orderings, causing exponential ambiguity in Teacher Forcing targets.
    Sorting by integral provides a unique, physically meaningful ordering
    (zero solution < small-amplitude branches < large-amplitude branches).

API:
    canonicalize_solutions(solutions)  → reordered tensor
    canonicalize_batch(solutions_list) → list of reordered tensors
"""
import torch


def canonicalize_solutions(solutions: torch.Tensor) -> torch.Tensor:
    """
    Sort a set of solutions at the same p by their spatial L1 integral (ascending).

    Args:
        solutions: [K, 1024]  — K solutions at the same p value

    Returns:
        [K, 1024]  — solutions reordered by ascending sum (integral proxy)
    """
    if solutions.shape[0] <= 1:
        return solutions

    # Integral proxy: sum of all grid values (proportional to trapezoidal integral)
    integrals = solutions.sum(dim=-1)       # [K]
    order = torch.argsort(integrals)        # ascending
    return solutions[order]


def canonicalize_batch(solutions_list: list) -> list:
    """
    Apply canonicalize_solutions to each element of a list.

    Args:
        solutions_list: list of Tensor [K_i, 1024]

    Returns:
        list of Tensor [K_i, 1024]  (each reordered)
    """
    return [canonicalize_solutions(s) for s in solutions_list]
