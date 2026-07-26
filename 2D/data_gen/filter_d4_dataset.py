"""
filter_d4_dataset.py
====================
Remove D4-rotationally-redundant solutions from p_solutions_lookup_2d.pt.

The 2D PDE -Δu - u² = -s·sin(πx)sin(πy) on [0,1]² has D4 symmetry.
Solutions come in two kinds of orbit:

  Size 1  : fully D4-symmetric (max at center, x_asym=0).  Keep as-is.
  Size 4  : 4 solutions related by 90° rotations of the unit square.
            Max nodes land in four distinct symmetric positions
            (e.g. four corner quadrants, or four edge midpoints).
            Keep only the canonical representative.

Canonical representative of a rotation orbit of 4:
    Among the 4 solutions, keep the one whose maximum-value node is most
    "north-east" of center, defined lexicographically as:
        1st key:  largest  (max_x − 0.5)   [furthest to the right]
        2nd key:  largest  (max_y − 0.5)   [highest, breaks ties]

At s=1600 this gives:
  • u_center≈-4.1 orbit: max at (0.75,0.75) selected  [top-right quadrant]
  • u_center≈14.9 orbit: max at (0.75,0.50) selected  [right-edge midpoint]

Result: ≤4 solutions per s  (2 symmetric + 2 orbit representatives),
        down from ≤10.

Input :  p_solutions_lookup_2d.pt
Output:  p_solutions_lookup_2d_filtered.pt  (same directory, same format)

Usage:
    python filter_d4_dataset.py
"""

import torch
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

IN_PATH  = Path(__file__).resolve().parent.parent / 'data' / 'p_solutions_lookup_2d.pt'
OUT_PATH = IN_PATH.parent / 'p_solutions_lookup_2d_filtered.pt'


# ---------------------------------------------------------------------------
# Canonical representative selection
# ---------------------------------------------------------------------------

def canonical_rep_index(sols_np: np.ndarray, coord_np: np.ndarray) -> int:
    """
    Given an orbit of solutions [K, 145], return the index (0..K-1) of
    the canonical representative: the solution whose maximum-value node is
    most north-east of center (0.5, 0.5).

    Selection key (lexicographic, descending):
        ( max_x − 0.5,   max_y − 0.5 )
    where (max_x, max_y) = coordinate of the node with the largest |u| value.
    """
    K = sols_np.shape[0]
    scores = []
    for k in range(K):
        imax = int(np.argmax(np.abs(sols_np[k])))
        x, y = float(coord_np[imax, 0]), float(coord_np[imax, 1])
        scores.append((x - 0.5, y - 0.5))
    # Return index with largest (dx, dy) in lexicographic order
    return max(range(K), key=lambda k: scores[k])


# ---------------------------------------------------------------------------
# Per-s filtering
# ---------------------------------------------------------------------------

def filter_one_s(sols: torch.Tensor, coord_np: np.ndarray,
                 cidx: int) -> torch.Tensor:
    """
    Filter a [K, 145] solution tensor for a single s-value.

    Groups solutions by round(u_center, 1):
      - Group of size 1: symmetric → keep
      - Group of size >1: rotation orbit → keep canonical representative only

    Returns [K', 145] with K' ≤ K.
    """
    if sols.shape[0] == 0:
        return sols

    sols_np = sols.float().numpy()           # [K, 145]

    # Group by rounded center value
    u_centers = np.round(sols_np[:, cidx], 1)
    groups = defaultdict(list)
    for k, uc in enumerate(u_centers):
        groups[uc].append(k)

    kept = []
    for uc_key in sorted(groups.keys()):
        idxs = groups[uc_key]
        if len(idxs) == 1:
            kept.append(idxs[0])
        else:
            orbit_sols = sols_np[idxs]        # [orbit_size, 145]
            local_best = canonical_rep_index(orbit_sols, coord_np)
            kept.append(idxs[local_best])

    # kept is already in ascending index order (groups sorted by u_center,
    # which matches the original canonicalized ordering)
    kept.sort()
    return sols[torch.tensor(kept, dtype=torch.long)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading: {IN_PATH}")
    data = torch.load(IN_PATH, weights_only=False)

    coord_np = data['coord'].float().numpy()          # [145, 2]
    cidx     = int(np.argmin(np.sum((coord_np - [0.5, 0.5]) ** 2, axis=1)))
    print(f"Center node index: {cidx}  coord={coord_np[cidx]}")

    solutions_by_p_in  = data['solutions_by_p']
    p_values           = data['p_values']
    N_s = len(p_values)

    # ── Filter each s-value ──────────────────────────────────────────────
    solutions_by_p_out = []
    k_before_all = []
    k_after_all  = []

    for i, sols in enumerate(solutions_by_p_in):
        before = sols.shape[0]
        filtered = filter_one_s(sols, coord_np, cidx)
        after  = filtered.shape[0]
        solutions_by_p_out.append(filtered)
        k_before_all.append(before)
        k_after_all.append(after)

    # ── Statistics ───────────────────────────────────────────────────────
    print(f"\nBefore filtering:")
    print(f"  Distribution: {dict(sorted(Counter(k_before_all).items()))}")
    print(f"  Total solutions: {sum(k_before_all)}")

    print(f"\nAfter filtering:")
    print(f"  Distribution: {dict(sorted(Counter(k_after_all).items()))}")
    print(f"  Total solutions: {sum(k_after_all)}")

    print(f"\nReduction: {sum(k_before_all)} → {sum(k_after_all)} "
          f"({100*(1 - sum(k_after_all)/sum(k_before_all)):.1f}% removed)")

    # Spot-check s=1600
    idx_1600 = (p_values == 1600).nonzero(as_tuple=True)[0]
    if len(idx_1600) > 0:
        i1600 = idx_1600[0].item()
        sols_kept = solutions_by_p_out[i1600].float().numpy()
        print(f"\ns=1600 spot check ({sols_kept.shape[0]} solutions kept):")
        for k, u in enumerate(sols_kept):
            imax = int(np.argmax(np.abs(u)))
            x, y = coord_np[imax]
            uc = round(float(u[cidx]), 3)
            left  = coord_np[:, 0] < 0.5
            right = coord_np[:, 0] > 0.5
            x_asym = float(np.mean(u[left]) - np.mean(u[right]))
            print(f"  [{k}] u_center={uc:>8.3f}  max_node=({x:.3f},{y:.3f})  "
                  f"x_asym={x_asym:+.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    out_data = dict(data)   # shallow copy keeps coord, elem, free_nodes, etc.
    out_data['solutions_by_p'] = solutions_by_p_out
    out_data['n_total_solutions'] = int(sum(k_after_all))
    out_data['note'] = (
        data.get('note', '') +
        " | D4-filtered: rotation-orbit redundancies removed; "
        "each orbit of 4 reduced to canonical representative "
        "(max-value node most north-east of center)."
    )

    torch.save(out_data, OUT_PATH)
    print(f"\nSaved: {OUT_PATH}")

    # Sanity check
    loaded = torch.load(OUT_PATH, weights_only=False)
    assert len(loaded['p_values']) == len(loaded['solutions_by_p'])
    print("Sanity check passed.")


if __name__ == '__main__':
    main()
