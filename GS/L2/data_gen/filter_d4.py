"""D4 symmetry reduction + canonical ordering for the Gray-Scott solution set.

The unit square with Neumann BC has D4 symmetry (8 elements: 4 rotations + 4
reflections), all EXACT on a 128x128 grid. Continuation found many solutions
that are just rotated/reflected copies of one another -> the same physical
solution. This collapses each D4 orbit to one representative, puts it in a
canonical pose, and sorts the distinct solutions deterministically.

Pipeline per parameter:
  1. D4-dedup: two solutions are equivalent if min over the 8 D4 transforms of
     ||A_i - g(A_j)||_inf < tol. Keep one representative per orbit.
  2. Canonical pose: rotate/reflect each representative to the transform that
     maximizes <A, W> with W a fixed NE-ramp template (robust, deterministic).
  3. Sort by (mean(A), energy) for a stable, unique autoregressive order.

Output: gs_dataset_d4.pt (same fields as gs_dataset.pt).
"""
import torch
import numpy as np

import os; HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
TOL = 1e-3          # D4-equivalence inf-norm tol on A (orbit members match to ~1e-8;
                    # genuinely distinct solutions differ by ~0.3 -> huge margin)


def d4_transforms(x):
    """x: [...,H,W] -> [8,...,H,W] the 8 dihedral transforms (last two dims)."""
    g = []
    for k in range(4):
        g.append(torch.rot90(x, k, dims=(-2, -1)))
    xt = x.transpose(-1, -2)
    for k in range(4):
        g.append(torch.rot90(xt, k, dims=(-2, -1)))
    return torch.stack(g, 0)            # [8, ..., H, W]


def reduce_param(sols, W):
    """sols: [K,2,128,128] -> (kept canonical-posed, sorted) [K',2,128,128]."""
    A = sols[:, 0]                                   # [K,128,128]
    # all 8 D4 transforms of each A: [8,K,128,128]
    Agt = d4_transforms(A)
    # ---- 1. greedy D4-dedup ----
    kept = []
    for i in range(sols.shape[0]):
        dup = False
        for j in kept:
            # min over 8 transforms of ||g(A_i) - A_j||_inf
            d = (Agt[:, i] - A[j][None]).abs().amax(dim=(1, 2)).min()
            if d < TOL:
                dup = True
                break
        if not dup:
            kept.append(i)
    kept = torch.tensor(kept, device=sols.device)
    reps = sols[kept]                                # [K',2,128,128]

    # ---- 2. canonical pose (argmax <g(A), W>) ----
    repA = reps[:, 0]
    g_all = d4_transforms(reps)                      # [8,K',2,128,128]
    proj = (d4_transforms(repA) * W).sum(dim=(2, 3))  # [8,K']
    best = proj.argmax(0)                            # [K']
    canon = g_all[best, torch.arange(reps.shape[0], device=sols.device)]  # [K',2,128,128]

    # ---- 3. stable sort by (mean(A), energy) ----
    ca = canon[:, 0]
    mean = ca.flatten(1).mean(1)
    energy = (ca ** 2).flatten(1).sum(1)
    key = mean + 1e-6 * energy                       # mean dominant, energy tiebreak
    order = torch.argsort(key)
    return canon[order]


def main():
    d = torch.load(f"{HERE}/gs_dataset.pt")
    A, S = d["A"], d["S"]; pid = d["param_id"]; params = d["params"]
    P = params.shape[0]
    sol = torch.stack([A, S], 1)                     # [N,2,128,128]

    # NE-ramp template on the A channel
    xy = torch.linspace(0, 1, 128, device=DEV)
    W = (xy[None, :] - 0.5) + 0.6180339 * (xy[:, None] - 0.5)   # [128,128]

    out_A, out_S, out_rho, out_mu, out_pid = [], [], [], [], []
    new_params, new_counts = [], []
    before = after = 0
    new_pid = 0
    for p in range(P):
        s = sol[pid == p].to(DEV)
        before += s.shape[0]
        red = reduce_param(s, W).cpu()
        after += red.shape[0]
        out_A.append(red[:, 0]); out_S.append(red[:, 1])
        rho = float(params[p, 0]); mu = float(params[p, 1])
        out_rho.append(torch.full((red.shape[0],), rho))
        out_mu.append(torch.full((red.shape[0],), mu))
        out_pid.append(torch.full((red.shape[0],), new_pid, dtype=torch.long))
        new_params.append([rho, mu]); new_counts.append(red.shape[0]); new_pid += 1

    A = torch.cat(out_A); S = torch.cat(out_S)
    rho = torch.cat(out_rho); mu = torch.cat(out_mu); pidx = torch.cat(out_pid)
    params_t = torch.tensor(new_params, dtype=torch.float32)
    counts = torch.tensor(new_counts, dtype=torch.long)
    info = dict(d.get("info", {})); info["N"] = A.shape[0]; info["P"] = params_t.shape[0]
    info["note"] = info.get("note", "") + "; D4-reduced + canonical pose + sorted(mean,energy)"
    torch.save(dict(A=A, S=S, rho=rho, mu=mu, param_id=pidx, params=params_t,
                    counts=counts, info=info), f"{HERE}/gs_dataset_d4.pt")

    import collections
    bc = collections.Counter(d["counts"].tolist())
    ac = collections.Counter(counts.tolist())
    print(f"D4 reduction: {before} -> {after} sols ({100*(1-after/before):.1f}% removed)")
    print(f"per-param before: {dict(sorted(bc.items()))}")
    print(f"per-param after : {dict(sorted(ac.items()))}")
    print(f"mean sols/param: {d['counts'].float().mean():.1f} -> {counts.float().mean():.1f} "
          f"(max {counts.max().item()})")
    print(f"saved gs_dataset_d4.pt: N={A.shape[0]} over P={params_t.shape[0]} params")


if __name__ == "__main__":
    main()
