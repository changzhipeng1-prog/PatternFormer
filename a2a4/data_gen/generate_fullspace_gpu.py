#!/usr/bin/env python3
"""
Full-space BFS verification + gap-filling.

Uses ALL existing dataset samples as seeds (snapped to a regular grid).
BFS starts from k=8 cells (most solutions) and propagates outward.
Multiple GPUs: frontier is split across N_GPU devices each round.

Output: corrected full-space dataset with verified/extended solution counts.
"""

import argparse, time
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp

# ── TDMA batched on a single device ──────────────────────────────────────────

def tdma_batch(a, b, c, d):
    B, N = d.shape
    c_p = torch.empty_like(d); d_p = torch.empty_like(d)
    inv0 = 1.0 / b[:, 0]
    c_p[:, 0] = c[:, 0] * inv0; d_p[:, 0] = d[:, 0] * inv0
    for i in range(1, N):
        denom = b[:, i] - a[:, i] * c_p[:, i-1]
        inv   = 1.0 / denom
        c_p[:, i] = c[:, i] * inv
        d_p[:, i] = (d[:, i] - a[:, i] * d_p[:, i-1]) * inv
    x = torch.empty_like(d)
    x[:, N-1] = d_p[:, N-1]
    for i in range(N-2, -1, -1):
        x[:, i] = d_p[:, i] - c_p[:, i] * x[:, i+1]
    return x


def newton_on_device(chunk, device, N=1024, max_iter=100, tol=1e-9):
    """
    chunk: list of (ci, a4, a2, u0_np)
    Returns: list of (ci, u_np or None)
    """
    if not chunk:
        return []
    h = 1.0 / N
    dtype = torch.float64

    a_fix = torch.full((1, N), -1.0/h, device=device, dtype=dtype)
    a_fix[0, 0] = 0.0
    c_fix = torch.full((1, N), -1.0/h, device=device, dtype=dtype)
    c_fix[0, 0] = -2.0/h; c_fix[0, -1] = 0.0

    B = len(chunk)
    ci_list = [x[0] for x in chunk]
    a4_t = torch.tensor([x[1] for x in chunk], device=device, dtype=dtype)
    a2_t = torch.tensor([x[2] for x in chunk], device=device, dtype=dtype)
    u    = torch.tensor(np.stack([x[3] for x in chunk]),
                        device=device, dtype=dtype)

    a_b = a_fix.expand(B, -1).clone()
    c_b = c_fix.expand(B, -1).clone()
    active = torch.ones(B, dtype=torch.bool, device=device)

    for _ in range(max_iter):
        if not active.any(): break
        f_u  = a4_t[:,None]*u**4 + a2_t[:,None]*u**2
        fp_u = 4*a4_t[:,None]*u**3 + 2*a2_t[:,None]*u
        Lu = torch.empty_like(u)
        Lu[:,0]    = 2*u[:,0] - 2*u[:,1]
        Lu[:,1:-1] = 2*u[:,1:-1] - u[:,:-2] - u[:,2:]
        Lu[:,-1]   = 2*u[:,-1]  - u[:,-2]
        Fu    = Lu/h + h*f_u
        b_jac = 2.0/h + h*fp_u
        res   = torch.linalg.norm(Fu, dim=1)
        active &= (res >= tol) & (res < 1e6)
        delta = tdma_batch(a_b, b_jac, c_b, -Fu)
        u += delta * active[:,None].to(dtype)

    Fu_f, _ = (None, None)
    f_u_f  = a4_t[:,None]*u**4 + a2_t[:,None]*u**2
    Lu_f   = torch.empty_like(u)
    Lu_f[:,0]    = 2*u[:,0] - 2*u[:,1]
    Lu_f[:,1:-1] = 2*u[:,1:-1] - u[:,:-2] - u[:,2:]
    Lu_f[:,-1]   = 2*u[:,-1]  - u[:,-2]
    Fu_f   = Lu_f/h + h*f_u_f
    conv   = (torch.linalg.norm(Fu_f, dim=1) < tol).cpu().numpy()
    u_np   = u.cpu().numpy()

    return [(ci_list[i], u_np[i] if conv[i] else None) for i in range(B)]


def gpu_newton_multi(params_list, inits_list, devices,
                     N=1024, max_iter=100, tol=1e-9, batch_limit=8192):
    """
    Distribute (cell, init) pairs across multiple GPU devices.
    Returns list of list[np.ndarray] — converged solutions per cell.
    """
    flat = []
    for ci, ((a4v, a2v), inits) in enumerate(zip(params_list, inits_list)):
        for u0 in inits:
            flat.append((ci, a4v, a2v, u0))

    n_cells = len(params_list)
    if not flat:
        return [[] for _ in range(n_cells)]

    # Split flat across devices
    n_dev = len(devices)
    per_dev = max(1, len(flat) // n_dev)
    chunks_by_dev = []
    for di, dev in enumerate(devices):
        start = di * per_dev
        end   = start + per_dev if di < n_dev-1 else len(flat)
        # Sub-chunk for batch_limit
        sub = flat[start:end]
        chunks_by_dev.append((dev, sub))

    cell_solutions = defaultdict(list)

    for dev, sub in chunks_by_dev:
        for start in range(0, len(sub), batch_limit):
            ch = sub[start:start+batch_limit]
            results = newton_on_device(ch, dev, N, max_iter, tol)
            for ci, u_np in results:
                if u_np is None: continue
                dup = any(np.linalg.norm(u_np - s) < 1e-6
                          for s in cell_solutions[ci])
                if not dup:
                    cell_solutions[ci].append(u_np)

    return [cell_solutions.get(ci, []) for ci in range(n_cells)]


# ── Grid helpers ──────────────────────────────────────────────────────────────

def build_grid(na4, na2, a4_min, a4_max, a2_min, a2_max):
    a4v = np.linspace(a4_min, a4_max, na4)
    a2v = np.linspace(a2_min, a2_max, na2)
    pt_map = {(ia, ib): (float(a4v[ia]), float(a2v[ib]))
              for ia in range(na4) for ib in range(na2)}
    return a4v, a2v, pt_map


def get_neighbor_inits(ia, ib, solved_dict):
    inits, seen = [], set()
    for (dia, dib) in [(0,1),(0,-1),(1,0),(-1,0)]:
        for sol in solved_dict.get((ia+dia, ib+dib), []):
            uid = sol.tobytes()[:64]
            if uid not in seen:
                inits.append(sol); seen.add(uid)
    return inits


def build_frontier(solved_dict, pt_map):
    frontier = {}
    for (ia, ib) in solved_dict:
        for (dia, dib) in [(0,1),(0,-1),(1,0),(-1,0)]:
            nb = (ia+dia, ib+dib)
            if nb in pt_map and nb not in solved_dict:
                frontier[nb] = True
    return list(frontier.keys())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',       default='./data/bvp_fullspace.pt')
    parser.add_argument('--na4',          type=int,   default=120)
    parser.add_argument('--na2',          type=int,   default=250)
    parser.add_argument('--a4_min',       type=float, default=0.05)
    parser.add_argument('--a4_max',       type=float, default=3.00)
    parser.add_argument('--a2_min',       type=float, default=-40.0)
    parser.add_argument('--a2_max',       type=float, default=-2.0)
    parser.add_argument('--seed_dataset',  default='../qwen/data/bvp_lookup.pt',
                        help='Single seed file (legacy)')
    parser.add_argument('--seed_datasets', default=None,
                        help='Comma-separated list of seed files (takes priority over --seed_dataset)')
    parser.add_argument('--min_k',        type=int,   default=2)
    parser.add_argument('--devices',      default='2,4,5,6',
                        help='Comma-separated CUDA device indices')
    parser.add_argument('--batch_limit',  type=int,   default=8192)
    args = parser.parse_args()

    devices = [torch.device(f'cuda:{d}') for d in args.devices.split(',')
               if torch.cuda.is_available()]
    if not devices:
        devices = [torch.device('cpu')]
    print(f"Devices: {devices}", flush=True)

    t0 = time.time()
    print("=" * 65)
    print("Full-space BFS verification (multi-GPU, TDMA Newton)")
    print(f"  Grid: a4∈[{args.a4_min},{args.a4_max}]×{args.na4}  "
          f"a2∈[{args.a2_min},{args.a2_max}]×{args.na2}")
    print(f"  Total grid cells: {args.na4*args.na2:,}")
    print("=" * 65, flush=True)

    a4_vals, a2_vals, pt_map = build_grid(
        args.na4, args.na2, args.a4_min, args.a4_max, args.a2_min, args.a2_max)

    # ── Snap ALL existing data as seeds ───────────────────────────────────────
    solved_dict = {}  # (ia,ib) -> list[np.ndarray N]

    seed_paths = ([p.strip() for p in args.seed_datasets.split(',')]
                  if args.seed_datasets else [args.seed_dataset])

    def snap_seed_file(path):
        print(f"\nLoading seeds from {path} …", flush=True)
        seed_data = torch.load(path, weights_only=False)
        if isinstance(seed_data, list):
            items = [(float(d['params'][0]), float(d['params'][1]), d['solutions'])
                     for d in seed_data]
            print(f"  {len(items)} samples (fullspace format).", flush=True)
            for (a4s, a2s, sols_list) in items:
                if not (args.a4_min <= a4s <= args.a4_max and
                        args.a2_min <= a2s <= args.a2_max):
                    continue
                ia = int(np.argmin(np.abs(a4_vals - a4s)))
                ib = int(np.argmin(np.abs(a2_vals - a2s)))
                sols_np = [np.array(s) if not isinstance(s, np.ndarray) else s
                           for s in sols_list]
                key = (ia, ib)
                if key not in solved_dict or len(solved_dict[key]) < len(sols_np):
                    solved_dict[key] = sols_np
        else:
            sols_raw = seed_data["solutions_by_idx"]
            prms_raw = seed_data["params_list"]
            print(f"  {len(prms_raw)} samples (lookup format).", flush=True)
            for idx in range(len(prms_raw)):
                a4s = float(prms_raw[idx][0]); a2s = float(prms_raw[idx][1])
                if not (args.a4_min <= a4s <= args.a4_max and
                        args.a2_min <= a2s <= args.a2_max):
                    continue
                ia = int(np.argmin(np.abs(a4_vals - a4s)))
                ib = int(np.argmin(np.abs(a2_vals - a2s)))
                k  = sols_raw[idx].shape[0]
                sols_np = [sols_raw[idx][j].numpy() for j in range(k)]
                key = (ia, ib)
                if key not in solved_dict or len(solved_dict[key]) < k:
                    solved_dict[key] = sols_np

    for path in seed_paths:
        snap_seed_file(path)

    kd = defaultdict(int)
    for s in solved_dict.values(): kd[len(s)] += 1
    print(f"  Snapped {len(solved_dict)} cells. k-dist: {dict(sorted(kd.items()))}", flush=True)
    print(f"  Unsolved: {len(pt_map)-len(solved_dict):,} cells", flush=True)

    # ── BFS from k=8 outward ──────────────────────────────────────────────────
    # Prioritise frontier cells adjacent to high-k cells by sorting frontier
    def frontier_priority(cell, solved_dict):
        """Higher priority = adjacent to higher k neighbour."""
        best = 0
        ia, ib = cell
        for (dia, dib) in [(0,1),(0,-1),(1,0),(-1,0)]:
            nb = (ia+dia, ib+dib)
            best = max(best, len(solved_dict.get(nb, [])))
        return -best  # negative so highest k comes first

    print(f"\nBFS starting from k=8 seeds …", flush=True)
    n_round = 0
    while True:
        frontier = build_frontier(solved_dict, pt_map)
        if not frontier:
            break
        n_round += 1

        # Sort: cells adjacent to highest-k neighbours processed first
        frontier.sort(key=lambda c: frontier_priority(c, solved_dict))

        params_f = [pt_map[c] for c in frontier]
        inits_f  = [get_neighbor_inits(c[0], c[1], solved_dict) for c in frontier]

        # Skip cells with no inits (shouldn't happen in BFS, but guard)
        valid = [(i, c) for i, c in enumerate(frontier) if inits_f[i]]
        if not valid:
            # Mark all as empty and continue
            for c in frontier: solved_dict[c] = []
            continue

        v_idx    = [v[0] for v in valid]
        v_cells  = [v[1] for v in valid]
        v_params = [params_f[i] for i in v_idx]
        v_inits  = [inits_f[i]  for i in v_idx]

        results = gpu_newton_multi(v_params, v_inits, devices,
                                   batch_limit=args.batch_limit)

        for cell, sols in zip(v_cells, results):
            solved_dict[cell] = sols

        # Cells with no valid inits get marked empty
        processed = {c for c in v_cells}
        for c in frontier:
            if c not in processed:
                solved_dict[c] = []

        elapsed = time.time() - t0
        kd_now = defaultdict(int)
        for s in solved_dict.values(): kd_now[len(s)] += 1
        new_k = sum(1 for c in frontier if len(solved_dict.get(c,[])) > 0)
        remaining = len(pt_map) - len(solved_dict)
        print(f"  Round {n_round:4d}: frontier={len(frontier):6d}  "
              f"found_k≥1={new_k:6d}  remaining={remaining:6d}  "
              f"k-dist={dict(sorted(kd_now.items()))}  {elapsed/60:.1f}min",
              flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    samples = []
    for (ia, ib), sols in solved_dict.items():
        if len(sols) < args.min_k: continue
        a4, a2 = pt_map[(ia, ib)]
        samples.append({'params': (a4, a2), 'solutions': sols})

    kf = defaultdict(int)
    for s in samples: kf[len(s['solutions'])] += 1
    print(f"\nFinal k-distribution: {dict(sorted(kf.items()))}")
    print(f"Total samples (k≥{args.min_k}): {len(samples)}")
    print(f"Saving to {args.output} …", flush=True)
    torch.save(samples, args.output)
    print(f"Done. Total time: {(time.time()-t0)/60:.1f} min")

    # ── Plot ──────────────────────────────────────────────────────────────────
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_a4 = np.array([s['params'][0] for s in samples])
    all_a2 = np.array([s['params'][1] for s in samples])
    all_k  = np.array([len(s['solutions']) for s in samples])

    palette = {2:'#2196F3', 3:'#00bcd4', 4:'#bbbbbb',
               5:'#ff7f00', 6:'#6a3d9a', 7:'#e31a1c', 8:'#00C853'}
    sizes   = {2:8, 3:10, 4:4, 5:18, 6:28, 7:6, 8:10}
    alphas_ = {2:0.6, 3:0.8, 4:0.15, 5:1.0, 6:1.0, 7:0.8, 8:1.0}
    ORDER   = [4, 2, 3, 7, 5, 6, 8]

    fig, ax = plt.subplots(figsize=(12, 8))
    for kk in ORDER:
        m = all_k == kk
        if m.sum() == 0: continue
        ax.scatter(all_a4[m], all_a2[m],
                   s=sizes.get(kk,8), c=palette.get(kk,'grey'),
                   alpha=alphas_.get(kk,0.5), linewidths=0,
                   zorder=5 if kk>=5 else 2,
                   label=f'k={kk}  (n={m.sum()})')
    ax.set_xlabel(r'$a_4$', fontsize=13); ax.set_ylabel(r'$a_2$', fontsize=13)
    ax.set_title(f'Full-space BFS corrected dataset  (N={len(samples):,})\n'
                 r'$-u\'\'+ a_4u^4+a_2u^2=0$,  $u\'(0)=0$, $u(1)=0$',
                 fontsize=12)
    ax.legend(fontsize=10, markerscale=2, loc='upper right',
              bbox_to_anchor=(1.18, 1.0))
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    plot_out = args.output.replace('.pt', '_distribution.png')
    fig.savefig(plot_out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {plot_out}")


if __name__ == '__main__':
    main()
