"""Multi-parameter Gray-Scott data generation by parameter continuation +
random perturbation (no companion method available in the FDM code).

BFS outward from the center (rho0,mu0) by Chebyshev distance. Each cell warm-
starts from solutions already found in strictly-inner neighbours, adds random
polynomial perturbations to probe other basins, batch-solves on GPU, then
filters trivial / dedupes / caps K per cell.

Run a small pilot first (--rings 3 --grid 7) then scale to the full grid.
"""
import argparse, json, os, time
import numpy as np
import scipy.io as sio
import torch
from gs_torch import GSOperator, solve_batch

import os; HERE = os.path.dirname(os.path.abspath(__file__))


def poly_perturb(x, scale, B, gen, device, dtype):
    """B independent random degree<=4 polynomial fields, normalised, *scale.
    Mirrors example3_2D.m: sum_{a,b=0..4} U(-1,1) x^a y^b, /max|.|, *scale."""
    n = x.numel()
    powx = torch.stack([x ** a for a in range(5)], 0)        # (5,n)
    fields = torch.zeros(B, n, n, dtype=dtype, device=device)
    coef = (torch.rand(B, 5, 5, generator=gen, device=device, dtype=dtype) * 2 - 1)
    # field_b[i,j] = sum_{a,c} coef[b,a,c] * x[i]^a * x[j]^c
    # = powx^T @ coef_b @ powx  (per b)
    fields = torch.einsum("ai,bac,cj->bij", powx, coef, powx)
    fields = fields / fields.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-30)
    return scale * fields


def dedup_cap(A, S, tol=1e-5, kmax=15, triv=0.05):
    """Greedy inf-norm dedup + trivial filter + cap by |A|_inf. A,S: (K,n,n)."""
    if A.shape[0] == 0:
        return A, S
    ainf = A.abs().amax(dim=(1, 2))
    keep_nontriv = ainf > triv
    A, S, ainf = A[keep_nontriv], S[keep_nontriv], ainf[keep_nontriv]
    if A.shape[0] == 0:
        return A, S
    order = torch.argsort(ainf, descending=True)
    A, S = A[order], S[order]
    kept = [0]
    flat = A.reshape(A.shape[0], -1)
    for i in range(1, A.shape[0]):
        d = (flat[i][None, :] - flat[kept]).abs().amax(dim=1)
        if d.min() > tol:
            kept.append(i)
        if len(kept) >= kmax:
            break
    idx = torch.tensor(kept, device=A.device)
    return A[idx], S[idx]


def run(args):
    dev = "cuda"
    op = GSOperator(f"{HERE}/op_n128.mat", device=dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    x = op.Tx.new_tensor(sio.loadmat(f"{HERE}/op_n128.mat")["x"].ravel())

    # rectangular parameter grid; BFS seeded at the cell nearest (seed_rho,seed_mu)
    rho_grid = np.linspace(args.rho_min, args.rho_max, args.nrho)
    mu_grid = np.linspace(args.mu_min, args.mu_max, args.nmu)
    nrho, nmu = args.nrho, args.nmu
    ci = int(np.argmin(np.abs(rho_grid - args.seed_rho)))
    cj = int(np.argmin(np.abs(mu_grid - args.seed_mu)))
    rings = max(ci, nrho - 1 - ci, cj, nmu - 1 - cj)
    print(f"grid {nrho}x{nmu}  rho[{rho_grid[0]:.4f},{rho_grid[-1]:.4f}] d={rho_grid[1]-rho_grid[0]:.4f} "
          f"mu[{mu_grid[0]:.4f},{mu_grid[-1]:.4f}] d={mu_grid[1]-mu_grid[0]:.4f}  "
          f"seed cell=({ci},{cj})=({rho_grid[ci]:.4f},{mu_grid[cj]:.4f})  rings={rings}")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # center cell from the 8 base seeds (already converged in matlab_ref)
    ref = sio.loadmat(f"{HERE}/matlab_ref.mat")
    A_c = torch.as_tensor(np.transpose(ref["A_out"], (2, 0, 1)), dtype=op.dtype, device=dev)
    S_c = torch.as_tensor(np.transpose(ref["S_out"], (2, 0, 1)), dtype=op.dtype, device=dev)
    # re-solve at the actual grid-center params (even grid -> center != exact base)
    cen = solve_batch(op, rho_grid[ci], mu_grid[cj], A_c, S_c, stepsize=args.stepsize,
                      tol=args.tol, maxiter=args.maxiter,
                      early_iter=args.early_iter, early_tol=args.early_tol)
    cm = cen["converged"]
    A_c, S_c = dedup_cap(cen["A"][cm], cen["S"][cm], kmax=args.kmax)
    store = {(ci, cj): (A_c, S_c)}
    np.savez(f"{outdir}/cell_{ci}_{cj}.npz",
             A=A_c.cpu().numpy().astype(np.float32), S=S_c.cpu().numpy().astype(np.float32),
             rho=rho_grid[ci], mu=mu_grid[cj])
    print(f"center ({ci},{cj}) seeded with {A_c.shape[0]} sols")

    t0 = time.time()
    for r in range(1, rings + 1):
        # cells at Chebyshev distance r
        cells = []
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                ii, jj = ci + di, cj + dj
                if 0 <= ii < nrho and 0 <= jj < nmu:
                    cells.append((ii, jj))
        if not cells:
            continue

        # build one big candidate batch for the whole ring
        cand_A2, cand_S2, rho2, mu2, cidx2, cellmap = [], [], [], [], [], []
        cnt = 0
        for (ii, jj) in cells:
            seeds_A, seeds_S = [], []
            for ei in (-1, 0, 1):
                for ej in (-1, 0, 1):
                    nb = (ii + ei, jj + ej)
                    if nb in store and max(abs(ii + ei - ci), abs(jj + ej - cj)) < r:
                        a, s = store[nb]
                        seeds_A.append(a); seeds_S.append(s)
            if not seeds_A:
                continue
            sA = torch.cat(seeds_A, 0); sS = torch.cat(seeds_S, 0)
            sA, sS = dedup_cap(sA, sS, kmax=args.seed_cap, triv=0.0)
            allA = [sA]; allS = [sS]
            for _ in range(args.nperturb):
                pA = poly_perturb(x, args.pscale, sA.shape[0], gen, dev, op.dtype)
                pS = poly_perturb(x, args.pscale, sA.shape[0], gen, dev, op.dtype)
                allA.append(sA + pA); allS.append(sS + pS)
            cA = torch.cat(allA, 0); cS = torch.cat(allS, 0)
            cand_A2.append(cA); cand_S2.append(cS)
            rho2.append(torch.full((cA.shape[0],), rho_grid[ii], dtype=op.dtype, device=dev))
            mu2.append(torch.full((cA.shape[0],), mu_grid[jj], dtype=op.dtype, device=dev))
            cidx2.append(torch.full((cA.shape[0],), cnt, dtype=torch.long, device=dev))
            cellmap.append((ii, jj)); cnt += 1
        if not cand_A2:
            print(f"ring {r}: no cells with inner neighbours")
            continue
        A0 = torch.cat(cand_A2, 0); S0 = torch.cat(cand_S2, 0)
        rho = torch.cat(rho2, 0); mu = torch.cat(mu2, 0); cidx = torch.cat(cidx2, 0)

        tr = time.time()
        # chunk the ring batch to bound GPU memory
        MB = args.max_batch
        convs, Afs, Sfs = [], [], []
        for s in range(0, A0.shape[0], MB):
            e = s + MB
            o = solve_batch(op, rho[s:e], mu[s:e], A0[s:e], S0[s:e],
                            stepsize=args.stepsize, tol=args.tol, maxiter=args.maxiter,
                            early_iter=args.early_iter, early_tol=args.early_tol)
            convs.append(o["converged"]); Afs.append(o["A"]); Sfs.append(o["S"])
        conv = torch.cat(convs); Af = torch.cat(Afs, 0); Sf = torch.cat(Sfs, 0)
        # scatter back per cell, dedup+cap, save
        nsol = 0
        for c, (ii, jj) in enumerate(cellmap):
            sel = (cidx == c) & conv
            if sel.sum() == 0:
                continue
            a, s = dedup_cap(Af[sel], Sf[sel], tol=args.dedup, kmax=args.kmax)
            if a.shape[0] == 0:
                continue
            store[(ii, jj)] = (a, s)
            np.savez(f"{outdir}/cell_{ii}_{jj}.npz",
                     A=a.cpu().numpy().astype(np.float32), S=s.cpu().numpy().astype(np.float32),
                     rho=rho_grid[ii], mu=mu_grid[jj])
            nsol += a.shape[0]
        print(f"ring {r}: {len(cellmap)} cells, {A0.shape[0]} candidates, "
              f"{int(conv.sum())} converged, {nsol} kept sols, "
              f"{time.time()-tr:.1f}s (total {time.time()-t0:.1f}s)")

    # manifest
    meta = dict(nrho=nrho, nmu=nmu, grid=max(nrho, nmu),
                rho_grid=rho_grid.tolist(), mu_grid=mu_grid.tolist(), n=op.n,
                DA=op.DA, DS=op.DS, center=[ci, cj], rings=rings,
                cells=[[int(i), int(j)] for (i, j) in store.keys()],
                ncells=len(store), nsols=int(sum(v[0].shape[0] for v in store.values())))
    json.dump(meta, open(f"{outdir}/manifest.json", "w"), indent=2)
    print(f"\nDONE: {meta['ncells']} cells, {meta['nsols']} solutions -> {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rho_min", type=float, default=0.030)
    p.add_argument("--rho_max", type=float, default=0.085)
    p.add_argument("--nrho", type=int, default=56)        # spacing 0.001
    p.add_argument("--mu_min", type=float, default=0.052)
    p.add_argument("--mu_max", type=float, default=0.075)
    p.add_argument("--nmu", type=int, default=24)         # spacing 0.001
    p.add_argument("--seed_rho", type=float, default=0.050)  # BFS start (in-lobe)
    p.add_argument("--seed_mu", type=float, default=0.065)
    p.add_argument("--kmax", type=int, default=15)
    p.add_argument("--seed_cap", type=int, default=24)
    p.add_argument("--nperturb", type=int, default=2)
    p.add_argument("--pscale", type=float, default=0.01)
    p.add_argument("--stepsize", type=float, default=0.1)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--maxiter", type=int, default=8000)
    p.add_argument("--early_iter", type=int, default=1500)
    p.add_argument("--early_tol", type=float, default=1e-2)
    p.add_argument("--dedup", type=float, default=1e-5)
    p.add_argument("--max_batch", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default=f"{HERE}/data_pilot")
    run(p.parse_args())
