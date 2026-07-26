"""L=2 Gray-Scott multi-solution GT generator (domain [0,2]^2 <=> DA,DS /= 4).

Same physics/solver as L=1, but on a 2x-larger domain so MANY more wavelengths
fit the box -> dense spot lattices / stripes / mazes (the xmorphia regime),
instead of the coarse few-blob L=1 steady states.

Why the atlas seed-bank (not gs_continuation): at L=2 the patterns live in
maze/lattice basins that the L=1 spot seeds (matlab_ref) do NOT reach. The rich
seed bank (synthetic stripes + checker/hole lattices + labyrinths + a large batch
of band-limited random fields) DOES land in those basins -- it already found 509
distinct L=2 states in the atlas probe. Here we keep the FULL D4-distinct (A,S)
set per cell (not just a count + one rep), so the output is a training dataset.

Per (F=rho, k=mu) cell: batch-solve all seeds (fp64 quasi-Newton), drop trivial
(homogeneous), greedy-D4 dedup, cap K, save A,S,rho,mu. Empty cells (no non-trivial
steady state) are skipped. Resumable; shard across GPUs with --j0/--j1 over F.
"""
import argparse, os, numpy as np, scipy.io as sio, torch
import sys
HERE = os.path.dirname(os.path.abspath(__file__))           # paper/GS/L2/data_gen/L2_gen
FDM = os.path.join(HERE, "..")                              # paper/GS/L2/data_gen (op_n128.mat, matlab_ref.mat, gs_torch.py)
sys.path.insert(0, FDM)
from gs_torch import GSOperator, solve_batch


def seed_bank(op, x, gen, dev, dt, nrand):
    n = op.n; X, Y = torch.meshgrid(x, x, indexing="ij")
    def nrm(A, amp=0.4):
        A = A - A.amin(); return amp * A / (A.amax() + 1e-12)
    seeds = []
    ref = sio.loadmat(f"{FDM}/matlab_ref.mat")                # 8 base spot seeds
    A_seed = torch.as_tensor(np.transpose(ref["A_out"], (2, 0, 1)), dtype=dt, device=dev)
    for kk in range(A_seed.shape[0]): seeds.append(A_seed[kk])
    # synthetic stripes (higher freqs -> finer L=2 patterns)
    for f in range(2, 14):
        for ph in (0.0, 0.5):
            seeds += [nrm(torch.sin(f*np.pi*X+ph)**2), nrm(torch.sin(f*np.pi*Y+ph)**2),
                      nrm(torch.sin(f*np.pi*(X+Y)+ph)**2), nrm(torch.sin(f*np.pi*(X-Y)+ph)**2)]
    # checker / hole lattices
    for d in (3, 4, 5, 6, 7, 8, 9, 10, 12):
        sp = (torch.cos(d*np.pi*X)*torch.cos(d*np.pi*Y)).clamp_min(0)
        seeds += [nrm(sp), nrm(1-sp)]
    # labyrinth-ish products
    for (a, b) in ((3, 4), (4, 5), (5, 6), (4, 7), (6, 7), (7, 9), (8, 10)):
        seeds += [nrm(torch.sin(a*np.pi*X)*torch.cos(b*np.pi*Y)),
                  nrm(torch.sin(a*np.pi*X)+torch.cos(b*np.pi*Y))]
    synth = torch.stack(seeds, 0)
    # band-limited random fields (the multiplicity workhorse)
    kx = torch.fft.fftfreq(n, device=dev)[:, None]; ky = torch.fft.rfftfreq(n, device=dev)[None, :]
    rad = (kx**2 + ky**2).sqrt(); rands = []
    for _ in range(nrand):
        sc = 4 + 16 * torch.rand(1, generator=gen, device=dev, dtype=dt).item()  # finer than L=1
        r = torch.randn(1, n, n, generator=gen, device=dev, dtype=dt)
        r = torch.fft.irfft2(torch.fft.rfft2(r) * (rad < sc/n).to(dt), s=(n, n))[0]
        r = r - r.amin(); rands.append(0.4 * r / (r.amax()+1e-12))
    return torch.cat([synth, torch.stack(rands, 0)], 0)


def d4_distinct_idx(A, thr=0.15, kmax=20):
    """greedy D4 dedup on A-channel; returns kept indices into A (highest-amp first)."""
    if A.shape[0] == 0: return []
    order = torch.argsort(A.abs().amax((1, 2)), descending=True)
    def poses(g):
        ps = []
        for r in range(4):
            b = torch.rot90(g, r, (0, 1)); ps += [b, torch.flip(b, (1,))]
        return torch.stack(ps, 0)
    kept, kp = [], []
    for i in order.tolist():
        na = A[i].norm() + 1e-9; dup = False
        for P in kp:
            if (P - A[i][None]).flatten(1).norm(dim=1).min()/na <= thr: dup = True; break
        if not dup: kept.append(i); kp.append(poses(A[i]))
        if len(kept) >= kmax: break
    return kept


def run(a):
    dev = "cuda"; dt = torch.float64
    op = GSOperator(f"{FDM}/op_n128.mat", device=dev, dtype=dt)
    op.DA /= a.Lscale**2; op.DS /= a.Lscale**2                 # enlarge domain L x
    x = op.Tx.new_tensor(sio.loadmat(f"{FDM}/op_n128.mat")["x"].ravel())
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    pool = seed_bank(op, x, gen, dev, dt, a.nrand)
    os.makedirs(a.outdir, exist_ok=True)
    Fs = np.round(np.linspace(a.F_min, a.F_max, a.nF), 5)
    ks = np.round(np.linspace(a.k_min, a.k_max, a.nk), 5)
    np.save(f"{a.outdir}/_axes.npy", dict(Fs=Fs, ks=ks, Lscale=a.Lscale,
            DA=op.DA, DS=op.DS), allow_pickle=True)
    # F-shard: contiguous [j0,j1) intersected with round-robin idx%jmod_m==jmod_r
    jmod_r, jmod_m = (int(v) for v in a.jmod.split(",")) if a.jmod else (0, 1)
    rows = [j for j in range(a.j0, min(a.j1, a.nF)) if j % jmod_m == jmod_r]
    print(f"L={a.Lscale} pool {pool.shape[0]} seeds | grid {a.nF}F x {a.nk}k | "
          f"DA={op.DA:.3e} DS={op.DS:.3e} | rows {rows}", flush=True)
    for jf in rows:
        F = float(Fs[jf]); row = []
        for ik in range(a.nk):
            k = float(ks[ik]); fn = f"{a.outdir}/cell_{jf}_{ik}.npz"
            if os.path.exists(fn) and not a.overwrite:
                row.append(int(np.load(fn)["A"].shape[0])); continue
            convs, Af, Sf = [], [], []
            for s in range(0, pool.shape[0], a.max_batch):
                o = solve_batch(op, F, k, pool[s:s+a.max_batch].clone(), tol=1e-9,
                                maxiter=a.maxiter, early_iter=a.early_iter, early_tol=a.early_tol)
                convs.append(o["converged"]); Af.append(o["A"]); Sf.append(o["S"])
            conv = torch.cat(convs); A = torch.cat(Af, 0)[conv]; S = torch.cat(Sf, 0)[conv]
            rng = A.amax((1, 2)) - A.amin((1, 2)); keep = rng > a.triv
            A, S = A[keep], S[keep]
            idx = d4_distinct_idx(A, thr=a.dedup, kmax=a.kmax)
            if idx:
                ii = torch.tensor(idx, device=A.device)
                np.savez(fn, A=A[ii].cpu().numpy().astype(np.float32),
                         S=S[ii].cpu().numpy().astype(np.float32), rho=F, mu=k)
                row.append(len(idx))
            else:
                row.append(0)
        print(f"F={F:.4f}: {row}  (nonzero {sum(c>0 for c in row)}, sols {sum(row)})", flush=True)
    print("SHARD DONE", a.j0, a.j1, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--F_min", type=float, default=0.038)
    p.add_argument("--F_max", type=float, default=0.104)
    p.add_argument("--nF", type=int, default=34)              # step ~0.002
    p.add_argument("--k_min", type=float, default=0.052)
    p.add_argument("--k_max", type=float, default=0.072)
    p.add_argument("--nk", type=int, default=41)              # step ~0.0005
    p.add_argument("--nrand", type=int, default=220)
    p.add_argument("--triv", type=float, default=0.05)
    p.add_argument("--dedup", type=float, default=0.15)
    p.add_argument("--kmax", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=9000)
    p.add_argument("--early_iter", type=int, default=1500)
    p.add_argument("--early_tol", type=float, default=3e-2)
    p.add_argument("--max_batch", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--Lscale", type=float, default=2.0)
    p.add_argument("--j0", type=int, default=0)
    p.add_argument("--j1", type=int, default=999)
    p.add_argument("--jmod", type=str, default="", help="r,m -> process F rows where idx%%m==r")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--outdir", type=str, default=f"{HERE}/data_L2")
    run(p.parse_args())
