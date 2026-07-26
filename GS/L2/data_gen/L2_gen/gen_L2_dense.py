"""L=2 dataset DENSIFICATION (step-2 of the L2 fix plan).

Interleaved 2x-finer grid: 67 F x 81 k (old 34x41 sits at even indices).
Only cells in the NEIGHBORHOOD of the known solvable band are attempted
(any old nonempty cell within 1 old-grid step), so compute goes to the
multi-solution region. Old grid points are ALSO re-run here with a NEW
seed bank (seed=7, nrand=300) -> extra solutions per existing param
(build merges by (rho,mu) with D4 dedup).

Same solver / physics / thresholds as gen_L2.py. Writes data_L2_dense/.
Shard:  python gen_L2_dense.py --jmod r,m   (F rows where idx%m==r)
"""
import argparse, os, sys
import numpy as np
import scipy.io as sio
import torch

HERE = os.path.dirname(os.path.abspath(__file__))           # paper/GS/L2/data_gen/L2_gen
FDM = os.path.join(HERE, "..")                              # paper/GS/L2/data_gen (op_n128.mat, gs_torch.py)
sys.path.insert(0, FDM); sys.path.insert(0, HERE)
from gs_torch import GSOperator, solve_batch
from gen_L2 import seed_bank, d4_distinct_idx


def band_mask(nF_d, nk_d):
    """dense cell (jf,ik) allowed if any old NONEMPTY cell within 1 old step."""
    import glob, re
    nonempty = set()
    for f in glob.glob(f"{HERE}/data_L2/cell_*.npz"):
        m = re.match(r".*cell_(\d+)_(\d+)\.npz", f)
        nonempty.add((int(m.group(1)), int(m.group(2))))
    ok = np.zeros((nF_d, nk_d), dtype=bool)
    for jf in range(nF_d):
        for ik in range(nk_d):
            jo, io = jf / 2.0, ik / 2.0
            for a in range(int(np.floor(jo)) - 1, int(np.ceil(jo)) + 2):
                for b in range(int(np.floor(io)) - 1, int(np.ceil(io)) + 2):
                    if (a, b) in nonempty:
                        ok[jf, ik] = True; break
                if ok[jf, ik]: break
    return ok


def run(a):
    dev = "cuda"; dt = torch.float64
    op = GSOperator(f"{FDM}/op_n128.mat", device=dev, dtype=dt)
    op.DA /= a.Lscale ** 2; op.DS /= a.Lscale ** 2
    x = op.Tx.new_tensor(sio.loadmat(f"{FDM}/op_n128.mat")["x"].ravel())
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    pool = seed_bank(op, x, gen, dev, dt, a.nrand)
    os.makedirs(a.outdir, exist_ok=True)
    Fs = np.round(np.linspace(0.038, 0.104, a.nF), 6)   # step 0.001, old at even idx
    ks = np.round(np.linspace(0.052, 0.072, a.nk), 6)   # step 0.00025, old at even idx
    np.save(f"{a.outdir}/_axes.npy", dict(Fs=Fs, ks=ks, Lscale=a.Lscale,
            DA=op.DA, DS=op.DS), allow_pickle=True)
    ok = band_mask(a.nF, a.nk)
    jmod_r, jmod_m = (int(v) for v in a.jmod.split(",")) if a.jmod else (0, 1)
    rows = [j for j in range(a.nF) if j % jmod_m == jmod_r and ok[j].any()]
    print(f"DENSE L={a.Lscale} pool {pool.shape[0]} seeds | grid {a.nF}x{a.nk} | "
          f"band cells {int(ok.sum())} | DA={op.DA:.3e} | rows {rows}", flush=True)
    for jf in rows:
        F = float(Fs[jf]); row = []
        for ik in range(a.nk):
            if not ok[jf, ik]:
                row.append(-1); continue
            k = float(ks[ik]); fn = f"{a.outdir}/cell_{jf}_{ik}.npz"
            if os.path.exists(fn) and not a.overwrite:
                row.append(int(np.load(fn)["A"].shape[0])); continue
            convs, Af, Sf = [], [], []
            for s in range(0, pool.shape[0], a.max_batch):
                o = solve_batch(op, F, k, pool[s:s + a.max_batch].clone(), tol=1e-9,
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
                # marker so resume skips known-empty cells (A shape 0)
                np.savez(fn, A=np.zeros((0, op.n, op.n), np.float32),
                         S=np.zeros((0, op.n, op.n), np.float32), rho=F, mu=k)
                row.append(0)
        print(f"F={F:.4f}: {[c for c in row if c >= 0]}  "
              f"(attempted {sum(c >= 0 for c in row)}, sols {sum(c for c in row if c > 0)})", flush=True)
    print("DENSE SHARD DONE", a.jmod, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nF", type=int, default=67)
    p.add_argument("--nk", type=int, default=81)
    p.add_argument("--nrand", type=int, default=300)
    p.add_argument("--triv", type=float, default=0.05)
    p.add_argument("--dedup", type=float, default=0.15)
    p.add_argument("--kmax", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=9000)
    p.add_argument("--early_iter", type=int, default=1500)
    p.add_argument("--early_tol", type=float, default=3e-2)
    p.add_argument("--max_batch", type=int, default=3000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--Lscale", type=float, default=2.0)
    p.add_argument("--jmod", type=str, default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--outdir", type=str, default=f"{HERE}/data_L2_dense")
    run(p.parse_args())
