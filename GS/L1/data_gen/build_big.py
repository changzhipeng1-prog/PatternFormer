"""Build the DENSIFIED dataset end-to-end, non-destructively.

Reads per-cell npz from BOTH data_merged/ (original 0.001-ish grid) and
data_dense/ (new offset-mu cells -> core lobe reaches 0.0005x0.0005), then runs
the full pipeline in one pass:
    package  -> group by (rho,mu)        (data_merged + data_dense)
    trivial  -> drop homogeneous (range < THR)
    D4       -> reduce each param's orbit + canonical pose + sort (reuse filter_d4)
    convert  -> lookup + param-level train/val/test split + norm_stats

Everything is written with a *_big suffix; NO existing file is overwritten.
Outputs:
    fdm/gs_dataset_big_d4.pt
    qwen/data/{gs_lookup_big.pt, train_p_idx_big.pt, val_p_idx_big.pt,
               test_p_idx_big.pt, norm_stats_big.pt, split_info_big.pt}
"""
import glob, os
import numpy as np
import torch

import filter_d4  # reuse reduce_param + d4 template (same dir)

import os; HERE = os.path.dirname(os.path.abspath(__file__))
QDATA = os.path.normpath(os.path.join(HERE, "..", "data"))
SRC_DIRS = [f"{HERE}/data_merged", f"{HERE}/data_dense"]
THR = 0.05                      # trivial cut (range of A), same as filter_trivial
DEV = "cuda"
SEED = 42
TRAIN, VAL = 0.70, 0.15


def load_cells():
    """All cells from both dirs -> (sol[N,2,128,128], rho[N], mu[N])."""
    A_list, S_list, rho_list, mu_list = [], [], [], []
    ncell = 0
    for D in SRC_DIRS:
        files = sorted(glob.glob(f"{D}/cell_*.npz"))
        ncell += len(files)
        for f in files:
            z = np.load(f)
            A, S = z["A"], z["S"]; rho = float(z["rho"]); mu = float(z["mu"])
            k = A.shape[0]
            A_list.append(A); S_list.append(S)
            rho_list.append(np.full(k, rho, np.float32))
            mu_list.append(np.full(k, mu, np.float32))
    A = torch.from_numpy(np.concatenate(A_list, 0))
    S = torch.from_numpy(np.concatenate(S_list, 0))
    rho = torch.from_numpy(np.concatenate(rho_list, 0))
    mu = torch.from_numpy(np.concatenate(mu_list, 0))
    print(f"loaded {ncell} cells from {len(SRC_DIRS)} dirs -> {A.shape[0]} raw sols")
    return torch.stack([A, S], 1), rho, mu


def main():
    sol, rho, mu = load_cells()

    # ---- trivial removal (range of A) ----
    A = sol[:, 0]
    rng = A.reshape(A.shape[0], -1).amax(1) - A.reshape(A.shape[0], -1).amin(1)
    keep = rng >= THR
    print(f"trivial: drop {int((~keep).sum())} homogeneous -> {int(keep.sum())} kept")
    sol, rho, mu = sol[keep], rho[keep], mu[keep]

    # ---- group by unique (rho,mu) ----
    key = torch.stack([rho, mu], 1).numpy()
    uniq, inv, cnt = np.unique(np.round(key, 6), axis=0, return_inverse=True, return_counts=True)
    pid = torch.from_numpy(inv.astype(np.int64))
    params = torch.from_numpy(uniq.astype(np.float32))
    P = params.shape[0]
    print(f"grouped into P={P} params (offset-mu merged by value); "
          f"raw per-param mean {cnt.mean():.1f}")

    # ---- D4 reduce per param (reuse filter_d4.reduce_param) ----
    xy = torch.linspace(0, 1, 128, device=DEV)
    W = (xy[None, :] - 0.5) + 0.6180339 * (xy[:, None] - 0.5)
    oA, oS, orho, omu, opid, nparams, ncounts = [], [], [], [], [], [], []
    before = after = 0; npid = 0
    for p in range(P):
        s = sol[pid == p].to(DEV)
        before += s.shape[0]
        red = filter_d4.reduce_param(s, W).cpu()
        after += red.shape[0]
        oA.append(red[:, 0]); oS.append(red[:, 1])
        r = float(params[p, 0]); m = float(params[p, 1])
        orho.append(torch.full((red.shape[0],), r)); omu.append(torch.full((red.shape[0],), m))
        opid.append(torch.full((red.shape[0],), npid, dtype=torch.long))
        nparams.append([r, m]); ncounts.append(red.shape[0]); npid += 1
    A = torch.cat(oA); S = torch.cat(oS)
    rho = torch.cat(orho); mu = torch.cat(omu); pidx = torch.cat(opid)
    params = torch.tensor(nparams, dtype=torch.float32)
    counts = torch.tensor(ncounts, dtype=torch.long)
    print(f"D4: {before} -> {after} sols ({100*(1-after/before):.1f}% removed); "
          f"per-param mean {counts.float().mean():.1f} max {counts.max().item()}")

    torch.save(dict(A=A, S=S, rho=rho, mu=mu, param_id=pidx, params=params, counts=counts,
                    info=dict(N=A.shape[0], P=P, n=128, domain="[0,1]^2", BC="Neumann",
                              note="densified (data_merged+data_dense); trivial-removed; "
                                   "D4-reduced+canonical+sorted")),
               f"{HERE}/gs_dataset_big_d4.pt")
    print(f"saved gs_dataset_big_d4.pt  N={A.shape[0]} P={P}")

    # ---- convert: lookup + split + norm_stats (mean/std + p_mean/p_std) ----
    sol = torch.stack([A, S], 1)
    solutions_by_p = [sol[pidx == p].contiguous() for p in range(P)]
    cc = torch.tensor([s.shape[0] for s in solutions_by_p])
    assert (cc > 0).all()

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(P, generator=g)
    n_tr = int(TRAIN * P); n_va = int(VAL * P)
    train_idx = perm[:n_tr].sort().values
    val_idx = perm[n_tr:n_tr + n_va].sort().values
    test_idx = perm[n_tr + n_va:].sort().values

    train_sols = torch.cat([solutions_by_p[i] for i in train_idx.tolist()], 0)
    mean = train_sols.mean(dim=(0, 2, 3)); std = train_sols.std(dim=(0, 2, 3))
    train_p = params[train_idx]
    p_mean = train_p.mean(0); p_std = train_p.std(0)

    torch.save({"p_values": params, "solutions_by_p": solutions_by_p}, f"{QDATA}/gs_lookup_big.pt")
    torch.save(train_idx, f"{QDATA}/train_p_idx_big.pt")
    torch.save(val_idx, f"{QDATA}/val_p_idx_big.pt")
    torch.save(test_idx, f"{QDATA}/test_p_idx_big.pt")
    torch.save({"mean": mean, "std": std, "p_mean": p_mean, "p_std": p_std},
               f"{QDATA}/norm_stats_big.pt")
    torch.save(dict(P=P, N=sol.shape[0], train=len(train_idx), val=len(val_idx),
                    test=len(test_idx), seed=SEED, channels=["A", "S"], shape=[2, 128, 128]),
               f"{QDATA}/split_info_big.pt")
    print(f"split: train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)} params")
    print(f"norm mean A,S={mean.tolist()} std={std.tolist()}")
    print(f"p_mean={p_mean.tolist()} p_std={p_std.tolist()}")
    print("DONE -> *_big files written (no existing file overwritten)")


if __name__ == "__main__":
    main()
