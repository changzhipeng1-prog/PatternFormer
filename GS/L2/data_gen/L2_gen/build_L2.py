"""Build the L=2 dataset end-to-end from L2_gen/data_L2 per-cell npz.

Mirrors fdm/build_big.py exactly (trivial-removal -> group by (rho,mu) ->
D4 reduce + canonical pose + sort -> lookup + 70/15/15 split + norm_stats),
but reads the L=2 cells and writes *_L2 files into qwen_L2/data/.

Outputs:
    fdm/L2_gen/gs_dataset_L2_d4.pt
    qwen_L2/data/{gs_lookup_L2.pt, train_p_idx_L2.pt, val_p_idx_L2.pt,
                  test_p_idx_L2.pt, norm_stats_L2.pt, split_info_L2.pt}
"""
import glob, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))           # paper/GS/L2/data_gen/L2_gen
FDM = os.path.join(HERE, "..")                              # paper/GS/L2/data_gen (filter_d4.py)
sys.path.insert(0, FDM)
import filter_d4  # reuse reduce_param + D4 template

QDATA = os.path.join(HERE, "qwen_L2_data")                  # local output
SRC_DIRS = [f"{HERE}/data_L2"]
THR = 0.05
DEV = "cuda"
SEED = 42
TRAIN, VAL = 0.70, 0.15


def load_cells():
    A_list, S_list, rho_list, mu_list = [], [], [], []
    ncell = 0
    for D in SRC_DIRS:
        files = sorted(glob.glob(f"{D}/cell_*.npz"))
        for f in files:
            z = np.load(f)
            A, S = z["A"], z["S"]
            if A.shape[0] == 0:
                continue
            ncell += 1
            rho = float(z["rho"]); mu = float(z["mu"])
            A_list.append(A); S_list.append(S)
            rho_list.append(np.full(A.shape[0], rho, np.float32))
            mu_list.append(np.full(A.shape[0], mu, np.float32))
    A = torch.from_numpy(np.concatenate(A_list, 0))
    S = torch.from_numpy(np.concatenate(S_list, 0))
    rho = torch.from_numpy(np.concatenate(rho_list, 0))
    mu = torch.from_numpy(np.concatenate(mu_list, 0))
    print(f"loaded {ncell} nonempty cells -> {A.shape[0]} raw sols")
    return torch.stack([A, S], 1), rho, mu


def main():
    os.makedirs(QDATA, exist_ok=True)
    sol, rho, mu = load_cells()

    A = sol[:, 0]
    rng = A.reshape(A.shape[0], -1).amax(1) - A.reshape(A.shape[0], -1).amin(1)
    keep = rng >= THR
    print(f"trivial: drop {int((~keep).sum())} -> {int(keep.sum())} kept")
    sol, rho, mu = sol[keep], rho[keep], mu[keep]

    key = torch.stack([rho, mu], 1).numpy()
    uniq, inv, cnt = np.unique(np.round(key, 6), axis=0, return_inverse=True, return_counts=True)
    pid = torch.from_numpy(inv.astype(np.int64))
    params = torch.from_numpy(uniq.astype(np.float32))
    P = params.shape[0]
    print(f"grouped into P={P} params; raw per-param mean {cnt.mean():.1f}")

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
                    info=dict(N=A.shape[0], P=P, n=128, domain="[0,2]^2 (L=2)", BC="Neumann",
                              note="L=2 (DA,DS /=4); trivial-removed; D4-reduced+canonical+sorted")),
               f"{HERE}/gs_dataset_L2_d4.pt")
    print(f"saved gs_dataset_L2_d4.pt  N={A.shape[0]} P={P}")

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

    torch.save({"p_values": params, "solutions_by_p": solutions_by_p}, f"{QDATA}/gs_lookup_L2.pt")
    torch.save(train_idx, f"{QDATA}/train_p_idx_L2.pt")
    torch.save(val_idx, f"{QDATA}/val_p_idx_L2.pt")
    torch.save(test_idx, f"{QDATA}/test_p_idx_L2.pt")
    torch.save({"mean": mean, "std": std, "p_mean": p_mean, "p_std": p_std},
               f"{QDATA}/norm_stats_L2.pt")
    torch.save(dict(P=P, N=sol.shape[0], train=len(train_idx), val=len(val_idx),
                    test=len(test_idx), seed=SEED, channels=["A", "S"], shape=[2, 128, 128]),
               f"{QDATA}/split_info_L2.pt")
    print(f"split: train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)} params")
    print(f"norm mean A,S={mean.tolist()} std={std.tolist()}")
    print(f"p_mean={p_mean.tolist()} p_std={p_std.tolist()}")
    print("DONE -> *_L2 files written")


if __name__ == "__main__":
    main()
