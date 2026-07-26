"""Build the EXPANDED L=2 dataset (data_L2 + data_L2_dense merged).

Mirrors build_L2.py (trivial-removal -> group by (rho,mu) -> D4 reduce ->
lookup + split + norm_stats) with two changes:
  1. SRC_DIRS includes the densified cells; same (rho,mu) groups merge and
     D4-dedup across dirs (old cells get topped-up by the new seed bank).
  2. Split PRESERVES the old qwen_L2 split: params matching old train/val/test
     keep their assignment (so the old 72-param test set stays test -> the new
     model remains comparable & uncontaminated). Only NEW params are randomly
     split 70/15/15.

Writes to qwen_ord_L2/data_expanded/ (NEVER touches frozen qwen_L2/data).
"""
import glob, os, sys
import numpy as np
import torch

import os; FDM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FDM)
import filter_d4

HERE = f"{FDM}/L2_gen"
OLDQ = os.path.normpath(os.path.join(FDM, "..", "data"))  # NOTE original 471-param L2 lookup (data-expansion history step)
QDATA = os.path.normpath(os.path.join(FDM, "..", "data"))
SRC_DIRS = [f"{HERE}/data_L2", f"{HERE}/data_L2_dense"]
THR = 0.05
DEV = "cuda"
SEED = 42
TRAIN, VAL = 0.70, 0.15


def load_cells():
    A_list, S_list, rho_list, mu_list = [], [], [], []
    ncell = 0
    for D in SRC_DIRS:
        for f in sorted(glob.glob(f"{D}/cell_*.npz")):
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
                              note="EXPANDED dense L=2; trivial-removed; D4-reduced")),
               f"{HERE}/gs_dataset_L2dense_d4.pt")
    print(f"saved gs_dataset_L2dense_d4.pt  N={A.shape[0]} P={P}")

    sol = torch.stack([A, S], 1)
    solutions_by_p = [sol[pidx == p].contiguous() for p in range(P)]
    cc = torch.tensor([s.shape[0] for s in solutions_by_p])
    assert (cc > 0).all()

    # ---- split: preserve old assignments, randomly split only NEW params ----
    old_lk = torch.load(f"{OLDQ}/gs_lookup_L2.pt", weights_only=False)
    old_params = np.round(old_lk["p_values"].numpy(), 6)
    old_assign = {}
    for name in ["train", "val", "test"]:
        idx = torch.load(f"{OLDQ}/{name}_p_idx_L2.pt").tolist()
        for i in idx:
            old_assign[tuple(old_params[i])] = name
    new_round = np.round(params.numpy(), 6)
    tr, va, te, fresh = [], [], [], []
    for p in range(P):
        a = old_assign.get(tuple(new_round[p]))
        if a == "train": tr.append(p)
        elif a == "val": va.append(p)
        elif a == "test": te.append(p)
        else: fresh.append(p)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.tensor(fresh)[torch.randperm(len(fresh), generator=g)]
    n_tr = int(TRAIN * len(fresh)); n_va = int(VAL * len(fresh))
    tr += perm[:n_tr].tolist(); va += perm[n_tr:n_tr + n_va].tolist(); te += perm[n_tr + n_va:].tolist()
    train_idx = torch.tensor(sorted(tr)); val_idx = torch.tensor(sorted(va)); test_idx = torch.tensor(sorted(te))
    print(f"split: train {len(train_idx)} val {len(val_idx)} test {len(test_idx)} "
          f"(old preserved: {len([1 for v in old_assign.values()])} matched, fresh {len(fresh)})")

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
                    test=len(test_idx), seed=SEED, channels=["A", "S"], shape=[2, 128, 128],
                    note="expanded dense; old split preserved"),
               f"{QDATA}/split_info_L2.pt")
    print(f"wrote {QDATA}/: lookup P={P} N={sol.shape[0]}")


if __name__ == "__main__":
    main()
