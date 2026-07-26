"""Convert fdm/gs_dataset.pt into the lookup format the Qwen pipeline expects.

Reference lookup (2D_multisolution) is:
    {"p_values": [P] scalar, "solutions_by_p": list of [K_i, dim]}
Here the parameter is 2D (rho,mu) and each solution is a 2-channel 128x128 image
(channel 0 = A activator, channel 1 = S substrate).

Outputs (in qwen/data/):
    gs_lookup.pt    {"p_values": [P,2], "solutions_by_p": list[[K_i,2,128,128]]}
    train_p_idx.pt / val_p_idx.pt / test_p_idx.pt   param-level split (LongTensor)
    norm_stats.pt   per-channel mean/std over train solutions
    split_info.pt   metadata
"""
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "data_gen", "gs_dataset_d4.pt"))  # D4-reduced, from data_gen pipeline
SEED = 42
TRAIN, VAL = 0.70, 0.15   # test = remainder

d = torch.load(SRC)
A, S = d["A"], d["S"]                       # (N,128,128) f32 each
pid = d["param_id"]; params = d["params"]   # (N,), (P,2)
P = params.shape[0]
sol = torch.stack([A, S], dim=1)            # (N,2,128,128)

# group solutions by param
solutions_by_p = []
for p in range(P):
    solutions_by_p.append(sol[pid == p].contiguous())
counts = torch.tensor([s.shape[0] for s in solutions_by_p])
assert (counts > 0).all(), "every param must have >=1 solution"

# param-level split
g = torch.Generator().manual_seed(SEED)
perm = torch.randperm(P, generator=g)
n_tr = int(TRAIN * P); n_va = int(VAL * P)
train_idx = perm[:n_tr].sort().values
val_idx   = perm[n_tr:n_tr + n_va].sort().values
test_idx  = perm[n_tr + n_va:].sort().values

# per-channel normalization stats over TRAIN solutions only
train_sols = torch.cat([solutions_by_p[i] for i in train_idx.tolist()], 0)  # (Ntr,2,128,128)
mean = train_sols.mean(dim=(0, 2, 3))          # (2,)
std = train_sols.std(dim=(0, 2, 3))            # (2,)

torch.save({"p_values": params, "solutions_by_p": solutions_by_p}, f"{HERE}/gs_lookup.pt")
torch.save(train_idx, f"{HERE}/train_p_idx.pt")
torch.save(val_idx, f"{HERE}/val_p_idx.pt")
torch.save(test_idx, f"{HERE}/test_p_idx.pt")
torch.save({"mean": mean, "std": std}, f"{HERE}/norm_stats.pt")
torch.save(dict(P=P, N=sol.shape[0], train=len(train_idx), val=len(val_idx), test=len(test_idx),
                seed=SEED, channels=["A", "S"], shape=[2, 128, 128]), f"{HERE}/split_info.pt")

print(f"P={P} params, N={sol.shape[0]} sols")
print(f"split: train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)} params")
print(f"sols per split: train {sum(counts[train_idx]).item()} "
      f"val {sum(counts[val_idx]).item()} test {sum(counts[test_idx]).item()}")
print(f"norm mean A,S = {mean.tolist()}  std A,S = {std.tolist()}")
print(f"saved gs_lookup.pt ({sum(s.numel() for s in solutions_by_p)*4/1e9:.2f} GB) + splits")
