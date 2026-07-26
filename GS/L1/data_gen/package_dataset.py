"""Package the merged per-cell npz solutions into a single torch .pt for the
downstream encoder-decoder / Qwen pipeline, and emit a coverage scatter."""
import glob, json, os
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import os; HERE = os.path.dirname(os.path.abspath(__file__))
D = f"{HERE}/data_merged"

A_list, S_list, rho_list, mu_list, pid_list = [], [], [], [], []
params, counts = [], []
for pid, f in enumerate(sorted(glob.glob(f"{D}/cell_*.npz"))):
    z = np.load(f)
    A, S = z["A"], z["S"]; rho = float(z["rho"]); mu = float(z["mu"])
    k = A.shape[0]
    A_list.append(A); S_list.append(S)
    rho_list.append(np.full(k, rho, np.float32)); mu_list.append(np.full(k, mu, np.float32))
    pid_list.append(np.full(k, pid, np.int64))
    params.append([rho, mu]); counts.append(k)

A = torch.from_numpy(np.concatenate(A_list, 0))            # (N,128,128) f32
S = torch.from_numpy(np.concatenate(S_list, 0))
rho = torch.from_numpy(np.concatenate(rho_list, 0))
mu = torch.from_numpy(np.concatenate(mu_list, 0))
pid = torch.from_numpy(np.concatenate(pid_list, 0))
params = torch.tensor(params, dtype=torch.float32)         # (P,2)
counts = torch.tensor(counts, dtype=torch.long)            # (P,)
N, P = A.shape[0], params.shape[0]

meta = json.load(open(f"{D}/manifest.json"))
out = dict(
    A=A, S=S, rho=rho, mu=mu, param_id=pid, params=params, counts=counts,
    info=dict(N=N, P=P, n=128, DA=meta["DA"], DS=meta["DS"],
              domain="[0,1]^2", BC="Neumann", equation="steady Gray-Scott Eq.66",
              note="A=activator, S=substrate; FDM Np=1 quasi-Newton; spots-only"))
torch.save(out, f"{HERE}/gs_dataset.pt")
print(f"saved gs_dataset.pt: N={N} solutions over P={P} params, "
      f"A {tuple(A.shape)} {A.dtype}, file ~{(A.numel()+S.numel())*4/1e9:.2f} GB")
print(f"per-param sols: min {counts.min().item()} max {counts.max().item()} "
      f"mean {counts.float().mean():.1f}")

# coverage scatter (rho,mu) colored by sols/param
plt.figure(figsize=(7, 5))
sc = plt.scatter(params[:, 0], params[:, 1], c=counts, s=14, cmap="viridis")
plt.colorbar(sc, label="solutions / param")
plt.xlabel("ρ"); plt.ylabel("μ"); plt.title(f"merged dataset: {N} sols over {P} params")
plt.tight_layout(); plt.savefig(f"{HERE}/dataset_coverage.png", dpi=120)
print("wrote dataset_coverage.png")
