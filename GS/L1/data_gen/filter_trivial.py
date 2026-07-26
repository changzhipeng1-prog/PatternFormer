"""Remove spatially-homogeneous (trivial) solutions: spatial range max-min
below threshold. Distribution is cleanly bimodal (trivial <0.01, patterns >0.35),
so the cut is unambiguous. Rebuilds param grouping and re-saves."""
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import os; HERE = os.path.dirname(os.path.abspath(__file__))
THR = 0.05

d = torch.load(f"{HERE}/gs_dataset.pt")
A, S = d["A"], d["S"]; rho, mu = d["rho"], d["mu"]
rng = (A.reshape(A.shape[0], -1).amax(1) - A.reshape(A.shape[0], -1).amin(1))
keep = rng >= THR
nrm = int((~keep).sum())
print(f"removing {nrm} trivial (homogeneous) sols of {A.shape[0]}  -> {int(keep.sum())} kept")

A, S, rho, mu = A[keep], S[keep], rho[keep], mu[keep]

# rebuild param grouping from surviving (rho,mu)
key = torch.stack([rho, mu], 1).numpy()
uniq, inv, cnt = np.unique(np.round(key, 6), axis=0, return_inverse=True, return_counts=True)
param_id = torch.from_numpy(inv.astype(np.int64))
params = torch.from_numpy(uniq.astype(np.float32))
counts = torch.from_numpy(cnt.astype(np.int64))
P = params.shape[0]

info = dict(d["info"]); info["N"] = A.shape[0]; info["P"] = P
info["note"] = info.get("note", "") + "; trivial(homogeneous) removed range>=%.2f" % THR
out = dict(A=A, S=S, rho=rho, mu=mu, param_id=param_id, params=params, counts=counts, info=info)
torch.save(out, f"{HERE}/gs_dataset.pt")
print(f"saved gs_dataset.pt: N={A.shape[0]} over P={P} params; "
      f"per-param min {counts.min().item()} max {counts.max().item()} mean {counts.float().mean():.1f}")

plt.figure(figsize=(7, 5))
sc = plt.scatter(params[:, 0], params[:, 1], c=counts, s=14, cmap="viridis")
plt.colorbar(sc, label="solutions / param")
plt.xlabel("ρ"); plt.ylabel("μ"); plt.title(f"after trivial removal: {A.shape[0]} sols over {P} params")
plt.tight_layout(); plt.savefig(f"{HERE}/dataset_coverage.png", dpi=120)
print("updated dataset_coverage.png")
