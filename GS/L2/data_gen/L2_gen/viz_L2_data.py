"""Visualize the L=2 GT dataset for the slides (Step 1: traditional generation).

  figs/L2_coverage.png     -- (k,F) scatter coloured by #solutions/param (the band)
  figs/L2_samples.png      -- representative patterns sampled across the band
  figs/L2_vs_L1.png        -- same (F,k) cells: L=1 (coarse) vs L=2 (fine) morphology
  figs/L2_multiplicity.png -- a few params, ALL their distinct GT solutions
"""
import os, glob, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; matplotlib.rcParams.update({"font.size":15,"axes.titlesize":16,"axes.labelsize":15,"figure.titlesize":18})
HERE = os.path.dirname(os.path.abspath(__file__))           # paper/GS/L2/data_gen/L2_gen
FDM = os.path.join(HERE, "..")                              # paper/GS/L2/data_gen (op_n128.mat, etc.)
OUT = os.path.join(HERE, "figs")                            # local output
os.makedirs(OUT, exist_ok=True)

d = torch.load(f"{HERE}/gs_dataset_L2_d4.pt", weights_only=False)
params = d["params"].numpy(); counts = d["counts"].numpy()
A = d["A"].numpy(); pid = d["param_id"].numpy()
rho = params[:, 0]; mu = params[:, 1]   # rho=F, mu=k
P = params.shape[0]; N = A.shape[0]
print(f"L2 dataset: P={P} params, N={N} sols, sols/param mean {counts.mean():.1f} max {counts.max()}")

# ---- coverage ----
plt.figure(figsize=(7.5, 5.5))
o = np.argsort(counts)
sc = plt.scatter(mu[o], rho[o], c=counts[o], s=24 + 7*counts[o], cmap="rainbow",
                 edgecolors="black", linewidths=0.3)
plt.colorbar(sc, label="# distinct GT solutions / param")
plt.xlabel(r"$\mu=k$ (kill)"); plt.ylabel(r"$\rho=F$ (feed)")
plt.title(f"L=2 traditional-solver GT coverage\n{N} solutions over {P} params (mean {counts.mean():.1f}/param)")
plt.tight_layout(); plt.savefig(f"{OUT}/L2_coverage.png", dpi=140); plt.close()
print("wrote L2_coverage.png")

# ---- representative samples across the band ----
order = np.argsort(rho + 0.0)  # by F
pick = order[np.linspace(0, P-1, 24).astype(int)]
fig, axs = plt.subplots(3, 8, figsize=(16, 6.4))
for ax, p in zip(axs.ravel(), pick):
    sel = np.where(pid == p)[0]
    a = A[sel[len(sel)//2]]   # a middle solution
    ax.imshow(a, cmap="viridis"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"F={rho[p]:.3f} k={mu[p]:.3f}\n({counts[p]} sols)", fontsize=12)
fig.suptitle("L=2 GT: representative patterns across the band (fine spots / stripes / mazes)", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/L2_samples.png", dpi=130); plt.close()
print("wrote L2_samples.png")

# ---- multiplicity: a few params, all their distinct solutions ----
rich = np.argsort(-counts)[:6]
maxk = int(min(10, counts[rich].max()))
fig, axs = plt.subplots(len(rich), maxk, figsize=(1.4*maxk, 1.4*len(rich)))
for r, p in enumerate(rich):
    sel = np.where(pid == p)[0]
    for c in range(maxk):
        ax = axs[r][c]; ax.set_xticks([]); ax.set_yticks([])
        if c < len(sel): ax.imshow(A[sel[c]], cmap="viridis")
        else: ax.axis("off")
        if c == 0: ax.set_ylabel(f"F={rho[p]:.3f}\nk={mu[p]:.3f}", fontsize=13)
fig.suptitle("L=2 GT multiplicity: distinct steady-state solutions at one parameter", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/L2_multiplicity.png", dpi=130); plt.close()
print("wrote L2_multiplicity.png")
