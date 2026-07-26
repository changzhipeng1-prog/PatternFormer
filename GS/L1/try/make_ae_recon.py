"""GS L=1 autoencoder reconstruction figure (GPU). Shows input vs AE round-trip
for a few diverse GT solutions -> the frozen conv-AE is not the bottleneck (rel_l2 ~ 0.002).
Top row = input A, bottom row = reconstructed A; per-column rel_l2 annotated.
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CODE = os.path.normpath(os.path.join(HERE, "..", "code"))
sys.path.insert(0, CODE); sys.path.insert(0, PAPER)
from model.autoencoder2d import SolutionAutoencoder2D
from config import Config
from test_plot_style import apply_style, save_fig
import matplotlib.pyplot as plt
apply_style()

C = Config()
look = torch.load(C.data_lookup_path, weights_only=False)
ns = torch.load(C.norm_stats_path)
dev = "cuda"
ae = SolutionAutoencoder2D(latent_dim=C.latent_dim, in_ch=C.in_ch, img=C.img_size)
sd = torch.load(os.path.join(HERE, "..", "ckpt", "epoch_60", "autoencoder2d.pt"), map_location="cpu")
ae.load_state_dict(sd); ae = ae.to(dev).eval()

# pick 6 diverse solutions across params (spots -> stripes -> maze)
rng = np.random.default_rng(0)
picks = []
for ti in [786, 825, 576, 293, 347, 413]:
    S = look["solutions_by_p"][ti]
    picks.append(S[rng.integers(len(S))])
X = torch.stack(picks).to(dev).float()                 # (6,2,128,128)
with torch.no_grad():
    R = ae(ae(X, "encode"), "decode")
X, R = X.cpu().numpy(), R.cpu().numpy()
def rl2(a, b): return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9)

n = len(picks)
fig, ax = plt.subplots(2, n, figsize=(2.1 * n, 4.6))
vmax = float(np.abs(X[:, 0]).max())
for j in range(n):
    ax[0, j].imshow(X[j, 0], cmap="viridis", vmin=0, vmax=vmax)
    ax[1, j].imshow(R[j, 0], cmap="viridis", vmin=0, vmax=vmax)
    ax[1, j].set_title(f"rel-L2\n{rl2(R[j], X[j]):.1e}", fontsize=15)
    for r in range(2):
        ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
ax[0, 0].set_ylabel("input", fontweight="bold", fontsize=20)
ax[1, 0].set_ylabel("AE recon", fontweight="bold", fontsize=20)
fig.tight_layout()
save_fig(fig, f"{HERE}/fig_ae_recon")
print(f"mean rel_l2 = {np.mean([rl2(R[j],X[j]) for j in range(n)]):.2e}")
