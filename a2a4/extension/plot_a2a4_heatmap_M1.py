"""a2a4 extrapolation heatmap, METHOD 1 (direct generation).

pcolor over the (a2,a4) plane coloured by COVERAGE FRACTION (# true branches the model
recovers / # that exist).  Training rectangle drawn as a dashed box.  Each cell is
annotated 'covered/exist'.  Cells where no solution exists are left blank.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from model.newton_fdm import newton_refine, residual_l2
from test_plot_style import apply_style, save_fig
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

apply_style()
import matplotlib as _mpl
for _k in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
           "ytick.labelsize", "legend.fontsize"):
    _mpl.rcParams[_k] = _mpl.rcParams[_k] + 2

N = 1024
H = 1.0 / N
COVER, DEDUP, REFINE_OK = 1e-2, 1e-3, 1e-6
# training rectangle
A2_LO, A2_HI = -14.934, -2.0
A4_LO, A4_HI = 0.301, 0.798


def rl2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def dedup(sols):
    out = []
    for u in sols:
        if not any(rl2(u, v) < DEDUP for v in out):
            out.append(u)
    return out


def refine_one(u, a4, a2):
    ru, ok, _ = newton_refine(u, a4, a2, h=H, max_iter=50, abs_tol=1e-9)
    ru = np.asarray(ru, np.float64).reshape(-1)
    return ru if (ok and residual_l2(ru, a4, a2, H) < REFINE_OK and np.linalg.norm(ru) > 1e-6) else None


def main():
    G = torch.load(os.path.join(HERE, "grid_direct.pt"), map_location="cpu", weights_only=False)
    GT = torch.load(os.path.join(HERE, "grid_gt_cont.pt"), map_location="cpu", weights_only=False)
    A2 = np.array(G["a2"]); A4 = np.array(G["a4"])
    cov = np.full((len(A4), len(A2)), np.nan)
    ann = {}
    for ia, a2 in enumerate(A2):
        for jb, a4 in enumerate(A4):
            gt = GT["gt"][(float(a2), float(a4))]
            gt = np.asarray(gt, np.float64)
            k = gt.shape[0]
            if k == 0:
                continue
            gen = G["gen"][(float(a2), float(a4))].numpy().astype(np.float64)
            distinct = dedup([r for r in (refine_one(u, float(a4), float(a2)) for u in gen) if r is not None])
            c = sum(1 for g in gt if any(rl2(d, g) < COVER for d in distinct))
            cov[jb, ia] = c / k
            ann[(ia, jb)] = f"{c}/{k}"

    torch.save({"a2": A2.tolist(), "a4": A4.tolist(), "cov": cov,
                "box": [A2_LO, A2_HI, A4_LO, A4_HI]},
               os.path.join(HERE, "a2a4_cov_M1.pt"))

    fig, ax = plt.subplots(figsize=(13, 8))
    pcm = ax.pcolormesh(A2, A4, cov, shading="nearest", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    cb = fig.colorbar(pcm, ax=ax, pad=0.015)
    cb.set_label("coverage  (recovered / existing branches)", fontsize=20, fontweight="bold")
    for (ia, jb), s in ann.items():
        ax.text(A2[ia], A4[jb], s, ha="center", va="center", fontsize=11,
                color="black", fontweight="bold")
    # training rectangle (dashed)
    ax.add_patch(Rectangle((A2_LO, A4_LO), A2_HI - A2_LO, A4_HI - A4_LO,
                           fill=False, ec="black", lw=3, ls="--", zorder=5))
    ax.text(0.5 * (A2_LO + A2_HI), A4_LO - 0.10, "training region", ha="center", va="top",
            fontsize=16, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1.5))
    ax.set_xlabel(r"$a_2$", fontsize=26, fontweight="bold")
    ax.set_ylabel(r"$a_4$", fontsize=26, fontweight="bold")
    ax.set_title("a2a4 extrapolation — M1 direct generation\n"
                 "(GT: inside box = dataset; outside = continuation)",
                 fontsize=18, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, os.path.join(HERE, "fig_a2a4_heatmap_M1"))
    print("saved fig_a2a4_heatmap_M1.png/.pdf")


if __name__ == "__main__":
    main()
