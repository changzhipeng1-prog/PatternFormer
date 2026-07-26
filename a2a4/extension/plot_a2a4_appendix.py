"""Appendix figure for Example 2 (a2,a4) extrapolation: two coverage maps,
   LEFT  = direct generation
   RIGHT = seed + single jump (Newton at the target from the nearest in-training output)
The main-text Fig.4C shows the further 'seed + stepped march' result.
Per-cell annotation = covered/existing branches.  Output -> fig_a2a4_appendix.{pdf,png}
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
N = 1024; H = 1.0 / N
COVER, DEDUP, REFINE_OK = 1e-2, 1e-3, 1e-6
A2_LO, A2_HI, A4_LO, A4_HI = -14.934, -2.0, 0.301, 0.798


def rl2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def dedup(sols):
    out = []
    for u in sols:
        if u is not None and not any(rl2(u, v) < DEDUP for v in out):
            out.append(u)
    return out


def refine_one(u, a4, a2):
    ru, ok, _ = newton_refine(u, a4, a2, h=H, max_iter=80, abs_tol=1e-9)
    ru = np.asarray(ru, np.float64).reshape(-1)
    return ru if (ok and residual_l2(ru, a4, a2, H) < REFINE_OK and np.linalg.norm(ru) > 1e-6) else None


def cov_counts(seeds, a2, a4, gt):
    distinct = dedup([refine_one(u, float(a4), float(a2)) for u in seeds])
    c = sum(1 for g in gt if any(rl2(d, g) < COVER for d in distinct))
    return c, len(gt)


def main():
    G = torch.load(os.path.join(HERE, "grid_direct.pt"), map_location="cpu", weights_only=False)
    GT = torch.load(os.path.join(HERE, "grid_gt_cont.pt"), map_location="cpu", weights_only=False)
    A2 = np.array(G["a2"]); A4 = np.array(G["a4"])
    a2r = A2.max() - A2.min(); a4r = A4.max() - A4.min()
    inbox = [(float(a2), float(a4)) for a2 in A2 for a4 in A4
             if A2_LO <= a2 <= A2_HI and A4_LO <= a4 <= A4_HI]
    nearest = lambda a2, a4: min(inbox, key=lambda b: ((a2 - b[0]) / a2r) ** 2 + ((a4 - b[1]) / a4r) ** 2)

    covD = np.full((len(A4), len(A2)), np.nan); annD = {}
    covS = np.full((len(A4), len(A2)), np.nan); annS = {}
    for ia, a2 in enumerate(A2):
        for jb, a4 in enumerate(A4):
            gt = np.asarray(GT["gt"][(float(a2), float(a4))], np.float64)
            k = gt.shape[0]
            if k == 0:
                continue
            cd, _ = cov_counts(G["gen"][(float(a2), float(a4))].numpy().astype(np.float64), a2, a4, gt)
            n2, n4 = nearest(float(a2), float(a4))
            cs, _ = cov_counts(G["gen"][(n2, n4)].numpy().astype(np.float64), a2, a4, gt)
            covD[jb, ia] = cd / k; annD[(ia, jb)] = f"{cd}/{k}"
            covS[jb, ia] = cs / k; annS[(ia, jb)] = f"{cs}/{k}"
        print(f"  a2={a2:+.2f} ({ia+1}/{len(A2)})", flush=True)

    fig, axs = plt.subplots(1, 2, figsize=(22, 8))
    for ax, cov, ann, title in [(axs[0], covD, annD, "direct generation"),
                                (axs[1], covS, annS, "seed + single jump")]:
        pcm = ax.pcolormesh(A2, A4, cov, shading="nearest", cmap="RdYlGn", vmin=0, vmax=1)
        for (ia, jb), s in ann.items():
            ax.text(A2[ia], A4[jb], s, ha="center", va="center", fontsize=10, color="black", fontweight="bold")
        ax.add_patch(Rectangle((A2_LO, A4_LO), A2_HI - A2_LO, A4_HI - A4_LO,
                               fill=False, ec="black", lw=3, ls="--", zorder=5))
        ax.text(0.5 * (A2_LO + A2_HI), A4_LO - 0.11, "training region", ha="center", va="top",
                fontsize=15, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1.5))
        ax.set_xlabel(r"$a_2$", fontweight="bold"); ax.set_title(title, fontsize=22, fontweight="bold")
    axs[0].set_ylabel(r"$a_4$", fontweight="bold")
    cb = fig.colorbar(pcm, ax=axs, fraction=0.025, pad=0.02)
    cb.set_label("coverage  (recovered / existing branches)", fontsize=18, fontweight="bold")
    save_fig(fig, os.path.join(HERE, "fig_a2a4_appendix"))
    print("saved fig_a2a4_appendix.png/.pdf")


if __name__ == "__main__":
    main()
