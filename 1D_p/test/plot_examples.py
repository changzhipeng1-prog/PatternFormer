"""1D_p test figure: 2 rows x 5 params.

Row 1 = Qwen direct output (M1).  Row 2 = after Newton refine (cbmfem Newton).
Each subplot: GT = black solid (all k coexisting solutions), Ours = Morandi-colored
dashed lines. The 5 params cover different GT solution counts. When the model emits
MORE solutions than GT, we Hungarian-match to GT and keep the rel-L2-smallest k.
Average rel-L2 (over the k matched pairs) is annotated in each subplot.
Legend distinguishes only Exact vs Ours.  Style from test_plot_style.py.
"""
import os
import sys
import numpy as np
import torch
import scipy.optimize as sciopt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))      # paper/ for test_plot_style
from test_plot_style import apply_style, save_fig
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from model.traditional_refine import refine_with_newton

apply_style()
# +2 to all default font sizes (ticks etc.) for this figure only
import matplotlib as _mpl
for _k in ("font.size","axes.labelsize","axes.titlesize","xtick.labelsize","ytick.labelsize","legend.fontsize","figure.titlesize"):
    _mpl.rcParams[_k] = _mpl.rcParams[_k] + 4


# Clear, high-contrast palette for the multiple solution curves
# (orange / blue / green / cyan / red / purple / pink / brown)
MORANDI = ["#FF7F0E", "#1F77B4", "#2CA02C", "#17BECF",
           "#D62728", "#9467BD", "#E377C2", "#8C564B"]

# Slim, GitHub-friendly data: the 5 representative records this figure plots
# (one per distinct GT count k=1,3,5 plus two k=7), pre-selected as the near-median
# rel-L2 example in each group. Regenerate the full pool with generate_test.py if a
# different selection is ever needed.
GEN = os.path.join(HERE, "examples_solutions.pt")
DX = 1.0 / 1023.0   # 1D_p grid: 1024 points on [0,1]


def rel_l2(pred, gt):
    g = float(np.linalg.norm(gt))
    return float(np.linalg.norm(pred - gt) / (g + 1e-12))


def match_best_k(preds, gt):
    """Hungarian-match preds (m,dim) to gt (k,dim); return the k pred indices
    (one per GT) minimizing total L2, plus the per-pair (pred_i, gt_j)."""
    m, k = preds.shape[0], gt.shape[0]
    cost = np.linalg.norm(preds[:, None, :] - gt[None, :, :], axis=2)  # (m,k)
    r, c = sciopt.linear_sum_assignment(cost)   # picks k rows for k cols
    return list(zip(r.tolist(), c.tolist()))


def main():
    # the 5 pre-selected representative records, in plotting order (k=1,3,5,7,7)
    chosen = torch.load(GEN, map_location="cpu", weights_only=False)
    print("plotting params (p, k):", [(round(float(r["p"]), 3), int(r["n_gt"])) for r in chosen])

    x = np.linspace(0.0, 1.0, 1024)
    # Single row: the DIRECT model output over the ground-truth set. The refined row
    # is dropped -- it was visually identical to the direct one, and the before/after
    # accuracy is already quantified in the box panel (panel B of the composite).
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.8))

    for col, rec in enumerate(chosen):
        p = float(rec["p"])
        gt = rec["gt"].numpy().astype(np.float64)
        gen = rec["generated"].numpy().astype(np.float64)
        k = gt.shape[0]
        pairs = match_best_k(gen, gt)               # direct-output selection (raw)
        sel_raw = [gen[pi] for pi, gj in pairs]
        gt_order = [gj for pi, gj in pairs]

        ax = axes[col]
        for j in range(k):                                  # GT background (black)
            ax.plot(x, gt[j], color="black", lw=3, zorder=1)
        errs = []
        for n, (u, gj) in enumerate(zip(sel_raw, gt_order)):   # ours (Morandi dashed)
            ax.plot(x, u, "--", color=MORANDI[n % len(MORANDI)], lw=2.5, zorder=2)
            errs.append(rel_l2(u, gt[gj]))
        mean_e = float(np.mean(errs)) if errs else float("nan")
        ax.text(0.04, 0.97, f"rel-L2={mean_e:.1e}", transform=ax.transAxes,
                va="top", ha="left", fontsize=18, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.6", alpha=0.85))
        ax.grid(False)
        ax.set_title(f"p = {p:.2f}  (k={k})", fontsize=26, fontweight="bold")
        ax.set_xlabel("x", fontsize=28, fontweight="bold")
        if col == 0:
            ax.set_ylabel("u(x)", fontsize=26, fontweight="bold")

    handles = [Line2D([0], [0], color="black", lw=3, label="Exact (GT)"),
               Line2D([0], [0], color="#6B6B6B", lw=2.5, ls="--", label="Ours (Qwen)")]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97], w_pad=0.2, h_pad=0.6)
    out = os.path.join(HERE, "fig_1Dp_examples_2x5")
    save_fig(fig, out)
    print(f"saved {out}.png/.pdf")


if __name__ == "__main__":
    main()
