"""p=19 coverage: model DIRECT output vs p=18-seeded Newton.

Left  : refine the model's OWN p=19 output -> collapses to 2/7 branches.
Right : refine the model's in-distribution p=18 output AT p=19 -> recovers 7/7.
Both overlaid on the 7 continued-GT branches (covered = solid colour, missed = faint).
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from test_plot_style import apply_style, save_fig
from model.traditional_refine import refine_with_newton_trace
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

apply_style()
import matplotlib as _mpl
for _k in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
           "ytick.labelsize", "legend.fontsize"):
    _mpl.rcParams[_k] = _mpl.rcParams[_k] + 1

P = 19.0
COVER, DEDUP, REFINE_OK = 1e-2, 1e-3, 1e-6
PAL = ["#E69F00", "#009E73", "#D55E00", "#0072B2", "#CC79A7", "#56B4E9", "#9467BD"]


def rl2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def refine_set(outs, p):
    sols = []
    for u in outs:
        ru, ok, _, tr = refine_with_newton_trace(torch.tensor(u), float(p), tol=1e-9, max_iter=30)
        res = float(tr[-1]["residual"]) if tr else float("inf")
        if ok and res < REFINE_OK:
            sols.append(np.asarray(ru.detach().cpu(), np.float64).reshape(-1))
    distinct = []
    for u in sols:
        if not any(rl2(u, v) < DEDUP for v in distinct):
            distinct.append(u)
    return distinct


def covered_map(distinct, gt):
    cov = {}
    for d in distinct:
        j = int(np.argmin([rl2(d, g) for g in gt]))
        if rl2(d, gt[j]) < COVER:
            cov[j] = d
    return cov


def panel(ax, gt, distinct, x, title):
    cov = covered_map(distinct, gt)
    for j in range(gt.shape[0]):
        if j in cov:
            ax.plot(x, gt[j], color=PAL[j % len(PAL)], lw=2.2, alpha=0.5, zorder=1)
        else:
            ax.plot(x, gt[j], color="0.8", lw=1.8, ls=":", zorder=1)
    for j, d in cov.items():
        ax.plot(x, d, color=PAL[j % len(PAL)], lw=3.2, zorder=4)
    ax.set_title(f"{title}  (covers {len(cov)}/{gt.shape[0]})", fontweight="bold", fontsize=15)
    ax.set_xlabel("x", fontweight="bold"); ax.set_ylabel("u(x)", fontweight="bold")
    ax.grid(False)
    ax.legend(handles=[Line2D([0], [0], color="0.4", lw=2.2, alpha=0.6, label="GT branch — covered"),
                       Line2D([0], [0], color="0.8", lw=1.8, ls=":", label="GT branch — missed"),
                       Line2D([0], [0], color="0.4", lw=3.2, label="recovered solution")],
              loc="upper right", framealpha=0.95, fontsize=12)


def main():
    g = torch.load(os.path.join(HERE, "gt_extension.pt"), map_location="cpu", weights_only=False)
    gt = g["gt"][P]
    x = np.linspace(0.0, 1.0, gt.shape[1])
    direct = torch.load(os.path.join(HERE, "generated_extension.pt"), map_location="cpu", weights_only=False)
    gen19 = next(r for r in direct if abs(float(r["p"]) - P) < 1e-6)["generated"].numpy().astype(np.float64)
    gen18 = torch.load(os.path.join(HERE, "p18seed_result.pt"),
                       map_location="cpu", weights_only=False)["gen18"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6.5))
    panel(axA, gt, refine_set(gen19, P), x, f"p={P:.0f}: refine model's DIRECT p={P:.0f} output")
    panel(axB, gt, refine_set(gen18, P), x, f"p={P:.0f}: refine model's p=18 output  (seeded)")
    fig.tight_layout(w_pad=3.0)
    save_fig(fig, os.path.join(HERE, "fig_p18seed_compare"))
    print("saved fig_p18seed_compare.png/.pdf")


if __name__ == "__main__":
    main()
