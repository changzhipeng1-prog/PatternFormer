"""a2a4 box-plot panel (1x4, grouped by non-zero multiplicity k), from stats.csv:
   1) rel-L2 BEFORE Newton refine
   2) Newton steps (FDM)
   3) rel-L2 AFTER Newton refine (+ lost markers/note if any)
   4) Newton refine residual ||F||  (FDM, data-gen convention)
The trivial u==0 branch is excluded upstream (compute_stats). Style: test_plot_style.
"""
import os
import sys
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from test_plot_style import apply_style, save_fig
import matplotlib.pyplot as plt

apply_style()
# +4 to all default font sizes (ticks etc.) for this figure only
import matplotlib as _mpl
for _k in ("font.size","axes.labelsize","axes.titlesize","xtick.labelsize","ytick.labelsize","legend.fontsize","figure.titlesize"):
    _mpl.rcParams[_k] = _mpl.rcParams[_k] + 4
rows = list(csv.DictReader(open(os.path.join(HERE, "stats.csv"))))
k_of = np.array([int(r["k"]) for r in rows])
direct = np.clip(np.array([float(r["direct_rel_l2"]) for r in rows]), 1e-9, None)
post = np.clip(np.array([float(r["post_rel_l2"]) for r in rows]), 1e-9, None)
resid = np.clip(np.array([float(r["post_residual"]) for r in rows]), 1e-13, None)
steps = np.array([float(r["newton_steps"]) for r in rows])
conv = np.array([int(r["converged"]) == 1 for r in rows])
a4_of = np.array([float(r["a4"]) for r in rows])
a2_of = np.array([float(r["a2"]) for r in rows])
lost = post > 1e-2
lost_params = (np.unique(np.stack([a4_of[lost], a2_of[lost]], 1).round(4), axis=0).shape[0]
               if lost.any() else 0)
ks = sorted(set(k_of.tolist()))
xpos = np.arange(len(ks))
klab = [f"k={k}" for k in ks]
C_BOX, C_LOST = "#6baed6", "darkorange"


def draw_box(ax, vals, log=True):
    bp = ax.boxplot([vals[k_of == k] for k in ks], positions=xpos, widths=0.55,
                    patch_artist=True, showfliers=False, whis=(5, 95))
    for patch in bp["boxes"]:
        patch.set_facecolor(C_BOX); patch.set_alpha(0.6); patch.set_edgecolor("black"); patch.set_linewidth(2)
    for m in bp["medians"]: m.set_color("black"); m.set_linewidth(3)
    for el in ("whiskers", "caps"):
        for ln in bp[el]: ln.set_color("black"); ln.set_linewidth(2)
    if log: ax.set_yscale("log")
    ax.set_xticks(xpos); ax.set_xticklabels(klab)
    ax.grid(axis="y", alpha=0.3)


fig, axes = plt.subplots(1, 4, figsize=(22, 7))

draw_box(axes[0], direct)
axes[0].set_ylabel("Relative L2 error", fontsize=28, fontweight="bold")
axes[0].set_title("Before Newton refine", fontsize=24, fontweight="bold")

# Newton steps bar (mean per k, whisker to p95)
ax = axes[1]
mean_s = np.array([steps[(k_of == k) & conv].mean() if (k_of == k).any() else 0 for k in ks])
p95_s = np.array([np.percentile(steps[(k_of == k) & conv], 95) if ((k_of == k) & conv).any() else 0 for k in ks])
ax.bar(xpos, mean_s, 0.6, color=C_BOX, alpha=0.85, edgecolor="black", lw=2, zorder=2)
ax.errorbar(xpos, mean_s, yerr=[np.zeros(len(ks)), np.maximum(p95_s - mean_s, 0)], fmt="none",
            ecolor="black", capsize=7, lw=2, zorder=3)
for x, m in zip(xpos, mean_s):
    ax.text(x, m + 0.05, f"{m:.2f}", ha="center", va="bottom", fontsize=21, fontweight="bold")
ax.set_xticks(xpos); ax.set_xticklabels(klab)
ax.set_ylim(0, max(p95_s.max() * 1.18, 1))
ax.set_ylabel("Newton steps", fontsize=28, fontweight="bold")
ax.set_title("Newton refine steps", fontsize=24, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# after rel-L2 + lost
draw_box(axes[2], post)
axes[2].set_ylabel("Relative L2 error", fontsize=28, fontweight="bold")
axes[2].set_title("After Newton refine", fontsize=24, fontweight="bold")
axes[2].set_ylim(1e-8, 2e1)
rng = np.random.default_rng(0)
n_total_p = np.unique(np.stack([a4_of, a2_of], 1).round(4), axis=0).shape[0]
for x, k in zip(xpos, ks):
    v = post[(k_of == k) & lost]
    if v.size:
        vs = v if v.size <= 8 else rng.choice(v, 8, replace=False)
        axes[2].scatter(x + rng.uniform(-0.20, 0.20, vs.size), vs, s=42, color=C_LOST,
                        edgecolor="none", alpha=0.9, zorder=5)
if lost.any():
    axes[2].annotate(f"missing branch\n{lost_params} params\n{lost_params/n_total_p*100:.1f}% of test params",
                     xy=(xpos[-1] - 0.22, 4e-1), xycoords="data",
                     xytext=(0.5, 0.46), textcoords="axes fraction",
                     fontsize=18, fontweight="bold", color=C_LOST, va="center", ha="center",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_LOST, lw=1.5),
                     arrowprops=dict(arrowstyle="->", color=C_LOST, lw=2))

draw_box(axes[3], resid)
axes[3].set_ylabel(r"Residual $\|F\|$", fontsize=28, fontweight="bold")
axes[3].set_title("Newton refine residual", fontsize=24, fontweight="bold")

fig.tight_layout(w_pad=0.2)
save_fig(fig, os.path.join(HERE, "fig_a2a4_box4"))
print("saved fig_a2a4_box4.png/.pdf")
