"""Efficiency comparison figures for 1D_p:  OUR method (Qwen + cascade Newton
refine) vs the TRADITIONAL multi-scale solver, on the two near-integer test
parameters p=7 and p=18, at resolutions 1024 / 2048 / 4096.

Data sources (already measured & stored):
  comparison.csv        wall-clock (ours_cumulative_s, trad_total_s, speedup)
  steps_comparison.csv  Newton iterations (ours_total, trad_total, ratio)

Output (png @300dpi + pdf):
  fig_1dp_efficiency_main      left grouped-bar (log time) + right summary table
"""
import os
import sys
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from test_plot_style import apply_style, save_fig
import matplotlib.pyplot as plt

apply_style()
# +2 to all default font sizes (ticks etc.) for this figure only
import matplotlib as _mpl
for _k in ("font.size","axes.labelsize","axes.titlesize","xtick.labelsize","ytick.labelsize","legend.fontsize","figure.titlesize"):
    _mpl.rcParams[_k] = _mpl.rcParams[_k] + 4


# ----- load -----------------------------------------------------------------
comp = list(csv.DictReader(open(os.path.join(HERE, "comparison.csv"))))
step = list(csv.DictReader(open(os.path.join(HERE, "steps_comparison.csv"))))


def grab(rows, p, res, col):
    for r in rows:
        if int(float(r["p"])) == p and int(r["resolution"]) == res:
            return r[col]
    raise KeyError((p, res, col))


PS = [7, 18]
RES = [1024, 2048, 4096]
C_OURS = "#0066CC"   # blue  -> our method
C_TRAD = "#FF0000"   # red   -> traditional

ours_t = {p: [float(grab(comp, p, r, "ours_cumulative_s")) for r in RES] for p in PS}
trad_t = {p: [float(grab(comp, p, r, "trad_total_s")) for r in RES] for p in PS}
spd = {p: [float(grab(comp, p, r, "speedup")) for r in RES] for p in PS}
ours_i = {p: [int(grab(step, p, r, "ours_newton_steps_total")) for r in RES] for p in PS}
trad_i = {p: [int(grab(step, p, r, "trad_newton_iters_total")) for r in RES] for p in PS}
iratio = {p: [float(grab(step, p, r, "total_iter_ratio_trad_over_ours")) for r in RES] for p in PS}


def fmt_time(s):
    if s < 60:
        return f"{s:.1f}s"
    m = s / 60.0
    return f"{m:.1f}m"


# ============================================================================
# FIGURE — grouped bars (log wall-time), ours vs traditional
# ============================================================================
fig = plt.figure(figsize=(13, 8.5))
axb = fig.add_subplot(1, 1, 1)

# x layout: two p-clusters, 3 resolutions each, 2 bars (ours/trad) per resolution
w = 0.38
gap_cluster = 1.4
xs = []
centers = []
x = 0.0
for ip, p in enumerate(PS):
    for ir, r in enumerate(RES):
        xs.append(x)
        x += 1.0
    centers.append(np.mean(xs[ip * 3:ip * 3 + 3]))
    x += gap_cluster
xs = np.array(xs)

ours_flat = np.array([ours_t[p][ir] for p in PS for ir in range(3)])
trad_flat = np.array([trad_t[p][ir] for p in PS for ir in range(3)])
spd_flat = np.array([spd[p][ir] for p in PS for ir in range(3)])

b1 = axb.bar(xs - w / 2, ours_flat, w, color=C_OURS, edgecolor="black", lw=1.5,
             label="Ours (Qwen + Newton refine)", zorder=3)
b2 = axb.bar(xs + w / 2, trad_flat, w, color=C_TRAD, edgecolor="black", lw=1.5,
             alpha=0.9, label="Traditional (multi-scale)", zorder=3)
axb.set_yscale("log")

# value labels on bars (horizontal)
ymax = trad_flat.max()
for xi, v in zip(xs - w / 2, ours_flat):
    axb.text(xi, v * 1.18, fmt_time(v), ha="center", va="bottom",
             fontsize=19, fontweight="bold", color=C_OURS, rotation=0)
for xi, v in zip(xs + w / 2, trad_flat):
    axb.text(xi, v * 1.18, fmt_time(v), ha="center", va="bottom",
             fontsize=19, fontweight="bold", color=C_TRAD, rotation=0)
# speedup: directly ABOVE the traditional time text, same x (no horizontal offset)
for xi, v, s in zip(xs + w / 2, trad_flat, spd_flat):
    axb.text(xi, v * 2.6, f"{s:.0f}x", ha="center", va="bottom",
             fontsize=21, fontweight="bold", color="darkorange")

axb.set_ylim(ours_flat.min() * 0.30, ymax * 600)
axb.set_xticks(xs)
axb.set_xticklabels([f"{r}" for p in PS for r in RES], fontsize=22, fontweight="bold")
axb.set_xlabel("grid resolution N", fontsize=28, fontweight="bold", labelpad=12)
axb.set_ylabel("wall-clock time (s)", fontsize=28, fontweight="bold")
axb.grid(False)
# p-cluster labels at the TOP of each cluster (inside the plot, above all bars)
nsol = {7: 3, 18: 7}
for p, c in zip(PS, centers):
    axb.text(c, ymax * 22, f"p = {p}   ({nsol[p]} solutions)", ha="center", va="center",
             fontsize=23, fontweight="bold")
# legend INSIDE the plot as TWO rows (ncol=1), pinned to the top so it sits ABOVE
# the lowered p-cluster labels and clear of the outer top-left "D" panel label
axb.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1,
           fontsize=21, framealpha=0.95, handlelength=1.4, borderpad=0.4, labelspacing=0.3)

axb.text(0.985, 0.980, "D", transform=axb.transAxes, ha="right", va="top",
         fontsize=46, fontweight="bold", zorder=30,
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", lw=1.5, alpha=0.95))
fig.tight_layout()
save_fig(fig, os.path.join(HERE, "fig_1dp_efficiency_main"))
print("saved fig_1dp_efficiency_main.png/.pdf")
print("DONE")
