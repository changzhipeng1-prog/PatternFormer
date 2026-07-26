"""Gray-Scott backbone-initialization ablation, single 1x4 figure:
  A  twin-axis bars: mean # distinct solutions (left) and coverage (right),
     for the three tiers random -> pretrained -> stop.
  B,C,D  per-parameter #distinct over (rho,mu), one map per tier, styled exactly
     like the main-text Fig.3 maps (apply_style, YlOrRd round markers s=26,
     bold rho/mu labels, method title, lower-left mean/max box) but with NO grid
     and colorbar vmax = 10; the three maps share one colorbar.

Output: paper/fig_gs_combined.{pdf,png}
"""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.insert(0, "/home/zfc5231/work/BBBB_qwen_pde_branch/paper")
from test_plot_style import apply_style
apply_style()

ROOT = "/home/zfc5231/work/BBBB_qwen_pde_branch/paper"
TIERS = ["random", "pretrained", "stop"]
MLAB = {"random": "untrained LLM", "pretrained": "pretrained LLM", "stop": "warm-start"}
XLAB = {"random": "untrained\nLLM", "pretrained": "pretrained\nLLM", "stop": "warm-start"}
JS = {"random": "GS/L1/ablation/eval/arm2_random_det.json",
      "pretrained": "GS/L1/ablation/eval/arm1_pretrained_det.json",
      "stop": "GS/L1/results/ord_eval_L1.json"}
d = {k: json.load(open(os.path.join(ROOT, v))) for k, v in JS.items()}
nsol = {k: d[k]["det_distinct_mean"] for k in TIERS}
cov = {k: d[k]["coverage_mean"] for k in TIERS}
MAP = np.load(os.path.join(ROOT, "gs_map_data.npz"))
rho, mu = MAP["rho"], MAP["mu"]
val = {k: MAP[k] for k in TIERS}
VMAX = 10
C_SOL, C_COV = "#3C6E9C", "#E0892B"

fig = plt.figure(figsize=(25, 5.4))
# col 1 is an empty spacer so A's right "coverage" label clears B's "mu" label;
# A is wider than the maps, and the three maps sit close together (small wspace).
gsf = fig.add_gridspec(1, 5, width_ratios=[1.7, 0.34, 1.0, 1.0, 1.0], wspace=0.16)

# ---- A: twin-axis bars (#sol left, coverage right) ----
axA = fig.add_subplot(gsf[0, 0]); axAr = axA.twinx()
x = np.arange(3); w = 0.38
b1 = axA.bar(x - w / 2, [nsol[k] for k in TIERS], width=w, color=C_SOL, label="# solutions")
b2 = axAr.bar(x + w / 2, [cov[k] for k in TIERS], width=w, color=C_COV, label="coverage")
for i, k in enumerate(TIERS):
    axA.text(i - w / 2, nsol[k] + 0.1, f"{nsol[k]:.2f}", ha="center", va="bottom", fontsize=17, fontweight="bold", color=C_SOL)
    axAr.text(i + w / 2, cov[k] + 0.004, f"{cov[k]:.2f}", ha="center", va="bottom", fontsize=17, fontweight="bold", color=C_COV)
axA.set_xticks(x); axA.set_xticklabels([XLAB[k] for k in TIERS], fontsize=19)
axA.set_ylabel("# distinct solutions / param", fontsize=21, fontweight="bold", color=C_SOL)
axAr.set_ylabel("coverage", fontsize=21, fontweight="bold", color=C_COV)
axA.tick_params(axis="y", labelcolor=C_SOL, labelsize=19); axAr.tick_params(axis="y", labelcolor=C_COV, labelsize=19)
axA.set_ylim(0, max(nsol.values()) * 1.2); axAr.set_ylim(0, max(cov.values()) * 1.2)
axA.legend(handles=[b1, b2], fontsize=18, loc="upper left")

# ---- B,C,D: GS maps (Fig.3 style, no grid, vmax=10) ----
mapax = []
for i, k in enumerate(TIERS):
    ax = fig.add_subplot(gsf[0, 2 + i]); mapax.append(ax)
    sc = ax.scatter(rho, mu, c=val[k], s=26, cmap="YlOrRd", vmin=0, vmax=VMAX,
                    edgecolor="0.5", linewidth=0.2)
    ax.text(0.05, 0.05, f"mean {val[k].mean():.1f}\nmax  {int(val[k].max())}",
            transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left",
            linespacing=1.25, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))
    ax.set_title(MLAB[k], fontsize=15, fontweight="bold")
    ax.set_xlabel(r"$\rho$", fontweight="bold")
    if i == 0:
        ax.set_ylabel(r"$\mu$", fontweight="bold")
    else:
        ax.tick_params(labelleft=False)   # drop mu tick labels on C, D (avoid overlap)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_axisbelow(True)
fig.canvas.draw()
p = mapax[-1].get_position()
cax = fig.add_axes([p.x1 + 0.008, p.y0, 0.008, p.height])
cb = fig.colorbar(sc, cax=cax); cb.set_label("# solutions", fontweight="bold", fontsize=11)

# panel letters A-D
for ax, lt in [(axA, "A"), (mapax[0], "B"), (mapax[1], "C"), (mapax[2], "D")]:
    pp = ax.get_position()
    fig.text(pp.x0 - 0.028, pp.y1 + 0.02, lt, fontsize=22, fontweight="bold", va="bottom", ha="left")

out = os.path.join(ROOT, "fig_gs_combined")
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf/.png")
