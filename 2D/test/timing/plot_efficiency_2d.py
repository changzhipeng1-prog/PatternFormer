"""2D efficiency comparison figure: OUR method (Qwen forward + FEM Newton refine) vs
the TRADITIONAL data-generation method (seed search @ s=1600 + continuation march down
to the target s), across selected test-set s values spanning march distance from the
anchor s=1600.

Reads:  trad_2d_timing.csv  (seed + march wall-time, Newton iters; k=raw D4 orbit)
        ours_2d_timing.csv  (forward + refine wall-time, Newton steps)
Writes: comparison_2d.csv  and  fig_2d_efficiency_main.png/.pdf
Single panel: grouped bars (log wall-time), speed-up annotated above each pair.
"""
import os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from test_plot_style import apply_style, save_fig
import matplotlib.pyplot as plt

apply_style()
# +4 to all default font sizes (ticks etc.) for this figure only
import matplotlib as _mpl
for _k in ("font.size","axes.labelsize","axes.titlesize","xtick.labelsize","ytick.labelsize","legend.fontsize","figure.titlesize"):
    _mpl.rcParams[_k] = _mpl.rcParams[_k] + 4
trad = {int(r["s"]): r for r in csv.DictReader(open(os.path.join(HERE, "trad_2d_timing.csv")))}
ours_path = os.path.join(HERE, "ours_2d_timing.csv")
if not os.path.exists(ours_path):
    raise SystemExit("ours_2d_timing.csv not found yet (GPU job 45855 still queued/running).")
ours = {int(r["s"]): r for r in csv.DictReader(open(ours_path))}

S = sorted(set(trad) & set(ours), reverse=True)          # high s (near seed) -> low s
ours_t = np.array([float(ours[s]["total_s"]) for s in S])
trad_t = np.array([float(trad[s]["total_time_s"]) for s in S])
ours_it = np.array([int(ours[s]["newton_steps_total"]) for s in S])
trad_it = np.array([int(trad[s]["total_newton_iters"]) for s in S])
steps_march = np.array([int(trad[s]["march_steps"]) for s in S])
spd = trad_t / ours_t
iratio = trad_it / np.maximum(ours_it, 1)

# ---- write combined comparison table ----
with open(os.path.join(HERE, "comparison_2d.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["s", "march_steps_from_1600", "ours_total_s", "trad_total_s", "speedup",
                "ours_newton_steps", "trad_newton_iters", "iter_ratio",
                "ours_forward_s", "ours_refine_s", "trad_seed_s", "trad_march_s"])
    for i, s in enumerate(S):
        w.writerow([s, steps_march[i], f"{ours_t[i]:.4f}", f"{trad_t[i]:.3f}", f"{spd[i]:.1f}",
                    ours_it[i], trad_it[i], f"{iratio[i]:.0f}",
                    ours[s]["forward_s"], ours[s]["refine_s"],
                    trad[s]["seed_time_s"], trad[s]["march_time_s"]])
print("wrote comparison_2d.csv")

# ---- figure: grouped bars, log wall-time ----
C_OURS, C_TRAD = "#0066CC", "#FF0000"
x = np.arange(len(S)); w = 0.38
fig, ax = plt.subplots(figsize=(13, 8.5))
ax.bar(x - w/2, ours_t, w, color=C_OURS, edgecolor="black", lw=1.5,
       label="Ours (Qwen + FEM Newton refine)", zorder=3)
ax.bar(x + w/2, trad_t, w, color=C_TRAD, edgecolor="black", lw=1.5, alpha=0.9,
       label="Traditional (seed search + continuation march)", zorder=3)
ax.set_yscale("log")
ymax = trad_t.max()
for xi, v in zip(x - w/2, ours_t):
    ax.text(xi, v*1.18, f"{v:.1f}s", ha="center", va="bottom", fontsize=17, fontweight="bold", color=C_OURS)
for xi, v in zip(x + w/2, trad_t):
    ax.text(xi, v*1.18, f"{v:.0f}s", ha="center", va="bottom", fontsize=17, fontweight="bold", color=C_TRAD)
for xi, v, sp in zip(x + w/2, trad_t, spd):
    ax.text(xi, v*2.6, f"{sp:.0f}x", ha="center", va="bottom", fontsize=21, fontweight="bold", color="darkorange")
ax.set_ylim(ours_t.min()*0.30, ymax*200)
ax.set_xticks(x)
ax.set_xticklabels([f"s={s}" for s in S], fontsize=20, fontweight="bold")
ax.set_ylabel("wall-clock time (s)", fontsize=28, fontweight="bold")
ax.grid(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=21,
          framealpha=0.95, handlelength=1.4, borderpad=0.4, labelspacing=0.3)
ax.text(0.985, 0.980, "D", transform=ax.transAxes, ha="right", va="top",
        fontsize=46, fontweight="bold", zorder=30,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", lw=1.5, alpha=0.95))
fig.tight_layout()
save_fig(fig, os.path.join(HERE, "fig_2d_efficiency_main"))
print("saved fig_2d_efficiency_main.png/.pdf")
