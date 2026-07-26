"""a2a4 efficiency comparison figure:  OUR method (Qwen forward + FDM Newton refine)
vs the TRADITIONAL classical solver (from-scratch multi-start damped-Newton search to
find ALL coexisting branches), across selected test parameters grouped by solution
multiplicity k.

a2a4 is single-resolution (N=1024) -- no multi-grid cascade -- so the classical cost is
the SEARCH (a battery of structured Newton starts), which grows with k (more branches /
smaller basins to discover), while ours is one forward pass + k cheap refines.

Reads : trad_a2a4_timing.csv  (total_time_s, total_newton_iters, coverage)
        ours_a2a4_timing.csv  (total_s, forward_s, refine_s, newton_steps_total)
Writes: comparison_a2a4.csv  and  fig_a2a4_efficiency_main.png/.pdf
Single panel: grouped bars (log wall-time), speed-up annotated above each pair.
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
# +4 to all default font sizes (ticks etc.) for this figure only
import matplotlib as _mpl
for _k in ("font.size","axes.labelsize","axes.titlesize","xtick.labelsize","ytick.labelsize","legend.fontsize","figure.titlesize"):
    _mpl.rcParams[_k] = _mpl.rcParams[_k] + 4
trad = list(csv.DictReader(open(os.path.join(HERE, "trad_a2a4_timing.csv"))))
ours_path = os.path.join(HERE, "ours_a2a4_timing.csv")
if not os.path.exists(ours_path):
    raise SystemExit("ours_a2a4_timing.csv not found yet (timing job still running).")
ours = list(csv.DictReader(open(ours_path)))


def key(r):
    return (round(float(r["a4"]), 3), round(float(r["a2"]), 3))


ours_by = {key(r): r for r in ours}
# keep only targets present in both, order by (k, a2)
recs = []
for t in trad:
    if key(t) in ours_by:
        o = ours_by[key(t)]
        recs.append({"a4": float(t["a4"]), "a2": float(t["a2"]), "k": int(t["k"]),
                     "ours_s": float(o["total_s"]), "trad_s": float(t["total_time_s"]),
                     "ours_fwd": float(o["forward_s"]), "ours_ref": float(o["refine_s"]),
                     "ours_it": int(o["newton_steps_total"]),
                     "trad_it": int(t["total_newton_iters"])})
recs.sort(key=lambda r: (r["k"], r["a2"]))

# ---- combined comparison table ----
with open(os.path.join(HERE, "comparison_a2a4.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["a4", "a2", "k", "ours_total_s", "trad_total_s", "speedup",
                "ours_newton_steps", "trad_newton_iters", "iter_ratio",
                "ours_forward_s", "ours_refine_s"])
    for r in recs:
        spd = r["trad_s"] / r["ours_s"]
        ir = r["trad_it"] / max(r["ours_it"], 1)
        w.writerow([f"{r['a4']:.3f}", f"{r['a2']:.3f}", r["k"], f"{r['ours_s']:.4f}",
                    f"{r['trad_s']:.3f}", f"{spd:.1f}", r["ours_it"], r["trad_it"],
                    f"{ir:.0f}", f"{r['ours_fwd']:.4f}", f"{r['ours_ref']:.4f}"])
print("wrote comparison_a2a4.csv")

# ---- x layout: k-clusters, targets within ----
C_OURS, C_TRAD = "#0066CC", "#FF0000"
w = 0.38
gap_cluster = 1.2
ks = sorted(set(r["k"] for r in recs))
xs, centers, idx = [], [], 0
x = 0.0
for k in ks:
    grp = [i for i, r in enumerate(recs) if r["k"] == k]
    cxs = []
    for _ in grp:
        xs.append(x); cxs.append(x); x += 1.0
    centers.append((k, np.mean(cxs)))
    x += gap_cluster
xs = np.array(xs)
ours_t = np.array([r["ours_s"] for r in recs])
trad_t = np.array([r["trad_s"] for r in recs])
spd = trad_t / ours_t


def fmt_time(s):
    return f"{s:.2f}s" if s < 60 else f"{s/60:.1f}m"


fig = plt.figure(figsize=(13, 8.5))
ax = fig.add_subplot(1, 1, 1)
ax.bar(xs - w/2, ours_t, w, color=C_OURS, edgecolor="black", lw=1.5,
       label="Ours (Qwen + FDM Newton refine)", zorder=3)
ax.bar(xs + w/2, trad_t, w, color=C_TRAD, edgecolor="black", lw=1.5, alpha=0.9,
       label="Traditional (multi-start Newton search)", zorder=3)
ax.set_yscale("log")
ymax = trad_t.max()
for xi, v in zip(xs - w/2, ours_t):
    ax.text(xi, v*1.18, fmt_time(v), ha="center", va="bottom", fontsize=17,
            fontweight="bold", color=C_OURS)
for xi, v in zip(xs + w/2, trad_t):
    ax.text(xi, v*1.18, fmt_time(v), ha="center", va="bottom", fontsize=17,
            fontweight="bold", color=C_TRAD)
for xi, v, s in zip(xs + w/2, trad_t, spd):
    ax.text(xi, v*2.6, f"{s:.0f}x", ha="center", va="bottom", fontsize=21,
            fontweight="bold", color="darkorange")
ax.set_ylim(ours_t.min()*0.30, ymax*40)
ax.set_xticks([c for k, c in centers])
ax.set_xticklabels([f"k={k}\n({k} sol.)" for k, c in centers], fontsize=22, fontweight="bold")
ax.set_ylabel("wall-clock time (s)", fontsize=28, fontweight="bold")
ax.grid(False)
# (k-cluster labels are now the x-ticks)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1,
          fontsize=21, framealpha=0.95, handlelength=1.4, borderpad=0.4, labelspacing=0.3)
ax.text(0.985, 0.980, "D", transform=ax.transAxes, ha="right", va="top",
        fontsize=46, fontweight="bold", zorder=30,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", lw=1.5, alpha=0.95))
fig.tight_layout()
save_fig(fig, os.path.join(HERE, "fig_a2a4_efficiency_main"))
print("saved fig_a2a4_efficiency_main.png/.pdf")
print("DONE")
