"""Regenerate the 1D-p bifurcation sub-figure (panel C of fig_1Dp_panel) from the
solution lookup, in the original 8-branch style, AND shade the two narrow p-windows
where the model misses a branch (the folds where new branches are born).
  python make_bifurcation_1Dp.py      # writes data_gen/fig_1Dp_bifurcation.{pdf,png}
"""
import os, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

# branch colours (Branch 0 = trivial u=0, black horizontal line; 1..7 from data)
COL = {0: "black", 1: "#E8A100", 2: "#1B9E77", 3: "#D95F02",
       4: "#1F6FB2", 5: "#E377C2", 6: "#66B3E0", 7: "#8E63B5"}
# the two p-windows where a branch is missed (from test/stats.csv: post rel-L2 > 1e-2;
# matches the windows quoted in the text, p in [16.1,16.2] and [17.8,17.9])
MISS = [(16.10, 16.20), (17.80, 17.90)]

d = torch.load(os.path.join(ROOT, "data", "p_solutions_lookup.pt"), weights_only=False)
pv = d["p_values"].numpy(); sbp = d["solutions_by_p"]; bid = d["branch_ids_by_p"]
br = {b: {"p": [], "I": []} for b in range(1, 8)}
for i in range(len(pv)):
    s = sbp[i]
    if s is None or len(s) == 0:
        continue
    s = s.numpy() if torch.is_tensor(s) else np.asarray(s)
    b = bid[i]; b = b.numpy() if torch.is_tensor(b) else np.asarray(b)
    dx = 1.0 / (s.shape[1] - 1)
    for j in range(s.shape[0]):
        I = float(np.trapz(s[j], dx=dx))
        br[int(b[j])]["p"].append(float(pv[i])); br[int(b[j])]["I"].append(I)

plt.rcParams.update({"font.size": 16, "font.weight": "bold", "axes.labelweight": "bold",
                     "axes.linewidth": 1.6})
fig, ax = plt.subplots(figsize=(6.2, 4.0))
# shaded miss-windows FIRST, behind the curves
for lo, hi in MISS:
    ax.axvspan(lo, hi, color="#F08080", alpha=0.70, zorder=0, linewidth=0)
# Branch 0: trivial solution u = 0
ax.axhline(0.0, color=COL[0], lw=2.4, zorder=2, label="Branch 0")
for b in range(1, 8):
    p = np.array(br[b]["p"]); I = np.array(br[b]["I"])
    o = np.argsort(p)
    ax.plot(p[o], I[o], color=COL[b], lw=2.6, zorder=3, label=f"Branch {b}")
ax.set_xlim(0, 18); ax.set_xlabel("$p$")
ax.set_ylabel(r"$\int_0^1 u\,\mathrm{d}x$")
ax.set_xticks(range(0, 19, 2))
ax.legend(ncol=2, fontsize=12, loc="upper left", framealpha=1.0,
          handlelength=1.5, columnspacing=1.0)
ax.text(0.985, 0.97, "C", transform=ax.transAxes, fontsize=22, fontweight="bold",
        va="top", ha="right", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6"))
fig.savefig(os.path.join(HERE, "fig_1Dp_bifurcation.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(HERE, "fig_1Dp_bifurcation.pdf"), bbox_inches="tight")
plt.close(fig)
print("wrote data_gen/fig_1Dp_bifurcation.{pdf,png}  (miss windows:", MISS, ")")
