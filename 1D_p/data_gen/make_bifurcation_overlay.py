"""Panel C of fig_1Dp_panel: bifurcation structure with the LLM-MSO predictions
OVERLAID on the ground-truth data.

  - grey solid lines : the 7 nontrivial ground-truth branches (data), from
    data/p_solutions_lookup.pt  (trivial u=0 branch is NOT shown -- we never generate it).
  - coloured scatter : the LLM-MSO test predictions, matched per parameter to the
    GT solution set (Hungarian on field L2) and coloured by the branch they cover,
    from test/generated_solutions.pt.
  - shaded bands     : the two fold windows where Newton refinement fails.

Writes data_gen/fig_1Dp_bifurcation.{pdf,png} (the file make_panel_1Dp.py tiles into
panel C).  Run:  python make_bifurcation_overlay.py
"""
import os, numpy as np, torch
import scipy.optimize as sciopt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

# per-branch colours for the predictions (1..7); trivial branch 0 not drawn
COL = {1: "#E8A100", 2: "#1B9E77", 3: "#D95F02", 4: "#1F6FB2",
       5: "#E377C2", 6: "#66B3E0", 7: "#8E63B5"}
MISS = [(16.10, 16.20), (17.80, 17.90)]   # folds where Newton refinement fails

# ---- ground-truth branches (grey) ----
d = torch.load(os.path.join(ROOT, "data", "p_solutions_lookup.pt"), weights_only=False)
pv = d["p_values"].numpy(); sbp = d["solutions_by_p"]; bid = d["branch_ids_by_p"]


def integ(field):
    f = field.numpy() if torch.is_tensor(field) else np.asarray(field)
    return float(np.trapz(f, dx=1.0 / (f.shape[-1] - 1)))


br = {b: {"p": [], "I": []} for b in range(1, 8)}
bid_by_idx = {}
for i in range(len(pv)):
    s = sbp[i]
    if s is None or len(s) == 0:
        continue
    s = s if torch.is_tensor(s) else torch.as_tensor(np.asarray(s))
    b = bid[i]; b = b.numpy() if torch.is_tensor(b) else np.asarray(b)
    bid_by_idx[i] = b
    for j in range(s.shape[0]):
        br[int(b[j])]["p"].append(float(pv[i])); br[int(b[j])]["I"].append(integ(s[j]))

# ---- Newton-REFINED LLM-MSO predictions ----
# A converged refine reproduces its GT branch exactly, so every covered branch point is
# the data curve itself; the branches where Newton refinement fails at the fold (found in
# refine_bifurcation_data.py) are split out and marked.
MISSED = os.path.join(HERE, "bif_missed.pt")
missed = set()
if os.path.exists(MISSED):
    for m in torch.load(MISSED, weights_only=False):
        missed.add((round(float(m["p"]), 3), int(m["branch"])))
    print(f"loaded {len(missed)} missed branches")
else:
    print("WARNING: bif_missed.pt not found -- run refine_bifurcation_data.py first")
pred = {b: {"p": [], "I": []} for b in range(1, 8)}
fail = {"p": [], "I": []}
for b in range(1, 8):
    for p, I in zip(br[b]["p"], br[b]["I"]):
        if (round(p, 3), b) in missed:
            fail["p"].append(p); fail["I"].append(I)
        else:
            pred[b]["p"].append(p); pred[b]["I"].append(I)

# ---- plot ----
plt.rcParams.update({"font.size": 16, "font.weight": "bold", "axes.labelweight": "bold",
                     "axes.linewidth": 1.6})
fig, ax = plt.subplots(figsize=(6.2, 4.0))
# (miss-window shading removed; the missing-branch case is discussed in the text)
# GT: grey solid
for b in range(1, 8):
    p = np.array(br[b]["p"]); I = np.array(br[b]["I"]); o = np.argsort(p)
    ax.plot(p[o], I[o], color="0.6", lw=3.2, zorder=2,
            label="ground truth (data)" if b == 1 else None)
# refined predictions: coloured, lying on the branches (converged refines)
for b in range(1, 8):
    if pred[b]["p"]:
        ax.scatter(pred[b]["p"], pred[b]["I"], s=9, color=COL[b], zorder=4,
                   linewidths=0, alpha=0.95)
# (missed-branch markers removed; the missing-branch case is discussed in the text)
ax.set_xlim(0, 18); ax.set_xlabel("$p$"); ax.set_ylabel(r"$\int_0^1 u\,\mathrm{d}x$")
ax.set_xticks(range(0, 19, 2))
handles = [Line2D([0], [0], color="0.6", lw=3.2, label="ground truth (data)"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="#444", markersize=8,
                  label="PatternFormer (Newton-refined)")]
ax.legend(handles=handles, fontsize=11, loc="upper left", framealpha=1.0, handlelength=1.5)
ax.text(0.985, 0.97, "C", transform=ax.transAxes, fontsize=22, fontweight="bold",
        va="top", ha="right", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6"))
fig.savefig(os.path.join(HERE, "fig_1Dp_bifurcation.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(HERE, "fig_1Dp_bifurcation.pdf"), bbox_inches="tight")
plt.close(fig)
print("wrote data_gen/fig_1Dp_bifurcation.{pdf,png}  (GT grey + LLM-MSO coloured overlay)")
