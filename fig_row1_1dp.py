"""Row 1 (standalone): Example 1 (1D single-parameter) pretrained-vs-random ablation.
   A  training val-MSE          B  test direct rel-L2 boxplot
   C  the seven branches at a k=7 parameter (full view, with a marked zoom box)
   Z  the zoom box, magnified into its OWN subplot, linked to C by an arrow.

Output: paper/fig_row1_1dp.{pdf,png}
"""
import os, re, numpy as np, torch
import scipy.optimize as sciopt
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch
sys.path.insert(0, "/home/zfc5231/work/BBBB_qwen_pde_branch/paper")
from test_plot_style import apply_style
apply_style()   # Nature-style big bold fonts, matching the other paper figures

ABL1D = "/home/zfc5231/work/BBBB_qwen_pde_branch/paper/1D_p/ablation_pretrain"
ROOT = "/home/zfc5231/work/BBBB_qwen_pde_branch/paper"
PRE_C, RND_C, GT_C = "#0066CC", "#E8730C", "0.55"
LOG = {"pretrained": "logs/arm1_pretrained_45991.out", "random": "logs/arm2_random_45990.out"}
GEN = {"pretrained": "gen_arm1_pretrained.pt", "random": "gen_arm2_random.pt"}

def parse_curve(path):
    stage, s1, s2 = 0, [], []
    for ln in open(os.path.join(ABL1D, path)):
        if "STAGE 1" in ln: stage = 1
        if "STAGE 2" in ln: stage = 2
        m = re.search(r"Epoch\s+(\d+)/\d+.*val_mse=([\d.eE+-]+)", ln)
        if m:
            (s1 if stage == 1 else s2).append((int(m.group(1)), float(m.group(2))))
    return s1, s2

def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

def direct_stats(genpath):
    g = torch.load(os.path.join(ABL1D, genpath), map_location="cpu", weights_only=False)
    rels, rec7 = [], None
    for rec in g:
        gt = rec["gt"].numpy().astype(np.float64); gen = rec["generated"]
        if gen is None or len(gen) == 0: continue
        gen = gen.numpy().astype(np.float64)
        cost = np.linalg.norm(gen[:, None, :] - gt[None, :, :], axis=2)
        r, c = sciopt.linear_sum_assignment(cost)
        rels += [rel_l2(gen[i], gt[j]) for i, j in zip(r, c)]
        if rec7 is None and gt.shape[0] == 7:
            rec7 = {"p": float(rec["p"]), "gt": gt, "gen": gen, "match": list(zip(r.tolist(), c.tolist()))}
    return np.array(rels), rec7

curves = {k: parse_curve(v) for k, v in LOG.items()}
stats = {k: direct_stats(v) for k, v in GEN.items()}

# ---------- figure: 1x4 (A, B, C, Z) + one legend row across the top ----------
fig = plt.figure(figsize=(24, 6.4))
gsf = fig.add_gridspec(1, 4, wspace=0.36)
axA = fig.add_subplot(gsf[0, 0]); axB = fig.add_subplot(gsf[0, 1])
axC = fig.add_subplot(gsf[0, 2]); axZ = fig.add_subplot(gsf[0, 3])

# A training
# pretrained LLM: solid wide; untrained LLM: dashed thin (same scheme as panel C)
for k, col, ls, lw in [("pretrained", PRE_C, "-", 3.0), ("random", RND_C, (0, (4, 2)), 2.2)]:
    s1, s2 = curves[k]
    e1 = [e for e, _ in s1]; v1 = [v for _, v in s1]
    off = max(e1) if e1 else 0
    e2 = [off + e for e, _ in s2]; v2 = [v for _, v in s2]
    axA.plot(e1 + e2, v1 + v2, color=col, lw=lw, ls=ls)
    if e1: axA.axvline(off, color="0.6", ls=":", lw=1.6, zorder=0)
axA.set_yscale("log"); axA.set_xlabel("epoch (stage 1 | stage 2)", fontweight="bold")
axA.set_ylabel("validation MSE", fontweight="bold")

# B test rel-L2
bp = axB.boxplot([stats["pretrained"][0], stats["random"][0]],
                 labels=["pretrained\nLLM", "untrained\nLLM"],
                 showfliers=False, patch_artist=True, widths=0.6)
for patch, col in zip(bp["boxes"], [PRE_C, RND_C]):
    patch.set_facecolor(col); patch.set_alpha(0.55)
axB.set_yscale("log"); axB.set_ylabel("test direct rel-$L^2$", fontweight="bold")

# C + Z: k=7 solutions, full view and external zoom
r7 = stats["pretrained"][1]; r7r = stats["random"][1]
x = np.linspace(0, 1, r7["gt"].shape[1])
gtmap = {j: i for i, j in r7["match"]}
rmap = {j: i for i, j in r7r["match"]} if r7r is not None else {}
def draw_C(ax, scale=1.0):
    for j in range(r7["gt"].shape[0]):                                  # exact: grey solid (background)
        ax.plot(x, r7["gt"][j], color=GT_C, lw=5.0 * scale, zorder=1)
    for j, i in gtmap.items():                                         # pretrained LLM: solid, wide (lw 4)
        ax.plot(x, r7["gen"][i], color=PRE_C, lw=4.0 * scale, zorder=2)
    for j, i in rmap.items():                                          # untrained LLM: dashed, thin (lw 2)
        ax.plot(x, r7r["gen"][i], color=RND_C, lw=2.0 * scale, ls=(0, (4, 2)), zorder=3)
draw_C(axC)
axC.set_xlabel("$x$", fontweight="bold"); axC.set_ylabel("$u$", fontweight="bold")
axC.set_xlim(0, 1)  # no legend in C; the single legend is the top row

# zoom region centred at (x, u) = (0.5, 3)
zx0, zx1, zy0, zy1 = 0.38, 0.62, 1.8, 4.2
axC.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec="0.35", lw=1.6, zorder=6))
draw_C(axZ, scale=1.25)
axZ.set_xlim(zx0, zx1); axZ.set_ylim(zy0, zy1)
axZ.set_xticks([]); axZ.set_yticks([])
for s in axZ.spines.values(): s.set_color("0.35"); s.set_linewidth(1.8)
axZ.set_xlabel("zoom", fontweight="bold", fontsize=18)
# arrow linking the box in C to the zoom subplot
con = ConnectionPatch(xyA=(zx1, 0.5 * (zy0 + zy1)), coordsA=axC.transData,
                      xyB=(0.0, 0.5), coordsB=axZ.transAxes,
                      arrowstyle="-|>", lw=2.2, color="0.35", mutation_scale=24)
fig.add_artist(con)

# single legend row across the top of the figure
fig.legend(handles=[Line2D([0], [0], color=GT_C, lw=5.0, label="exact"),
                    Line2D([0], [0], color=PRE_C, lw=4.0, label="pretrained LLM"),
                    Line2D([0], [0], color=RND_C, lw=2.5, ls=(0, (4, 2)), label="untrained LLM")],
           loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10), frameon=False, fontsize=23)

# panel letters at the top-left corner of A, B, C
fig.canvas.draw()
for ax, lt in [(axA, "A"), (axB, "B"), (axC, "C")]:
    p = ax.get_position()
    fig.text(p.x0 - 0.012, p.y1 + 0.014, lt, fontsize=26, fontweight="bold", va="bottom", ha="left")

out = os.path.join(ROOT, "fig_row1_1dp")
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
print("wrote", out + ".pdf/.png")
