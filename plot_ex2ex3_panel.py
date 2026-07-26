"""Ex2 -> Ex3 cross-equation generalization panel (1D two-param -> 2D forced, a
DIFFERENT PDE). Mimics the four-panel layout of the manuscript ablation figure but
for warm-start (initialize from the previous PDE's checkpoint) vs scratch (fresh
LoRA on the pretrained Qwen). Only the DIRECT generation is shown (no refinement).

  A  training process   : validation MSE over the two stages, warm vs scratch (logs)
  B  direct relative L2  : test-set box plots, warm vs scratch (no refinement)
  C  error field |pred-GT|, warm-start   \  same test parameter + branch,
  D  error field |pred-GT|, scratch      /  shared error colour scale

Output -> paper/fig_ex2ex3_panel.{png,pdf}
"""
import os
import re
import sys
import numpy as np
import torch
import scipy.optimize as sciopt
from matplotlib.tri import Triangulation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_plot_style import apply_style, save_fig
import matplotlib.pyplot as plt

apply_style()
C_WARM, C_SCR = "#0066CC", "#FF8C00"

# ---- chosen test parameter / branch (largest warm-vs-scratch error contrast) ----
SEL_IDX, SEL_S, SEL_J = 873, 735.0, 2

WARM_LOG = "2D/train_log/2d_2stage_45773.out"
SCR_LOG = "2D/code/logs/ex3_scratch_46112.out"
WARM_STATS = os.path.join(HERE, "2D", "test", "stats.pt")
SCR_STATS = os.path.join(HERE, "2D", "test", "scratch", "stats.pt")
WARM_GEN = os.path.join(HERE, "2D", "test", "generated_solutions.pt")
SCR_GEN = os.path.join(HERE, "2D", "test", "scratch", "generated_solutions.pt")

EP_RE = re.compile(r"(?:\[phase2\]\s*Ep|Epoch)\s+(\d+)/(\d+).*?val_mse=([0-9.eE+-]+)")
ST_RE = re.compile(r"===\s*\[.*?\]\s*STAGE\s*(\d+)")


def parse_log(path):
    stage, s1, s2 = 0, [], []
    for ln in open(os.path.join(HERE, path)):
        m = ST_RE.search(ln)
        if m:
            stage = int(m.group(1)); continue
        m = EP_RE.search(ln)
        if m:
            (s1 if stage == 1 else s2).append(float(m.group(3)))
    return np.array(s1), np.array(s2)


def match(gen, gt):
    c = np.linalg.norm(gen[:, None, :] - gt[None, :, :], axis=2)
    r, cc = sciopt.linear_sum_assignment(c)
    return dict(zip(cc.tolist(), r.tolist()))


# =============================== figure ===============================
fig = plt.figure(figsize=(27, 7.0))
gs = fig.add_gridspec(1, 4, width_ratios=[1.9, 0.7, 1.0, 1.0], wspace=0.30)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])
axD = fig.add_subplot(gs[0, 3])

# ---- A: training process ----
w1, w2 = parse_log(WARM_LOG)
r1, r2 = parse_log(SCR_LOG)


def draw_curve(ax, s1, s2, color, label):
    n1 = len(s1)
    ax.plot(np.arange(1, n1 + 1), s1, color=color, lw=2.6, alpha=0.9)
    ax.plot(np.arange(n1 + 1, n1 + len(s2) + 1), s2, color=color, lw=2.6, alpha=0.9, label=label)
    jb = int(np.argmin(s2))
    ax.plot([n1 + 1 + jb], [s2[jb]], "o", color=color, ms=10, mec="black", mew=1.4, zorder=5)
    return n1, float(s2[jb])


n1, bw = draw_curve(axA, w1, w2, C_WARM, "Warm-start")
_, bs = draw_curve(axA, r1, r2, C_SCR, "Scratch")
axA.set_yscale("log")
axA.axvline(n1 + 0.5, color="0.45", ls="--", lw=1.8)
ytop = axA.get_ylim()[1]
axA.text(n1 * 0.5, ytop, "with context", ha="center", va="top", fontsize=16, color="0.4", fontweight="bold")
axA.text(n1 + len(w2) * 0.5, ytop, "without context", ha="center", va="top", fontsize=16, color="0.4", fontweight="bold")
axA.annotate(f"best {bw:.3f}", xy=(axA.get_xlim()[1], bw), xytext=(-6, -4), textcoords="offset points",
             ha="right", va="top", fontsize=14, color=C_WARM, fontweight="bold")
axA.annotate(f"best {bs:.3f}", xy=(axA.get_xlim()[1], bs), xytext=(-6, 6), textcoords="offset points",
             ha="right", va="bottom", fontsize=14, color=C_SCR, fontweight="bold")
axA.set_xlabel("Training epoch", fontsize=20, fontweight="bold")
axA.set_ylabel("Validation MSE", fontsize=20, fontweight="bold")
axA.grid(alpha=0.3)
axA.legend(loc="upper right", fontsize=15, frameon=False, bbox_to_anchor=(0.99, 0.80))

# ---- B: direct relative L2 (no refine), warm vs scratch ----
sw = torch.load(WARM_STATS, map_location="cpu", weights_only=False)
ss = torch.load(SCR_STATS, map_location="cpu", weights_only=False)
kw, ks = np.asarray(sw["k"]).astype(int), np.asarray(ss["k"]).astype(int)
kc = list(set(kw.tolist()) & set(ks.tolist()))
dw = np.clip(np.asarray(sw["direct"])[np.isin(kw, kc)].astype(float), 1e-9, None)
ds = np.clip(np.asarray(ss["direct"])[np.isin(ks, kc)].astype(float), 1e-9, None)
bp = axB.boxplot([dw, ds], positions=[1, 2], widths=0.55, patch_artist=True, showfliers=False, whis=(5, 95))
for patch, col in zip(bp["boxes"], [C_WARM, C_SCR]):
    patch.set_facecolor(col); patch.set_alpha(0.65); patch.set_edgecolor("black"); patch.set_linewidth(2)
for m in bp["medians"]:
    m.set_color("black"); m.set_linewidth(3)
for el in ("whiskers", "caps"):
    for ln in bp[el]:
        ln.set_color("black"); ln.set_linewidth(2)
axB.set_yscale("log")
axB.set_xticks([1, 2]); axB.set_xticklabels(["Warm", "Scratch"])
axB.set_xlim(0.4, 2.6)
axB.set_ylabel("Direct relative $L^2$", fontsize=20, fontweight="bold")
axB.grid(axis="y", alpha=0.3)
axB.text(1, np.median(dw), f"{np.median(dw):.1e}", ha="center", va="bottom", fontsize=13, color=C_WARM, fontweight="bold")
axB.text(2, np.median(ds), f"{np.median(ds):.1e}", ha="center", va="bottom", fontsize=13, color=C_SCR, fontweight="bold")

# ---- C, D: error fields at the selected parameter/branch ----
GW = torch.load(WARM_GEN, map_location="cpu", weights_only=False)
GS = torch.load(SCR_GEN, map_location="cpu", weights_only=False)
COORD = GW["coord"].numpy(); ELEM = GW["elem"].numpy()
TRI = Triangulation(COORD[:, 0], COORD[:, 1], ELEM)
wr = {r["idx"]: r for r in GW["results"]}[SEL_IDX]
srr = {r["idx"]: r for r in GS["results"]}[SEL_IDX]
gt = wr["gt"].numpy().astype(float)
gw = wr["generated"].numpy().astype(float)
gsc = srr["generated"].numpy().astype(float)
mw = match(gw, gt); ms = match(gsc, gt)
gt_j = gt[SEL_J]
err_w = np.abs(gw[mw[SEL_J]] - gt_j)
err_s = np.abs(gsc[ms[SEL_J]] - gt_j)
rl_w = np.linalg.norm(gw[mw[SEL_J]] - gt_j) / np.linalg.norm(gt_j)
rl_s = np.linalg.norm(gsc[ms[SEL_J]] - gt_j) / np.linalg.norm(gt_j)
# error colour scale: light where error is ~0, DEEP blue where error is large
# (deliberately the opposite brightness convention from the rainbow/viridis solution
# fields, and on the paper's blue accent). Shared 0..vmax across both panels.
vmax = max(err_w.max(), err_s.max())
for ax, err, col, tag, rl in [(axC, err_w, C_WARM, "Warm-start", rl_w), (axD, err_s, C_SCR, "Scratch", rl_s)]:
    im = ax.tripcolor(TRI, err, shading="gouraud", cmap="Blues", vmin=0, vmax=vmax)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("0.5"); sp.set_linewidth(1.5)
    ax.set_title(tag, fontsize=20, fontweight="bold", color=col, pad=8)
    ax.text(0.03, 0.03, f"rel $L^2$ = {rl:.1e}", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=15, fontweight="bold", color="0.15",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.8))
cb = fig.colorbar(im, ax=[axC, axD], fraction=0.045, pad=0.02)
cb.set_label(r"$|u_{\mathrm{pred}}-u_{\mathrm{GT}}|$", fontsize=17, fontweight="bold")

# panel letters
for ax, L in [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")]:
    ax.text(-0.02, 1.10, L, transform=ax.transAxes, fontsize=28, fontweight="bold", va="top", ha="right")

save_fig(fig, os.path.join(HERE, "fig_ex2ex3_panel"))
print(f"saved -> fig_ex2ex3_panel.{{png,pdf}}   warm rel-L2={rl_w:.2e} scratch rel-L2={rl_s:.2e} vmax={vmax:.3f}")
