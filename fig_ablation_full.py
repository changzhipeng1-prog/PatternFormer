"""Combined backbone-initialization ablation figure (single 2-row figure).

Row 1 (A-D): Example 1 (1D single-parameter) -- pretrained LLM vs untrained LLM.
   A training val-MSE | B test rel-L2 | C the seven branches at a k=7 p | D zoom of C.
Row 2 (E-H): Gray-Scott -- untrained / pretrained / warm-start LLM (fixed-K, deterministic).
   E #sol & coverage bars | F,G,H per-parameter #distinct over (rho,mu).

Output: paper/fig_ablation_full.{pdf,png}
"""
import os, re, sys, json, numpy as np, torch
import scipy.optimize as sciopt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.tri import Triangulation
sys.path.insert(0, "/home/zfc5231/work/BBBB_qwen_pde_branch/paper")
from test_plot_style import apply_style
apply_style()

ROOT = "/home/zfc5231/work/BBBB_qwen_pde_branch/paper"
ABL1D = os.path.join(ROOT, "1D_p/ablation_pretrain")
PRE_C, RND_C, GT_C = "#0066CC", "#E8730C", "0.55"
C_SOL, C_COV = "#3C6E9C", "#E0892B"

# ---------------- 1D-p data ----------------
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

# ---------------- GS data ----------------
TIERS = ["random", "pretrained", "stop"]
MLAB = {"random": "untrained LLM", "pretrained": "pretrained LLM", "stop": "warm-start"}
XLAB = {"random": "untrained\nLLM", "pretrained": "pretrained\nLLM", "stop": "warm-start"}
JS = {"random": "GS/L1/ablation/eval/arm2_random_det.json",
      "pretrained": "GS/L1/ablation/eval/arm1_pretrained_det.json",
      "stop": "GS/L1/results/ord_eval_L1.json"}
dj = {k: json.load(open(os.path.join(ROOT, v))) for k, v in JS.items()}
nsol = {k: dj[k]["det_distinct_mean"] for k in TIERS}
cov = {k: dj[k]["coverage_mean"] for k in TIERS}
MAP = np.load(os.path.join(ROOT, "gs_map_data.npz"))
rho, mu = MAP["rho"], MAP["mu"]
val = {k: MAP[k] for k in TIERS}
VMAX = 15

# ---------------- Ex2->Ex3 cross-equation data (middle row E-H) ----------------
# Colours track the initialization type consistently with row 1: pretrained LLM is
# the same blue as A-D, warm-start gets a distinct green (untrained LLM = orange only
# appears in row 1). So each method reads as one colour across the whole figure.
M_WARM, M_SCR = "#009E73", "#0066CC"
SEL_IDX, SEL_S, SEL_J = 873, 735.0, 2           # test param/branch with the clearest contrast
EX_WARM_LOG = os.path.join(ROOT, "2D/train_log/2d_2stage_45773.out")
EX_SCR_LOG = os.path.join(ROOT, "2D/code/logs/ex3_scratch_46112.out")
EX_WARM_STATS = os.path.join(ROOT, "2D/test/stats.pt")
EX_SCR_STATS = os.path.join(ROOT, "2D/test/scratch/stats.pt")
EX_WARM_GEN = os.path.join(ROOT, "2D/test/generated_solutions.pt")
EX_SCR_GEN = os.path.join(ROOT, "2D/test/scratch/generated_solutions.pt")


def parse_curve_abs(path):
    """validation MSE per epoch, split by STAGE 1 / STAGE 2 (absolute path)."""
    stage, s1, s2 = 0, [], []
    for ln in open(path):
        if "STAGE 1" in ln:
            stage = 1
        if "STAGE 2" in ln:
            stage = 2
        m = re.search(r"(?:\[phase2\]\s*Ep|Epoch)\s+(\d+)/\d+.*val_mse=([\d.eE+-]+)", ln)
        if m:
            (s1 if stage == 1 else s2).append(float(m.group(2)))
    return s1, s2


def match_idx(gen, gt):
    c = np.linalg.norm(gen[:, None, :] - gt[None, :, :], axis=2)
    r, cc = sciopt.linear_sum_assignment(c)
    return dict(zip(cc.tolist(), r.tolist()))

# ---------------- figure (3 rows) ----------------
fig = plt.figure(figsize=(24, 19.0))
outer = fig.add_gridspec(3, 1, hspace=0.50)
g1 = outer[0].subgridspec(1, 4, wspace=0.36)
gM = outer[1].subgridspec(1, 4, wspace=0.36)   # mirror row 1 (A-D): four equal columns
g2 = outer[2].subgridspec(1, 5, width_ratios=[1.7, 0.34, 1.0, 1.0, 1.0], wspace=0.16)
axA = fig.add_subplot(g1[0, 0]); axB = fig.add_subplot(g1[0, 1])
axC = fig.add_subplot(g1[0, 2]); axZ = fig.add_subplot(g1[0, 3])
axMA = fig.add_subplot(gM[0, 0]); axMB = fig.add_subplot(gM[0, 1])
axMC = fig.add_subplot(gM[0, 2]); axMD = fig.add_subplot(gM[0, 3])
axE = fig.add_subplot(g2[0, 0]); axEr = axE.twinx()
mapax = [fig.add_subplot(g2[0, 2 + i]) for i in range(3)]

# ---- A: training ----
for k, col, ls, lw in [("pretrained", PRE_C, "-", 3.0), ("random", RND_C, (0, (4, 2)), 2.2)]:
    s1, s2 = curves[k]
    e1 = [e for e, _ in s1]; v1 = [v for _, v in s1]
    off = max(e1) if e1 else 0
    e2 = [off + e for e, _ in s2]; v2 = [v for _, v in s2]
    axA.plot(e1 + e2, v1 + v2, color=col, lw=lw, ls=ls)
    if e1: axA.axvline(off, color="0.6", ls=":", lw=1.6, zorder=0)
axA.set_yscale("log"); axA.set_xlabel("epoch (stage 1 | stage 2)", fontweight="bold")
axA.set_ylabel("validation MSE", fontweight="bold")

# ---- B: test rel-L2 ----
bp = axB.boxplot([stats["pretrained"][0], stats["random"][0]],
                 labels=["pretrained\nLLM", "untrained\nLLM"], showfliers=False,
                 patch_artist=True, widths=0.6)
for patch, col in zip(bp["boxes"], [PRE_C, RND_C]):
    patch.set_facecolor(col); patch.set_alpha(0.55)
axB.set_yscale("log"); axB.set_ylabel("test direct rel-$L^2$", fontweight="bold")

# ---- C + D(zoom): k=7 solutions ----
r7 = stats["pretrained"][1]; r7r = stats["random"][1]
x = np.linspace(0, 1, r7["gt"].shape[1])
gtmap = {j: i for i, j in r7["match"]}
rmap = {j: i for i, j in r7r["match"]} if r7r is not None else {}
def draw_C(ax, scale=1.0):
    for j in range(r7["gt"].shape[0]):
        ax.plot(x, r7["gt"][j], color=GT_C, lw=5.0 * scale, zorder=1)
    for j, i in gtmap.items():
        ax.plot(x, r7["gen"][i], color=PRE_C, lw=4.0 * scale, zorder=2)
    for j, i in rmap.items():
        ax.plot(x, r7r["gen"][i], color=RND_C, lw=2.0 * scale, ls=(0, (4, 2)), zorder=3)
draw_C(axC)
axC.set_xlabel("$x$", fontweight="bold"); axC.set_ylabel("$u$", fontweight="bold"); axC.set_xlim(0, 1)
zx0, zx1, zy0, zy1 = 0.38, 0.62, 1.8, 4.2
axC.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec="0.35", lw=1.6, zorder=6))
draw_C(axZ, scale=1.25)
axZ.set_xlim(zx0, zx1); axZ.set_ylim(zy0, zy1); axZ.set_xticks([]); axZ.set_yticks([])
for s in axZ.spines.values(): s.set_color("0.35"); s.set_linewidth(1.8)
axZ.set_xlabel("zoom", fontweight="bold", fontsize=18)
con = ConnectionPatch(xyA=(zx1, 0.5 * (zy0 + zy1)), coordsA=axC.transData,
                      xyB=(0.0, 0.5), coordsB=axZ.transAxes,
                      arrowstyle="-|>", lw=2.2, color="0.35", mutation_scale=24)
fig.add_artist(con)

# ---- E: GS bars (#sol left, coverage right) ----
xx = np.arange(3); w = 0.38
b1 = axE.bar(xx - w / 2, [nsol[k] for k in TIERS], width=w, color=C_SOL, label="# solutions")
b2 = axEr.bar(xx + w / 2, [cov[k] for k in TIERS], width=w, color=C_COV, label="coverage")
for i, k in enumerate(TIERS):
    axE.text(i - w / 2, nsol[k] + 0.1, f"{nsol[k]:.2f}", ha="center", va="bottom", fontsize=17, fontweight="bold", color=C_SOL)
    axEr.text(i + w / 2, cov[k] + 0.004, f"{cov[k]:.2f}", ha="center", va="bottom", fontsize=17, fontweight="bold", color=C_COV)
axE.set_xticks(xx); axE.set_xticklabels([XLAB[k] for k in TIERS], fontsize=19)
axE.set_ylabel("# distinct solutions / param", fontsize=21, fontweight="bold", color=C_SOL)
axEr.set_ylabel("coverage", fontsize=21, fontweight="bold", color=C_COV)
axE.tick_params(axis="y", labelcolor=C_SOL, labelsize=19); axEr.tick_params(axis="y", labelcolor=C_COV, labelsize=19)
axE.set_ylim(0, max(nsol.values()) * 1.2); axEr.set_ylim(0, max(cov.values()) * 1.2)
axE.legend(handles=[b1, b2], fontsize=18, loc="upper left")

# ---- F,G,H: GS maps ----
sc = None
for i, k in enumerate(TIERS):
    ax = mapax[i]
    sc = ax.scatter(rho, mu, c=val[k], s=26, cmap="YlOrRd", vmin=0, vmax=VMAX, edgecolor="0.5", linewidth=0.2)
    ax.text(0.05, 0.05, f"mean {val[k].mean():.1f}\nmax  {int(val[k].max())}", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left", linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))
    ax.set_title(MLAB[k], fontsize=15, fontweight="bold")
    ax.set_xlabel(r"$\rho$", fontweight="bold")
    if i == 0:
        ax.set_ylabel(r"$\mu$", fontweight="bold")
    else:
        ax.tick_params(labelleft=False)
    ax.yaxis.set_major_locator(MultipleLocator(0.01)); ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_axisbelow(True)

# ---- middle row (E-H): Ex2 -> Ex3 cross-equation warm-start vs scratch ----
mw1, mw2 = parse_curve_abs(EX_WARM_LOG)
mr1, mr2 = parse_curve_abs(EX_SCR_LOG)


def draw_run(ax, s1, s2, col, label):
    n1 = len(s1)
    ax.plot(range(1, n1 + 1), s1, color=col, lw=2.8, alpha=0.9)
    ax.plot(range(n1 + 1, n1 + len(s2) + 1), s2, color=col, lw=2.8, alpha=0.9, label=label)
    return n1


# E: training process (styled like A -- two curves, stage divider, no annotations)
n1 = draw_run(axMA, mw1, mw2, M_WARM, "warm-start")
draw_run(axMA, mr1, mr2, M_SCR, "pretrained LLM")
axMA.set_yscale("log"); axMA.grid(alpha=0.3)
axMA.axvline(n1, color="0.6", ls=":", lw=1.6, zorder=0)
axMA.set_xlabel("epoch (stage 1 | stage 2)", fontweight="bold"); axMA.set_ylabel("validation MSE", fontweight="bold")
axMA.legend(loc="upper right", fontsize=20, frameon=False)

# F: direct relative L2 (no refinement), warm vs scratch
sw = torch.load(EX_WARM_STATS, map_location="cpu", weights_only=False)
ss = torch.load(EX_SCR_STATS, map_location="cpu", weights_only=False)
kw = np.asarray(sw["k"]).astype(int); ks = np.asarray(ss["k"]).astype(int)
kc = list(set(kw.tolist()) & set(ks.tolist()))
dw = np.clip(np.asarray(sw["direct"])[np.isin(kw, kc)].astype(float), 1e-9, None)
ds = np.clip(np.asarray(ss["direct"])[np.isin(ks, kc)].astype(float), 1e-9, None)
bpm = axMB.boxplot([dw, ds], positions=[1, 2], widths=0.6, patch_artist=True, showfliers=False, whis=(5, 95))
for patch, col in zip(bpm["boxes"], [M_WARM, M_SCR]):
    patch.set_facecolor(col); patch.set_alpha(0.65); patch.set_edgecolor("black"); patch.set_linewidth(2)
for m in bpm["medians"]:
    m.set_color("black"); m.set_linewidth(3)
for el in ("whiskers", "caps"):
    for ln in bpm[el]:
        ln.set_color("black"); ln.set_linewidth(2)
axMB.set_yscale("log"); axMB.set_xticks([1, 2]); axMB.set_xticklabels(["warm-\nstart", "pretrained\nLLM"], fontsize=19)
axMB.set_xlim(0.4, 2.6); axMB.set_ylabel("test direct rel-$L^2$", fontweight="bold"); axMB.grid(axis="y", alpha=0.3)

# G, H: error fields |pred - GT| at the selected parameter/branch, shared 0..vmax
GW = torch.load(EX_WARM_GEN, map_location="cpu", weights_only=False)
GSg = torch.load(EX_SCR_GEN, map_location="cpu", weights_only=False)
COORD = GW["coord"].numpy(); ELEM = GW["elem"].numpy(); TRI = Triangulation(COORD[:, 0], COORD[:, 1], ELEM)
wr = {r["idx"]: r for r in GW["results"]}[SEL_IDX]
sr = {r["idx"]: r for r in GSg["results"]}[SEL_IDX]
gt = wr["gt"].numpy().astype(float); gw = wr["generated"].numpy().astype(float); gsc = sr["generated"].numpy().astype(float)
mmw = match_idx(gw, gt); mms = match_idx(gsc, gt); gtj = gt[SEL_J]
ew = np.abs(gw[mmw[SEL_J]] - gtj); es = np.abs(gsc[mms[SEL_J]] - gtj)
rlw = np.linalg.norm(gw[mmw[SEL_J]] - gtj) / np.linalg.norm(gtj)
rls = np.linalg.norm(gsc[mms[SEL_J]] - gtj) / np.linalg.norm(gtj)
evmax = max(ew.max(), es.max())
imM = None
for ax, err, col, tag, rl in [(axMC, ew, M_WARM, "warm-start", rlw), (axMD, es, M_SCR, "pretrained LLM", rls)]:
    imM = ax.tripcolor(TRI, err, shading="gouraud", cmap="Blues", vmin=0, vmax=evmax)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("$x$", fontweight="bold")
    ax.set_ylabel("$y$", fontweight="bold")
    ax.set_title(tag, fontsize=21, fontweight="bold", color=col, pad=8)
    ax.text(0.04, 0.04, f"rel $L^2$ = {rl:.1e}", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=17, fontweight="bold", color="0.15",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.85))

fig.canvas.draw()
# colorbar for the maps
pH = mapax[-1].get_position()
cax = fig.add_axes([pH.x1 + 0.008, pH.y0, 0.008, pH.height])
cb = fig.colorbar(sc, cax=cax); cb.set_label("# solutions", fontweight="bold", fontsize=13)

# middle-row error colorbar (shared by G, H)
pD = axMD.get_position()
caxM = fig.add_axes([pD.x1 + 0.012, pD.y0, 0.010, pD.height])
cbM = fig.colorbar(imM, cax=caxM)
cbM.set_label(r"$|u_{\mathrm{pred}}-u_{\mathrm{GT}}|$", fontweight="bold", fontsize=22)
cbM.ax.tick_params(labelsize=16)

# 1D-p line legend, centred just above row 1
p_top = max(ax.get_position().y1 for ax in [axA, axB, axC, axZ])
fig.legend(handles=[Line2D([0], [0], color=GT_C, lw=5.0, label="exact"),
                    Line2D([0], [0], color=PRE_C, lw=4.0, label="pretrained LLM"),
                    Line2D([0], [0], color=RND_C, lw=2.5, ls=(0, (4, 2)), label="untrained LLM")],
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, p_top + 0.015), frameon=False, fontsize=22)

# panel letters A-L at each panel's top-left corner (row1 1D, row2 Ex2->Ex3, row3 GS)
letters = [(axA, "A"), (axB, "B"), (axC, "C"), (axZ, "D"),
           (axMA, "E"), (axMB, "F"), (axMC, "G"), (axMD, "H"),
           (axE, "I"), (mapax[0], "J"), (mapax[1], "K"), (mapax[2], "L")]
for ax, lt in letters:
    p = ax.get_position()
    fig.text(p.x0 - 0.006, p.y1 + 0.008, lt, fontsize=26, fontweight="bold", va="bottom", ha="right")

out = os.path.join(ROOT, "fig_ablation_full")
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=140, bbox_inches="tight")
print("wrote", out + ".pdf/.png")
