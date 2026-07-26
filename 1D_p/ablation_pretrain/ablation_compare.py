"""Pretrained-vs-random ablation comparison (Example 1 only), DIRECT predictions
(no Newton post-processing).  Produces fig_ablation.{pdf,png} and prints a table.

Panels:
  A  training curves: val_mse vs epoch (stage1 with-context | stage2 no-context),
     pretrained vs random.
  B  test-set DIRECT rel-L2 (per matched solution), pretrained vs random.
  C  a representative k=7 parameter: GT vs pretrained vs random solution fields.
"""
import os, re, numpy as np, torch
import scipy.optimize as sciopt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
PRE_C, RND_C, GT_C = "#0066CC", "#E8730C", "black"
LOG = {"pretrained": "logs/arm1_pretrained_45991.out", "random": "logs/arm2_random_45990.out"}
GEN = {"pretrained": "gen_arm1_pretrained.pt", "random": "gen_arm2_random.pt"}


def parse_curve(path):
    stage, s1, s2 = 0, [], []
    for ln in open(os.path.join(HERE, path)):
        if "STAGE 1" in ln: stage = 1
        if "STAGE 2" in ln: stage = 2
        m = re.search(r"Epoch\s+(\d+)/\d+.*val_mse=([\d.eE+-]+)", ln)
        if m:
            (s1 if stage == 1 else s2).append((int(m.group(1)), float(m.group(2))))
    return s1, s2


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def direct_stats(genpath, tol=0.1):
    """Per-solution direct rel-L2 (closest output per GT) and exact count-match rate."""
    g = torch.load(os.path.join(HERE, genpath), map_location="cpu", weights_only=False)
    rels, exact, npar = [], 0, 0
    rec7 = None
    for rec in g:
        gt = rec["gt"].numpy().astype(np.float64)
        gen = rec["generated"]
        npar += 1
        if gen is None or len(gen) == 0:
            continue
        gen = gen.numpy().astype(np.float64)
        cost = np.linalg.norm(gen[:, None, :] - gt[None, :, :], axis=2)
        r, c = sciopt.linear_sum_assignment(cost)
        per = [rel_l2(gen[i], gt[j]) for i, j in zip(r, c)]
        rels += per
        # exact count-match: every GT branch covered within tol by its closest output
        if len(per) == gt.shape[0] and max(per) < tol:
            exact += 1
        if rec7 is None and gt.shape[0] == 7:
            rec7 = {"p": float(rec["p"]), "gt": gt, "gen": gen,
                    "match": list(zip(r.tolist(), c.tolist()))}
    return np.array(rels), exact / max(npar, 1), rec7


# ---------- gather ----------
curves = {k: parse_curve(v) for k, v in LOG.items()}
have_gen = all(os.path.exists(os.path.join(HERE, v)) for v in GEN.values())
stats = {}
if have_gen:
    for k, v in GEN.items():
        stats[k] = direct_stats(v)

# ---------- table ----------
print("\n=== ABLATION: pretrained vs random-init (Example 1, DIRECT) ===")
print(f"{'metric':32s} {'pretrained':>14s} {'random-init':>14s}")
b_pre = min(v for _, v in curves["pretrained"][0] + curves["pretrained"][1])
b_rnd = min(v for _, v in curves["random"][0] + curves["random"][1])
print(f"{'best val_mse':32s} {b_pre:14.3e} {b_rnd:14.3e}")
if have_gen:
    rp, ep, _ = stats["pretrained"]; rr, er, _ = stats["random"]
    print(f"{'test direct rel-L2 (median)':32s} {np.median(rp):14.3e} {np.median(rr):14.3e}")
    print(f"{'test direct rel-L2 (mean)':32s} {rp.mean():14.3e} {rr.mean():14.3e}")
    print(f"{'exact count-match rate':32s} {ep:14.3f} {er:14.3f}")

# ---------- figure ----------
fig = plt.figure(figsize=(16, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.1], wspace=0.32)
plt.rcParams.update({"font.size": 14})

# A: training curves
axA = fig.add_subplot(gs[0, 0])
for k, col in [("pretrained", PRE_C), ("random", RND_C)]:
    s1, s2 = curves[k]
    e1 = [e for e, _ in s1]; v1 = [v for _, v in s1]
    off = (max(e1) if e1 else 0)
    e2 = [off + e for e, _ in s2]; v2 = [v for _, v in s2]
    lab = "pretrained" if k == "pretrained" else "random init"
    axA.plot(e1 + e2, v1 + v2, color=col, lw=2.6, label=lab)
    if e1:
        axA.axvline(off, color="0.6", ls=":", lw=1.4, zorder=0)
axA.set_yscale("log"); axA.set_xlabel("epoch (stage 1 | stage 2)")
axA.set_ylabel("validation MSE"); axA.legend(frameon=True)
axA.set_title("A  Training", loc="left", fontweight="bold")
axA.text(0.5, 0.93, "with-context", transform=axA.transAxes, ha="center", fontsize=11, color="0.4")
axA.text(0.88, 0.93, "no-context", transform=axA.transAxes, ha="center", fontsize=11, color="0.4")

# B: test direct rel-L2
axB = fig.add_subplot(gs[0, 1])
if have_gen:
    data = [stats["pretrained"][0], stats["random"][0]]
    bp = axB.boxplot(data, labels=["pretrained", "random"], showfliers=False,
                     patch_artist=True, widths=0.6)
    for patch, col in zip(bp["boxes"], [PRE_C, RND_C]):
        patch.set_facecolor(col); patch.set_alpha(0.55)
    axB.set_yscale("log")
else:
    axB.text(0.5, 0.5, "gen pending", ha="center", transform=axB.transAxes)
axB.set_ylabel("test direct rel-$L^2$")
axB.set_title("B  Test prediction (direct)", loc="left", fontweight="bold")

# C: representative k=7 solutions
axC = fig.add_subplot(gs[0, 2])
if have_gen and stats["pretrained"][2] is not None:
    r7 = stats["pretrained"][2]; r7r = stats["random"][2]
    x = np.linspace(0, 1, r7["gt"].shape[1])
    for i, j in r7["match"]:
        axC.plot(x, r7["gt"][j], color=GT_C, lw=2.6, zorder=1)
        axC.plot(x, r7["gen"][i], color=PRE_C, lw=1.8, ls=(0, (5, 2)), zorder=2)
    if r7r is not None:
        for i, j in r7r["match"]:
            axC.plot(x, r7r["gen"][i], color=RND_C, lw=1.6, ls=(0, (1, 1.6)), zorder=3)
    axC.set_xlabel("$x$"); axC.set_xlim(0, 1)
    axC.set_title(f"C  $p={r7['p']:.1f}$ ($k=7$)", loc="left", fontweight="bold")
    axC.legend(handles=[Line2D([0], [0], color=GT_C, lw=2.6, label="exact"),
                        Line2D([0], [0], color=PRE_C, lw=1.8, ls=(0, (5, 2)), label="pretrained"),
                        Line2D([0], [0], color=RND_C, lw=1.6, ls=(0, (1, 1.6)), label="random")],
               fontsize=11, loc="best")
else:
    axC.text(0.5, 0.5, "gen pending", ha="center", transform=axC.transAxes)

for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(HERE, f"fig_ablation.{ext}"), dpi=200 if ext == "png" else None,
                bbox_inches="tight")
plt.close(fig)
print("\nwrote ablation_pretrain/fig_ablation.{pdf,png}")
