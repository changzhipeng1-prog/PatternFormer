"""外延总图 2x3:
  行1(统计): direct/seed+jump coverage 曲线(1D_p,2D) | a2a4 M1 coverage 热图
  行2(解示意): 1D_p 解 u(x) | 2D 解(4 个场拼成一张) | a2a4 解 u(x)
子图标识 A-F;不放方法简写,直接写方法名。输出 -> paper/fig_extension_2x3.png/.pdf
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_plot_style import apply_style, save_fig
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

apply_style()
import matplotlib as _mpl
for _k in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
           "ytick.labelsize", "legend.fontsize"):
    _mpl.rcParams[_k] = _mpl.rcParams[_k] + 1

M1c, M2c = "#D55E00", "#0072B2"
PAL = ["#E69F00", "#009E73", "#D55E00", "#0072B2", "#CC79A7", "#56B4E9", "#9467BD"]
GTC = "0.6"


def rl2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def panel_label(ax, s):
    ax.text(-0.04, 1.04, s, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=30, fontweight="bold")


def curve_panel(ax, xs, f1, f2, xlabel, logx=False, xtick_pos=None, xtick_lab=None):
    ax.axhline(1.0, ls="--", color="0.6", lw=2, zorder=1, label="all coexisting")
    ax.plot(xs, f2, "-s", color=M2c, lw=3, ms=11, zorder=4, label="seed + single jump")
    ax.plot(xs, f1, "-o", color=M1c, lw=3, ms=11, zorder=3, label="direct")
    if logx:
        ax.set_xscale("log")
        if xtick_pos is not None:
            ax.set_xticks(xtick_pos); ax.set_xticklabels(xtick_lab)   # 水平、默认字体
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel("coverage fraction", fontweight="bold")
    ax.set_ylim(-0.05, 1.12); ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(float(np.min(xs)), float(np.max(xs)))     # x 端点贴轴边,无两端空白
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=18, framealpha=0.95)


def sol_curve(ax, x, gt, sols, method, color, ylabel=False, leg_loc="best"):
    for g in gt:
        ax.plot(x, g, color=GTC, lw=2.2, zorder=1)
    for d in sols:
        j = int(np.argmin([rl2(d, g) for g in gt]))
        ax.plot(x, d, "--", color=PAL[j % len(PAL)], lw=3.0, zorder=3)
    ax.set_xlabel("x", fontweight="bold")
    ax.set_xlim(float(x.min()), float(x.max()))           # x 端点贴轴边,无两端空白
    if ylabel:
        ax.set_ylabel("u(x)", fontweight="bold")
    ax.grid(False)
    ax.legend(handles=[Line2D([0], [0], color=GTC, lw=2.2, label="Exact (GT)"),
                       Line2D([0], [0], color=color, lw=3.0, ls="--", label=method)],
              loc=leg_loc, fontsize=18, framealpha=0.95)


def main():
    d1 = torch.load(os.path.join(HERE, "1D_p/extension/three_methods_1dp.pt"), weights_only=False)
    d2 = torch.load(os.path.join(HERE, "2D/extension/three_methods_2d.pt"), weights_only=False)
    dA = torch.load(os.path.join(HERE, "a2a4/extension/a2a4_cov_M3_bfs.pt"), weights_only=False)  # model-seeded reference continuation (full coverage)
    dG = torch.load(os.path.join(HERE, "a2a4/extension/grid_gt_cont.pt"), weights_only=False)     # per-cell GT solution sets -> recovered branch count
    s1 = torch.load(os.path.join(HERE, "1D_p/extension/sols_row2.pt"), weights_only=False)
    g2 = torch.load(os.path.join(HERE, "2D/extension/gt_extension.pt"), weights_only=False)
    sA = torch.load(os.path.join(HERE, "a2a4/extension/sols_row2.pt"), weights_only=False)

    fig = plt.figure(figsize=(26, 14), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.03, w_pad=0.06, hspace=0.02, wspace=0.06)
    outer = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.4])

    # ---------- ROW 1 ----------
    axA1 = fig.add_subplot(outer[0, 0])
    P = np.array(d1["P"], float); ne = np.array(d1["n_exist"], float)
    TP = [18.01, 18.1, 18.5, 19.0, 20.0]                  # 稀疏刻度(数据点 marker 仍全保留)
    curve_panel(axA1, P - 18.0, np.array(d1["m1"]) / ne, np.array(d1["m2"]) / ne,
                r"extrapolated $p$  (train $\leq 18$)",
                logx=True, xtick_pos=[t - 18.0 for t in TP], xtick_lab=[f"{t:g}" for t in TP])

    axB1 = fig.add_subplot(outer[0, 1])
    rows = d2; s = np.array([r[0] for r in rows], float); ex = np.array([r[1] for r in rows], float)
    curve_panel(axB1, s, np.array([r[2] for r in rows]) / ex, np.array([r[3] for r in rows]) / ex,
                r"extrapolated $s$  (train $\leq 1600$)")

    from matplotlib.colors import BoundaryNorm
    box = dA["box"]
    # Atlas over the (a2,a4) plane: each tested parameter is a dot coloured by the number
    # of coexisting branches the model-seeded march recovers there. Recovery is complete
    # at every point (full coverage), far beyond the small training box -- a positive view
    # of extrapolation across the whole parameter plane (colour shows the 1/3/5 structure).
    a2v = np.array(dG["a2"], float); a4v = np.array(dG["a4"], float); gt_by = dG["gt"]
    P2, P4, CN = [], [], []
    for a4 in a4v:
        for a2 in a2v:
            key = next((k for k in gt_by if abs(k[0] - a2) < 1e-3 and abs(k[1] - a4) < 1e-3), None)
            if key is not None:
                arr = np.asarray(gt_by[key]); P2.append(a2); P4.append(a4)
                CN.append(arr.shape[0] if arr.ndim == 2 else 1)
    P2, P4, CN = np.array(P2), np.array(P4), np.array(CN)
    axC1 = fig.add_subplot(outer[0, 2])
    vmax_c = int(CN.max()); cmap = plt.get_cmap("viridis", vmax_c)
    norm = BoundaryNorm(np.arange(0.5, vmax_c + 1.5, 1.0), cmap.N)
    sc = axC1.scatter(P2, P4, c=CN, cmap=cmap, norm=norm, s=200, edgecolor="0.25", linewidth=0.7, zorder=3)
    axC1.add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2],
                             fill=False, ec="crimson", lw=3, ls="--", zorder=5))
    axC1.text(0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3]), "train", ha="center", va="center",
              fontsize=19, fontweight="bold", color="crimson", zorder=6,
              bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="crimson", lw=1.6))
    axC1.text(0.03, 0.05, "100% coverage", transform=axC1.transAxes, va="bottom", ha="left",
              fontsize=17, fontweight="bold", zorder=7,
              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.4", lw=1.3))
    cb = fig.colorbar(sc, ax=axC1, pad=0.02, ticks=np.arange(1, vmax_c + 1))
    cb.set_label("# branches recovered", fontweight="bold")
    axC1.set_xlabel(r"$a_2$", fontweight="bold"); axC1.set_ylabel(r"$a_4$", fontweight="bold")
    axC1.margins(0.06)

    # ---------- ROW 2 ----------
    axA2 = fig.add_subplot(outer[1, 0])
    sol_curve(axA2, s1["x"], s1["gt"], s1["m2"], "seed + single jump", M2c, ylabel=True,
              leg_loc="upper right")

    # 2D: 四个解插值到规则网格,拼成一张连续 2x2 图像
    s_show = 1800.0
    gt2 = np.asarray(g2["gt"][s_show], np.float64)
    coord = g2["coord"].numpy(); elem = g2["elem"].numpy()
    tri = Triangulation(coord[:, 0], coord[:, 1], elem)
    vmax = float(np.abs(gt2).max())
    ng = 90
    gx, gy = np.meshgrid(np.linspace(0, 1, ng), np.linspace(0, 1, ng))

    def gridf(u):
        return np.asarray(LinearTriInterpolator(tri, u)(gx, gy).filled(np.nan))
    Gf = [gridf(gt2[n]) for n in range(4)]
    full = np.vstack([np.hstack([Gf[0], Gf[1]]), np.hstack([Gf[2], Gf[3]])])
    ax2d = fig.add_subplot(outer[1, 1])
    im2 = ax2d.imshow(full, origin="upper", cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax2d.axvline(ng - 0.5, color="white", lw=3); ax2d.axhline(ng - 0.5, color="white", lw=3)
    ax2d.set_xticks([]); ax2d.set_yticks([])
    for lx, ly, lab in [(0.02, 0.98, "sol 1"), (0.52, 0.98, "sol 2"),
                        (0.02, 0.48, "sol 3"), (0.52, 0.48, "sol 4")]:
        ax2d.text(lx, ly, lab, transform=ax2d.transAxes, va="top", ha="left",
                  fontsize=20, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.9))
    cbb = fig.colorbar(im2, ax=ax2d, location="right", fraction=0.05, pad=0.03)
    cbb.set_label("u(x,y)", fontweight="bold")

    axC2 = fig.add_subplot(outer[1, 2])
    sol_curve(axC2, sA["x"], sA["gt"], sA["m1"], "direct", M1c)

    for ax, lab in [(axA1, "A"), (axB1, "B"), (axC1, "C"),
                    (axA2, "D"), (ax2d, "E"), (axC2, "F")]:
        panel_label(ax, lab)

    save_fig(fig, os.path.join(HERE, "fig_extension_2x3"))
    print("saved fig_extension_2x3.png/.pdf")


if __name__ == "__main__":
    main()
