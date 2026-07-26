"""Fig. 5 -- "Is it the network or the solver?"  Cross-project evidence that the
model finds the solutions and Newton refinement only sets their precision.
Nature-style (test_plot_style.apply_style): large bold fonts, thick lines.

Layout (2x2, columns aligned):
    row 1:  A = 1D one-parameter overlay     C = 2D before/after triptych
    row 2:  B = 1D two-parameter overlay      D = tolerance dial
Colours match Fig. 1 / the other figures: exact=black, direct=orange, refined=blue.

Run each test/_who_npz.py first, then:  python fig5_who_solves.py
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.tri import Triangulation
HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE)
from test_plot_style import apply_style, CMAP_FIELD
apply_style()

# colours unified with Fig. 1 (blue + orange family) -- no green
GTC, DIRC, REFC = "black", "#E8730C", "#0066CC"          # exact / direct / after Newton
# dial: colourblind-safe (Okabe-Ito), all SOLID lines (distinguished by colour only)
DIAL = {"1D_p": ("#0072B2", "-"), "a2a4": ("#D55E00", "-"), "2D": ("#009E73", "-")}
EXLAB = {"1D_p": "Example 1", "a2a4": "Example 2", "2D": "Example 3"}


def load(p):
    return dict(np.load(os.path.join(HERE, p, "test", "who_npz.npz"), allow_pickle=True))


# ---------- 1D overlay: direct vs refined vs GT ----------
def draw_overlay_1d(ax, z, legend=False):
    x, gt, direct, refined = z["x"], z["gt"], z["direct"], z["refined"]
    for j in range(gt.shape[0]):
        ax.plot(x, gt[j], color=GTC, lw=3.4, zorder=1, solid_capstyle="round")
        ax.plot(x, direct[j], color=DIRC, lw=3.0, ls=(0, (5, 2)), zorder=2)
        ax.plot(x, refined[j], color=REFC, lw=2.2, ls=(0, (1, 1.7)), zorder=3)
    ax.set_xlabel("$x$"); ax.set_xlim(0, 1)
    ax.set_title(f"{EXLAB[str(z['key'])]}\n{str(z['param_str'])}", fontsize=20, pad=10)
    ax.text(0.04, 0.04, f"direct rel-$L^2$ {float(z['med_direct']):.0e} "
            r"$\rightarrow$ " + f"{float(z['med_post']):.0e}", transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=1.5, alpha=0.92))
    if legend:
        ax.legend(handles=[Line2D([0], [0], color=GTC, lw=3.4, label="exact"),
                           Line2D([0], [0], color=DIRC, lw=3.0, ls=(0, (5, 2)), label="direct"),
                           Line2D([0], [0], color=REFC, lw=2.4, ls=(0, (1, 1.7)), label="after Newton")],
                  loc="upper right", fontsize=18, frameon=True, framealpha=0.95)


# ---------- 2D triptych: GT | direct | refined ----------
def draw_triptych_2d(spec, fig, z):
    tri = Triangulation(z["coord"][:, 0], z["coord"][:, 1], z["elem"])
    j = int(np.argmax([np.linalg.norm(g) for g in z["gt"]]))
    fields = [z["gt"][j], z["direct"][j], z["refined"][j]]
    names, cols = ["exact", "direct", "after Newton"], [GTC, DIRC, REFC]
    subs = [None, f"rel-$L^2$ {float(z['direct_rel'][j]):.0e}", f"{float(z['post_rel'][j]):.0e}"]
    vmin = min(f.min() for f in fields); vmax = max(f.max() for f in fields)
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.08)
    xlo, xhi = float(tri.x.min()), float(tri.x.max())
    ylo, yhi = float(tri.y.min()), float(tri.y.max())
    for c, (f, nm, cc, sub) in enumerate(zip(fields, names, cols, subs)):
        ax = fig.add_subplot(gs[0, c])
        ax.tripcolor(tri, f, shading="gouraud", cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)   # tight: field touches the frame
        ax.margins(0); ax.set_aspect("equal")
        ax.set_title(nm, fontsize=21, color=cc, pad=6)
        if sub:
            ax.set_xlabel(sub, fontsize=17)
    # problem header (with concrete s) centred above the triptych
    bb = spec.get_position(fig)
    fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.012, f"{EXLAB[str(z['key'])]},  {str(z['param_str'])}",
             ha="center", va="bottom", fontsize=20, fontweight="bold")
    return gs


# ---------- tolerance dial ----------
def draw_dial(ax, zs):
    for key, z in zs.items():
        col, lsty = DIAL[key]
        ax.plot(z["tol"], 100 * z["frac"], ls=lsty, lw=4.5, color=col, label=EXLAB[key],
                solid_capstyle="round")
    ax.set_xscale("log"); ax.invert_xaxis()
    # light decade bands marking the precision regime (~1, 2, 3 digits)
    for lo, hi, c in [(1e-1, 1e0, "#DCEBF7"), (1e-2, 1e-1, "#DFF2DC"), (1e-3, 1e-2, "#FCE8D2")]:
        ax.axvspan(lo, hi, color=c, alpha=0.7, zorder=0, lw=0)
    tols = np.concatenate([np.asarray(z["tol"], float).ravel() for z in zs.values()])
    ax.set_xlim(float(tols.max()), float(tols.min()))   # tight: no side padding
    ax.axhspan(0, 10, color="0.45", alpha=0.16, zorder=0.5)
    ax.text(0.97, 0.055, "network alone suffices", transform=ax.transAxes, fontsize=17,
            color="0.3", ha="right", va="center")
    ax.set_xlabel(r"required accuracy (rel-$L^2$ tolerance) $\rightarrow$ tighter")
    ax.set_ylabel("% needing refinement")
    ax.set_ylim(-3, 105); ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=19, frameon=True, framealpha=0.92,
              title=None, title_fontsize=17, ncol=1, handlelength=2.4,
              columnspacing=1.1)


def build_combined(zs):
    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.45], height_ratios=[1, 1],
                          wspace=0.22, hspace=0.42)
    axA = fig.add_subplot(gs[0, 0]); draw_overlay_1d(axA, zs["1D_p"], legend=True)
    axB = fig.add_subplot(gs[1, 0]); draw_overlay_1d(axB, zs["a2a4"])
    draw_triptych_2d(gs[0, 1], fig, zs["2D"])                       # C
    draw_dial(fig.add_subplot(gs[1, 1]), zs)                        # D
    for spec, lt in [(gs[0, 0], "A"), (gs[0, 1], "C"), (gs[1, 0], "B"), (gs[1, 1], "D")]:
        bb = spec.get_position(fig)
        fig.text(bb.x0 - 0.010, bb.y1 + 0.018, lt, fontsize=30, fontweight="bold", va="bottom")
    for ext, dpi in [("pdf", None), ("png", 150)]:
        fig.savefig(os.path.join(HERE, f"fig5_who_solves.{ext}"), dpi=dpi, bbox_inches="tight")
    dst = os.path.join(HERE, "..", "manuscript_NMI", "Figures", "fig5_who_solves.pdf")
    if os.path.isdir(os.path.dirname(dst)):
        fig.savefig(dst, bbox_inches="tight")
    plt.close(fig); print("saved fig5_who_solves.pdf/.png (+ manuscript Figures/)")


def build_standalone(key, z):
    fig = plt.figure(figsize=(15, 6.4))
    g = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.30)
    if str(z["kind"]) == "1d":
        draw_overlay_1d(fig.add_subplot(g[0]), z, legend=True)
    else:
        draw_triptych_2d(g[0], fig, z)
    draw_dial(fig.add_subplot(g[1]), {key: z})
    out = os.path.join(HERE, key, "test", "fig_who_solves")
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"saved {key}/test/fig_who_solves.pdf")


if __name__ == "__main__":
    zs = {k: load(k) for k in ["1D_p", "a2a4", "2D"]}
    for k in zs:
        zs[k]["key"] = k
    build_combined(zs)
    for k in zs:
        build_standalone(k, zs[k])
