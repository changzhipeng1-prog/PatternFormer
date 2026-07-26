"""Per-setup composite figure (L=1 -> combined_L1, L=2 -> combined_L2), identical layout:
  row1: [ Fig1 max-distinct solution montage | Fig5 (rho,mu) solution atlas ]  (equal width)
  row2: Fig2 four-method (rho,mu) distinction maps (random / STOP / fixed-K / +noise)
  row3: [ noise-sweep curve | matched-vs-beyond bars ]
Reuses the standalone scripts' logic (make_fig1.panel, _gridload, extrap atlas).
  python make_combined.py            # writes combined_L1 and combined_L2
"""
import os, sys, json, glob
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
import make_fig1
import make_fig5_atlas as F5
from _gridload import load_m, test_set, DIFF, METHODS, EXAMPLE_TI
from test_plot_style import apply_style, COLORS
apply_style()

VMAX = 15
BARLAB = [("random", "trad.\n(random)"), ("stop", "STOP"), ("det", "fixed-K"), ("noise", "fixed-K\n+noise")]


# ---------- row1 left: Fig1 montage ----------
def draw_fig1(fig, spec, setup):
    A, is_gt, param = make_fig1.load(setup, EXAMPLE_TI[setup], "_noise")
    make_fig1.panel(fig, spec, A, is_gt, 6, setup, param, show_diff=False, aspect="auto")


# ---------- row1 right: Fig5 atlas ----------
def draw_atlas(fig, spec, setup):
    p = F5.PATHS[setup]
    lk = torch.load(f"{HERE}/{p['look']}", weights_only=False)
    params, sols = lk["p_values"], lk["solutions_by_p"]
    recs = []
    for f in glob.glob(f"{HERE}/{setup}/results/full_chunks_*/chunk_*.json"):
        recs += json.load(open(f)).get("recs", [])
    detm = {r["ti"]: r["distinct"] for r in recs}
    tis = sorted(detm.keys())
    rho = np.array([float(params[ti, 0]) for ti in tis]); mu = np.array([float(params[ti, 1]) for ti in tis])
    det = np.array([detm[ti] for ti in tis])
    r0, r1 = rho.min(), rho.max(); u0, u1 = mu.min(), mu.max()
    dwr = (r1 - r0) / F5.NBr_in; dwu = (u1 - u0) / F5.NBu_in
    re = r0 + (np.arange(F5.NBr_in + F5.ELr + F5.ERr + 1) - F5.ELr) * dwr
    me = u0 + (np.arange(F5.NBu_in + F5.ELu + F5.EUu + 1) - F5.ELu) * dwu
    NBc, NBrow = len(re) - 1, len(me) - 1

    def colrow(rr, uu):
        return (int(np.clip(np.searchsorted(re, rr) - 1, 0, NBc - 1)),
                int(np.clip(np.searchsorted(me, uu) - 1, 0, NBrow - 1)))
    cell = {}
    for k, ti in enumerate(tis):
        ij = colrow(rho[k], mu[k])
        if ij not in cell or det[k] > detm[cell[ij]]:
            cell[ij] = ti
    extrap_cell = {}; all_ood = []
    try:
        ef = np.load(f"{HERE}/{p['ef']}")
        for rr, uu, Af in zip(ef["rho"], ef["mu"], ef["A"]):
            extrap_cell[colrow(float(rr), float(uu))] = Af
        for gr in json.load(open(f"{HERE}/{p['eg']}")).get("rows", []):
            if gr.get("extrap"):
                all_ood.append(gr.get("n_solutions", 0))
    except FileNotFoundError:
        pass
    gs = GridSpecFromSubplotSpec(NBrow, NBc, subplot_spec=spec, hspace=0.05, wspace=0.05)
    # background axes spanning the whole atlas, only to carry the rho / mu axis names
    axlab = fig.add_subplot(spec, zorder=-1)
    axlab.set_xticks([]); axlab.set_yticks([]); axlab.patch.set_alpha(0)
    for sp in axlab.spines.values():
        sp.set_visible(False)
    axlab.set_xlabel(r"$\rho$", fontweight="bold", fontsize=20, labelpad=20)
    axlab.set_ylabel(r"$\mu$", fontweight="bold", fontsize=20, labelpad=26)
    # cell of the montage parameter (panel B), to highlight and link with an arrow
    mti = EXAMPLE_TI[setup]; tgt = colrow(float(params[mti, 0]), float(params[mti, 1])); hl_ax = None
    bfield = make_fig1.load(setup, mti, "_noise")[0][0]   # B's first solution tile, dropped into the linked cell
    for jj in range(NBrow):
        for ii in range(NBc):
            ax = fig.add_subplot(gs[NBrow - 1 - jj, ii]); ax.set_xticks([]); ax.set_yticks([])
            if (ii, jj) in extrap_cell:                       # OOD point: orange frame only (no per-point count)
                ax.imshow(extrap_cell[(ii, jj)], cmap="viridis", aspect="auto")
                for sp in ax.spines.values():
                    sp.set_edgecolor("orange"); sp.set_linewidth(2.2)
            elif (ii, jj) in cell:
                ax.imshow(sols[cell[(ii, jj)]][0, 0].cpu().numpy(), cmap="viridis", aspect="auto")
                for sp in ax.spines.values():
                    sp.set_visible(False)
            else:
                ax.set_facecolor("#f0f0f0")
                for sp in ax.spines.values():
                    sp.set_visible(False)
            if (ii, jj) == tgt:
                hl_ax = ax
                ax.imshow(bfield, cmap="viridis", aspect="auto")   # overlay B's first solution -> link obvious at a glance
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_edgecolor("#17b3c4"); sp.set_linewidth(3.2)
            if jj == 0 and ii % 3 == 0:
                ax.set_xlabel(f"{0.5*(re[ii]+re[ii+1]):.3f}", fontsize=12.5, rotation=0)
            if ii == 0 and jj % 3 == 0:
                ax.set_ylabel(f"{0.5*(me[jj]+me[jj+1]):.3f}", fontsize=12.5, rotation=0, ha="right", va="center")
    if all_ood:   # summary in the blank bottom-left (low rho, low mu) region — not per-point counts
        succ = sorted([n for n in all_ood if n >= 1], reverse=True)
        rng = f"{min(succ)}–{max(succ)}" if succ else "0"
        msg = ("OOD extrapolation\n"
               f"{len(all_ood)} points tested\n"
               f"{len(succ)} gave non-trivial solutions\n"
               f"({rng} each, {sum(all_ood)} total)")
        axS = fig.add_subplot(gs[NBrow - 6:NBrow - 1, 0:9]); axS.axis("off")
        axS.text(0.0, 0.5, msg, transform=axS.transAxes, va="center", ha="left", fontsize=13.5,
                 bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="orange", lw=1.8, alpha=0.95))
    return hl_ax


# ---------- row2: Fig2 four-method maps ----------
def draw_grid_row(fig, spec, setup):
    TE = test_set(setup)
    gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=0.38)
    sc = None; axs = []
    for c, (m, lab) in enumerate(METHODS):
        ax = fig.add_subplot(gs[0, c]); axs.append(ax); rs = load_m(setup, m, TE)
        if rs:
            ti = list(rs)
            rho = np.array([rs[t]["rho"] for t in ti]); mu = np.array([rs[t]["mu"] for t in ti])
            tot = np.array([rs[t]["total"] for t in ti])
            sc = ax.scatter(rho, mu, c=tot, s=26, cmap="YlOrRd", vmin=0, vmax=VMAX, edgecolor="0.5", linewidth=0.2)
            ax.text(0.05, 0.05, f"mean {tot.mean():.1f}\nmax  {int(tot.max())}", transform=ax.transAxes, fontsize=12,
                    fontweight="bold", va="bottom", ha="left", linespacing=1.25,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))
        ax.set_title(lab, fontsize=15, fontweight="bold")
        ax.set_xlabel(r"$\rho$", fontweight="bold")
        if c == 0:
            ax.set_ylabel(r"$\mu$", fontweight="bold")
        ax.grid(alpha=0.2); ax.set_axisbelow(True)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axs, fraction=0.015, pad=0.01)
        cb.set_label("# solutions", fontweight="bold", fontsize=11)


# ---------- row3 left: noise-sweep curve ----------
def draw_noise(ax, setup):
    d = torch.load(f"{HERE}/{setup}/results/fig1_{setup}_ti{EXAMPLE_TI[setup]}_noise.pt", weights_only=False)
    sig = d["sigmas"]; curves = d["curves"]; det0 = d["det0"]
    x = np.arange(1, len(curves[str(sig[0])]) + 1)
    for i, s in enumerate(sig):
        ax.plot(x, curves[str(s)], "-o", color=COLORS["palette"][i % len(COLORS["palette"])],
                lw=2.8, ms=8, label=f"$\\sigma$ = {s}")
    ax.axhline(det0, ls="--", color="0.5", lw=2.2, label=f"det only ({det0})")
    ax.set_xlabel("# forward passes", fontweight="bold", fontsize=11)
    ax.set_ylabel("# solutions", fontweight="bold", fontsize=11)
    ax.set_xticks(x); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(r"$(\rho,\,\mu)=(%.3f,\ %.3f)$" % (d["param"][0], d["param"][1]),
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=12, loc="upper left")


# ---------- row3 right: matched vs beyond ----------
def draw_matched_beyond(ax, setup):
    TE = test_set(setup)
    labs, mm, bb = [], [], []
    for m, lab in BARLAB:
        rs = load_m(setup, m, TE)
        if not rs:
            continue
        labs.append(lab); mm.append(np.mean([r["matched"] for r in rs.values()])); bb.append(np.mean([r["beyond"] for r in rs.values()]))
    x = np.arange(len(labs))
    ax.bar(x, mm, color=COLORS["pred"], edgecolor="black", linewidth=1.5, label="in dataset (matched)")
    ax.bar(x, bb, bottom=mm, color=COLORS["pred2"], edgecolor="black", linewidth=1.5, label="new (beyond-GT)")
    for i, (a, b) in enumerate(zip(mm, bb)):
        ax.text(i, a + b + 0.1, f"{a + b:.2f}", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=11)
    ax.set_ylabel("# solutions / param", fontweight="bold", fontsize=11)
    ax.set_ylim(0, max(np.array(mm) + np.array(bb)) * 1.18)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.legend(fontsize=12, loc="upper left")


def build(setup):
    fig = plt.figure(figsize=(18, 17.5))
    outer = fig.add_gridspec(3, 1, height_ratios=[2.9, 1.05, 1.15], hspace=0.40)
    top = outer[0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.20)
    hl_ax = draw_atlas(fig, top[0], setup)   # A (left): (rho,mu) solution atlas -- distribution over parameter space
    draw_fig1(fig, top[1], setup)    # B (right): per-parameter solution montage (2 cover-data + new)
    if hl_ax is not None:            # arrow: highlighted atlas cell (A) -> montage (B)
        from matplotlib.patches import FancyArrowPatch
        fig.canvas.draw()
        cp = hl_ax.get_position(); bp = top[1].get_position(fig)
        arr = FancyArrowPatch((cp.x1, 0.5 * (cp.y0 + cp.y1)), (bp.x0, bp.y0 + 0.55 * bp.height),
                              transform=fig.transFigure, arrowstyle="-|>", mutation_scale=30,
                              lw=3.0, color="#17b3c4", zorder=300, shrinkA=3, shrinkB=3,
                              connectionstyle="arc3,rad=-0.15")
        fig.add_artist(arr)
    draw_grid_row(fig, outer[1], setup)
    bot = outer[2].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.34)
    draw_noise(fig.add_subplot(bot[0]), setup)
    draw_matched_beyond(fig.add_subplot(bot[1]), setup)
    # panel labels A-E at each panel's top-left, outside the content (in the inter-row gap)
    for spec, lt in [(top[0], "A"), (top[1], "B"), (outer[1], "C"), (bot[0], "D"), (bot[1], "E")]:
        bb = spec.get_position(fig)
        fig.text(bb.x0 - 0.012, bb.y1 + 0.008, lt, fontsize=22, fontweight="bold", va="bottom", ha="left")
    out = f"{HERE}/combined_{setup}"
    fig.savefig(f"{out}.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved combined_{setup}.png/.pdf")


if __name__ == "__main__":
    for s in ["L1", "L2"]:
        build(s)
