"""Fig 1: the max-distinct param's full solution set, L=1 (left) | L=2 (right).
Each panel = A-fields packed tight, sharing one horizontal colorbar at the bottom.
Top of each panel: row 1 = the diffusion params (D_A, D_S) + the (rho,mu) point;
row 2 = legend (cyan frame = covers a GT solution; orange frame = beyond-GT / new).
Project colormap = viridis.

  python make_fig1.py --l1_ti 786 --l2_ti 402            # deterministic
  python make_fig1.py --l1_ti 786 --l2_ti 402 --noise    # det + noise multipass union
"""
import os, argparse, math
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
GT_C, NEW_C = "#00d5d5", "#ff7f0e"   # cover-data (cyan) / new-solution (orange)
# diffusion coefficient value-strings per setup (L2 = L1 / 4, i.e. the 2x-larger domain)
DIFF = {"L1": (r"2.5{\times}10^{-4}", r"5{\times}10^{-4}"),
        "L2": (r"6.25{\times}10^{-5}", r"1.25{\times}10^{-4}")}


def load(setup, ti, suf):
    d = torch.load(f"{HERE}/{setup}/results/fig1_{setup}_ti{ti}{suf}.pt", weights_only=False)
    A = d["sols"][:, 0].numpy()
    is_gt = d["is_gt"].numpy().astype(bool)
    mA = A.reshape(len(A), -1).mean(1)
    order = sorted(range(len(A)), key=lambda i: (not is_gt[i], mA[i]))  # GT first, then by mean(A)
    rho, mu = d["param"]
    return A[order], is_gt[order], (float(rho), float(mu))


def panel(fig, spec, A, is_gt, ncols, setup, param, show_diff=True, aspect="equal"):
    N = len(A); nrows = math.ceil(N / ncols)
    vmax = float(A.max())
    inner = GridSpecFromSubplotSpec(nrows + 3, ncols, subplot_spec=spec,
                                    height_ratios=[0.30, 0.26] + [1] * nrows + [0.16],
                                    hspace=0.06, wspace=0.06)
    # row 0: header — full (D_A,D_S,rho,mu) for the standalone fig; just (rho,mu) when the
    # composite already states the diffusion regime once for the whole figure
    da, ds = DIFF[setup]
    axt = fig.add_subplot(inner[0, :]); axt.axis("off")
    htxt = (r"$(D_A,\,D_S,\,\rho,\,\mu)=(%s,\ %s,\ %.3f,\ %.3f)$" % (da, ds, param[0], param[1])
            if show_diff else r"$(\rho,\,\mu)=(%.3f,\ %.3f)$" % (param[0], param[1]))
    axt.text(0.5, 0.5, htxt, ha="center", va="center", fontsize=15)
    # row 1: legend
    axl = fig.add_subplot(inner[1, :]); axl.axis("off")
    axl.legend(handles=[Patch(facecolor="none", edgecolor=GT_C, lw=3, label="cover data"),
                        Patch(facecolor="none", edgecolor=NEW_C, lw=3, label="new solution")],
               loc="center", ncol=2, frameon=False, fontsize=13, handlelength=1.4, columnspacing=2.0)
    # rows 2..: solution fields
    im = None
    for i in range(nrows * ncols):
        ax = fig.add_subplot(inner[2 + i // ncols, i % ncols])
        ax.set_xticks([]); ax.set_yticks([])
        if i < N:
            im = ax.imshow(A[i], cmap="viridis", vmin=0, vmax=vmax, aspect=aspect)
            c = GT_C if is_gt[i] else NEW_C
            for s in ax.spines.values():
                s.set_color(c); s.set_linewidth(3)
        else:
            ax.set_axis_off()
    cax = fig.add_subplot(inner[nrows + 2, :])
    fig.colorbar(im, cax=cax, orientation="horizontal")


def main(a):
    suf = "_noise" if a.noise else ""
    ncols = 6 if a.noise else 5
    L1 = load("L1", a.l1_ti, suf); L2 = load("L2", a.l2_ti, suf)
    fig = plt.figure(figsize=(15, 9) if a.noise else (13, 8))
    outer = fig.add_gridspec(1, 2, wspace=0.10)
    panel(fig, outer[0, 0], L1[0], L1[1], ncols, "L1", L1[2])
    panel(fig, outer[0, 1], L2[0], L2[1], ncols, "L2", L2[2])
    out = f"{HERE}/fig1_maxsol_L1-ti{a.l1_ti}_L2-ti{a.l2_ti}{suf}"
    fig.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"L1 ti={a.l1_ti}: {len(L1[0])} sols ({int(L1[1].sum())} GT / {int((~L1[1]).sum())} beyond)")
    print(f"L2 ti={a.l2_ti}: {len(L2[0])} sols ({int(L2[1].sum())} GT / {int((~L2[1]).sum())} beyond)")
    print(f"saved {os.path.basename(out)}.png/.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--l1_ti", type=int, required=True)
    p.add_argument("--l2_ti", type=int, required=True)
    p.add_argument("--noise", action="store_true")
    main(p.parse_args())
