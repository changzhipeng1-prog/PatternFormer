"""Fig 5: solution atlas over (rho, mu). The plane is tiled into equal SQUARE cells over
the GT training-parameter box (+ extension cells). Each cell shows a representative
steady-state A-field, so morphology varies continuously (spots -> stripes -> mazes).
Only the OOD EXTRAPOLATION cells are framed (orange + circled # = solutions the model
generated there with no GT); in-domain GT cells are shown unframed; cells where only the
trivial homogeneous state exists are light gray.
L=1 (left) | L=2 (right), independent (rho,mu) axes, sized like Fig1. -> GS/fig5_atlas.png/.pdf
Mirrors 2d_GrayScott/qwen_ord{,_L2}/slides/make_param_atlas.py but combined + paper-styled.
"""
import os, sys, json, glob
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _gridload import DIFF

PATHS = {"L1": dict(look="L1/data/gs_lookup_big.pt",
                    ef="L1/extension/experiments/extrap_fields.npz",
                    eg="L1/extension/experiments/extrap_grid_L1.json"),
         "L2": dict(look="L2/data/gs_lookup_L2.pt",
                    ef="L2/extension/experiments/extrap_fields.npz",
                    eg="L2/extension/experiments/extrap_grid_L2.json")}
NBr_in, NBu_in = 9, 10        # in-box cells (rho, mu)
ELr, ERr = 1, 3              # rho extension: few low-rho (left), more high-rho (right -> OOD region)
ELu, EUu = 3, 1              # mu extension: more low-mu (bottom -> OOD region), few high-mu (top)
OOD_C = "orange"


def panel(sf, setup):
    p = PATHS[setup]
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
    dwr = (r1 - r0) / NBr_in; dwu = (u1 - u0) / NBu_in
    re = r0 + (np.arange(NBr_in + ELr + ERr + 1) - ELr) * dwr
    me = u0 + (np.arange(NBu_in + ELu + EUu + 1) - ELu) * dwu
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

    axes = sf.subplots(NBrow, NBc, gridspec_kw=dict(wspace=0.05, hspace=0.05))
    for jj in range(NBrow):
        for ii in range(NBc):
            ax = axes[NBrow - 1 - jj][ii]; ax.set_xticks([]); ax.set_yticks([])
            if (ii, jj) in extrap_cell:                       # OOD extrapolation: orange frame only
                ax.imshow(extrap_cell[(ii, jj)], cmap="viridis")
                for sp in ax.spines.values():
                    sp.set_edgecolor(OOD_C); sp.set_linewidth(2.6)
            elif (ii, jj) in cell:                            # in-domain GT cell: solution only, no frame
                ti = cell[(ii, jj)]; ax.imshow(sols[ti][0, 0].cpu().numpy(), cmap="viridis")
                for sp in ax.spines.values():
                    sp.set_visible(False)
            else:                                             # trivial homogeneous state only
                ax.set_facecolor("#f0f0f0")
                for sp in ax.spines.values():
                    sp.set_visible(False)
    for ii in range(0, NBc, 3):
        axes[NBrow - 1, ii].set_xlabel(f"{0.5*(re[ii]+re[ii+1]):.3f}", fontsize=11, rotation=0)
    for jj in range(0, NBrow, 3):
        axes[NBrow - 1 - jj, 0].set_ylabel(f"{0.5*(me[jj]+me[jj+1]):.3f}", fontsize=11, rotation=0, ha="right", va="center")
    if all_ood:   # summary in the blank bottom-left region (not per-point counts)
        succ = sorted([n for n in all_ood if n >= 1], reverse=True)
        rng = f"{min(succ)}–{max(succ)}" if succ else "0"
        sf.text(0.09, 0.13, "OOD extrapolation\n" f"{len(all_ood)} points tested\n"
                f"{len(succ)} gave non-trivial solutions\n" f"({rng} each, {sum(all_ood)} total)",
                ha="left", va="bottom", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="orange", lw=1.8, alpha=0.95))
    sf.suptitle(r"$(D_A,\,D_S)=(%s,\,%s)$" % DIFF[setup], fontsize=15, fontweight="bold", y=0.99)
    sf.text(0.55, 0.005, r"$\rho$ (feed) $\rightarrow$", ha="center", fontsize=12, fontweight="bold")
    sf.text(0.003, 0.5, r"$\mu$ (kill) $\rightarrow$", va="center", rotation=90, fontsize=12, fontweight="bold")


def main():
    fig = plt.figure(figsize=(16.5, 8.4))
    subfigs = fig.subfigures(1, 2, wspace=0.04)
    for sf, setup in zip(subfigs, ["L1", "L2"]):
        panel(sf, setup)
    leg = fig.legend(handles=[Patch(facecolor="none", edgecolor="orange", lw=2.6, label="OOD extrapolation (no GT)  —  circled # = solutions generated"),
                              Patch(facecolor="#f0f0f0", edgecolor="none", label="trivial homogeneous state only")],
                     loc="upper center", ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.savefig(f"{HERE}/fig5_atlas.png", dpi=140, bbox_inches="tight", bbox_extra_artists=[leg])
    fig.savefig(f"{HERE}/fig5_atlas.pdf", bbox_inches="tight", bbox_extra_artists=[leg])
    print("saved fig5_atlas (OOD framed orange; GT unframed; trivial gray)")


if __name__ == "__main__":
    main()
