"""a2a4 test figure: 2 rows x 5 params (mirror of 1D_p).

PDE: -u'' + a4 u^4 + a2 u^2 = 0.  Row 1 = Qwen direct output, Row 2 = after
Newton refine (FDM Newton). GT = black solid (only the NON-ZERO coexisting
branches; the trivial u==0 branch is excluded), Ours = clear-palette dashed.
The 5 params cover different non-zero solution counts. When the model emits more
solutions than (non-zero) GT, we Hungarian-match and keep the rel-L2-smallest k.
Average rel-L2 (over matched pairs) annotated per subplot. Style: test_plot_style.
"""
import os
import sys
import numpy as np
import torch
import scipy.optimize as sciopt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))      # paper/ for test_plot_style
from test_plot_style import apply_style, save_fig
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from model.newton_fdm import newton_refine

apply_style()
# +4 to all default font sizes (ticks etc.) for this figure only
import matplotlib as _mpl
for _k in ("font.size","axes.labelsize","axes.titlesize","xtick.labelsize","ytick.labelsize","legend.fontsize","figure.titlesize"):
    _mpl.rcParams[_k] = _mpl.rcParams[_k] + 4
MORANDI = ["#FF7F0E", "#1F77B4", "#2CA02C", "#17BECF",
           "#D62728", "#9467BD", "#E377C2", "#8C564B"]
GEN = os.path.join(HERE, "generated_solutions.pt")
ZERO_NORM = 1e-6


def rel_l2(pred, gt):
    return float(np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-12))


def match_best_k(preds, gt):
    cost = np.linalg.norm(preds[:, None, :] - gt[None, :, :], axis=2)
    r, c = sciopt.linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def nonzero_gt(rec):
    gt = rec["gt"].numpy().astype(np.float64)
    return gt[np.linalg.norm(gt, axis=1) > ZERO_NORM]


def main():
    data = torch.load(GEN, map_location="cpu", weights_only=False)

    by_k = {}
    for rec in data:
        gt = nonzero_gt(rec)
        gen = rec["generated"].numpy().astype(np.float64)
        if gt.shape[0] == 0 or gen.shape[0] < gt.shape[0]:
            continue
        pairs = match_best_k(gen, gt)
        errs = [rel_l2(gen[pi], gt[gj]) for pi, gj in pairs]
        # score by the WORST branch so the examples show params whose ALL coexisting
        # branches are captured (the rare missed-branch cases are quantified in box4)
        by_k.setdefault(gt.shape[0], []).append((float(np.max(errs)), rec))

    # Use the WELL-REPRESENTED multiplicities (most params). a2a4's dominant
    # non-zero counts are k=1,3,5; k=4 is a rare near-bifurcation case (~17 params,
    # the model nearly always misses its deep branch) -> not shown here, quantified
    # in box4. Build 5 columns from the most-populated k (distinct k first, then
    # repeat the richest with several quantiles).
    kvals = sorted(by_k.keys(), key=lambda k: -len(by_k[k]))
    distinct = [k for k in kvals if len(by_k[k]) >= 30]
    specs = [(k, 0.35) for k in distinct[:5]]
    qs = [0.2, 0.4, 0.55, 0.7, 0.3]
    i = 0
    while len(specs) < 5 and distinct:
        specs.append((distinct[i % len(distinct)], qs[len(specs) % len(qs)])); i += 1
    specs = sorted(specs, key=lambda t: t[0])
    chosen, used = [], set()
    for ki, frac in specs:
        cands = sorted([c for c in by_k.get(ki, []) if id(c[1]) not in used], key=lambda t: t[0])
        if not cands:
            continue
        pick = cands[min(len(cands) - 1, int(frac * len(cands)))]
        used.add(id(pick[1])); chosen.append(pick[1])
    print("chosen (a4,a2,k_nonzero):",
          [(round(float(r["params"][0]), 2), round(float(r["params"][1]), 2),
            nonzero_gt(r).shape[0]) for r in chosen])

    x = np.linspace(0.0, 1.0, 1024)
    fig, axes = plt.subplots(2, 5, figsize=(21, 9))

    for col, rec in enumerate(chosen):
        a4, a2 = float(rec["params"][0]), float(rec["params"][1])
        gt = nonzero_gt(rec)
        gen = rec["generated"].numpy().astype(np.float64)
        k = gt.shape[0]
        pairs = match_best_k(gen, gt)
        sel_raw = [gen[pi] for pi, gj in pairs]
        gord = [gj for pi, gj in pairs]

        refined_all = []
        for u in gen:
            ru, ok, _ = newton_refine(u, a4, a2, h=1.0 / len(u), max_iter=50, abs_tol=1e-9)
            refined_all.append(np.asarray(ru, np.float64).reshape(-1))
        refined_all = np.stack(refined_all)
        pairs_r = match_best_k(refined_all, gt)
        sel_ref = [refined_all[pi] for pi, gj in pairs_r]
        gord_r = [gj for pi, gj in pairs_r]

        for row, (sel, go) in enumerate([(sel_raw, gord), (sel_ref, gord_r)]):
            ax = axes[row, col]
            for j in range(k):
                ax.plot(x, gt[j], color="black", lw=3, zorder=1)
            errs = []
            for n, (u, gj) in enumerate(zip(sel, go)):
                ax.plot(x, u, "--", color=MORANDI[n % len(MORANDI)], lw=2.5, zorder=2)
                errs.append(rel_l2(u, gt[gj]))
            me = float(np.mean(errs)) if errs else float("nan")
            ax.text(0.04, 0.96, f"rel-L2={me:.1e}", transform=ax.transAxes, va="top",
                    ha="left", fontsize=18, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.6", alpha=0.85))
            ax.grid(False)
            if row == 0:
                ax.set_title(f"$a_4$={a4:.2f}, $a_2$={a2:.2f} (k={k})", fontsize=18, fontweight="bold")
            if row == 1:
                ax.set_xlabel("x", fontsize=28, fontweight="bold")
            if col == 0:
                ax.set_ylabel("Direct output\nu(x)" if row == 0 else "Newton-refined\nu(x)",
                              fontsize=26, fontweight="bold")

    handles = [Line2D([0], [0], color="black", lw=3, label="Exact (GT)"),
               Line2D([0], [0], color="#6B6B6B", lw=2.5, ls="--", label="Ours (Qwen)")]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97], w_pad=0.2, h_pad=0.6)
    save_fig(fig, os.path.join(HERE, "fig_a2a4_examples_2x5"))
    print("saved fig_a2a4_examples_2x5.png/.pdf")


if __name__ == "__main__":
    main()
