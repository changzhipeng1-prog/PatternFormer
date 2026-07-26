"""Shared scientific-plot style for the BBBB paper test figures.

Distilled from the AAAA project's figure scripts (Nature-style: sans-serif,
large bold fonts, thick lines, dpi-300 export). Import and call apply_style()
at the top of any plotting script so all test figures look consistent.

    from test_plot_style import apply_style, COLORS, save_fig
    apply_style()
    ...
    save_fig(fig, "test/fig_xxx")   # writes .png (dpi300) + .pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Nature-style rcParams (matches AAAA Figure_pde / Figure_ff)
_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 20,
    "font.weight": "bold",
    "axes.labelsize": 24,
    "axes.labelweight": "bold",
    "axes.titlesize": 26,
    "axes.titleweight": "bold",
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 22,
    "lines.linewidth": 4,
    "grid.alpha": 0.3,
    "figure.titlesize": 28,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "axes.linewidth": 2,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Curve / category colors (AAAA convention)
COLORS = {
    "exact": "black",       # GT / exact solution -> black solid
    "pred": "#0066CC",      # model prediction -> blue
    "pred2": "#FF8C00",     # secondary method -> orange
    "newton": "#009E73",    # Newton-refined -> green
    # Nature palette for multi-series
    "palette": ["#56B4E9", "#FF0000", "#009E73", "#CC79A7", "#FF8C00", "#0066CC"],
}

# 2D field rendering defaults
CMAP_FIELD = "rainbow"      # solution field (pcolormesh, shading='gouraud')
CMAP_ERROR = "viridis"      # error field (LogNorm)
CBAR_KW = dict(fraction=0.046, pad=0.04)


def apply_style():
    plt.rcParams.update(_RC)


def save_fig(fig, path_noext, pdf=True):
    """Save dpi-300 PNG (+ optional PDF) with tight bbox."""
    fig.savefig(path_noext + ".png", dpi=300, bbox_inches="tight")
    if pdf:
        fig.savefig(path_noext + ".pdf", bbox_inches="tight")
