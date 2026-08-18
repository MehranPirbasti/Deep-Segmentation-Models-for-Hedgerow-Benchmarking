"""
Single publication figure style.

Every figure in the paper is produced through this module, so line weights,
font sizes, colors and export resolution are identical across the whole
manuscript. Sizes are chosen so that the figure remains legible after reduction
to IEEE single-column width (~8.8 cm).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DPI = 300
INDEX_CMAP = "viridis"          # one colormap for every matrix / heatmap figure
SEQ_PALETTE = "colorblind"      # categorical palette, color-vision safe

RC = {
    "figure.dpi": 110,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
    "lines.linewidth": 2.6,
    "lines.markersize": 9,
    "lines.markeredgewidth": 0.9,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "grid.linewidth": 0.9,
    "grid.alpha": 0.35,
    "axes.grid": True,
    "axes.axisbelow": True,
    "figure.autolayout": False,
}


def apply():
    sns.set_theme(style="whitegrid", context="talk", palette=SEQ_PALETTE)
    matplotlib.rcParams.update(RC)


def save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


def legend_outside(ax, title=None, ncol=1):
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True,
              title=title, ncol=ncol, borderaxespad=0.0)
