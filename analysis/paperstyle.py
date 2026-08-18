"""Plot style, sized for the printed page.

The paper is two-column, so a single-column figure ends up about 3.4 inches
wide. Drawing at 12 inches and letting LaTeX shrink it is what made the labels
unreadable in the first submission. Everything here is drawn at final size, and
save() iterates the canvas until the saved width and aspect actually match.

Color = backbone, marker shape = decoder, same as the original figures.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

# IEEEtran two-column geometry
TEXTWIDTH_IN = 7.16
def W(frac: float) -> float:
    return TEXTWIDTH_IN * frac

W_048, W_049, W_050, W_044, W_LINE, W_FULL = (W(.48), W(.49), W(.50),
                                              W(.44), W(.48), TEXTWIDTH_IN)

DPI = 600
MATRIX_CMAP = "viridis"

_serif = "Liberation Serif" if any(
    f.name == "Liberation Serif" for f in fm.fontManager.ttflist) else "DejaVu Serif"

RC = {
    "font.family": "serif",
    "font.serif": [_serif, "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.4,
    "legend.title_fontsize": 6.8,
    "figure.dpi": 160,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.1,
    "lines.markersize": 4.2,
    "lines.markeredgewidth": 0.5,
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.edgecolor": "#333333",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.30,
    "grid.color": "#9aa0a6",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#666666",
    "legend.borderpad": 0.35,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.45,
    "legend.columnspacing": 0.9,
    "legend.labelspacing": 0.30,
    "patch.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

# Encoding convention, identical to the figures of the original submission:
# COLOUR identifies the encoder backbone family and MARKER SHAPE identifies the
# decoder architecture. Keeping this convention means a reader comparing the two
# submissions sees the same clusters in the same colors.
BACKBONE_COLOR = {
    "MiT-B0":       "#FFA3A3",   # sampled from the legend of the source figures
    "MiT-B2":       "#DA3C3D",
    "MiT-B4":       "#973232",
    "ResNet-50":    "#3584BB",
    "ResNet-34":    "#FF8B26",
    "MobileNet-V2": "#41A941",
}
DECODER_MARKER = {
    "DPT": "o", "DeepLabV3": "X", "DeepLabV3+": "s", "FPN": "P",
    "LinkNet": "D", "PAN": (4, 1, 0), "SegFormer": "^", "U-Net": (8, 1, 0),
    "U-Net++": "v", "UPerNet": (6, 1, 0),
}
# Kept for the panels that group by decoder rather than by configuration.
DECODER_COLOR = {
    "U-Net++":    "#0173B2", "U-Net":      "#DE8F05",
    "UPerNet":    "#029E73", "DeepLabV3":  "#D55E00",
    "DeepLabV3+": "#CC78BC", "FPN":        "#CA9161",
    "LinkNet":    "#7F7F7F", "PAN":        "#ECE133",
    "SegFormer":  "#56B4E9", "DPT":        "#9467BD",
}
BACKBONE_MARKER = {
    "ResNet-50": "o", "ResNet-34": "s", "MobileNet-V2": "P",
    "MiT-B0": "^", "MiT-B2": "D", "MiT-B4": "v",
}
GROUP_COLOR = {"Main": "#0173B2", "Hybrid": "#D55E00"}


def apply():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams.update(RC)


VECTOR = True   # also write an editable .svg and a vector .pdf beside the .png


def save(fig, path, target_w=None, target_ar=None, quiet=False):
    """
    Save at the exact width the figure will occupy on the printed page.

    A tight bounding box adds the tick labels, axis labels and color bar to the
    requested canvas, so a figure asked for at 3.44 in is typically written at
    4.0 in and is then scaled DOWN by LaTeX, shrinking the type below the size it
    was designed at. We therefore iterate the canvas size until the saved width
    matches the printed width, which keeps the point sizes on the page equal to
    the point sizes set in this module.
    """
    if target_ar is not None:
        # Tick and axis labels are added outside the requested canvas by the
        # tight bounding box, so the saved aspect ratio is always squarer than
        # the axes. The canvas height is iterated until the SAVED figure has the
        # intended proportions, which is what determines how the figure sits on
        # the page.
        for _ in range(6):
            fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.02,
                        facecolor="white")
            im = Image.open(path)
            ar = im.width / im.height
            if abs(ar - target_ar) < 0.02:
                break
            w, h = fig.get_size_inches()
            fig.set_size_inches(w, h * ar / target_ar)
    if target_w is not None:
        for _ in range(4):
            fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02,
                        facecolor="white")
            got = Image.open(path).width / 150.0
            if abs(got - target_w) < 0.01:
                break
            w, h = fig.get_size_inches()
            k = target_w / got
            fig.set_size_inches(w * k, h * k)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    if VECTOR:
        # Vector companions. Text stays as text rather than being converted to
        # outlines, so every label, tick and number in the figure can be
        # selected and edited in any vector editor, and can be searched.
        import matplotlib as _mpl, os as _os
        _mpl.rcParams["svg.fonttype"] = "none"
        _mpl.rcParams["pdf.fonttype"] = 42
        for ext in ("svg", "pdf"):
            d = _os.path.join(_os.path.dirname(path) or ".", "vector")
            _os.makedirs(d, exist_ok=True)
            fig.savefig(_os.path.join(d, _os.path.basename(path)[:-4] + "." + ext),
                        bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    im = Image.open(path)
    if not quiet:
        flag = "" if target_w is None else f"  (target {target_w:.2f} in)"
        print(f"  {path.split('/')[-1]:<38} {im.width/DPI:.2f} x "
              f"{im.height/DPI:.2f} in @ {DPI} dpi{flag}")


def short(model: str) -> str:
    """Compact model label for dense scatter plots."""
    d, b = model.split(" (")
    b = b.rstrip(")")
    dd = {"U-Net++": "U++", "U-Net": "U", "UPerNet": "UPer", "DeepLabV3": "DLv3",
          "DeepLabV3+": "DLv3+", "FPN": "FPN", "LinkNet": "Link", "PAN": "PAN",
          "SegFormer": "SegF", "DPT": "DPT"}[d]
    bb = {"ResNet-50": "R50", "ResNet-34": "R34", "MobileNet-V2": "MNv2",
          "MiT-B0": "B0", "MiT-B2": "B2", "MiT-B4": "B4"}[b]
    return f"{dd}/{bb}"
