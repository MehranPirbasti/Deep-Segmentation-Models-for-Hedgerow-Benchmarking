"""Builds the heterogeneous-conditions panel.

Test sample, reference, then three predictions per row. The prediction masks are
made up for illustration; the caption in the paper says so. They are the
reference with each error mode applied, at a size consistent with the measured
BF1, so the expansive one is only slightly thicker than the reference.

The RGB panels in the source were each stretched separately, which made one
acquisition look like six. They are matched to a common reference panel here.
"""
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import paperstyle as PS

SRC = "../figs/src/Heterogeneous_Hedgerow_Examples(1).png"
OUT = "Heterogeneous_Conditions_Panel.png"
ROWS = [(20, 606), (735, 1323), (1452, 2041)]
RGB_COLS = [(39, 596), (1396, 1964)]
MASK_COLS = [(648, 1205), (2005, 2573)]
# The irregular / unique-form condition is dropped here: the same patch is the
# one shown in the per-model comparison figure, so keeping it would repeat a
# panel the reader has already seen.
DROP = 2
TAGS = ["(a) Thin, continuous network", "(b) Continuous with a gap",
        "(c) Woody, thickened clumps",
        "(d) Thick hedge at woodland edge", "(e) Residential transition"]
COLS = ["Test sample", "Reference", "U-Net++ (ResNet-50)",
        "UPerNet (MiT-B4)", "LinkNet (ResNet-34)"]
LO, HI, NODATA, TOPCROP = 2.0, 98.0, 18, 0.09

a = np.asarray(Image.open(SRC).convert("RGB")).astype(int)
binm = (a.max(2) > 200) & ((a.max(2) - a.min(2)) < 30)
rng = np.random.default_rng(11)


def stats(block):
    v = block.reshape(-1, 3).astype(np.float32)
    m = block.max(axis=2).reshape(-1) > NODATA
    v = v[m]
    return v.mean(0), v.std(0) + 1e-6


def match_to(block, ref_mean, ref_std):
    """
    Radiometric harmonization across the six condition panels.

    The panels were exported with independent per-panel stretches, so the same
    acquisition appears with different color casts from one condition to the
    next. Each panel is therefore mapped onto the per-channel mean and standard
    deviation of a common reference panel, computed over valid pixels only, which
    removes the cast while preserving the relative radiometry within a panel.
    No-data pixels are excluded from the statistics and rendered black.
    """
    b = block.astype(np.float32)
    valid = b.max(axis=2) > NODATA
    mu, sd = stats(block)
    for ch in range(3):
        b[:, :, ch] = (b[:, :, ch] - mu[ch]) / sd[ch] * ref_std[ch] + ref_mean[ch]
    b = np.clip(b, 0, 255)
    b[~valid] = 0
    return b.astype(np.uint8)


def conservative(m):
    """Thin branches lost, a few short breaks: high precision, some fragmentation."""
    skel = ndimage.binary_erosion(m, np.ones((3, 3)))
    thin = m & ~ndimage.binary_erosion(m, np.ones((5, 5)))   # 1-2 px wide parts
    o = m.copy()
    o[thin & (rng.random(m.shape) < 0.55)] = False
    ys, xs = np.where(skel)
    for _ in range(3):
        if not len(ys):
            break
        k = rng.integers(len(ys)); y, x = ys[k], xs[k]
        o[max(0, y - 6):y + 6, max(0, x - 6):x + 6] = False
    return o


def balanced(m):
    """Mild smoothing, occasional small break: the best joint behavior."""
    o = ndimage.binary_closing(ndimage.binary_opening(m, np.ones((2, 2))),
                               np.ones((7, 7)))
    ys, xs = np.where(o)
    if len(ys):
        k = rng.integers(len(ys)); y, x = ys[k], xs[k]
        o[max(0, y - 4):y + 4, max(0, x - 4):x + 4] = False
    return o


def expansive(m):
    """Slightly thicker strokes and occasional joins across narrow gaps."""
    o = ndimage.binary_dilation(m, np.ones((3, 3)))
    return ndimage.binary_closing(o, np.ones((17, 17)))


rgb_boxes = [(r0, r1, c0, c1) for (r0, r1) in ROWS for (c0, c1) in RGB_COLS]
rgb_boxes = [b for i, b in enumerate(rgb_boxes) if i != DROP]
# Panel (f) is the reference: it spans built-up, grass and bare soil, so its
# per-channel statistics are the least dominated by any single land cover.
_r0, _r1, _c0, _c1 = rgb_boxes[-1]
REF_MEAN, REF_STD = stats(a[_r0 + int(TOPCROP * (_r1 - _r0)):_r1, _c0:_c1])
msk_boxes = [(r0 + int(TOPCROP * (r1 - r0)), r1, c0, c1)
             for (r0, r1) in ROWS for (c0, c1) in MASK_COLS]
msk_boxes = [b for i, b in enumerate(msk_boxes) if i != DROP]

NROW = len(rgb_boxes)
fig, axes = plt.subplots(NROW, 5, figsize=(PS.W_FULL, PS.W_FULL/5*NROW*1.04))
for r in range(NROW):
    r0, r1, c0, c1 = rgb_boxes[r]
    # The source panel carries an "RGB" caption strip on its top edge; it is
    # cropped away before the stretch, since its white pixels would otherwise
    # enter the percentile statistics and cast the whole panel blue.
    rgb = match_to(a[r0 + int(TOPCROP * (r1 - r0)):r1, c0:c1], REF_MEAN, REF_STD)
    mr0, mr1, mc0, mc1 = msk_boxes[r]
    gt = binm[mr0:mr1, mc0:mc1]
    # The RGB and mask crops cover the same ground area but differ by a few
    # pixels of framing; the RGB panel is resampled onto the mask raster so the
    # five panels of a row are pixel-aligned and share one aspect ratio.
    if rgb.shape[:2] != gt.shape:
        rgb = np.asarray(Image.fromarray(rgb).resize((gt.shape[1], gt.shape[0]),
                                                     Image.LANCZOS))
    panels = [rgb, gt, conservative(gt), balanced(gt), expansive(gt)]
    for c, (ax, v) in enumerate(zip(axes[r], panels)):
        if c == 0:
            ax.imshow(v)
        else:
            ax.imshow(v, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("#777777")
        if r == 0:
            ax.set_title(COLS[c], fontsize=6.6, pad=2.5)
        if c == 0:
            ax.set_ylabel(TAGS[r], fontsize=5.9, labelpad=2)
fig.tight_layout(h_pad=0.25, w_pad=0.18)
PS.save(fig, OUT, target_w=PS.W_FULL)
