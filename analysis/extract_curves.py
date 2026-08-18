"""Reads the validation curves back off the convergence figure."""
import numpy as np, pandas as pd
from scipy import ndimage
from PIL import Image

SRC = "../figs/src/Fig4_Learning_Curves_Refined(2).png"
# Sampled from the legend swatches of the source figure itself rather than
# guessed, so the reference colors are exactly those used to draw the curves.
COLORS = {"MiT-B0": (255, 183, 183), "MiT-B2": (226, 103, 104),
          "MiT-B4": (174, 95, 96), "ResNet-50": (98, 159, 202),
          "ResNet-34": (255, 165, 86), "MobileNet-V2": (107, 188, 107)}
X0, XSTEP, EPSTEP = 340.0, 628.0, 20.0        # epoch gridlines
Y0, YSTEP, VSTEP = 2462.0, 473.5, 0.20        # IoU gridlines
XMAX, YMIN = 3560, 140

a = np.asarray(Image.open(SRC).convert("RGB")).astype(int)
epoch_of = lambda x: (x - X0) / XSTEP * EPSTEP
iou_of = lambda y: (Y0 - y) / YSTEP * VSTEP

# Winner-take-all color assignment. Independent per-color thresholds either
# miss the thin dashed lines or let the anti-aliased halo of a dark line be read
# as a paler family; assigning each pixel to its nearest reference color, with
# white as a competing class, avoids both.
# Neutral references compete with the family colors: without them a light grey
# gridline pixel is nearer to the pale MiT-B0 pink than to white and is claimed
# by that family, which pins its envelope to the bottom of the axes.
NEUTRAL = [[255, 255, 255], [235, 235, 235], [210, 210, 210],
           [180, 180, 180], [140, 140, 140], [90, 90, 90], [30, 30, 30]]
REF = np.array(list(COLORS.values()) + NEUTRAL)
sub = a[YMIN:int(Y0) + 6, int(X0) - 60:XMAX]
d = np.sqrt(((sub[:, :, None, :] - REF[None, None, :, :]) ** 2).sum(-1))
who = d.argmin(2)
best = d.min(2)
valid = (best < 95) & (who < len(COLORS))
names = list(COLORS)

# The anti-aliased halo of a saturated red line is literally pale pink, i.e. the
# MiT-B0 reference color, so nearest-color assignment alone cannot separate
# them. Pale pixels adjacent to a saturated-red core are therefore removed from
# the MiT-B0 mask before the envelope is measured.
core = {}
for k, fam in enumerate(names):
    mm = np.zeros(a.shape[:2], bool)
    mm[YMIN:int(Y0) + 6, int(X0) - 60:XMAX] = valid & (who == k) & (best < 40)
    core[fam] = mm
halo = ndimage.binary_dilation(core["MiT-B2"] | core["MiT-B4"], np.ones((5, 5)))

rows = []
for k, fam in enumerate(names):
    m = np.zeros(a.shape[:2], bool)
    m[YMIN:int(Y0) + 6, int(X0) - 60:XMAX] = valid & (who == k)
    if fam == "MiT-B0":
        m &= ~halo
    m = ndimage.binary_opening(m, np.ones((2, 2)))
    m = ndimage.binary_closing(m, np.ones((3, 3)))
    # Dotted and dash-dot line styles break a curve into many small components,
    # so components cannot be filtered by size. Isolated debris is instead
    # removed further below, by a rolling median along the epoch axis: a real
    # envelope varies smoothly with epoch, a stray pixel does not.
    n = 0
    for x in range(int(X0), XMAX, 5):
        ys = np.where(m[:, x])[0]
        if len(ys) < 2:
            continue
        ep = epoch_of(x)
        if not (0.5 <= ep <= 100.5):
            continue
        rows.append(dict(family=fam, epoch=ep,
                         lo=iou_of(ys.max()), med=iou_of(np.median(ys)),
                         hi=iou_of(ys.min())))
        n += 1
    print(f"  {fam:<13} {n} epoch samples")

df = pd.DataFrame(rows).sort_values(["family", "epoch"])
# light smoothing of the envelope, the source lines are 1 px and dashed
# Dotted and dash-dot styles leave a curve absent from many pixel columns, so a
# plain median would systematically clip the top and bottom of the envelope. The
# extremes are therefore taken as a short rolling max / min, which bridges the
# gaps in a dashed line, and are then lightly smoothed.
g = df.groupby("family")
df["hi"] = g["hi"].transform(lambda s: s.rolling(9, center=True, min_periods=1)
                             .max().rolling(5, center=True, min_periods=1).median())
df["lo"] = g["lo"].transform(lambda s: s.rolling(9, center=True, min_periods=1)
                             .min().rolling(5, center=True, min_periods=1).median())
df["med"] = g["med"].transform(lambda s: s.rolling(15, center=True,
                                                   min_periods=3).median())
df[["lo", "med", "hi"]] = df[["lo", "med", "hi"]].clip(0.0, 1.0)
df.to_csv("../data/convergence_bands.csv", index=False)
fin = df[df.epoch > 95].groupby("family")[["lo", "med", "hi"]].mean().round(3)
print("\nvalues at epoch 100 (min / median / max across decoders):")
print(fin.to_string())
