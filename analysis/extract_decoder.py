"""Reads best-validation IoU back off the decoder distribution figure."""
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage

SRC = "../figs/src/Fig3_Decoder_Boxplot(2).png"
FAMILY = {"MiT (transformer)": (211, 80, 79), "ResNet-50": (83, 147, 195),
          "ResNet-34": (230, 162, 101), "MobileNet-V2": (132, 195, 129)}
EXPECTED = {"MiT (transformer)": 3, "ResNet-50": 1, "ResNet-34": 1,
            "MobileNet-V2": 1}
DECODERS = ["U-Net++", "UPerNet", "U-Net", "DeepLabV3", "FPN", "SegFormer",
            "DPT", "DeepLabV3+", "LinkNet", "PAN"]
Y_TOP, V_TOP, Y_BOT, V_BOT = 445.0, 0.90, 1891.0, 0.65

a = np.asarray(Image.open(SRC).convert("RGB")).astype(int)
H, W = a.shape[:2]
val = lambda y: V_TOP + (y - Y_TOP) * (V_BOT - V_TOP) / (Y_BOT - Y_TOP)


def centers_of(mask, radius):
    """Centers of overlapping filled circles, via distance-transform peaks."""
    d = ndimage.distance_transform_edt(mask)
    mx = ndimage.maximum_filter(d, size=int(1.5 * radius) | 1)
    peaks = (d > 0.55 * radius) & (d >= mx - 1e-6)
    lab, n = ndimage.label(peaks)
    return [(c[1], c[0]) for c in ndimage.center_of_mass(peaks, lab,
                                                         range(1, n + 1))]


masks, radius = {}, []
for fam, rgb in FAMILY.items():
    m = np.sqrt(((a - np.array(rgb)) ** 2).sum(axis=2)) < 60
    m[:, int(W * 0.78):] = False                     # exclude legend swatches
    m = ndimage.binary_opening(m, np.ones((3, 3)))
    masks[fam] = m
    lab, n = ndimage.label(m)
    sz = ndimage.sum(m, lab, range(1, n + 1))
    if len(sz):
        radius.append(np.sqrt(np.median(sz[sz > 60]) / np.pi))
R = float(np.median(radius))

pts = []
for fam, m in masks.items():
    for cx, cy in centers_of(m, R):
        pts.append((float(cx), float(cy), fam))
print(f"dot radius {R:.1f} px | {len(pts)} centers found (expect 60)")

xs = sorted(p[0] for p in pts)
groups, cur = [], [xs[0]]
for x in xs[1:]:
    if x - cur[-1] < 90:
        cur.append(x)
    else:
        groups.append(cur); cur = [x]
groups.append(cur)
col = sorted(float(np.mean(g)) for g in groups)
assert len(col) == len(DECODERS), f"{len(col)} columns, expected 10"

rows = [(DECODERS[int(np.argmin([abs(cx - c) for c in col]))], fam,
         round(float(val(cy)), 4)) for cx, cy, fam in pts]
df = pd.DataFrame(rows, columns=["Decoder", "EncoderFamily", "val_iou"])

chk = df.groupby(["Decoder", "EncoderFamily"]).size().unstack(fill_value=0)
chk = chk.reindex(index=DECODERS, columns=list(FAMILY), fill_value=0)
print("\nrecovered count per decoder x encoder family (target 3/1/1/1):")
print(chk.to_string())

df = df.sort_values(["Decoder", "val_iou"], ascending=[True, False])
df.to_csv("../data/decoder_val_iou.csv", index=False)
top = df[(df.Decoder == "U-Net++") & (df.EncoderFamily == "ResNet-50")].val_iou.max()
print(f"\ncalibration check: best U-Net++ (ResNet-50) = {top:.3f} "
      f"(source figure shows 0.942)")
