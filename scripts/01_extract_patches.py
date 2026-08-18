"""Cuts the scene into 416x416 patches and writes patch_inventory.csv.

The imagery and the reference layer are licensed and are not in this repo. This
is the code that produced our patches; run it on data you have licensed.
"""
#!/usr/bin/env python3
"""
Sliding-window patch extraction from the orthorectified scene and the rasterized
hedgerow reference (manuscript Section III-B).

Window 416x416 px, stride 208 px (50 % step). Windows are discarded when they
contain no-data pixels or fall entirely inside land-cover classes excluded from
the UKCEH hedgerow product. Writes the patches plus patch_inventory.csv, which
is the input to 02_make_splits.py.

LICENSE NOTE. The imagery and the reference layer are licensed products and are
not distributed with this repository. This script is the exact code that
produced our patches; run it on data obtained from Airbus Defence and Space and
from UKCEH / EDINA Digimap under their own terms.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hedgebench import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="4-band orthorectified GeoTIFF")
    ap.add_argument("--mask", required=True, help="rasterized hedgerow reference")
    ap.add_argument("--eligible", default=None,
                    help="optional 0/1 raster of terrain eligible under the UKCEH "
                         "product (woodland/urban/water/moor excluded)")
    ap.add_argument("--out", required=True, help="output dataset root")
    args = ap.parse_args()

    xdir = os.path.join(args.out, "X")
    ydir = os.path.join(args.out, "Y", "Detection")
    os.makedirs(xdir, exist_ok=True)
    os.makedirs(ydir, exist_ok=True)

    S, P = C.EXTRACT_STRIDE, C.PATCH_SIZE
    rows = []
    pid = 0

    with rasterio.open(args.image) as src, rasterio.open(args.mask) as msk:
        elig = rasterio.open(args.eligible) if args.eligible else None
        H, W = src.height, src.width
        prof_x = src.profile.copy()
        prof_y = msk.profile.copy()
        prof_x.update(height=P, width=P, count=src.count)
        prof_y.update(height=P, width=P, count=1)

        n_r = (H - P) // S + 1
        n_c = (W - P) // S + 1
        print(f"scene {W}x{H} px -> window grid {n_c} x {n_r} = {n_c*n_r} windows")

        for r in range(n_r):
            for c in range(n_c):
                win = Window(c * S, r * S, P, P)
                img = src.read(window=win)
                if img.shape[1] != P or img.shape[2] != P:
                    continue
                nod = src.nodata
                valid = 1.0 if nod is None else float((img != nod).all(axis=0).mean())
                if valid < C.MIN_VALID_FRACTION:
                    continue
                if elig is not None:
                    e = elig.read(1, window=win)
                    if float((e > 0).mean()) < C.MIN_ELIGIBLE_FRACTION:
                        continue
                m = (msk.read(1, window=win) > 0).astype(np.uint8)

                name_x = f"RGBN_{pid:05d}.tif"
                name_y = f"Dtct_{pid:05d}.tif"
                tx = src.window_transform(win)
                with rasterio.open(os.path.join(xdir, name_x), "w",
                                   **{**prof_x, "transform": tx}) as dst:
                    dst.write(img)
                with rasterio.open(os.path.join(ydir, name_y), "w",
                                   **{**prof_y, "transform": tx, "dtype": "uint8"}) as dst:
                    dst.write(m[None])

                rows.append(dict(patch_id=f"P{pid:05d}", image_file=name_x,
                                 mask_file=name_y, row=r, col=c,
                                 hedgerow_fraction=float(m.mean()),
                                 valid_fraction=valid))
                pid += 1
        if elig is not None:
            elig.close()

    inv = pd.DataFrame(rows)
    inv_path = os.path.join(args.out, "patch_inventory.csv")
    inv.to_csv(inv_path, index=False)
    area_km2 = len(inv) * (P * C.GSD_M) ** 2 / 1e6
    print(f"kept {len(inv)} windows | mean hedgerow fraction "
          f"{inv['hedgerow_fraction'].mean():.4f} | window footprint area "
          f"{area_km2:.1f} km^2 (overlapping)")
    print(f"wrote {inv_path}")


if __name__ == "__main__":
    main()
