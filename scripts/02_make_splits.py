"""Assigns patches to train/val/test and writes the split.

Blocks of adjacent patches are assigned as units, stratified by hedgerow
density, and any patch sitting on the seam between two differently assigned
blocks is dropped. That is the one-patch buffer described in Section III-B.
"""
#!/usr/bin/env python3
"""
Input
---
patch_inventory.csv produced by 01_extract_patches.py, with at least:
    patch_id, row, col, hedgerow_fraction
where (row, col) are indices on the sliding-window grid (stride 208 px).

Procedure
-----
1. Group windows into spatial blocks of BLOCK_SIZE_WINDOWS x BLOCK_SIZE_WINDOWS.
2. Rank blocks by hedgerow density and cut into DENSITY_STRATA quantile strata.
3. Within each stratum, assign WHOLE BLOCKS to train/validation/test in the
   target 70/15/15 proportions, using a fixed seed. Stratifying by density makes
   foreground prevalence comparable across partitions.
4. Apply the buffer. Because the extraction stride is half the window width, two
   windows share pixels iff they are 8-neighbours on the window grid. Every
   window that is an 8-neighbour of a window belonging to a DIFFERENT partition
   is therefore a leakage path. We drop the window on the LOWER-PRECEDENCE side
   of each seam (train < validation < test), which:
       (a) guarantees that no image pixel is shared between any two partitions,
           i.e. the partitions are pixel-disjoint by design; and
       (b) erodes the large training partition rather than the small evaluation
           partitions, which would otherwise lose a disproportionate share of
           their windows and inflate the variance of the reported scores.

Output
---
split_metadata.csv with: patch_id, image_file, mask_file, row, col, block_row,
block_col, block_id, density_stratum, hedgerow_fraction, split, dropped, reason.

Only rows with dropped == False take part in training or evaluation.
"""
from __future__ import annotations
import argparse, os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hedgebench import config as C


def assign_blocks(blocks: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Assign whole blocks to partitions in target proportions, per density stratum."""
    names = ["train", "validation", "test"]
    props = np.array([C.TARGET_SPLIT[n] for n in names], dtype=float)
    props = props / props.sum()

    blocks = blocks.copy()
    blocks["split"] = ""
    for stratum, grp in blocks.groupby("density_stratum"):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        # Largest-remainder allocation so small strata still receive eval blocks.
        raw = props * n
        counts = np.floor(raw).astype(int)
        while counts.sum() < n:
            counts[np.argmax(raw - counts)] += 1
        # Guarantee at least one block per partition when the stratum allows it.
        if n >= 3:
            for k in range(3):
                if counts[k] == 0:
                    counts[int(np.argmax(counts))] -= 1
                    counts[k] = 1
        start = 0
        for k, name in enumerate(names):
            blocks.loc[idx[start:start + counts[k]], "split"] = name
            start += counts[k]
    return blocks


def apply_buffer(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the lower-precedence window on every cross-partition seam."""
    split_of = {}
    for r, c, s in zip(df["row"], df["col"], df["split"]):
        split_of[(int(r), int(c))] = s

    rad = C.BUFFER_NEIGHBOURHOOD
    prec = C.SPLIT_PRECEDENCE
    dropped, reason = [], []
    for r, c, s in zip(df["row"], df["col"], df["split"]):
        r, c = int(r), int(c)
        drop = False
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if dr == 0 and dc == 0:
                    continue
                other = split_of.get((r + dr, c + dc))
                if other is None or other == s:
                    continue
                # Overlapping windows in different partitions: the lower-
                # precedence side is removed.
                if prec[s] < prec[other]:
                    drop = True
                elif prec[s] == prec[other]:
                    drop = True   # cannot happen (splits differ), kept for safety
        dropped.append(drop)
        reason.append("cross_partition_buffer" if drop else "")
    out = df.copy()
    out["dropped"] = dropped
    out["reason"] = reason
    return out


def verify_pixel_disjoint(df: pd.DataFrame) -> None:
    """Hard check: no two kept windows from different partitions overlap."""
    kept = df[~df["dropped"]]
    lut = {(int(r), int(c)): s for r, c, s in zip(kept["row"], kept["col"], kept["split"])}
    bad = 0
    for (r, c), s in lut.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                o = lut.get((r + dr, c + dc))
                if o is not None and o != s:
                    bad += 1
    if bad:
        raise AssertionError(
            f"{bad} overlapping window pairs remain across partitions; the split "
            f"is NOT pixel-disjoint. Refusing to write split_metadata.csv.")
    print("  [verify] partitions are pixel-disjoint: no overlapping window pair "
          "spans two splits.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True, help="patch_inventory.csv")
    ap.add_argument("--out", default="split_metadata.csv")
    ap.add_argument("--report", default="split_report.json")
    args = ap.parse_args()

    df = pd.read_csv(args.inventory)
    for col in ("patch_id", "row", "col", "hedgerow_fraction"):
        if col not in df.columns:
            raise SystemExit(f"inventory is missing required column '{col}'")

    B = C.BLOCK_SIZE_WINDOWS
    df["block_row"] = df["row"] // B
    df["block_col"] = df["col"] // B
    df["block_id"] = df["block_row"].astype(str) + "_" + df["block_col"].astype(str)

    blocks = (df.groupby("block_id")["hedgerow_fraction"].mean()
                .rename("density").reset_index().set_index("block_id"))
    # Quantile strata; duplicates='drop' guards against degenerate densities.
    blocks["density_stratum"] = pd.qcut(
        blocks["density"].rank(method="first"),
        q=min(C.DENSITY_STRATA, max(1, len(blocks))),
        labels=False, duplicates="drop")

    rng = np.random.default_rng(C.SPLIT_SEED)
    blocks = assign_blocks(blocks, rng)

    df = df.merge(blocks[["split", "density_stratum", "density"]],
                  left_on="block_id", right_index=True, how="left")
    df = apply_buffer(df)
    verify_pixel_disjoint(df)

    kept = df[~df["dropped"]]
    counts = kept["split"].value_counts().to_dict()
    total = int(len(kept))
    realised = {k: round(100.0 * v / max(total, 1), 1) for k, v in counts.items()}

    df.to_csv(args.out, index=False)

    report = {
        "windows_before_buffer": int(len(df)),
        "windows_dropped_by_buffer": int(df["dropped"].sum()),
        "unique_source_patches": total,
        "counts": {k: int(v) for k, v in counts.items()},
        "target_proportions_pct": {k: round(100 * v, 1) for k, v in C.TARGET_SPLIT.items()},
        "realised_proportions_pct": realised,
        "block_size_windows": B,
        "extract_stride_px": C.EXTRACT_STRIDE,
        "patch_size_px": C.PATCH_SIZE,
        "buffer": "one-sided, full-window-width (Chebyshev radius 1 on the window grid)",
        "pixel_disjoint_verified": True,
        "split_seed": C.SPLIT_SEED,
        "foreground_prevalence_by_split": {
            k: round(float(v), 5) for k, v in
            kept.groupby("split")["hedgerow_fraction"].mean().items()},
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
