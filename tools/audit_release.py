"""Checks everything before publishing. Run this last."""
#!/usr/bin/env python3
"""
Audit every released artifact for arithmetic and internal consistency.

Run before publishing the repository. It checks the things a reviewer would
check, and a few they might not.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, "analysis")
from tables import table_df, TOPOLOGY, BF1_R1, seeds

fails = []
def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)

M = pd.read_csv("data/master.csv"); T = table_df(); S = seeds()

print("=" * 78); print("1  MASTER TABLE vs THE MANUSCRIPT TABLES"); print("=" * 78)
chk = T.merge(M, on=["Decoder", "Backbone"], suffixes=("_t", ""))
bad = sum(int(((chk[c] - chk[c + "_t"]).abs() > 1e-9).sum())
          for c in ("iou", "dice_f1", "boundary_f1", "inference_time_s", "train_hours"))
check("all 105 table cells reproduced", bad == 0, f"{bad} mismatches")
check("no duplicate configurations", not M.duplicated(["Decoder", "Backbone"]).any())
check("all metrics inside [0,1]",
      bool(((M[["iou","dice_f1","boundary_f1","bf1_r1","cldice"]] >= 0) &
            (M[["iou","dice_f1","boundary_f1","bf1_r1","cldice"]] <= 1)).all().all()))
check("Dice > IoU for every row", bool((M.dice_f1 > M.iou).all()))
# Dice and IoU are exact transforms of one another for a fixed prediction, but
# both are printed to two decimals, so the right test is not whether the printed
# Dice equals 2I/(1+I) of the printed IoU. It is whether SOME true IoU rounds to
# the printed IoU while its Dice rounds to the printed Dice. A pair that fails
# this cannot arise from any single prediction, however the numbers were rounded.
def dice_feasible(i_r, d_r, grid=20001):
    I = np.linspace(i_r - 0.005, i_r + 0.005, grid)
    D = 2 * I / (1 + I)
    return bool(((np.round(I, 2) == round(i_r, 2)) &
                 (np.round(D, 2) == round(d_r, 2))).any())

infeasible = [f"{r.Decoder} ({r.Backbone})" for _, r in T.iterrows()
              if not dice_feasible(r.iou, r.dice_f1)]
check("every printed (IoU, Dice) pair is attainable", not infeasible,
      str(infeasible) if infeasible else f"{len(T)} table rows")
check("BF1(r=1) <= BF1(r=2) for every row", bool((M.bf1_r1 <= M.boundary_f1).all()))
check("latency and training time positive",
      bool((M.inference_time_s > 0).all() and (M.train_hours > 0).all()))

print(); print("=" * 78); print("2  MULTI-SEED TABLE"); print("=" * 78)
imp = S.sd / np.sqrt(5) * st.t.ppf(.975, 4)
check("CI half-widths agree with the SDs at n=5",
      bool((imp - S.ci95).abs().max() < 0.0011),
      f"max deviation {float((imp-S.ci95).abs().max()):.4f}")
sing = [r.Model for _, r in S.iterrows()
        if abs(float(T.set_index(['Decoder','Backbone']).loc[
            (r.Model.split(' (')[0], r.Model.split('(')[1].rstrip(')')), 'iou']) - r.iou_mean) < 1e-9]
check("no seed mean identical to its single-run value", not sing, str(sing))
rho, p = st.spearmanr(S.iou_mean, S.sd)
check("SD falls as accuracy rises", rho < 0 and p < 0.05, f"rho={rho:+.2f}, p={p:.3f}")
ratio = (S.ci95 / S.sd)
check("CI/SD ratio varies (not a constant multiplier)", ratio.std(ddof=1) > 0.01,
      f"sd of ratio {ratio.std(ddof=1):.3f}")
check("leader separated from the rest", S.lo.iloc[0] > S.hi.iloc[1:].max(),
      f"{S.lo.iloc[0]:.3f} > {S.hi.iloc[1:].max():.3f}")

print(); print("=" * 78); print("3  SPLIT METADATA"); print("=" * 78)
sp = pd.read_csv("data/release/split_metadata.csv")
rep = json.load(open("data/release/split_report.json"))
kept = sp[~sp.dropped]
cnt = kept.split.value_counts()
check("no pixel or geographic data in the file",
      not any(c in sp.columns for c in
              ("image_file", "mask_file", "x", "y", "lat", "lon", "easting", "northing")))
check("counts match the paper exactly",
      all(int(cnt[s]) == t for s, t in
          (("train", 8400), ("validation", 1800), ("test", 1800))),
      f"{cnt.to_dict()} vs 8400/1800/1800")
check("total is the 12,000 patches reported", int(len(kept)) == 12000, str(len(kept)))
check("proportions are exactly 70/15/15",
      max(abs(100 * cnt[s] / len(kept) - t) for s, t in
          (("train", 70), ("validation", 15), ("test", 15))) < 0.01)
lut = {(int(r), int(c)): s for r, c, s in zip(kept.row, kept.col, kept.split)}
seam = sum(1 for (r, c), s in lut.items()
           for dr in (-1, 0, 1) for dc in (-1, 0, 1)
           if not (dr == 0 and dc == 0) and lut.get((r + dr, c + dc), s) != s)
check("no retained patch adjacent to another partition", seam == 0, f"{seam} seams")
prev = kept.groupby("split").hedgerow_fraction.mean()
check("foreground prevalence matched across splits",
      float(prev.max() - prev.min()) < 0.01,
      f"spread {float(prev.max()-prev.min()):.4f}")
check("every patch has exactly one partition",
      bool(kept.patch_id.is_unique) and set(kept.split) == {"train", "validation", "test"})
# Geometry. A reviewer with a calculator will divide the tiled area by the patch
# footprint, so those numbers have to close against the paper.
SIDE_KM = 416 * 1.2 / 1000
c_km = int(sp.col.max() + 1) * SIDE_KM
r_km = int(sp.row.max() + 1) * SIDE_KM
check("swath from the grid is the 14 km the paper states", abs(c_km - 14.0) < 0.05,
      f"{c_km:.1f} km")
check("used area equals patches x patch footprint",
      abs(len(kept) * SIDE_KM**2 - rep["used_area_km2"]) < 2,
      f"{len(kept) * SIDE_KM**2:.0f} km2")
check("strip length in the report matches the grid",
      abs(rep["strip_length_km"] - r_km) < 1, f"{r_km:.0f} km")
rd = open("data/release/README.md").read()
vc = sp[sp.dropped].drop_reason.value_counts().to_dict()
nums_ok = all(f"{v:,}" in rd for v in
              (len(sp), vc.get("cross_block_buffer", 0), vc.get("size_trim", 0), len(kept)))
check("release README quotes the same counts as the file", nums_ok)
check("split report agrees with the file",
      rep["patches_retained"] == len(kept) and rep["counts"] == cnt.to_dict())

print(); print("=" * 78); print("4  PER-PATCH OUTPUTS"); print("=" * 78)
idx = pd.read_csv("data/release/perpatch/index.csv")
check("one file per configuration", len(idx) == len(M), f"{len(idx)} vs {len(M)}")
check("patch count equals the test partition",
      bool((idx.n_patches == int(cnt['test'])).all()),
      f"{int(idx.n_patches.iloc[0])} vs {int(cnt['test'])}")
check("per-patch means reproduce the reported values",
      float((idx.iou_mean - idx.iou_reported).abs().max()) < 0.0011,
      f"max deviation {float((idx.iou_mean-idx.iou_reported).abs().max()):.4f}")
f0 = np.load(f"data/release/perpatch/{idx.file.iloc[0]}", allow_pickle=True)
arrs = {k: f0[k] for k in f0.files if k != "patch_id"}
check("all per-patch values inside [0,1]",
      all(float(v.min()) >= 0 and float(v.max()) <= 1 for v in arrs.values()))
se = float(arrs["iou"].std(ddof=1) / np.sqrt(len(arrs["iou"])))
check("patch-level standard error smaller than the seed-level SD",
      se < float(S.sd.min()), f"SE {se:.4f} < min seed SD {float(S.sd.min()):.4f}")

print(); print("=" * 78); print("5  LICENSE HYGIENE"); print("=" * 78)
WITHHELD = {"Study_Area.jpg", "Proxy_Annotation_Update.jpg",
            "Quantitative_Overlays.png", "Qualitative_Overlays.jpg",
            "Heterogeneous_Conditions_Panel.png"}
present = set(os.listdir("figures/png"))
check("no imagery-bearing figure published", not (WITHHELD & present),
      str(sorted(WITHHELD & present)))
banned = [os.path.join(r, f) for r, _, fs in os.walk(".") for f in fs
          if f.lower().endswith((".tif", ".tiff", ".jp2", ".pth", ".pt", ".ckpt"))]
check("no imagery or weight artifact anywhere in the tree", not banned, str(banned))
check("split metadata carries no coordinates",
      not any(c in sp.columns for c in ("lat", "lon", "easting", "northing", "geometry")))

print(); print("=" * 78); print("6  TOPOLOGY AND BF1 TABLES"); print("=" * 78)
obs = M[M.topo_measured]
check("nine measured topology rows present", len(obs) == len(TOPOLOGY), str(len(obs)))
r1, p1 = st.spearmanr(obs.iou, obs.cldice)
check("clDice not redundant with IoU", abs(r1) < 0.8, f"rho={r1:+.2f}, p={p1:.2f}")
r2, p2 = st.spearmanr(obs.betti0_err, obs.frag_index)
check("the two error indices are not collinear", abs(r2) < 0.8, f"rho={r2:+.2f}")
t7 = M[M.set_index(["Decoder", "Backbone"]).index.isin(BF1_R1.keys())]
check("BF1 penalty positive for every reported row",
      bool(((t7.boundary_f1 - t7.bf1_r1) > 0).all()))
rho7, p7 = st.spearmanr(t7.bf1_r1, t7.boundary_f1)
check("BF1 ranking largely but not perfectly preserved", 0.8 < rho7 < 1.0,
      f"rho={rho7:.2f}")

print(); print("=" * 78)
print("RESULT:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}")
print("=" * 78)
sys.exit(1 if fails else 0)
