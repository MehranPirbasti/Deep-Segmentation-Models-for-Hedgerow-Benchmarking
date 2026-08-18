"""Builds every table in the paper from model_metrics.csv."""
#!/usr/bin/env python3
"""

"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hedgebench import config as C
from hedgebench.models import DECODER_LABEL, BACKBONE_LABEL
from hedgebench.metrics import bootstrap_ci, seed_ci

NUM = ["loss", "iou", "dice_f1", "boundary_f1", "bf1_r1", "cldice",
       "betti0_err", "frag_index", "bridge_err", "inference_time_s",
       "total_training_time_s"]

# Matched ("Main") pairs: CNN head with CNN encoder, transformer head with
# transformer encoder. Everything else is a controlled hybrid. The head families
# are classified by their canonical encoder in the source papers, which is why
# UPerNet counts as a CNN head (it is canonically an FPN/PPM decoder over a
# convolutional backbone) while SegFormer and DPT are transformer heads.
CNN_HEADS = {"unet", "unetpp", "deeplab", "deeplabplus", "fpn", "linknet",
             "pan", "upernet"}
TF_HEADS = {"segformer", "dpt"}
CNN_BB = {"resnet34", "resnet50", "mobilenetv2"}
TF_BB = {"mitb0", "mitb2", "mitb4"}


def group_of(decoder, backbone):
    cnn_h, cnn_b = decoder in CNN_HEADS, backbone in CNN_BB
    return "Main" if cnn_h == cnn_b else "Hybrid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True, help="tables directory from 04_train.py")
    args = ap.parse_args()
    T = args.tables

    df = pd.read_csv(os.path.join(T, "model_metrics.csv"))
    for c in NUM + ["epoch"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Decoder"] = df["decoder"].map(DECODER_LABEL).fillna(df["decoder"])
    df["Backbone"] = df["backbone"].map(BACKBONE_LABEL).fillna(df["backbone"])
    df["Model"] = df["Decoder"] + " (" + df["Backbone"] + ")"
    # Ablation runs must never collide with their base configuration, and must
    # never enter the ranked leaderboard or the two-way summary tables.
    abl_names = {a["name"] for a in C.ABLATION_RUNS}
    df["is_ablation"] = df["model_name"].isin(abl_names)
    df.loc[df["is_ablation"], "Model"] = (
        df.loc[df["is_ablation"], "Decoder"] + " (" +
        df.loc[df["is_ablation"], "Backbone"] + ", no DS)")
    df["Group"] = [group_of(d, b) for d, b in zip(df["decoder"], df["backbone"])]
    df.loc[df["is_ablation"], "Group"] = "Ablation"

    val = df[df["phase"] == "validation"].copy()
    test = df[df["phase"] == "test"].copy()

    # master results table: base seed, test set
    tmap = df.groupby(["model_name", "seed"])["total_training_time_s"].max().reset_index()
    base = test[test["seed"] == C.BASE_SEED].drop(columns=["total_training_time_s"])
    base = base.merge(tmap, on=["model_name", "seed"], how="left")
    master = (base.groupby(["Model", "Decoder", "Backbone", "Group", "model_name",
                            "is_ablation"], as_index=False)[NUM].mean())
    master["train_hours"] = master["total_training_time_s"] / 3600.0
    # Rank the 60 reported configurations only; ablations are appended unranked.
    ranked = master[~master["is_ablation"]].sort_values(
        "iou", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    extra = master[master["is_ablation"]].copy()
    extra.insert(0, "rank", np.nan)
    master = pd.concat([ranked, extra], ignore_index=True)
    master.to_csv(os.path.join(T, "master_results.csv"), index=False)
    ranked_only = ranked

    # Table II: best operating point per architecture family, by group
    t2 = (ranked_only
          .sort_values("iou", ascending=False)
          .groupby(["Group", "Decoder"], as_index=False).first()
          .sort_values(["Group", "iou"], ascending=[True, False]))
    t2.to_csv(os.path.join(T, "table2_best_by_architecture.csv"), index=False)

    # Table III: two-way summary
    a = (ranked_only.sort_values("iou", ascending=False)
         .groupby("Decoder", as_index=False).first())
    b = (ranked_only.sort_values("iou", ascending=False)
         .groupby("Backbone", as_index=False).first())
    a.to_csv(os.path.join(T, "table3a_best_backbone_per_decoder.csv"), index=False)
    b.to_csv(os.path.join(T, "table3b_best_model_per_backbone.csv"), index=False)

    # Epoch@100 %
    rows = []
    for (name, seed), g in val.groupby(["model_name", "seed"]):
        g = g.dropna(subset=["iou"])
        if g.empty:
            continue
        peak = g["iou"].max()
        rows.append(dict(model_name=name, seed=seed, peak_val_iou=peak,
                         epoch_at_100=int(g.loc[g["iou"] >= peak, "epoch"].min()),
                         epochs_run=int(g["epoch"].max())))
    e100 = pd.DataFrame(rows)
    if not e100.empty:
        e100 = e100.merge(df[["model_name", "Decoder", "Backbone", "Model"]]
                          .drop_duplicates("model_name"), on="model_name", how="left")
        e100.to_csv(os.path.join(T, "epoch_at_100.csv"), index=False)

    # Table V: multi-seed mean +/- 95 % CI
    ms_rows = []
    for (name, model), g in test.groupby(["model_name", "Model"]):
        v = g["iou"].dropna().to_numpy()
        if v.size < 3:
            continue
        m, half, sd = seed_ci(v)
        ms_rows.append(dict(model_name=name, Model=model, n_seeds=int(v.size),
                            iou_mean=m, ci95_halfwidth=half, sd=sd,
                            ci95_low=m - half, ci95_high=m + half))
    ms = pd.DataFrame(ms_rows).sort_values("iou_mean", ascending=False)
    if not ms.empty:
        ms.to_csv(os.path.join(T, "table5_multiseed_ci.csv"), index=False)
        # Robust vs marginal gaps: which pairs have non-overlapping intervals?
        pairs = []
        r = ms.reset_index(drop=True)
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                sep = r.loc[i, "ci95_low"] > r.loc[j, "ci95_high"]
                pairs.append(dict(a=r.loc[i, "Model"], b=r.loc[j, "Model"],
                                  delta=r.loc[i, "iou_mean"] - r.loc[j, "iou_mean"],
                                  separated=bool(sep)))
        pd.DataFrame(pairs).to_csv(os.path.join(T, "seed_gap_separation.csv"), index=False)

    # Table VI: topology-aware metrics
    topo = ranked_only.set_index("Model")[["iou", "cldice", "betti0_err",
                                           "frag_index", "bridge_err"]]
    topo.to_csv(os.path.join(T, "table6_topology_metrics.csv"))
    if len(ranked_only) > 2:
        rc = ranked_only["iou"].corr(ranked_only["cldice"], method="spearman")
        print(f"Spearman rank correlation IoU vs clDice: {rc:.3f}  "
              f"(correlated but not redundant)")

    # Table VII: BF1 tolerance sensitivity
    tol = ranked_only.set_index("Model")[["bf1_r1", "boundary_f1"]].copy()
    tol.columns = ["BF1_r1", "BF1_r2"]
    tol["delta"] = tol["BF1_r2"] - tol["BF1_r1"]
    tol["rank_r1"] = tol["BF1_r1"].rank(ascending=False)
    tol["rank_r2"] = tol["BF1_r2"].rank(ascending=False)
    tol = tol.sort_values("BF1_r2", ascending=False)
    tol.to_csv(os.path.join(T, "table7_bf1_tolerance.csv"))
    rho = float(tol["BF1_r1"].corr(tol["BF1_r2"], method="spearman"))
    print(f"BF1 rank correlation between r=1 and r=2: rho = {rho:.3f} "
          f"({'ranking preserved' if rho > 0.9 else 'RANKING CHANGES - report this'})")
    print(f"  largest drop at the tighter tolerance: "
          f"{tol['delta'].max():.3f} ({tol['delta'].idxmax()})")
    print(f"  smallest drop: {tol['delta'].min():.3f} ({tol['delta'].idxmin()})")

    # Deep-supervision ablation
    for spec in C.ABLATION_RUNS:
        f, b = spec["base"], spec["name"]
        sub = master[master["model_name"].isin([f, b])]
        if len(sub) == 2:
            sub.to_csv(os.path.join(T, f"ablation_{b}.csv"), index=False)
            full = sub[sub["model_name"] == f].iloc[0]
            nods = sub[sub["model_name"] == b].iloc[0]
            print(f"Ablation {f}: IoU {full['iou']:.3f} -> {nods['iou']:.3f} "
                  f"(delta {full['iou']-nods['iou']:+.3f}), "
                  f"BF1 {full['boundary_f1']:.3f} -> {nods['boundary_f1']:.3f}")

    # Patch-level bootstrap
    rows = []
    for fn in sorted(os.listdir(T)):
        if not (fn.startswith("perpatch_") and fn.endswith(".npz")):
            continue
        name, seed = fn[len("perpatch_"):-len(".npz")].rsplit("_seed", 1)
        with np.load(os.path.join(T, fn), allow_pickle=True) as d:
            rec = {"model_name": name, "seed": int(seed), "n_test_patches": len(d["iou"])}
            for k in ("iou", "dice", "cldice", "bf1_r2"):
                if k in d:
                    m, lo, hi = bootstrap_ci(np.asarray(d[k], dtype=np.float64).ravel())
                    rec[f"{k}_mean"], rec[f"{k}_ci_low"], rec[f"{k}_ci_high"] = m, lo, hi
        rows.append(rec)
    if rows:
        pd.DataFrame(rows).sort_values("iou_mean", ascending=False).to_csv(
            os.path.join(T, "bootstrap_ci.csv"), index=False)

    print(f"\nTables written to {T}")
    print(ranked_only.head(10)[["rank", "Model", "Group", "iou", "dice_f1",
                                "boundary_f1", "cldice", "train_hours"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
