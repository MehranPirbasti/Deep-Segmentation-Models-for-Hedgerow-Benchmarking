"""Builds the figures from a finished training run."""
#!/usr/bin/env python3
"""
Regenerate every data-driven figure from master_results.csv.

"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hedgebench import config as C
from hedgebench.plotstyle import apply, save, legend_outside, INDEX_CMAP

METRIC_LABEL = {"iou": "Test IoU", "dice_f1": "Test Dice / F1",
                "boundary_f1": "Test BF1 (r = 2)", "bf1_r1": "Test BF1 (r = 1)",
                "cldice": "clDice", "betti0_err": "Betti-0 error",
                "frag_index": "Fragmentation index"}


def norm_matrix(df, higher_better, lower_better):
    out = df.copy()
    for c in higher_better:
        r = df[c].max() - df[c].min()
        out[c] = (df[c] - df[c].min()) / r if r > 1e-12 else 0.5
    for c in lower_better:
        r = df[c].max() - df[c].min()
        out[c] = 1 - ((df[c] - df[c].min()) / r) if r > 1e-12 else 0.5
    return out


def matrix_figure(raw, cols, labels, higher, lower, title, path, width=11):
    nrm = norm_matrix(raw[cols], higher, lower)
    # Fixed 2-decimal annotation so the figure prints the same precision as the
    # manuscript tables (a bare round() would drop trailing zeros).
    ann = raw[cols].map(lambda v: f"{v:.2f}")
    fig, ax = plt.subplots(figsize=(width, max(5.5, 0.52 * len(raw))))
    sns.heatmap(nrm, annot=ann, fmt="", cmap=INDEX_CMAP,
                vmin=0, vmax=1, linewidths=0.8, linecolor="white", ax=ax,
                annot_kws={"fontsize": 13, "fontweight": "bold"},
                cbar_kws={"label": "relative score (brighter = better)"})
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel("")
    save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--figures", required=True)
    ap.add_argument("--metrics-csv", default=None,
                    help="model_metrics.csv, needed for the learning curves")
    args = ap.parse_args()
    apply()
    T, F = args.tables, args.figures
    os.makedirs(F, exist_ok=True)

    master = pd.read_csv(os.path.join(T, "master_results.csv"))
    # Ablation runs are diagnostics, not reported operating points: they are kept
    # out of every leaderboard, matrix, frontier and box plot.
    if "is_ablation" in master.columns:
        ablations = master[master["is_ablation"].astype(bool)].copy()
        master = master[~master["is_ablation"].astype(bool)].copy()
    else:
        ablations = master.iloc[0:0].copy()
    main_g = master[master["Group"] == "Main"]
    hyb_g = master[master["Group"] == "Hybrid"]

    # leaderboard matrices (main and hybrid groups)
    cols = ["iou", "dice_f1", "boundary_f1", "bf1_r1", "cldice",
            "betti0_err", "frag_index"]
    labels = ["IoU", "Dice/F1", "BF1 r=2", "BF1 r=1", "clDice",
              "Betti-0 err", "Frag."]
    for grp, fname, title in ((main_g, "Leaderboard_Matrix_Main.png",
                               "Test-set operating points - matched configurations"),
                              (hyb_g, "Leaderboard_Matrix_Hybrid.png",
                               "Test-set operating points - hybrid configurations")):
        if grp.empty:
            continue
        matrix_figure(grp.set_index("Model"), cols, labels,
                      ["iou", "dice_f1", "boundary_f1", "bf1_r1", "cldice"],
                      ["betti0_err", "frag_index"], title, os.path.join(F, fname))

    # topology matrix
    tcols = ["iou", "cldice", "betti0_err", "frag_index"]
    if set(tcols).issubset(master.columns):
        matrix_figure(master.set_index("Model").head(25), tcols,
                      ["IoU", "clDice", "Betti-0 error", "Fragmentation"],
                      ["iou", "cldice"], ["betti0_err", "frag_index"],
                      "Topology-aware evaluation matrix (test set)",
                      os.path.join(F, "Topology_Matrix.png"), width=9)

    # radar
    radar = [("iou", "IoU", False), ("dice_f1", "Dice/F1", False),
             ("boundary_f1", "BF1 (r=2)", False), ("cldice", "clDice", False),
             ("betti0_err", "Betti-0 (inv)", True),
             ("frag_index", "Fragmentation (inv)", True)]
    top = master.head(10).set_index("Model")
    if not top.empty:
        norm = pd.DataFrame(index=top.index)
        for col, lab, inv in radar:
            v = top[col].astype(float); rng = v.max() - v.min()
            sc = (v - v.min()) / rng if rng > 1e-12 else pd.Series(0.5, index=v.index)
            norm[lab] = 1.0 - sc if inv else sc
        labs = [r[1] for r in radar]
        ang = np.linspace(0, 2 * np.pi, len(labs), endpoint=False).tolist(); ang += ang[:1]
        fig = plt.figure(figsize=(11.5, 9))
        ax = fig.add_subplot(111, polar=True)
        pal = sns.color_palette("husl", len(norm))
        for (m, row), col in zip(norm.iterrows(), pal):
            vals = [float(x) for x in row.tolist()]; vals += vals[:1]
            ax.plot(ang, vals, linewidth=2.8, label=m, color=col)
            ax.fill(ang, vals, alpha=0.07, color=col)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels(labs)
        ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=11)
        ax.set_title("Normalized test-set performance profile", pad=30)
        ax.legend(loc="upper left", bbox_to_anchor=(1.14, 1.08), title="Configuration")
        save(fig, os.path.join(F, "Test_Set_Radar_All.png"))

    # efficiency frontier and training cost
    for col in ("iou", "dice_f1", "boundary_f1"):
        if col not in master.columns:
            continue
        fig, ax = plt.subplots(figsize=(12.5, 7.5))
        sns.scatterplot(data=master, x="inference_time_s", y=col, hue="Decoder",
                        style="Backbone", s=210, edgecolor="black",
                        linewidth=0.8, ax=ax)
        fr = master.sort_values("inference_time_s")
        ax.step(fr["inference_time_s"], fr[col].cummax(), where="pre",
                color="crimson", linestyle="-", linewidth=2.4, label="Pareto frontier")
        for _, r in master.head(4).iterrows():
            ax.annotate(r["Model"], (r["inference_time_s"], r[col]),
                        textcoords="offset points", xytext=(8, 6), fontsize=12)
        ax.set_xlabel("Mean per-patch inference time (s)")
        ax.set_ylabel(METRIC_LABEL.get(col, col))
        ax.set_title(f"Accuracy vs inference cost - {METRIC_LABEL.get(col, col)}")
        legend_outside(ax)
        save(fig, os.path.join(F, f"Efficiency_Frontier_{col}.png"))

        fig, ax = plt.subplots(figsize=(12.5, 7.5))
        sns.scatterplot(data=master, x="train_hours", y=col, hue="Backbone",
                        style="Decoder", s=210, edgecolor="black",
                        linewidth=0.8, ax=ax)
        ax.set_xlabel("Training wall-clock time (hours)")
        ax.set_ylabel(METRIC_LABEL.get(col, col))
        ax.set_title(f"Accuracy vs training cost - {METRIC_LABEL.get(col, col)}")
        legend_outside(ax)
        save(fig, os.path.join(F, f"Training_Efficiency_{col}.png"))

        # backbone-family box plot
        fig, ax = plt.subplots(figsize=(12, 7))
        order = master.groupby("Backbone")[col].median().sort_values(ascending=False).index
        sns.boxplot(data=master, x="Backbone", y=col, order=order, width=0.55,
                    linewidth=1.8, fliersize=0, boxprops={"alpha": 0.35}, ax=ax)
        sns.stripplot(data=master, x="Backbone", y=col, order=order, hue="Decoder",
                      size=10, edgecolor="black", linewidth=0.7, jitter=0.2, ax=ax)
        ax.set_title(f"Impact of encoder family on {METRIC_LABEL.get(col, col)}")
        ax.set_xlabel("Backbone"); ax.set_ylabel(METRIC_LABEL.get(col, col))
        ax.tick_params(axis="x", rotation=18)
        legend_outside(ax, title="Decoder")
        save(fig, os.path.join(F, f"Boxplot_{col}.png"))

    # BF1 tolerance sensitivity
    tol_path = os.path.join(T, "table7_bf1_tolerance.csv")
    if os.path.exists(tol_path):
        tol = pd.read_csv(tol_path).head(12)
        fig, ax = plt.subplots(figsize=(12, 7))
        y = np.arange(len(tol))
        ax.hlines(y, tol["BF1_r1"], tol["BF1_r2"], color="#9aa5b1", linewidth=4, zorder=1)
        ax.scatter(tol["BF1_r1"], y, s=170, label="r = 1 px", zorder=3,
                   edgecolor="black", linewidth=0.9)
        ax.scatter(tol["BF1_r2"], y, s=170, label="r = 2 px", zorder=3,
                   edgecolor="black", linewidth=0.9)
        ax.set_yticks(y); ax.set_yticklabels(tol["Model"])
        ax.invert_yaxis(); ax.set_xlabel("Boundary F1")
        ax.set_title("BF1 sensitivity to the boundary tolerance radius")
        legend_outside(ax)
        save(fig, os.path.join(F, "BF1_Tolerance.png"))

    # multi-seed dispersion
    ms_path = os.path.join(T, "table5_multiseed_ci.csv")
    if os.path.exists(ms_path):
        ms = pd.read_csv(ms_path)
        per_seed = []
        for _, r in ms.iterrows():
            for fn in os.listdir(T):
                if fn.startswith(f"perpatch_{r['model_name']}_seed") and fn.endswith(".npz"):
                    with np.load(os.path.join(T, fn)) as d:
                        per_seed.append({"Model": r["Model"],
                                         "iou": float(np.mean(d["iou"]))})
        fig, ax = plt.subplots(figsize=(12.5, 7))
        if per_seed:
            ps = pd.DataFrame(per_seed)
            sns.boxplot(data=ps, x="Model", y="iou", order=ms["Model"], width=0.5,
                        showmeans=True, boxprops={"alpha": 0.35}, linewidth=1.8,
                        meanprops={"marker": "D", "markerfacecolor": "white",
                                   "markeredgecolor": "black", "markersize": 10}, ax=ax)
            sns.stripplot(data=ps, x="Model", y="iou", order=ms["Model"],
                          color="black", size=8, jitter=0.15, ax=ax)
        else:
            ax.errorbar(np.arange(len(ms)), ms["iou_mean"], yerr=ms["ci95_halfwidth"],
                        fmt="D", capsize=6, elinewidth=2.6, markersize=10,
                        markerfacecolor="white", markeredgecolor="black")
            ax.set_xticks(np.arange(len(ms))); ax.set_xticklabels(ms["Model"])
        ax.set_title(f"Test IoU across {int(ms['n_seeds'].max())} random seeds")
        ax.set_xlabel(""); ax.set_ylabel("Test IoU")
        ax.tick_params(axis="x", rotation=30)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
        save(fig, os.path.join(F, "MultiSeed_Boxplot_IoU.png"))

    # deep-supervision ablation
    if not ablations.empty:
        pairs = []
        for spec in C.ABLATION_RUNS:
            f_row = master[master["model_name"] == spec["base"]]
            a_row = ablations[ablations["model_name"] == spec["name"]]
            if f_row.empty or a_row.empty:
                continue
            for metric, lab in (("iou", "IoU"), ("boundary_f1", "BF1 (r=2)"),
                                ("cldice", "clDice")):
                pairs.append({"metric": lab, "variant": "with deep supervision",
                              "value": float(f_row.iloc[0][metric])})
                pairs.append({"metric": lab, "variant": "without deep supervision",
                              "value": float(a_row.iloc[0][metric])})
            base_ref = master[master["Decoder"] == "U-Net"]
            base_ref = base_ref[base_ref["Backbone"] == f_row.iloc[0]["Backbone"]]
            if not base_ref.empty:
                for metric, lab in (("iou", "IoU"), ("boundary_f1", "BF1 (r=2)"),
                                    ("cldice", "clDice")):
                    pairs.append({"metric": lab, "variant": "plain U-Net baseline",
                                  "value": float(base_ref.iloc[0][metric])})
        if pairs:
            pdf = pd.DataFrame(pairs)
            fig, ax = plt.subplots(figsize=(10.5, 6.5))
            sns.barplot(data=pdf, x="metric", y="value", hue="variant",
                        edgecolor="black", linewidth=1.2, ax=ax)
            for c in ax.containers:
                ax.bar_label(c, fmt="%.2f", fontsize=12, padding=3)
            ax.set_ylim(0.80, 1.0)
            ax.set_xlabel(""); ax.set_ylabel("Test-set score")
            ax.set_title("Deep-supervision ablation, U-Net++ (ResNet-50)")
            legend_outside(ax, title="")
            save(fig, os.path.join(F, "Ablation_DeepSupervision.png"))

    # learning curves
    mpath = args.metrics_csv or os.path.join(T, "model_metrics.csv")
    if os.path.exists(mpath):
        df = pd.read_csv(mpath)
        for c in ("iou", "loss", "epoch"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        from hedgebench.models import DECODER_LABEL, BACKBONE_LABEL
        df["Decoder"] = df["decoder"].map(DECODER_LABEL).fillna(df["decoder"])
        df["Backbone"] = df["backbone"].map(BACKBONE_LABEL).fillna(df["backbone"])
        df["Model"] = df["Decoder"] + " (" + df["Backbone"] + ")"
        v = df[(df["phase"] == "validation") & (df["seed"] == C.BASE_SEED)]
        if not v.empty:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.lineplot(data=v, x="epoch", y="iou", hue="Backbone", style="Decoder",
                         units="Model", estimator=None, linewidth=2.0, alpha=0.85, ax=ax)
            ax.set_title("Validation IoU convergence")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Validation IoU")
            legend_outside(ax)
            save(fig, os.path.join(F, "Learning_Curves_IoU.png"))

            fig, ax = plt.subplots(figsize=(14, 8))
            t = df[(df["phase"] == "train") & (df["seed"] == C.BASE_SEED)]
            sns.lineplot(data=t, x="epoch", y="loss", hue="Backbone", units="Model",
                         estimator=None, linewidth=1.3, alpha=0.45, legend=False, ax=ax)
            sns.lineplot(data=v, x="epoch", y="loss", hue="Backbone", units="Model",
                         estimator=None, linewidth=2.2, alpha=0.95, ax=ax)
            ax.set_title("Training (thin) and validation (thick) loss")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Combined loss")
            legend_outside(ax, title="Backbone")
            save(fig, os.path.join(F, "Learning_Curves_Loss.png"))

    # consistency audit against the manuscript tables
    print("\n[audit] figure/table consistency (R2.m4)")
    checked = mismatch = 0
    for f in ("table2_best_by_architecture.csv", "table3a_best_backbone_per_decoder.csv",
              "table3b_best_model_per_backbone.csv"):
        p = os.path.join(T, f)
        if not os.path.exists(p):
            continue
        tb = pd.read_csv(p)
        for _, r in tb.iterrows():
            src = master[master["Model"] == r["Model"]]
            if src.empty:
                continue
            for col in ("iou", "dice_f1", "boundary_f1", "inference_time_s"):
                if col in tb.columns:
                    checked += 1
                    if abs(float(src.iloc[0][col]) - float(r[col])) > 1e-9:
                        mismatch += 1
                        print(f"  MISMATCH {r['Model']} {col}")
    print(f"  {checked} table cells checked against the figure source, "
          f"{mismatch} mismatches")
    print(f"\nFigures written to {F}")


if __name__ == "__main__":
    main()
