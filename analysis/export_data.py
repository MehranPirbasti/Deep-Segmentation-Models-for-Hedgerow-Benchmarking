"""Writes one CSV per figure, so each can be checked or edited on its own."""
import os, shutil
import numpy as np, pandas as pd
import tables as T

OUT = "../data"
os.makedirs(OUT, exist_ok=True)
M = pd.read_csv("../data/master.csv")
w = lambda name, df, note: (
    df.to_csv(os.path.join(OUT, name), index=False),
    print(f"  {name:<44} {len(df):3d} rows   {note}"))

print("per-figure data tables")
COLS = ["Model","Decoder","Backbone","Group","iou","dice_f1","boundary_f1",
        "cldice","betti0_err","frag_index","inference_time_s","train_hours",
        "loss","topo_measured","src"]

w("Leaderboard_Matrix_Main.csv", M[M.Group=="Main"][COLS],
  "one row per cell row of the matrix")
w("Leaderboard_Matrix_Hybrid.csv", M[M.Group=="Hybrid"][COLS], "")
w("Topology_Matrix.csv",
  M[M.topo_measured][["Model","iou","cldice","betti0_err","frag_index"]],
  "the nine measured rows of Table VI")
w("Test_Set_Radar_All.csv", M[COLS], "one polygon per row")
for m in ("iou","dice_f1","boundary_f1","cldice","betti0_err"):
    w(f"Efficiency_Frontier_{m}.csv",
      M[["Model","Decoder","Backbone","inference_time_s",m]], "x = latency")
for m in ("iou","dice_f1","boundary_f1"):
    w(f"Training_Efficiency_{m}.csv",
      M[["Model","Decoder","Backbone","train_hours",m]], "x = training hours")
for m in ("iou","dice_f1","boundary_f1","cldice","betti0_err","frag_index"):
    w(f"Boxplot_{m}.csv", M[["Model","Decoder","Backbone",m]], "grouped by Backbone")
w("Decoder_Boxplot.csv", pd.read_csv("../data/decoder_val_iou.csv"),
  "best validation IoU, from the original decoder figure")
w("Topology_Indices.csv",
  M[["Model","Decoder","Backbone","iou","cldice","betti0_err","frag_index",
     "topo_measured"]], "three panels against IoU")
w("Topology_Frontier.csv",
  M[["Model","Decoder","Backbone","inference_time_s","cldice","betti0_err",
     "topo_measured"]], "")
w("Encoder_Family_Frontier.csv",
  M[["Model","Backbone","inference_time_s","iou","cldice"]],
  "medians and quartiles are computed per Backbone")
t7 = M[M.set_index(["Decoder","Backbone"]).index.isin(T.BF1_R1.keys())]
w("BF1_Tolerance.csv",
  t7[["Model","bf1_r1","boundary_f1"]].assign(
      delta=(t7.boundary_f1-t7.bf1_r1).round(2)), "Table VII")
w("MultiSeed_CI_IoU.csv", T.seeds(), "Table V")
w("Learning_Curves_IoU.csv", pd.read_csv("../data/convergence_bands.csv"),
  "per-family min/median/max envelope by epoch")

# provenance of every value
w("_master_all_configurations.csv", M, "the single source for all of the above")
prov = pd.DataFrame({
 "quantity": ["iou / dice_f1 / boundary_f1 / inference_time_s / train_hours",
              "the same, for configurations no table reports",
              "dice_f1 where no table reports it",
              "cldice / betti0_err / frag_index",
              "the same, remaining configurations",
              "bf1_r1", "multi-seed mean, CI, SD", "best validation IoU",
              "convergence envelopes", "validation loss"],
 "source": ["Tables II and III of the manuscript, verbatim",
            "recovered marker by marker from the box plots of the original submission",
            "computed as 2*IoU/(1+IoU), exact for a fixed prediction",
            "Table VI of the manuscript, verbatim",
            "extended by a behavior-family model fitted on the nine measured rows; flagged topo_measured = False",
            "Table VII of the manuscript, verbatim",
            "Table V of the manuscript, verbatim",
            "recovered from the decoder-distribution figure of the original submission",
            "recovered from the validation-convergence figure of the original submission",
            "the two performance matrices of the original submission"]})
w("_provenance.csv", prov, "where each column comes from")
print(f"\nwritten to {OUT}/")
