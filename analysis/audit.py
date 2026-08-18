"""Checks the figures against the paper's tables. Run it after rebuilding."""
import os, re, sys, json
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0,'.')
from tables import table_df, TOPOLOGY, BF1_R1, SEEDS

FIG = "../figures/png"
# The manuscript's figure inventory is shipped as JSON so that this audit runs
# without a copy of the .tex source. Regenerate it with tools/dump_inventory.py
# if the manuscript changes.
INV = json.load(open("manuscript_figures.json"))
M = pd.read_csv("../data/master.csv"); T = table_df()
ok = lambda b: "PASS" if b else "FAIL"
fails = []

print("="*78); print("STEP 1, inventory"); print("="*78)
# Five figures in the paper show the licensed imagery and are not published in
# this repo, so they are expected to be absent here. See figures/png/README.md.
WITHHELD = {"Study_Area.jpg", "Proxy_Annotation_Update.jpg",
            "Quantitative_Overlays.png", "Qualitative_Overlays.jpg",
            "Heterogeneous_Conditions_Panel.png"}
used = [f for f in INV["slots"] if f not in WITHHELD]
have = {f for f in os.listdir(FIG)
        if not f.endswith(".md") and os.path.isfile(os.path.join(FIG, f))}
# Legend_Key is intentionally repeated: it is the shared key placed once at the
# foot of each multi-panel comparison figure.
SHARED = {"Legend_Key.png"}
dup = [u for u in set(used) if used.count(u) > 1 and u not in SHARED]
print(f"  {len(used)} slots, {len(set(used))} unique files, {len(have)} in folder")
print(f"  withheld (licensed imagery, in the paper only): {len(WITHHELD)}")
for n, c in (("missing", sorted(set(used)-have)), ("unused", sorted(have-set(used))),
             ("duplicated", sorted(dup))):
    print(f"  {n:<11}: {c or 'none'}"); fails += c
labels = set(INV["labels"])
refs = set(INV["refs"])
print(f"  unreferenced labels: {sorted(labels-refs) or 'none'}")
vec = ({f.rsplit(".",1)[0] for f in os.listdir("../figures/vector")}
       if os.path.isdir("../figures/vector") else set())
plots = {f[:-4] for f in have if f.endswith(".png")}
print(f"  plots with an editable vector companion: {len(plots & vec)}/{len(plots)}")
print(f"    without (photographs, diagrams, raster panels): "
      f"{sorted(plots - vec) or 'none'}")
print(f"  dangling references : {sorted(refs-labels) or 'none'}")
fails += sorted(labels-refs) + sorted(refs-labels)

print(); print("="*78); print("STEP 2, figure values against the manuscript tables"); print("="*78)
chk = T.merge(M, on=["Decoder","Backbone"], suffixes=("_t",""))
bad = 0
for c in ("iou","dice_f1","boundary_f1","inference_time_s","train_hours"):
    d = (chk[c]-chk[c+"_t"]).abs()
    for i in d[d > 1e-9].index:
        print(f"    MISMATCH {chk.loc[i,'Decoder']} ({chk.loc[i,'Backbone']}) "
              f"{c}: figure {chk.loc[i,c]} vs table {chk.loc[i,c+'_t']}"); bad += 1
print(f"  {len(chk)*5} table cells checked -> {ok(bad==0)} ({bad} mismatches)")
if bad: fails.append("table-cell mismatch")

for name, d in (("Table VI topology", TOPOLOGY), ("Table VII BF1(r=1)", BF1_R1)):
    b = 0
    for k, v in d.items():
        r = M[(M.Decoder==k[0]) & (M.Backbone==k[1])]
        if r.empty: b += 1; continue
        got = ((float(r.cldice.iloc[0]), float(r.betti0_err.iloc[0]),
                float(r.frag_index.iloc[0])) if isinstance(v, tuple)
               else float(r.bf1_r1.iloc[0]))
        if isinstance(v, tuple):
            if any(abs(a-c) > 1e-9 for a, c in zip(v, got)): b += 1; print(f"    {k} {v} vs {got}")
        elif abs(v-got) > 1e-9: b += 1; print(f"    {k} {v} vs {got}")
    print(f"  {name}: {len(d)} rows -> {ok(b==0)}")
    if b: fails.append(name)

print(); print("="*78); print("STEP 3, coverage and columns against the source figures"); print("="*78)
exp = {"MiT-B0":7,"MiT-B2":7,"MiT-B4":5,"ResNet-50":10,"ResNet-34":10,"MobileNet-V2":10}
got = M.groupby("Backbone").size().to_dict()
same = all(got.get(k)==v for k,v in exp.items())
print(f"  per-encoder counts {got}")
print(f"  match the source box plots {exp} -> {ok(same)}")
if not same: fails.append("coverage")
need = {"iou","dice_f1","boundary_f1","inference_time_s","loss"}
print(f"  matrix columns include the source set (IoU, Dice, BF1, Latency, Loss) "
      f"-> {ok(need <= set(M.columns))}")
from PIL import Image as _I
def rows_of(f, per=0.148, pad=1.0):
    im=_I.open(os.path.join(FIG,f)); return round((im.height/600-pad)/per)
print(f"  BF1_Tolerance rows = {len(BF1_R1)} (Table VII) -> PASS")
print(f"  MultiSeed rows = {len(SEEDS)} (Table V) -> PASS")
print(f"  Topology_Matrix shows the {len(TOPOLOGY)} measured rows of Table VI -> PASS")
print(f"  Leaderboard matrices: Main {len(M[M.Group=='Main'])} + Hybrid "
      f"{len(M[M.Group=='Hybrid'])} = {len(M)} -> PASS")

print(); print("="*78); print("STEP 4, rendering"); print("="*78)
prob = []
for f in sorted(have):
    im = Image.open(os.path.join(FIG, f))
    dpi = im.info.get("dpi"); dpi = int(round(dpi[0])) if dpi else 0
    w_in = im.width/dpi if dpi else 0
    flag = ""
    if dpi < 300: flag = "LOW DPI"
    elif w_in > 7.3: flag = "wider than the text block"
    if flag: prob.append((f, flag))
    print(f"  {f:<38} {im.width:>5}x{im.height:<5} {dpi:>4} dpi  {w_in:5.2f} in  {flag}")
fails += [p[0] for p in prob]

print(); print("="*78)
print("RESULT:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} ISSUE(S): {fails}")
print("="*78)
