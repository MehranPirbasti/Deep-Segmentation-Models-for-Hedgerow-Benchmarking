"""Draws the figures. Reads master.csv, writes PNG plus SVG and PDF."""
import os, sys, argparse
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import paperstyle as PS
from tables import seeds, ABLATION

ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True)
ap.add_argument("--topology",action="store_true"); A=ap.parse_args()
OUT=A.out; os.makedirs(OUT,exist_ok=True); PS.apply(); p=lambda f:os.path.join(OUT,f)
M=pd.read_csv("../data/master.csv"); S=seeds()
HI={"iou","dice_f1","boundary_f1","bf1_r1","cldice"}
LBL={"iou":"Test IoU","dice_f1":"Test Dice F1","boundary_f1":"Test Boundary F1",
     "bf1_r1":r"Test BF1 ($r=1$)","cldice":"Test clDice","betti0_err":"Test Betti-0 Error",
     "frag_index":"Test Fragmentation Index","inference_time_s":"Inference Latency (s)",
     "train_hours":"Total Training Time (Hours)","loss":"Validation loss"}
SHORT={"iou":"IoU","dice_f1":"Dice / F1","boundary_f1":r"BF1 ($r{=}2$)",
       "bf1_r1":r"BF1 ($r{=}1$)","cldice":"clDice","betti0_err":"Betti-0 err.",
       "frag_index":"Frag.","inference_time_s":"Latency (s)","loss":"Loss"}

# Two decimals for every metric: the third decimal is zero for the values fixed
# by the manuscript tables, so printing it would advertise a precision the study
# does not have. The validation loss is the exception, since at two decimals it
# would collapse to two distinct values.
FMT = {"loss": "{:.3f}"}
def matrix(df, cols, title, path, width, fmt="{:.2f}"):
    n=df[cols].copy()
    for c in cols:
        r=df[c].max()-df[c].min()
        v=(df[c]-df[c].min())/r if r>1e-12 else 0.5
        n[c]= v if c in HI else 1-v
    fig,ax=plt.subplots(figsize=(width,0.148*len(df)+1.0))
    im=ax.imshow(n.values,cmap=PS.MATRIX_CMAP,vmin=0,vmax=1,aspect="auto")
    for i in range(len(df)):
        for j,c in enumerate(cols):
            ax.text(j,i,FMT.get(c,fmt).format(df[c].iloc[i]),ha="center",va="center",fontsize=5.2,
                    color="black" if n[c].iloc[i]>0.55 else "white")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([SHORT[c] for c in cols],fontsize=6.4,rotation=24,ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df.index,fontsize=5.4)
    ax.set_xticks(np.arange(-.5,len(cols),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(df),1),minor=True)
    ax.grid(which="minor",color="white",linewidth=0.45); ax.grid(which="major",visible=False)
    ax.tick_params(which="both",length=0); ax.set_title(title,fontsize=8.0,pad=4)
    cb=fig.colorbar(im,ax=ax,fraction=0.026,pad=0.014,ticks=[0,1])
    cb.ax.set_yticklabels(["worst","best"],fontsize=5.8); cb.outline.set_linewidth(0.5)
    PS.save(fig,path,target_w=width)

# Column set of the original performance matrices, plus the topology indices.
cols = ["iou","dice_f1","boundary_f1","cldice","betti0_err","frag_index",
        "inference_time_s","loss"]
print("matrices")
for grp,fn,ti in (("Main","Leaderboard_Matrix_Main.png",
                   "Elite Models Performance Matrix (Test Set)"),
                  ("Hybrid","Leaderboard_Matrix_Hybrid.png",
                   "Hybrid Models Performance Matrix (Test Set)")):
    g=M[M.Group==grp].set_index("Model")
    matrix(g,cols,ti,p(fn),PS.W_050)
if A.topology:
    T=M[M.topo_measured].sort_values("iou",ascending=False).set_index("Model")
    matrix(T,["iou","cldice","betti0_err","frag_index"],
           "Topology-Aware Evaluation Matrix (Test Set)",
           p("Topology_Matrix.png"),PS.W_050)

# radar
print("radar")
AX=[("iou","IoU",False),("dice_f1","Dice / F1",False),("boundary_f1",r"BF1 ($r{=}2$)",False)]
if A.topology: AX+=[("cldice","clDice",False),("betti0_err","Betti-0\n(inv.)",True),
                    ("frag_index","Frag.\n(inv.)",True)]
AX+=[("inference_time_s","Inference\nspeed",True)]
rng={c:(M[c].min(),M[c].max()) for c,_,_ in AX}
ang=np.linspace(0,2*np.pi,len(AX),endpoint=False).tolist(); ang+=ang[:1]
fig,axes=plt.subplots(1,2,figsize=(PS.W_FULL,3.15),subplot_kw={"polar":True})
for ax,grp in zip(axes,("Main","Hybrid")):
    g=M[M.Group==grp]
    for _,r in g.iterrows():
        v=[]
        for c,_,inv in AX:
            lo,hi=rng[c]; t=(r[c]-lo)/(hi-lo) if hi-lo>1e-12 else .5
            v.append(1-t if inv else t)
        v+=v[:1]
        ax.plot(ang,v,lw=0.85,alpha=0.85,color=PS.BACKBONE_COLOR[r.Backbone],
                ls="-" if r.Backbone.startswith(("ResNet","Mobile")) else "-")
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([a[1] for a in AX],fontsize=6.0)
    ax.set_ylim(0,1); ax.set_yticks([.25,.5,.75])
    ax.set_yticklabels(["0.25","0.50","0.75"],fontsize=5.0,color="#888888")
    ax.set_rlabel_position(200); ax.grid(lw=0.35,alpha=0.45); ax.tick_params(pad=-1)
    ax.set_title(f"({'a' if grp=='Main' else 'b'}) {grp} pairings ($n={len(g)}$)",
                 fontsize=7.6,pad=10)
hd=[Line2D([],[],color=c,lw=1.2,label=b) for b,c in PS.BACKBONE_COLOR.items()]
hs=[Line2D([],[],color="0.35",lw=1.2,ls=s,label=l) for s,l in
    (("-","Convolutional encoder"),("-","Transformer encoder"))]
lg=fig.legend(handles=hd+hs,loc="lower center",ncol=8,fontsize=6.0,
              bbox_to_anchor=(0.5,-0.11),columnspacing=0.8,handlelength=1.3)
lg.get_frame().set_linewidth(0.5)
fig.suptitle("Cross-Metric Operating-Point Comparison (Test Set)",fontsize=8.2,y=1.03)
PS.save(fig,p("Test_Set_Radar_All.png"),target_w=PS.W_FULL)

# scatter
def scatter(x,y,path,title,w=PS.W_048,h=None,annot=0):
    """
    Accuracy against cost, one point per configuration.

    Color identifies the encoder backbone and marker shape the decoder, as in
    the corresponding figures of the original submission, so the same clusters
    appear in the same colors.
    """
    d=M.dropna(subset=[x,y])
    h = h or w/1.50
    fig,ax=plt.subplots(figsize=(w,h))
    for _,r in d.iterrows():
        ax.scatter(r[x],r[y],s=26,color=PS.BACKBONE_COLOR[r.Backbone],
                   marker=PS.DECODER_MARKER[r.Decoder],edgecolor="black",
                   linewidth=0.35,zorder=3)
    ax.set_xlabel(LBL[x]); ax.set_ylabel(LBL[y])
    ax.set_title(title,fontsize=8.0,pad=3)
    hb=[Line2D([],[],lw=0,marker="o",ms=3.2,mec="black",mew=0.3,color=c,label=b)
        for b,c in PS.BACKBONE_COLOR.items()]
    hd=[Line2D([],[],lw=0,marker=mk,ms=3.2,mec="black",mew=0.3,color="0.45",label=dd)
        for dd,mk in PS.DECODER_MARKER.items()]
    lg=ax.legend(handles=hb+[Line2D([],[],lw=0,label="")]+hd,
                 loc="center left",bbox_to_anchor=(1.01,0.5),fontsize=5.2,
                 handletextpad=0.4,labelspacing=0.22,borderaxespad=0.0,
                 title="Backbone / Architecture",title_fontsize=5.6)
    lg.get_frame().set_linewidth(0.5)
    PS.save(fig,path,target_w=w,target_ar=1.50)

print("frontiers")
mets=["iou","dice_f1","boundary_f1"]
for m in mets:
    scatter("inference_time_s",m,p(f"Efficiency_Frontier_{m}.png"),
            f"Efficiency Frontier: Accuracy ({LBL[m]}) vs. Latency")
for m in ["iou","dice_f1","boundary_f1"]:
    scatter("train_hours",m,p(f"Training_Efficiency_{m}.png"),
            f"Training Cost vs {LBL[m]}")

# encoder-family effect on the frontier (new)
if A.topology:
    # A per-family summary rather than a second copy of the frontier scatter:
    # each family is shown at its median cost and median score with the
    # interquartile range on both axes, over faint individual configurations.
    # Fitted trend lines were tried and removed, since with five to ten points
    # per family they invited a reading the data does not support.
    fig,axes=plt.subplots(1,2,figsize=(PS.W_FULL,PS.W_FULL/2/1.50))
    for ax,m in zip(axes,("iou","cldice")):
        for _,r in M.iterrows():
            ax.scatter(r.inference_time_s,r[m],s=13,
                       color=PS.BACKBONE_COLOR[r.Backbone],alpha=0.28,
                       marker="o",linewidth=0,zorder=2)
        for fam,c in PS.BACKBONE_COLOR.items():
            sub=M[M.Backbone==fam]
            if sub.empty: continue
            x,y=sub.inference_time_s.median(),sub[m].median()
            xe=[[x-sub.inference_time_s.quantile(.25)],[sub.inference_time_s.quantile(.75)-x]]
            ye=[[y-sub[m].quantile(.25)],[sub[m].quantile(.75)-y]]
            ax.errorbar(x,y,xerr=xe,yerr=ye,fmt="o",ms=5.0,color=c,mec="black",
                        mew=0.6,ecolor=c,elinewidth=1.2,capsize=2.0,zorder=4)
            OFF={"MiT-B0":(6,-9),"MiT-B2":(6,4),"MiT-B4":(-30,4),
                 "ResNet-50":(6,-9),"ResNet-34":(6,5),"MobileNet-V2":(6,4)}
            ax.annotate(fam,(x,y),textcoords="offset points",
                        xytext=OFF.get(fam,(6,4)),fontsize=5.6,color="black")
        ax.set_xlabel(LBL["inference_time_s"]); ax.set_ylabel(LBL[m])
        ax.set_title(f"({'a' if m=='iou' else 'b'}) {SHORT[m]}",fontsize=8.0,pad=3)
    hb=[Line2D([],[],lw=0,marker="o",ms=3.6,mec="black",mew=0.4,color=c,label=b)
        for b,c in PS.BACKBONE_COLOR.items()]
    lg=fig.legend(handles=hb,loc="lower center",ncol=6,fontsize=6.0,
                  bbox_to_anchor=(0.5,-0.11),
                  title="Encoder backbone. Marker: median; bars: interquartile "
                        "range; faint dots: individual configurations",
                  title_fontsize=5.8)
    lg.get_frame().set_linewidth(0.5)
    fig.suptitle("Impact of Encoder Family on the Efficiency Frontier",
                 fontsize=8.2,y=1.02)
    PS.save(fig,p("Encoder_Family_Frontier.png"),target_w=PS.W_FULL)

# box plots by backbone
print("box plots")
box_metrics=["iou","dice_f1","boundary_f1"]
for m in box_metrics:
    # Encoder order is fixed as in the original figures rather than sorted, so
    # a reader comparing the two submissions sees the same columns in the same
    # places.
    order=[b for b in ["MiT-B0","MiT-B2","MiT-B4","ResNet-50","ResNet-34",
                       "MobileNet-V2"] if b in set(M.Backbone)]
    fig,ax=plt.subplots(figsize=(PS.W_048,PS.W_048/1.62))
    # Boxes are tinted in the encoder's own color, as in the original figures.
    bp=ax.boxplot([M[M.Backbone==b][m].values for b in order],widths=0.55,
                  patch_artist=True,showfliers=False,
                  medianprops=dict(color="black",lw=1.0),
                  whiskerprops=dict(lw=0.6),capprops=dict(lw=0.6))
    for patch,b in zip(bp["boxes"],order):
        patch.set_facecolor(PS.BACKBONE_COLOR[b]); patch.set_alpha(0.42)
        patch.set_edgecolor("#333333"); patch.set_linewidth(0.6)
    for i,b in enumerate(order,1):
        sub=M[M.Backbone==b]; xs=i+np.linspace(-0.17,0.17,len(sub))
        for x,(_,r) in zip(xs,sub.iterrows()):
            ax.scatter(x,r[m],s=15,color=PS.BACKBONE_COLOR[b],
                       marker=PS.DECODER_MARKER[r.Decoder],edgecolor="black",
                       linewidth=0.25,zorder=3)
    ax.set_xticks(range(1,len(order)+1))
    ax.set_xticklabels(order,fontsize=5.9,rotation=12,ha="right",rotation_mode="anchor")
    ax.set_ylabel("Best "+LBL[m]); ax.set_xlabel("Encoder Architecture")
    ax.set_title(f"Impact of Encoder on {LBL[m].replace('Test ','Test ')}",fontsize=8.0,pad=3)
    hd=[Line2D([],[],lw=0,marker=mk,ms=3.2,mec="black",mew=0.3,color="0.45",label=dd)
        for dd,mk in PS.DECODER_MARKER.items()]
    lg=ax.legend(handles=hd,loc="center left",bbox_to_anchor=(1.01,0.5),
                 fontsize=5.4,title="Decoder",title_fontsize=5.8,
                 handletextpad=0.4,labelspacing=0.24,borderaxespad=0.0)
    lg.get_frame().set_linewidth(0.5)
    PS.save(fig,p(f"Boxplot_{m}.png"),target_w=PS.W_048,target_ar=1.62)

# decoder-family distributions
print("decoder distributions")
# The decoder-distribution panel of the original submission plots BEST VALIDATION
# IoU and colors the points by encoder family, so both are reproduced here.
DV=pd.read_csv("../data/decoder_val_iou.csv")
FAMCOL={"MiT (transformer)":"#DA3C3D","ResNet-50":"#3584BB",
        "ResNet-34":"#FF8B26","MobileNet-V2":"#41A941"}
for m in ["iou"]:
    if m=="iou":
        src=DV.rename(columns={"val_iou":"v"}); ylab="Best Validation IoU"
        colof=lambda r: FAMCOL[r.EncoderFamily]; mkof=lambda r: "o"
        handles=[Line2D([],[],lw=0,marker="o",ms=3.4,mec="black",mew=0.3,color=c,label=k)
                 for k,c in FAMCOL.items()]; ltitle="Encoder family"
    else:
        src=M.rename(columns={m:"v"}); ylab=LBL[m]
        colof=lambda r: PS.BACKBONE_COLOR[r.Backbone]
        mkof=lambda r: PS.DECODER_MARKER[r.Decoder]
        handles=[Line2D([],[],lw=0,marker="o",ms=3.4,mec="black",mew=0.3,color=c,label=b)
                 for b,c in PS.BACKBONE_COLOR.items()]; ltitle="Backbone family"
    order=src.groupby("Decoder")["v"].median().sort_values(ascending=False).index
    fig,ax=plt.subplots(figsize=(PS.W_FULL*0.62,PS.W_FULL*0.62/1.71))
    bp=ax.boxplot([src[src.Decoder==d]["v"].values for d in order],widths=0.55,
                  patch_artist=True,showfliers=False,
                  medianprops=dict(color="black",lw=1.0),
                  whiskerprops=dict(lw=0.6),capprops=dict(lw=0.6))
    for patch in bp["boxes"]:
        patch.set_facecolor("white"); patch.set_edgecolor("#555555"); patch.set_linewidth(0.6)
    for i,d in enumerate(order,1):
        sub=src[src.Decoder==d]; xs=i+np.linspace(-0.17,0.17,len(sub))
        for x,(_,r) in zip(xs,sub.iterrows()):
            ax.scatter(x,r["v"],s=15,color=colof(r),marker=mkof(r),
                       edgecolor="black",linewidth=0.25,zorder=3)
    ax.set_xticks(range(1,len(order)+1))
    ax.set_xticklabels(order,rotation=16,ha="right",rotation_mode="anchor",fontsize=6.2)
    ax.set_ylabel(ylab); ax.set_xlabel("Decoder Architecture")
    ax.set_title(f"Decoder Performance Distribution: {SHORT[m]}",fontsize=8.0,pad=3)
    lg=ax.legend(handles=handles,loc="center left",bbox_to_anchor=(1.01,0.5),
                 fontsize=5.6,title=ltitle,title_fontsize=6.0,
                 handletextpad=0.4,labelspacing=0.28,borderaxespad=0.0)
    lg.get_frame().set_linewidth(0.5)
    PS.save(fig,p("Decoder_Boxplot.png"),target_w=PS.W_FULL*0.62,target_ar=1.71)

# shared key
fig,ax=plt.subplots(figsize=(PS.W_044,0.95)); ax.axis("off")
hb=[Line2D([],[],color=c,lw=0,marker="o",ms=3.4,mec="black",mew=0.25,label=b)
    for b,c in PS.BACKBONE_COLOR.items()]
hd=[Line2D([],[],color="0.45",lw=0,marker=mk,ms=3.4,mec="black",mew=0.25,label=d)
    for d,mk in PS.DECODER_MARKER.items()]
l1=ax.legend(handles=hb,loc="upper center",ncol=6,fontsize=5.8,
             title="Backbone family (color)",bbox_to_anchor=(0.5,1.06))
l1.get_title().set_fontsize(6.2); ax.add_artist(l1)
l2=ax.legend(handles=hd,loc="lower center",ncol=5,fontsize=5.8,
             title="Architecture (marker)",bbox_to_anchor=(0.5,-0.06))
l2.get_title().set_fontsize(6.2)
PS.save(fig,p("Legend_Key.png"),target_w=PS.W_044)

# topology-aware indices, the nine configurations of Table VI
if A.topology:
    print("topology panels")
    # Topology indices in the scatter idiom used by the frontier panels, so the
    # whole paper reads in one style. All configurations are shown, with the
    # nine measured rows of Table VI ringed so that measured and extended values
    # are distinguishable at a glance.
    meas = M.topo_measured.values
    spec=[("cldice","clDice",True),("betti0_err","Betti-0 error",False),
          ("frag_index","Fragmentation index",False)]

    fig,axes=plt.subplots(1,3,figsize=(PS.W_FULL,PS.W_FULL/3/1.28))
    for ax,(col,lab,hi) in zip(axes,spec):
        for k,(_,r) in enumerate(M.iterrows()):
            ax.scatter(r.iou,r[col],s=26 if meas[k] else 15,
                       color=PS.BACKBONE_COLOR[r.Backbone],
                       marker=PS.DECODER_MARKER[r.Decoder],
                       alpha=1.0 if meas[k] else 0.42,
                       edgecolor="black",linewidth=1.0 if meas[k] else 0.25,
                       zorder=4 if meas[k] else 3)
        # The fit and the coefficient are computed on the MEASURED rows only.
        # The extended values are derived from accuracy, so including them would
        # report a correlation that the extension itself created.
        Mm=M[M.topo_measured]
        z=np.polyfit(Mm.iou,Mm[col],1)
        xs=np.linspace(M.iou.min(),M.iou.max(),40)
        rho=pd.Series(Mm.iou).corr(pd.Series(Mm[col]),method="spearman")
        ax.plot(xs,np.polyval(z,xs),lw=0.8,ls="-",color="#777777",zorder=2,
                label=f"fit on measured, $\\rho={rho:+.2f}$")
        ax.legend(loc="best",fontsize=5.6)
        ax.set_xlabel("Test IoU"); ax.set_ylabel(lab)
        ax.set_title(f"({'abc'[spec.index((col,lab,hi))]}) {lab}: "
                     f"{'higher' if hi else 'lower'} is better",fontsize=7.2,pad=3)
    hb=[Line2D([],[],lw=0,marker="o",ms=3.2,mec="black",mew=0.3,color=c,label=b)
        for b,c in PS.BACKBONE_COLOR.items()]
    hd=[Line2D([],[],lw=0,marker=mk,ms=3.2,mec="black",mew=0.3,color="0.45",label=d)
        for d,mk in PS.DECODER_MARKER.items()]
    hm=[Line2D([],[],lw=0,marker="o",ms=3.6,mfc="none",mec="black",mew=0.9,
               label="measured (Table VI); faint = extended")]
    lg=fig.legend(handles=hb+hd+hm,loc="lower center",ncol=9,fontsize=5.6,
                  bbox_to_anchor=(0.5,-0.20),columnspacing=0.7,handlelength=1.2)
    lg.get_frame().set_linewidth(0.5)
    fig.suptitle("Topology-Aware Indices against Region Overlap (Test Set)",
                 fontsize=8.2,y=1.03)
    PS.save(fig,p("Topology_Indices.png"),target_w=PS.W_FULL)

    # connectivity against inference cost, all configurations
    fig,axes=plt.subplots(1,2,figsize=(PS.W_FULL,PS.W_FULL/2/1.50))
    for ax,(col,lab,hi) in zip(axes,[spec[0],spec[1]]):
        for k,(_,r) in enumerate(M.iterrows()):
            ax.scatter(r.inference_time_s,r[col],s=26 if meas[k] else 15,
                       color=PS.BACKBONE_COLOR[r.Backbone],
                       marker=PS.DECODER_MARKER[r.Decoder],
                       alpha=1.0 if meas[k] else 0.42,edgecolor="black",
                       linewidth=1.0 if meas[k] else 0.25,
                       zorder=4 if meas[k] else 3)
        d=M.sort_values("inference_time_s")
        ax.step(d.inference_time_s,d[col].cummax() if hi else d[col].cummin(),
                where="post",color="#B00020",ls="-",lw=0.9,zorder=2,
                label="Pareto frontier")
        ax.set_xlabel(LBL["inference_time_s"]); ax.set_ylabel(lab)
        ax.set_title(f"({'a' if hi else 'b'}) {lab}: "
                     f"{'higher' if hi else 'lower'} is better",fontsize=7.6,pad=3)
        ax.legend(loc="best",fontsize=6.0)
    lg=fig.legend(handles=hb+hd+hm,loc="lower center",ncol=9,fontsize=5.6,
                  bbox_to_anchor=(0.5,-0.16),columnspacing=0.7,handlelength=1.2)
    lg.get_frame().set_linewidth(0.5)
    fig.suptitle("Connectivity against Inference Cost (Test Set)",fontsize=8.2,y=1.02)
    PS.save(fig,p("Topology_Frontier.png"),target_w=PS.W_FULL)

# shared key
fig,ax=plt.subplots(figsize=(PS.W_044,0.95)); ax.axis("off")
hb=[Line2D([],[],color=c,lw=0,marker="o",ms=3.4,mec="black",mew=0.25,label=b)
    for b,c in PS.BACKBONE_COLOR.items()]
hd=[Line2D([],[],color="0.45",lw=0,marker=mk,ms=3.4,mec="black",mew=0.25,label=d)
    for d,mk in PS.DECODER_MARKER.items()]
l1=ax.legend(handles=hb,loc="upper center",ncol=6,fontsize=5.8,
             title="Backbone family (color)",bbox_to_anchor=(0.5,1.06))
l1.get_title().set_fontsize(6.2); ax.add_artist(l1)
l2=ax.legend(handles=hd,loc="lower center",ncol=5,fontsize=5.8,
             title="Architecture (marker)",bbox_to_anchor=(0.5,-0.06))
l2.get_title().set_fontsize(6.2)
PS.save(fig,p("Legend_Key.png"),target_w=PS.W_044)

# topology-aware indices, the nine configurations of Table VI
# BF1 tolerance
# Exactly the nine configurations of Table VII, so the figure and the table
# list the same models in the same order.
from tables import BF1_R1 as _T7
t=M[M.set_index(["Decoder","Backbone"]).index.isin(_T7.keys())].sort_values(
    "boundary_f1",ascending=False)
fig,ax=plt.subplots(figsize=(PS.W_048,2.55)); y=np.arange(len(t))
ax.hlines(y,t.bf1_r1,t.boundary_f1,color="#b8c0cc",lw=2.4,zorder=1)
ax.scatter(t.bf1_r1,y,s=20,color="#D55E00",marker="o",zorder=3,edgecolor="black",lw=0.3,label=r"$r=1$ px")
ax.scatter(t.boundary_f1,y,s=20,color="#0173B2",marker="s",zorder=3,edgecolor="black",lw=0.3,label=r"$r=2$ px")
for i,(a_,b_) in enumerate(zip(t.bf1_r1,t.boundary_f1)):
    ax.text(b_+0.005,i,f"{b_-a_:.2f}",va="center",fontsize=5.4)
ax.set_yticks(y); ax.set_yticklabels(t.Model,fontsize=5.8); ax.invert_yaxis()
ax.set_xlim(0.70,1.00); ax.set_xlabel("Boundary F1")
ax.set_title("BF1 Sensitivity to the Tolerance Radius",fontsize=8.0,pad=3)
ax.legend(loc="lower left",fontsize=6.0)
PS.save(fig,p("BF1_Tolerance.png"),target_w=PS.W_048)

# multi-seed
fig,ax=plt.subplots(figsize=(PS.W_LINE,2.35)); y=np.arange(len(S))
ax.axvspan(S.hi[1:].max(),S.lo[0],color="#029E73",alpha=0.13,lw=0,
           label="separation from the second tier")
ax.errorbar(S.iou_mean,y,xerr=S.ci95,fmt="D",ms=3.4,mfc="white",mec="#0173B2",mew=1.0,
            ecolor="#0173B2",elinewidth=1.0,capsize=2.2,capthick=0.8,ls="none",zorder=3)
for i,(mu,ci) in enumerate(zip(S.iou_mean,S.ci95)):
    ax.text(mu+ci+0.004,i,f"{mu:.2f}",va="center",fontsize=5.8)
ax.set_yticks(y); ax.set_yticklabels(S.Model,fontsize=6.4); ax.invert_yaxis()
ax.set_xlim(min(S.lo)-0.012,max(S.hi)+0.021)
ax.set_xlabel("Test IoU (mean $\\pm$ 95% CI over five seeds)")
ax.set_title("Multi-Seed Variability (Test IoU)",fontsize=8.0,pad=3)
ax.legend(loc="lower right",fontsize=6.0)
PS.save(fig,p("MultiSeed_CI_IoU.png"),target_w=PS.W_LINE)

# The deep-supervision ablation is reported in the text and its table; a
# three-bar figure would carry no information the two numbers do not.

print("done")
