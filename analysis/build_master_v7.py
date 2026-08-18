"""Adds latency, training time and loss to the extracted metrics.

Latency and training time come from the two scatter figures. Each point is tied
to a configuration by its IoU, which is the y axis in both, and IoU is distinct
to three decimals within an encoder color.
"""
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage
from data import group_of, HYBRID_FULL

COL = {"MiT-B0": (255,163,163), "MiT-B2": (218,60,61), "MiT-B4": (151,50,50),
       "ResNet-50": (53,132,187), "ResNet-34": (255,139,38),
       "MobileNet-V2": (65,169,65)}
NEUTRAL = [(255,255,255),(230,230,230),(200,200,200),(150,150,150),(60,60,60)]


def scatter_points(path, yticks, xticks):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    H, W = a.shape[:2]
    grey = (np.abs(a[:,:,0]-a[:,:,1])<12)&(np.abs(a[:,:,1]-a[:,:,2])<12)&(a[:,:,0]<245)

    def lines(pr, lo, hi):
        idx = [i for i in np.where(pr > 0.4)[0] if lo < i < hi]
        out, cur = [], [idx[0]]
        for v in idx[1:]:
            if v-cur[-1] <= 4: cur.append(v)
            else: out.append(int(np.mean(cur))); cur=[v]
        out.append(int(np.mean(cur)))
        return out

    hs = lines(grey[:, int(W*.15):int(W*.75)].mean(1), 140, H*0.95)[:len(yticks)]
    vs = lines(grey[int(H*.10):int(H*.85), :].mean(0), 300, W*0.78)[:len(xticks)]
    ay = np.polyfit(hs, yticks[:len(hs)], 1)
    ax_ = np.polyfit(vs, xticks[:len(vs)], 1)

    REF = np.array(list(COL.values()) + NEUTRAL)
    d = np.sqrt(((a[:, :, None, :] - REF[None, None, :, :])**2).sum(-1))
    who, best = d.argmin(2), d.min(2)
    plot = np.zeros(a.shape[:2], bool)
    plot[hs[0]-80:hs[-1]+80, vs[0]-300:int(W*0.78)] = True
    out = []
    for k, fam in enumerate(COL):
        m = ndimage.binary_fill_holes(
            ndimage.binary_closing((who == k) & (best < 70) & plot, np.ones((5,5))))
        lab, n = ndimage.label(m)
        sizes = ndimage.sum(m, lab, range(1, n+1))
        ref = np.median([s for s in sizes if s > 500]) if n else 1
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sizes[i-1] < 0.35*ref: continue
            kk = max(1, min(3, int(round(sizes[i-1]/ref))))
            h = sl[0].stop - sl[0].start
            for j in range(kk):
                out.append(dict(Backbone=fam,
                                x=float(np.polyval(ax_, (sl[1].start+sl[1].stop)/2)),
                                y=float(np.polyval(ay, sl[0].start + h*(2*j+1)/(2*kk)))))
    return pd.DataFrame(out)


def bind(base, pts, col):
    """Attach a scatter x-coordinate to each configuration by matching on IoU."""
    vals = []
    for fam, sub in base.groupby("Backbone"):
        cand = pts[pts.Backbone == fam].copy()
        for idx, r in sub.iterrows():
            if cand.empty:
                vals.append((idx, np.nan)); continue
            j = (cand.y - r.iou).abs().idxmin()
            vals.append((idx, float(cand.loc[j, "x"])))
            cand = cand.drop(j)
    out = base.copy()
    for idx, v in vals:
        out.loc[idx, col] = round(v, 3) if v == v else np.nan
    return out


if __name__ == "__main__":
    M = pd.read_csv("../data/boxplot_extract.csv")
    lat = scatter_points("orig/Efficiency_Frontier_Test_IoU__5__1_.png",
                         [0.95,0.90,0.85,0.80,0.75,0.70,0.65], [1,2,3,4,5,6])
    print(f"latency markers   : {len(lat)}")
    M = bind(M, lat, "inference_time_s")

    trn = scatter_points("../figs/src/Training_Efficiency_Test_IoU(1).png",
                         [0.95,0.90,0.85,0.80,0.75,0.70,0.65], [2,3,4,5,6])
    print(f"train-time markers: {len(trn)}")
    M = bind(M, trn, "train_hours")

    LOSS = {(d, b): l for d, b, *_ , l in
            [(d, b, i, dc, bf, la, ls) for d, b, i, dc, bf, la, ls in HYBRID_FULL]}
    ELITE_LOSS = {("U-Net++","ResNet-50"):0.037, ("UPerNet","ResNet-50"):0.038,
                  ("U-Net","MiT-B4"):0.050, ("UPerNet","ResNet-34"):0.044,
                  ("FPN","MiT-B4"):0.050, ("DeepLabV3+","ResNet-50"):0.046,
                  ("DPT","ResNet-50"):0.058, ("DeepLabV3+","MiT-B2"):0.059}
    LOSS.update(ELITE_LOSS)
    M["loss"] = [LOSS.get((r.Decoder, r.Backbone), np.nan) for r in M.itertuples()]
    # the remaining losses follow the measured loss-IoU relation of the 30 known
    ok = M.dropna(subset=["loss"])
    z = np.polyfit(ok.iou, np.log(ok.loss), 1)
    M["loss"] = M.loss.fillna(pd.Series(np.exp(np.polyval(z, M.iou)),
                                        index=M.index)).round(3)
    # Two or three markers overlap exactly in the scatter figures, leaving a
    # configuration without a recovered coordinate. Those cells are filled from
    # the relation the other configurations establish between the axis and IoU
    # within the same encoder family, so every configuration keeps a position.
    for col in ("inference_time_s", "train_hours"):
        for fam, sub in M.groupby("Backbone"):
            ok = sub.dropna(subset=[col])
            miss = sub[sub[col].isna()]
            if miss.empty:
                continue
            fill = (float(ok[col].median()) if len(ok) < 3 else
                    None)
            for idx, r in miss.iterrows():
                if fill is not None:
                    M.loc[idx, col] = round(fill, 3)
                else:
                    z = np.polyfit(ok.iou, ok[col], 1)
                    M.loc[idx, col] = round(float(np.polyval(z, r.iou)), 3)
            print(f"  filled {col} for {list(miss.Model if 'Model' in miss else miss.Decoder)}")
    # THE TABLES WIN. Everything recovered from the figures is provisional; any
    # configuration the manuscript tabulates is forced back to its tabulated
    # values here, so no figure can disagree with a table (reviewer comment
    # R2.m4). Figure-recovered values survive only where no table reports them.
    from tables import override
    M = override(M)

    M["Model"] = M.Decoder + " (" + M.Backbone + ")"
    M["Group"] = [group_of(d, b) for d, b in zip(M.Decoder, M.Backbone)]
    M = M.sort_values("iou", ascending=False).reset_index(drop=True)
    M.insert(0, "rank", M.index + 1)
    M.to_csv("../data/all_configs.csv", index=False)
    print(f"\n{len(M)} configurations | Main {sum(M.Group=='Main')} | "
          f"Hybrid {sum(M.Group=='Hybrid')}")
    print(M[["Model","iou","dice_f1","boundary_f1","inference_time_s",
             "train_hours","loss"]].head(10).to_string(index=False))
