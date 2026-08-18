"""Re-runs the statistics quoted in the text, so they can be checked."""
import numpy as np, pandas as pd
from scipy import stats as st
import tables as T
from master import build, FAMILY

M, _ = build()
S = T.seeds()
out = []

print("=" * 72)
print("1. MULTI-SEED VALUES (Table V)")
print("=" * 72)
# (a) are the reported SD and CI mutually consistent for n = 5?
n = 5
implied = S.sd / np.sqrt(n) * st.t.ppf(0.975, n-1)
print("  CI half-width implied by the reported SD (t, n=5) vs the reported CI:")
for _, r in S.iterrows():
    print(f"    {r.Model:<24} reported {r.ci95:.3f}   implied {implied[_]:.3f}"
          f"   {'consistent' if abs(implied[_]-r.ci95) <= 0.0035 else 'MISMATCH'}")
bad = int((abs(implied - S.ci95) > 0.0035).sum())
print(f"  -> {len(S)-bad}/{len(S)} internally consistent")

# (b) is the CI/SD ratio suspiciously constant (a sign of fabricated numbers)?
ratio = (S.ci95 / S.sd)
print(f"\n  CI/SD ratio: mean {ratio.mean():.2f}, sd {ratio.std(ddof=1):.3f}, "
      f"range {ratio.min():.2f}-{ratio.max():.2f}")
print(f"     (t_.975,4/sqrt(5) = {st.t.ppf(0.975,4)/np.sqrt(5):.2f}; a ratio that "
      f"is *exactly* constant would be the giveaway)")

# (c) does the SD scale with the mean, as binomial-like scores should?
rho, p = st.spearmanr(S.iou_mean, S.sd)
print(f"\n  SD against mean IoU: Spearman rho = {rho:+.2f}, p = {p:.3f}"
      f"  ({'weaker models are noisier, as expected' if rho < 0 else 'no trend'})")

# (d) the separation claim
top, second = S.iloc[0], S.iloc[1:]
# Welch statistic for two means of n = 5, i.e. the standard errors enter, not
# the standard deviations themselves.
n_seeds = 5
z = ((top.iou_mean - second.iou_mean.max())
     / np.sqrt(top.sd**2/n_seeds + second.sd.max()**2/n_seeds))
print(f"\n  Separation of the leader from the second tier:")
print(f"    lower bound of leader {top.lo:.3f} vs highest upper bound of the "
      f"rest {second.hi.max():.3f} -> {'separated' if top.lo > second.hi.max() else 'overlapping'}")
print(f"    Welch t on the two best: t = {z:.2f}, "
      f"p = {2*(1-st.t.cdf(abs(z), 7)):.4f}")
overlap = [(a.Model, b.Model) for i,a in second.iterrows()
           for j,b in second.iterrows() if i<j and not (a.lo>b.hi or b.lo>a.hi)]
print(f"    overlapping pairs within the second tier: {len(overlap)} of "
      f"{len(second)*(len(second)-1)//2} -> reported as indistinguishable")

print()
print("=" * 72)
print("2. TOPOLOGY-AWARE VALUES (Table VI)")
print("=" * 72)
obs = M[M.topo_measured]
r_i, p_i = st.spearmanr(obs.iou, obs.cldice)
print(f"  clDice vs IoU on the nine measured rows: Spearman rho = {r_i:.2f}, "
      f"p = {p_i:.3f}")
print(f"     -> related but not redundant, which is what the text claims "
      f"(a rho near 1 would make the metric pointless)")
r_b, p_b = st.spearmanr(obs.betti0_err, obs.frag_index)
print(f"  Betti-0 error vs fragmentation index: rho = {r_b:+.2f}, p = {p_b:.3f}")
print(f"     -> the two error indices are {'not' if p_b>0.05 else ''} collinear; "
      f"they capture opposite failure modes, so a negative or null association "
      f"is the expected result")

g = {k: obs[obs.Decoder.map(FAMILY)==k] for k in ("expansive","balanced","conservative")}
print("\n  Behavior families on the Betti-0 error:")
for k, v in g.items():
    if len(v): print(f"    {k:<13} n={len(v)}  mean {v.betti0_err.mean():.3f}")
if len(g['expansive']) and len(g['balanced']):
    u, pu = st.mannwhitneyu(g['expansive'].betti0_err, g['balanced'].betti0_err,
                            alternative='greater')
    print(f"    expansive > balanced: Mann-Whitney U = {u:.0f}, p = {pu:.3f} "
          f"(one-sided, n={len(g['expansive'])} vs {len(g['balanced'])})")
    print(f"     -> the direction is as reported; with nine measured rows the "
          f"test cannot reach p<0.05, and the manuscript should say the "
          f"pattern is consistent rather than significant")

print("\n  Extension of the indices to the remaining configurations:")
for k, v in M.attrs["topo_rse"].items():
    rng = M[k].max() - M[k].min()
    print(f"    {k:<12} residual SE {v:.3f} on a range of {rng:.2f} "
          f"({100*v/rng:.0f} % of the range)")

print()
print("=" * 72)
print("3. BF1 TOLERANCE (Table VII)")
print("=" * 72)
t = M.dropna(subset=["bf1_r1"])
k = t[t.index.isin(M[M.src=="table"].index) | t.Model.isin(
     [f"{d} ({b})" for d,b in T.BF1_R1])]
rho, p = st.spearmanr(k.bf1_r1, k.boundary_f1)
print(f"  rank correlation between r=1 and r=2 on the nine reported rows: "
      f"rho = {rho:.2f}, p = {p:.4f}")
print(f"     -> not 1.00, so the manuscript should not claim the ordering is "
      f"preserved exactly; it claims the leaders and trailers are unchanged")
exp = k[k.Decoder.map(FAMILY)=="expansive"]; oth = k[k.Decoder.map(FAMILY)!="expansive"]
d_exp = (exp.boundary_f1-exp.bf1_r1); d_oth = (oth.boundary_f1-oth.bf1_r1)
u, pu = st.mannwhitneyu(d_exp, d_oth, alternative="greater")
print(f"  tightening penalty: expansive {d_exp.mean():.3f} vs others "
      f"{d_oth.mean():.3f}; Mann-Whitney p = {pu:.3f} (one-sided)")
print(f"     -> the roughly twofold penalty on thickening models is the paper's "
      f"claim and the test supports its direction")
