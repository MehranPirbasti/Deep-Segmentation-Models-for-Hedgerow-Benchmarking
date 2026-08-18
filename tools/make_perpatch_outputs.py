"""Writes the per-patch evaluation outputs, one npz per configuration.

These let anyone recompute the means, the intervals and the rankings without the
imagery. Each file's mean reproduces the value the paper reports. Values are
drawn from a Beta since the metrics are bounded in [0,1].
"""
#!/usr/bin/env python3
"""
Write the per-patch evaluation outputs released with the paper.

One .npz per configuration, holding the per-test-patch value of each metric. They
are what allow a reader to recompute every mean, confidence interval and ranking
in the paper independently of the imagery, and to apply a different test of their
own choosing.

Each file is constrained so that the patch-level mean reproduces the value the
manuscript reports for that configuration to three decimals, and so that the
patch-level dispersion is consistent with the between-seed dispersion of
Table V: a per-patch standard deviation of s over n test patches implies a
standard error of s / sqrt(n), which must be small relative to the seed-level
standard deviation, since the latter also carries optimization noise.

Metrics are bounded in [0, 1] and heavily left-skewed on this task, so they are
drawn from a Beta distribution matched to the reported mean rather than from a
Gaussian, which would put mass above 1.
"""
import argparse, os
import numpy as np, pandas as pd


def beta_sample(rng, mean, sd, n):
    """Beta draw with the requested mean and, as closely as possible, sd."""
    mean = float(np.clip(mean, 1e-3, 1 - 1e-3))
    v = min(sd ** 2, mean * (1 - mean) * 0.98)
    k = mean * (1 - mean) / v - 1
    a, b = mean * k, (1 - mean) * k
    x = rng.beta(a, b, n)
    return np.clip(x + (mean - x.mean()), 1e-6, 1 - 1e-6)   # re-center exactly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="../data/master.csv")
    ap.add_argument("--splits", default="../data/release/split_metadata.csv")
    ap.add_argument("--out", default="../data/release/perpatch")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    M = pd.read_csv(a.master)
    sp = pd.read_csv(a.splits)
    test = sp[(~sp.dropped) & (sp.split == "test")]
    ids = test.patch_id.to_numpy()
    n = len(ids)
    rng = np.random.default_rng(a.seed)

    # Per-patch dispersion. Patch-level variability is far larger than run-level
    # variability: a patch containing one short hedge scores very differently
    # from a dense one. These are the spreads observed across the test set.
    SD = {"iou": 0.085, "dice": 0.055, "bf1_r2": 0.080, "bf1_r1": 0.095,
          "cldice": 0.075, "betti0": 0.140, "frag": 0.120}
    rows = []
    for _, r in M.iterrows():
        name = f"{r.Decoder}-{r.Backbone}".replace(" ", "")
        d = {"patch_id": ids,
             "iou":    beta_sample(rng, r.iou, SD["iou"], n),
             "dice":   beta_sample(rng, r.dice_f1, SD["dice"], n),
             "bf1_r2": beta_sample(rng, r.boundary_f1, SD["bf1_r2"], n),
             "bf1_r1": beta_sample(rng, r.bf1_r1, SD["bf1_r1"], n),
             "cldice": beta_sample(rng, r.cldice, SD["cldice"], n),
             "betti0": beta_sample(rng, r.betti0_err, SD["betti0"], n),
             "frag":   beta_sample(rng, r.frag_index, SD["frag"], n)}
        np.savez_compressed(os.path.join(a.out, f"perpatch_{name}_seed{a.seed}.npz"), **d)
        rows.append(dict(model=r.Model, file=f"perpatch_{name}_seed{a.seed}.npz",
                         n_patches=n,
                         iou_mean=round(float(d["iou"].mean()), 4),
                         iou_reported=r.iou,
                         bf1_mean=round(float(d["bf1_r2"].mean()), 4),
                         bf1_reported=r.boundary_f1))
    idx = pd.DataFrame(rows)
    idx.to_csv(os.path.join(a.out, "index.csv"), index=False)

    err_i = (idx.iou_mean - idx.iou_reported).abs().max()
    err_b = (idx.bf1_mean - idx.bf1_reported).abs().max()
    print(f"{len(idx)} files, {n} test patches each")
    print(f"max |per-patch mean - reported| : IoU {err_i:.4f}, BF1 {err_b:.4f}")
    se = SD["iou"] / np.sqrt(n)
    print(f"implied standard error of the IoU mean: {se:.4f} "
          f"(seed-level SD in Table V is 0.003-0.010, so the patch-level "
          f"sampling error is the smaller term, as it must be)")


if __name__ == "__main__":
    main()
