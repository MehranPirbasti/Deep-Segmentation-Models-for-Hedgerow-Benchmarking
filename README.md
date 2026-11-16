# Benchmarking deep segmentation models for hedgerow mapping

Code, configurations and the split-construction procedure accompanying:

> **Benchmarking Deep Segmentation Models for Hedgerow Mapping:
> A Controlled Comparative Study of Vegetated Linear Structures
> in High-Resolution Remote Sensing** *IEEE Transactions on Geoscience and Remote Sensing*, accepted for publication (2026).

Ten decoder families are paired with six encoder backbones and trained under a
single frozen protocol, so that architecture is the only variable that changes.
Models are scored on region overlap, boundary alignment at two tolerance radii,
and three topology-aware indices.


## What is here, and what is not

| | |
|---|---|
| **Included** | training and evaluation code . all model and protocol configuration files . the split-construction script with its fixed seed (`split_seed: 42`) . every figure as PNG and as editable SVG/PDF |
| **Not included** | the Pleiades Neo imagery . the UKCEH hedgerow reference . derived image patches or masks . trained weights . the per-patch split assignment and per-patch evaluation outputs (both are regenerated from the licensed inputs by the released scripts, see below) |

The two source products are licensed and may not be redistributed, and trained
weights are derived directly from the licensed imagery. Both products are
obtainable by any researcher from their providers on the same terms under which
they were obtained for this study; see [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md).
Given those inputs the pipeline below is deterministic and reproduces the
reported partition, models and numbers: `scripts/02_make_splits.py` rebuilds the
partition from `patch_inventory.csv` with the recorded seed, and
`hedgebench/metrics.py` recomputes every per-patch metric.

**Figure provenance.** The figure scripts read the manuscript tables
(`analysis/tables.py`) verbatim. For configurations that no table lists, the
plotted values were recovered from the figures of the original submission and
carry `src = figure`; a small number of latency, training-time and loss cells
for which no coordinate could be read are imputed by `analysis/master.py` and
flagged as such. No imputed value is used for any statement in the paper.


## Quick start

```bash
pip install -r requirements.txt
python scripts/00_selftest.py          # verify the implementation before spending GPU hours
```

Full pipeline, once the imagery and reference layer are in hand:

```bash
python scripts/01_extract_patches.py --image scene.tif --mask hedgerows.tif \
       --eligible eligible_mask.tif --out DATA
python scripts/02_make_splits.py --inventory DATA/patch_inventory.csv \
       --out DATA/split_metadata.csv --report DATA/split_report.json
python scripts/04_train.py --data-root DATA --splits DATA/split_metadata.csv \
       --out-root OUT
python scripts/05_make_tables.py  --tables  OUT/tables
python scripts/06_make_figures.py --tables  OUT/tables --figures OUT/figures
```

Reproduce the paper's figures from the published numbers, without any imagery:

```bash
cd analysis
python master.py                        # build the master table
python build_v6.py --out ../figures/png --topology
python audit.py                         # figures against the manuscript tables
python stats_audit.py                   # the statistical checks reported in the text
```

---

## The frozen protocol

Applied identically to every configuration
([`hedgebench/config.py`](hedgebench/config.py), mirrored into
[`configs/run_config.json`](configs/run_config.json)).

| | |
|---|---|
| Patch | 416 x 416 px at 1.2 m GSD (approx. 25 ha), four bands (RGB + NIR) |
| Tiling | non-overlapping, stride 416 px |
| Split | whole spatial blocks, stratified by hedgerow density, target 70/15/15, one-patch geographic buffer on cross-block seams |
| Augmentation | random 256 x 256 crop, flips, 90 deg rotations, coarse dropout, grid distortion |
| Evaluation | native 416 x 416, fixed threshold 0.5 |
| Loss | 0.3 Dice + 0.4 Focal + 0.3 BCEWithLogits |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4, cosine annealing (T_max 100, eta_min 1e-6) |
| Budget | 100 epochs, batch size 8, full budget for every configuration |
| Encoder init | standard three-channel ImageNet; the NIR channel by channel-mean inflation |
| Deep supervision | U-Net++ only, with an explicit ablation |
| Selection | best validation IoU; that checkpoint is evaluated on the test set |
| Seeds | base 42; nine configurations repeated over five seeds |

## Metrics

- **Region** ,  IoU, Dice/F1
- **Boundary** ,  BF1 at tolerance r = 1 and r = 2 px (r = 2 is the headline value)
- **Topology** ,  centerline Dice (clDice), normalized Betti-0 error, fragmentation index
- **Uncertainty** ,  five-seed mean +/- 95 % CI

Breaks and bridges degrade a hedgerow network in opposite directions, so the
fragmentation index and the Betti-0 error are reported separately: a model can
score well on one while failing badly on the other, which is what the expansive
decoders do.

---

## Layout

```
hedgebench/      config.py  data.py  models.py  losses.py  metrics.py  engine.py
scripts/         00_selftest  01_extract_patches  02_make_splits
                 04_train  05_make_tables  06_make_figures
configs/         run_config.json ,  snapshot of the frozen protocol
analysis/        the figure pipeline, the audits, and the recovery scripts
figures/png/     final figures, 600 dpi
figures/vector/  the same figures as SVG and PDF, with text left editable
docs/            FIGURES.md . DATA_ACCESS.md . FIGURE_PROVENANCE.md . PROTOCOL.md
```

## Verification

Two commands, both of which run without any imagery:

```bash
python analysis/tables.py       # the manuscript tables as the figure scripts read them
python scripts/00_selftest.py   # the protocol implementation itself
```

`analysis/audit.py` and `analysis/stats_audit.py` check the figure values and
the statistical claims against the manuscript tables; they read the figure
pipeline's local working table (`data/master.csv`), which is built by
`analysis/master.py` and is not committed.

Five figures of the manuscript contain licensed imagery and are therefore not
published here; see [`figures/png/README.md`](figures/png/README.md). Nothing in
this repository depends on them.

## Citation

```bibtex
@article{hedgerow_benchmark_2026,
  title   = {Benchmarking Deep Segmentation Models for Hedgerow Mapping:  
             A Controlled Comparative Study of Vegetated Linear Structures
             in High-Resolution Remote Sensing},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  note    = {Accepted for publication}
}
```

## License

Code in this repository: MIT (see [`LICENSE`](LICENSE)).
The imagery and reference layer are **not** covered by that license and are not
distributed here; see [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md).
