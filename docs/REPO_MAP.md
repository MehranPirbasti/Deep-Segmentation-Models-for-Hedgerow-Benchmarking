# What every file is

## Code that produced the results

| Path | Purpose |
|---|---|
| `hedgebench/config.py` | the frozen protocol; every constant used by every run |
| `hedgebench/data.py` | dataset, transforms, loaders; reads the split assignment |
| `hedgebench/models.py` | the decoder x backbone grid, four-band adaptation, the deep-supervision wrapper |
| `hedgebench/losses.py` | the combined loss, deep-supervision aware |
| `hedgebench/metrics.py` | IoU, Dice, BF1 at any tolerance, clDice, Betti-0 error, fragmentation index, bootstrap |
| `hedgebench/engine.py` | train and evaluate loops |
| `scripts/00_selftest.py` | asserts the protocol before any GPU time is spent |
| `scripts/01_extract_patches.py` | tiles the scene, writes `patch_inventory.csv` |
| `scripts/02_make_splits.py` | block assignment, density stratification, buffer; writes the split |
| `scripts/04_train.py` | runs the grid; interrupt-safe, resumes and skips finished runs |
| `scripts/05_make_tables.py` | every table in the paper |
| `scripts/06_make_figures.py` | figures from a completed run |
| `configs/run_config.json` | the protocol as executed, written at run time |

## The figure pipeline

| Path | Purpose |
|---|---|
| `analysis/tables.py` | the manuscript tables, transcribed. **Authoritative** |
| `analysis/master.py` | merges tables, recovered values and imputation into one table |
| `analysis/paperstyle.py` | typography, colors, markers, printed-size handling |
| `analysis/build_v6.py` | draws every data figure |
| `analysis/make_hetero_v2.py` | the heterogeneous-conditions panel |
| `analysis/audit.py` | verifies every figure against the manuscript tables |
| `analysis/stats_audit.py` | re-runs the statistical claims made in the text |
| `analysis/extract_*.py` | recovery of values from the original submission figures |
| `analysis/export_data.py` | writes one CSV per figure |
| `analysis/make_workbook.py` | builds `data/Figure_Data.xlsx` |
| `tools/dump_inventory.py` | refreshes the figure inventory from the .tex source |

## Data

| Path | Purpose |
|---|---|
| `data/release/split_metadata.csv` | the per-patch split assignment, 12,000 used patches |
| `data/release/perpatch/` | per-test-patch metric values, one file per configuration |
| `data/master.csv` | the single source table for every figure |
| `data/<Figure>.csv` | one file per figure, exactly what it plots |
| `data/Figure_Data.xlsx` | the same, in one editable workbook with a legend |
| `data/_provenance.csv` | where each quantity comes from |
| `data/seed_runs.csv` | the five-seed runs behind Table V |
| `data/decoder_val_iou.csv`, `data/convergence_bands.csv` | series recovered from the original figures |

## Figures

`figures/png/` at 600 dpi (the five imagery-bearing figures are withheld; see `figures/png/README.md`); `figures/vector/` as SVG and PDF with text left
editable. `docs/FIGURES.md` lists all twenty with their captions and their data.
