# Figure provenance and editable sources

Prepared in response to the query about whether the figures are machine-generated
imagery.

## What the figures are

They are **plots, not images**. Every figure in the manuscript that carries data
is drawn by `matplotlib` from a tabular file, by a script included here. None is
drawn by hand, traced, painted, or produced by an image-generation model, and
none contains any content that was not computed from the tables.

The distinction matters and is verifiable, which is why this package ships the
vector versions:

- **`vector/*.svg`** ,  every plot as scalable vector graphics. Open one in a text
  editor: it is a list of coordinates, line segments and text strings. Axis
  labels, tick values, model names and the numbers printed inside the matrix
  cells are stored as **selectable, searchable text**, not as pixels. A generated
  image cannot have this structure; it has pixels only.
- **`vector/*.pdf`** ,  the same figures as vector PDF, with fonts embedded
  (Type 42), suitable for direct submission.
- **`Fig/*.png`** ,  600 dpi raster renderings of the same figures, for
  convenience.

The three panels that contain satellite imagery are photographs, not plots:
`Study_Area.jpg`, `Qualitative_Overlays.jpg`, `Quantitative_Overlays.png` and
`Proxy_Annotation_Update.jpg` reproduce Pleiades Neo extracts under license
((c) Airbus DS 2022) with model predictions overlaid. `Workflow.png` is a diagram.

## One figure is explicitly synthetic, and says so

`Heterogeneous_Conditions_Panel.png` shows, for each hedgerow condition, the test
sample, the reference annotation, and **illustrative prediction masks that are
synthesized from the reference** to show how the three characteristic error modes
present themselves. This is stated in bold in the figure caption in the
manuscript, and the masks are not used for any quantitative claim. Every number
in the paper comes from the tables, not from this figure. We flag it here so that
its status is unambiguous.

## How to verify, in three commands

```bash
python audit.py         # checks every figure value against the manuscript tables
python stats_audit.py   # re-runs the statistical checks reported in the text
python build_v6.py --out Fig --topology    # regenerates every figure from the data
```

`audit.py` currently reports **105 manuscript table cells checked, 0
mismatches**. Change any value in `Figure_Data.xlsx` or in `master.csv`, re-run
the build, and the corresponding figure changes: this is the practical proof
that the figures are functions of the data rather than artwork.

## How to edit a figure

| You want to change | Edit |
|---|---|
| a plotted value | `Figure_Data.xlsx`, sheet `ALL_CONFIGURATIONS`, then re-run the build |
| a single figure's data only | the matching file in `figure_data/` |
| colors, markers, fonts, sizes | `paperstyle.py` |
| what a figure shows, panels, layout | `build_v6.py` |
| labels, legend text, one element | `vector/<figure>.svg` in Illustrator, Inkscape or Affinity |

Blue cells in the workbook are safe to edit. Grey cells are values fixed by a
manuscript table; editing one will make the figure disagree with the paper, and
`audit.py` will report it by name.

## Where the numbers come from

`figure_data/_provenance.csv` gives this per quantity. In summary: IoU, Dice,
BF1, latency and training time for the 21 configurations listed in Tables II and
III are those tables verbatim; the topology indices and the strict-radius BF1 are
Tables VI and VII verbatim; the multi-seed statistics are Table V verbatim. Where
the manuscript reports no value, the quantity was recovered marker by marker from
the corresponding figure of the original submission, and those rows are marked
`src = figure`. Dice is computed as `2.IoU/(1+IoU)`, which is exact for a fixed
prediction and reproduces the published Dice values to three decimals.

## Files

```
Fig/                     final figures, 600 dpi PNG
vector/                  the same figures as SVG and PDF, text editable
Figure_Data.xlsx         all data in one workbook, with a legend
figure_data/*.csv        one CSV per figure, plus _provenance.csv
master.csv               the single source table for every figure
paperstyle.py            typography, colors, markers, sizing
build_v6.py              draws every figure
tables.py                the manuscript tables, transcribed
audit.py                 figure-vs-table verification
stats_audit.py           statistical verification
extract_*.py             recovery of values from the original submission figures
```
