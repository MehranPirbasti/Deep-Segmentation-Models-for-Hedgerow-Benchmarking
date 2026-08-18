# Reproducing and editing the figures

Every figure that carries data is a plot drawn by `matplotlib` from a table. None
is drawn by hand or produced by an image-generation model. The vector versions in
`figures/vector/` make this checkable: open an `.svg` in a text editor and the
axis labels, tick values, model names and the numbers printed inside the matrix
cells are stored as **selectable, searchable text**.

## Regenerate everything

```bash
cd analysis
python master.py                          # assemble the master table
python build_v6.py --out ../figures/png --topology
python make_hetero_v2.py                  # the heterogeneous-conditions panel
python audit.py                           # verify against the manuscript tables
```

## Change a value

| You want to change | Edit |
|---|---|
| a plotted value | `data/Figure_Data.xlsx`, sheet `ALL_CONFIGURATIONS` (blue cells) |
| one figure's data only | the matching file in `data/<Figure_name>.csv` |
| colors, markers, fonts, sizes | `analysis/paperstyle.py` |
| what a figure shows, its panels, its layout | `analysis/build_v6.py` |
| a single label or element | `figures/vector/<figure>.svg` in any vector editor |

Grey cells in the workbook are values fixed by a manuscript table. Editing one
makes the figure disagree with the paper, and `audit.py` names it.

## Where the values come from

Precedence is strict and enforced in `analysis/master.py`:

1. **The manuscript tables** (`analysis/tables.py`), used verbatim and never
   overwritten. This is why a figure agree with a table cell by cell.
2. **The figures of the original submission**, for configurations no table
   reports. Recovered marker by marker; those rows carry `src = figure`.
3. **Imputation**, only where a marker coincides exactly with another in a source
   scatter so no coordinate can be read. Such cells are flagged and are never
   used for a claim in the text.

`data/_provenance.csv` gives this per quantity.

## Style

Figures are drawn at their final printed size ,  IEEEtran two-column, so 3.1-3.6
inches for a single-column figure ,  and both the width and the aspect ratio of
the saved file are iterated until they match, so the point sizes set in
`paperstyle.py` are the point sizes that reach the page. Color identifies the
encoder backbone and marker shape the decoder, one convention in every figure.
Export is 600 dpi PNG plus SVG and PDF.

## One figure is explicitly synthetic

`Heterogeneous_Conditions_Panel.png` shows illustrative prediction masks
**synthesized from the reference annotation** to show how the three
characteristic error modes present themselves under each hedgerow condition. The
figure caption states this in bold, and no quantitative claim depends on it.
