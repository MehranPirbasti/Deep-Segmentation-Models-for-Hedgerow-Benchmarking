# Uploading this repository

Total size is about 20 MB across 254 files. Nothing here approaches GitHub's
100 MB per-file limit, so **Git LFS is not needed**; a plain `git push` is enough.

---

## Step 1, create the repository

Create an **empty** repository on GitHub named `hedgerow-benchmark`, with no
README, no `.gitignore` and no license, since this folder already contains all
three. Keep it **private** until the paper is accepted.

## Step 2, replace the three placeholders

Do this before the first commit; the URL in particular has to match what the
paper prints.

| Where | Replace |
|---|---|
| `README.md`, `CITATION.cff` | `XXXX` in the repository URL |
| `CITATION.cff` | `FIRST AUTHOR` / `AFFILIATION` |
| `LICENSE` | the copyright holder |

The same URL appears in the manuscript's Data and Code Availability subsection.
All three must agree.

## Step 3, verify before committing

```bash
python tools/audit_release.py    # 31 checks: arithmetic, consistency, license hygiene
python analysis/audit.py         # every figure value against the manuscript tables
python analysis/stats_audit.py   # the statistical claims made in the text
python scripts/00_selftest.py    # the protocol implementation
```

All four should pass. The last check in `audit_release.py` confirms that no
imagery, no model weights and no coordinates have found their way in.

## Step 4, push

```bash
cd hedgerow-benchmark
git init
git add .
git status                       # read this list before committing
git commit -m "Code, protocol, split assignment and figure data for the hedgerow benchmark"
git branch -M main
git remote add origin https://github.com/<you>/hedgerow-benchmark.git
git push -u origin main
```

`git status` should list no `.tif`, `.pth` or `DATA/` entry. `.gitignore` covers
them, but read the list once rather than trusting it.

## Step 5, on acceptance

1. Make the repository public.
2. Cut a release, `v1.0.0`.
3. Connect the repository to Zenodo before tagging, so the release is archived
   and receives a DOI.
4. Add the DOI to `CITATION.cff` and to the paper's final proof.

---

## What is mandatory, and what is not

### Mandatory, the paper states these are released

Omitting any of these leaves a promise in the Data and Code Availability
subsection unmet, which is exactly what Reviewer 2's fourth comment asked us to
avoid.

| Path | Why it is mandatory |
|---|---|
| `hedgebench/` | the training and evaluation code |
| `scripts/` | preprocessing, split construction, training, tables, figures |
| `configs/run_config.json` | the frozen protocol as executed |
| `data/release/split_metadata.csv` | the per-patch split assignment; the one artifact a reader cannot regenerate without it |
| `data/release/perpatch/` | the per-patch evaluation outputs behind every reported statistic |
| `hedgebench/metrics.py` | the boundary-F1 and topology-aware evaluation code |
| `README.md`, `LICENSE`, `docs/DATA_ACCESS.md` | how to obtain the two licensed inputs, and under what terms |

About 5 MB.

### Strongly recommended, not promised, but they are what makes the release credible

| Path | Why |
|---|---|
| `analysis/` | reproduces every figure from the published numbers, with no imagery needed |
| `tools/audit_release.py` | lets a reviewer re-run the consistency checks rather than take them on trust |
| `analysis/stats_audit.py` | the same for the statistical claims |
| `data/*.csv`, `data/Figure_Data.xlsx` | the numbers behind each figure, in a form anyone can open |
| `docs/` | protocol, figure list with captions, provenance |

`tools/audit_release.py` and `analysis/audit.py` answer a question the editor
raised directly: they demonstrate that the figures are computed from the tables
rather than drawn. Without them it is just an assertion.

### Optional

| Path | Note |
|---|---|
| `figures/png/` | convenience copies; the figures are in the article |
| `figures/vector/` | **keep these.** They are the concrete evidence that the figures are plots and not generated images: the labels and numbers are stored as selectable text |
| `analysis/extract_*.py` | recovery scripts; of interest only if someone wants to trace where a value came from |
| `HedgeBench_Colab.ipynb` | if you want to ship the notebook, add it at the top level |

Dropping all of `figures/` would take the repository to about 6 MB. We would
still keep `figures/vector/`, for the reason above.

---

## What must never be uploaded

- The Pleiades Neo imagery, any patch derived from it, or any figure that
  displays it. Five figures of the manuscript are already withheld for this
  reason; see `figures/png/README.md`.
- The UKCEH reference layer or masks rasterized from it.
- Trained model weights: they are derived from the licensed imagery.
- Anything with geographic coordinates attached to a patch.

`.gitignore` blocks the file types. The last section of
`tools/audit_release.py` checks the rest.
