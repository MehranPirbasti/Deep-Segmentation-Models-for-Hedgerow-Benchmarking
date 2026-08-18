# The frozen protocol, in full

Every value below is applied identically to all configurations. It is defined
once in `hedgebench/config.py` and written to `configs/run_config.json` at run
time, so the settings that produced a result travel with it.

Nothing here may be tuned per model. Changing a value means the run is no longer
the protocol reported in the paper.

## Data

| Setting | Value |
|---|---|
| Patch size | 416 x 416 px |
| Ground sampling distance | 1.2 m (each patch approx.  25 ha) |
| Extraction stride | 416 px, i.e. non-overlapping tiling |
| Bands | 4 (R, G, B, NIR) |
| Radiometric scaling | divide by a fixed 1419.0, clip to [0, 1] |
| Excluded windows | any containing no-data, or falling entirely within land-cover classes excluded from the UKCEH hedgerow product (woodland, urban and suburban, open water, mountain/moor/heath) |

## Split

| Setting | Value |
|---|---|
| Unit of assignment | spatial blocks of 4 x 4 adjacent patches |
| Target proportions | 70 / 15 / 15 |
| Stratification | block-level hedgerow density, quartiles |
| Buffer | patches on the seam between two differently-assigned blocks are dropped |
| Seed | 42 |

Assigning whole blocks keeps adjacent patches in the same partition; the buffer
removes the direct cross-split adjacency that is the mechanism by which spatial
autocorrelation would inflate test scores. A leakage check against a purely
random patch split is reported in the paper.

## Training

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Schedule | cosine annealing, T_max 100, eta_min 1e-6 |
| Epochs | 100, run in full for every configuration |
| Batch size | 8 |
| Loss | 0.3 Dice + 0.4 Focal (gamma = 2) + 0.3 BCEWithLogits |
| Augmentation | random 256 x 256 crop, horizontal and vertical flips, 90 deg rotations, coarse dropout, grid distortion |
| Mixed precision | enabled |
| Model selection | highest validation IoU |
| Seeds | 42 (base); 0, 1, 2, 3 additionally for nine configurations |

## Encoder initialisation and four-band adaptation

Every encoder starts from standard three-channel ImageNet weights. The first
convolution, or the patch-embedding projection for the MiT encoders, is expanded
from three to four input channels: **the RGB weights are copied unchanged** and
the NIR weights are set to **the mean of those copied RGB weights**. No rescaling
is applied, no encoder is trained from scratch, and the rule is identical for
every architecture, so the adaptation cannot confound the comparison.

`scripts/00_selftest.py` asserts both properties before any training begins.

## Deep supervision

U-Net++ is the only decoder in the suite that defines intermediate nested
outputs, so deep supervision cannot be applied uniformly. It is enabled for
U-Net++ ,  auxiliary heads on X^{0,1}, X^{0,2}, X^{0,3}, the same combined loss on
each, averaged with equal weight, and only the final head used at inference ,  and
ablated explicitly, so that the architectural effect and the auxiliary
supervision are not conflated.

## Evaluation

| Setting | Value |
|---|---|
| Resolution | native 416 x 416 |
| Threshold | 0.5, fixed for every model |
| BF1 tolerances | r = 1 and r = 2 px |
| Topology indices | clDice, normalized Betti-0 error, fragmentation index |
| Bootstrap | 10 000 patch-level resamples, alpha = 0.05 |
