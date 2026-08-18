# Obtaining the data

Neither source product is redistributed here. Both are obtainable directly from
their providers, on the same terms under which they were obtained for this study.

## Pleiades Neo imagery ,  (c) Airbus DS 2022

A commercial product supplied by Airbus Defence and Space. The end-user license
under which it was obtained permits internal processing and the development and
training of machine-learning algorithms, and permits small non-georeferenced
extracts to be reproduced in a paper with the credit displayed. It does **not**
permit redistributing the imagery, or products that still contain imagery data,
to third parties.

Trained model weights are derived directly from that imagery, so they are not
distributed either. Running the released configurations on imagery licensed in
your own name reproduces them.

Contact: Airbus Defence and Space, <https://www.intelligence-airbusds.com>

**Acquisition used in this study:** Pleiades Neo, 2022, four bands (R, G, B,
NIR) at 1.2 m GSD, orthorectified, over St Albans, Hertfordshire, England;
approximately -0.440 deg to -0.240 deg longitude and 51.760 deg to 51.880 deg latitude.

## UKCEH Land Cover Plus: Hedgerows ,  (c) UKCEH

Supplied by the UK Center for Ecology & Hydrology under license, **not** as open
data. Academic users can obtain it through the EDINA Environment Digimap service,
or directly from the UKCEH Data Licensing Team. The product contains data derived
from the Environment Agency National LIDAR Programme.

- EDINA Environment Digimap: <https://digimap.edina.ac.uk>
- UKCEH Data Licensing: <https://www.ceh.ac.uk/data>

## Reproducing the partition without any imagery

`scripts/02_make_splits.py` operates on `patch_inventory.csv` ,  patch identifier,
window row and column, hedgerow fraction ,  and nothing else. It verifies, and
refuses to write its output unless, no patch in one partition is adjacent to a
patch in another. The released split assignment therefore lets anyone reconstruct
and inspect the exact partition used in the paper without any pixel data changing
hands.

## Credits required when reproducing a figure

Any figure containing the imagery must carry `(c) Airbus DS 2022`. Any figure
showing the reference layer must credit UKCEH. Both credits appear in the
corresponding captions in the manuscript.
