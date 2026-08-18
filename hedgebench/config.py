"""Protocol settings. Same values for every model.

These are the numbers in Section IV-E of the paper. Don't tune anything here
per model, otherwise the runs stop being comparable.
"""
from __future__ import annotations
import os, json
from typing import Dict


# Geometry of the scene and of patch extraction (manuscript Section III-B)
PATCH_SIZE     = 416    # px; native evaluation patch, divisible by the 32-px stride
EXTRACT_STRIDE = 416    # px; non-overlapping tiling, 0 overlap between patches
GSD_M          = 1.2    # m; native multispectral ground sampling distance
IN_CHANNELS    = 4      # R, G, B, NIR

# Windows whose valid-data fraction falls below this are discarded.
MIN_VALID_FRACTION = 0.99
# Windows lying entirely in land-cover classes excluded from the UKCEH hedgerow
# product (woodland, urban/suburban, open water, mountain/moor/heath) are
# discarded; a window is kept if at least this fraction is eligible terrain.
MIN_ELIGIBLE_FRACTION = 0.05


# Spatial split (manuscript Section III-B)
BLOCK_SIZE_WINDOWS = 4          # each spatial block groups BLOCK x BLOCK windows
TARGET_SPLIT       = {"train": 0.70, "validation": 0.15, "test": 0.15}
DENSITY_STRATA     = 4          # quartiles of block-level hedgerow density
SPLIT_SEED         = 42

# One-patch geographic buffer: patches lying on the seam between two
# differently-assigned blocks are dropped, which removes direct cross-split
# adjacency (manuscript Section III-B).
BUFFER_NEIGHBOURHOOD = 1        # Chebyshev radius on the patch grid
# Which side of a seam loses its patch. Higher precedence is kept, so the
# smaller evaluation partitions are not eroded disproportionately.
SPLIT_PRECEDENCE = {"test": 3, "validation": 2, "train": 1}



# Training protocol
BASE_SEED   = 42
EXTRA_SEEDS = [0, 1, 2, 3]      # multi-seed configurations

NUM_EPOCHS     = 100            # every configuration is trained for the full budget
BATCH_SIZE     = 8
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
COSINE_T_MAX   = 100
COSINE_ETA_MIN = 1e-6

CROP_SIZE       = 256           # on-line random crop used for training
EVAL_SIZE       = 416           # native patch size for validation and test
RADIOMETRIC_MAX = 1419.0        # fixed radiometric scaling divisor
THRESHOLD       = 0.5           # fixed decision threshold, all models

LOSS_WEIGHTS = dict(dice=0.3, focal=0.4, bce=0.3)
FOCAL_GAMMA  = 2.0

# Encoder initialisation
# All reported configurations use standard 3-channel ImageNet pretraining,
# adapted to 4 bands by channel-mean inflation (see models.py). No encoder is
# trained from scratch and the scheme is not tuned per model.
ENCODER_INIT = "imagenet"

# Deep supervision (manuscript Section IV-E, ablated in Section V)
DEEP_SUPERVISION_HEADS = {"unetpp": True}
DS_AUX_WEIGHT = 1.0             # equal weight; head losses are averaged

# Evaluation
BF1_TOLERANCES      = (1, 2)    # pixels; r=2 is the headline value
TOPO_EVERY_N_EPOCHS = 20        # CPU-bound metrics during validation only
BOOTSTRAP_N         = 10000
BOOTSTRAP_ALPHA     = 0.05

# Run plan
# "single"      : 60 configurations, base seed only
# "paper_table" : the above + EXTRA_SEEDS on MULTI_SEED_MODELS   <- reported
# "full"        : 60 configurations x 5 seeds
RUN_PLAN = "paper_table"

MULTI_SEED_MODELS = [
    "unetpp-resnet50", "upernet-resnet50", "unet-resnet50", "linknet-resnet34",
    "deeplab-resnet50", "segformer-mitb4", "deeplabplus-resnet50",
    "fpn-resnet50", "pan-resnet34",
]

ABLATION_RUNS = [
    {"name": "unetpp-resnet50-nods", "base": "unetpp-resnet50",
     "override": {"deep_supervision": False}},
]

USE_AMP            = True
NUM_WORKERS        = 2
PIN_MEMORY         = True
PERSISTENT_WORKERS = True


def as_dict() -> Dict:
    """Serialisable snapshot of the frozen protocol, written to run_config.json."""
    out = {}
    for k in sorted(k for k in globals() if k.isupper() and not k.startswith("_")):
        v = globals()[k]
        if isinstance(v, (int, float, str, bool, list, tuple, dict, type(None))):
            out[k.lower()] = list(v) if isinstance(v, tuple) else v
    return out


def dump(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(as_dict(), f, indent=2, sort_keys=True)
