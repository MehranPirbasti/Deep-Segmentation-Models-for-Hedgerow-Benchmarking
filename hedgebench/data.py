"""Dataset and loaders. Split membership comes from split_metadata.csv."""
from __future__ import annotations
import os, random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from . import config as C

try:
    import rasterio
except ImportError:
    rasterio = None
import albumentations as A
from albumentations.pytorch import ToTensorV2


def set_seed(seed: int) -> None:
    """Deterministic seeding across python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_image(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)          # (C, H, W)
    img = np.transpose(img, (1, 2, 0))               # (H, W, C)
    return np.clip(img / C.RADIOMETRIC_MAX, 0.0, 1.0)


def read_mask(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return (src.read(1) > 0).astype(np.float32)


class HedgerowDataset(Dataset):
    """
    4-band (RGB+NIR) GeoTIFF patches and their binary hedgerow masks.

    The partition used here is exactly the partition published in
    split_metadata.csv, so results are tied to a stated, inspectable split.
    """

    def __init__(self, records: pd.DataFrame, image_dir: str, mask_dir: str,
                 transform=None):
        self.records = records.reset_index(drop=True)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records.iloc[idx]
        img = read_image(os.path.join(self.image_dir, row["image_file"]))
        mask = read_mask(os.path.join(self.mask_dir, row["mask_file"]))
        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        return img, mask


def build_transforms():
    """
    On-line augmentation of the frozen protocol (manuscript Section IV-E, item 3).

    Applied identically to every architecture. This is in addition to the fixed
    offline expansion of the training partition performed by
    scripts/03_expand_train.py, which touches the training partition only.
    """
    train_tf = A.Compose([
        A.RandomCrop(height=C.CROP_SIZE, width=C.CROP_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.CoarseDropout(max_holes=12, max_height=32, max_width=32,
                        min_holes=4, min_height=8, min_width=8,
                        fill_value=0, mask_fill_value=0, p=0.5),
        A.GridDistortion(p=0.3),
        ToTensorV2(),
    ])
    eval_tf = A.Compose([
        A.Resize(height=C.EVAL_SIZE, width=C.EVAL_SIZE),
        ToTensorV2(),
    ])
    for tf in (train_tf, eval_tf):
        if hasattr(tf, "check_shapes"):
            tf.check_shapes = False
    return train_tf, eval_tf


def build_loaders(meta_csv: str, image_dir: str, mask_dir: str,
                  train_image_dir: str = None, train_mask_dir: str = None):
    """
    Returns (train_loader, val_loader, test_loader, summary_dataframe).

    Rows flagged dropped == True by the split buffer are excluded here, so a
    buffered patch can never enter any partition.
    """
    meta = pd.read_csv(meta_csv)
    if "dropped" in meta.columns:
        meta = meta[~meta["dropped"].astype(bool)]
    train_tf, eval_tf = build_transforms()

    tr_img = train_image_dir or image_dir
    tr_msk = train_mask_dir or mask_dir

    parts = {}
    for name in ("train", "validation", "test"):
        sub = meta[meta["split"] == name]
        idir, mdir = (tr_img, tr_msk) if name == "train" else (image_dir, mask_dir)
        parts[name] = HedgerowDataset(
            sub, idir, mdir, transform=train_tf if name == "train" else eval_tf)

    common = dict(num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY,
                  persistent_workers=C.PERSISTENT_WORKERS and C.NUM_WORKERS > 0)
    train_loader = DataLoader(parts["train"], batch_size=C.BATCH_SIZE, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(parts["validation"], batch_size=C.BATCH_SIZE,
                            shuffle=False, **common)
    test_loader = DataLoader(parts["test"], batch_size=C.BATCH_SIZE,
                             shuffle=False, **common)
    return train_loader, val_loader, test_loader, meta


def summarize(meta: pd.DataFrame) -> str:
    lines = ["=" * 74, "DATASET SUMMARY (from split_metadata.csv)", "=" * 74]
    tot = 0
    for name in ("train", "validation", "test"):
        sub = meta[meta["split"] == name]
        n_unique = sub["patch_id"].nunique() if "patch_id" in sub.columns else len(sub)
        tot += n_unique
        px = len(sub) * C.EVAL_SIZE * C.EVAL_SIZE
        lines.append(f"  {name:11s} samples={len(sub):6d}  unique source patches="
                     f"{n_unique:5d}  labeled pixels={px/1e6:8.1f}M")
    lines.append(f"  {'TOTAL':11s} unique source patches={tot:5d}")
    lines.append("=" * 74)
    return "\n".join(lines)
