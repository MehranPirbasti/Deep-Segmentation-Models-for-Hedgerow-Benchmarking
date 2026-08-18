"""Train and eval loops. AMP, cosine schedule, resumable."""
from __future__ import annotations
import os, csv, time
from typing import Dict

import numpy as np
import torch
from tqdm.auto import tqdm
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score

from . import config as C
from .losses import supervised_loss, main_logits
from .metrics import (boundary_f1, topology_metrics, per_patch_iou_dice,
                      bootstrap_ci)

CSV_FIELDS = ["model_name", "decoder", "backbone", "seed", "epoch", "phase",
              "loss", "iou", "dice_f1", "boundary_f1", "bf1_r1",
              "cldice", "betti0_err", "frag_index", "bridge_err",
              "inference_time_s", "total_training_time_s"]


def append_metrics(csv_path: str, row: Dict) -> None:
    exists = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    running, n = 0.0, 0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.float().to(device, non_blocking=True)
        masks = masks.float().to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=C.USE_AMP and device == "cuda"):
            out = model(images)                     # list if deep supervision
            loss = supervised_loss(out, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.item()); n += 1
    return running / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, with_topology: bool = True,
             collect_per_patch: bool = False):
    """
    Metrics dict. Boundary and topology measures are CPU-bound and dominate epoch
    time, so they are optional during validation and always computed on the test
    set. In eval() mode a deep-supervised model returns only its main head, so
    all reported metrics come from the head used at inference.
    """
    model.eval()
    iou_m = BinaryJaccardIndex().to(device)
    f1_m = BinaryF1Score().to(device)
    total_loss, total_time, nb = 0.0, 0.0, 0
    P, M = [], []

    for images, masks in tqdm(loader, desc="eval", leave=False):
        images = images.float().to(device, non_blocking=True)
        masks = masks.float().to(device, non_blocking=True).unsqueeze(1)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast(device_type="cuda", enabled=C.USE_AMP and device == "cuda"):
            out = model(images)
        if device == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - t0

        logits = main_logits(out).float()
        total_loss += float(supervised_loss(logits, masks).item()); nb += 1
        preds = (torch.sigmoid(logits) > C.THRESHOLD).int()
        iou_m.update(preds, masks.int())
        f1_m.update(preds, masks.int())
        if with_topology or collect_per_patch:
            P.append(preds.detach().cpu()); M.append(masks.int().detach().cpu())

    out = {"loss": total_loss / max(nb, 1),
           "iou": iou_m.compute().item(),
           "dice_f1": f1_m.compute().item(),
           "inference_time_s": total_time,
           "boundary_f1": float("nan"), "bf1_r1": float("nan"),
           "cldice": float("nan"), "betti0_err": float("nan"),
           "frag_index": float("nan"), "bridge_err": float("nan")}

    if with_topology or collect_per_patch:
        Pc, Mc = torch.cat(P, 0), torch.cat(M, 0)
        if with_topology:
            out["boundary_f1"] = boundary_f1(Pc, Mc, dist_thresh=2)
            out["bf1_r1"] = boundary_f1(Pc, Mc, dist_thresh=1)
            cld, bet, frg, brd = topology_metrics(Pc, Mc)
            out.update(cldice=cld, betti0_err=bet, frag_index=frg, bridge_err=brd)
        if collect_per_patch:
            iou_pp, dice_pp = per_patch_iou_dice(Pc, Mc)
            cld_pp, bet_pp, frg_pp, brd_pp = topology_metrics(Pc, Mc, per_patch=True)
            out["_per_patch"] = {
                "iou": iou_pp, "dice": dice_pp, "cldice": cld_pp,
                "betti0": bet_pp, "frag": frg_pp, "bridge": brd_pp,
                "bf1_r2": boundary_f1(Pc, Mc, 2, per_patch=True),
                "bf1_r1": boundary_f1(Pc, Mc, 1, per_patch=True)}
    return out
