"""IoU, Dice, boundary F1 and the three topology indices.

All computed on the foreground class, at the fixed threshold in config, on the
416x416 patch. The boundary and topology parts are CPU-bound and slow, so the
training loop only runs them every so often on validation.
"""
from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize

from . import config as C

_CONN1 = ndimage.generate_binary_structure(rank=2, connectivity=1)
_CONN2 = ndimage.generate_binary_structure(rank=2, connectivity=2)
_EPS = 1e-7


# Boundary F1 (manuscript Eqs. 3a, 3b, 3)
def _bf1_single(args):
    p, t, r = args
    pb = ndimage.binary_dilation(p, structure=_CONN1) ^ p
    tb = ndimage.binary_dilation(t, structure=_CONN1) ^ t
    if not pb.any() and not tb.any():
        return 1.0
    if not pb.any() or not tb.any():
        return 0.0
    dt = ndimage.distance_transform_edt(~tb)     # distance to GT boundary
    dp = ndimage.distance_transform_edt(~pb)     # distance to predicted boundary
    prec = np.sum(dt[pb] <= r) / (pb.sum() + _EPS)
    rec = np.sum(dp[tb] <= r) / (tb.sum() + _EPS)
    return (2 * prec * rec) / (prec + rec + _EPS)


def boundary_f1(preds: torch.Tensor, targets: torch.Tensor,
                dist_thresh: int = 2, per_patch: bool = False):
    """Mean per-patch boundary F1 at a given pixel tolerance r."""
    p = preds.squeeze(1).cpu().numpy().astype(bool)
    t = targets.squeeze(1).cpu().numpy().astype(bool)
    tasks = [(p[i], t[i], dist_thresh) for i in range(p.shape[0])]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        res = list(ex.map(_bf1_single, tasks))
    return np.asarray(res) if per_patch else float(np.mean(res)) if res else float("nan")


# Topology-aware metrics (manuscript Section IV-D)
def _topo_single(args):
    p, t = args
    sp = skeletonize(p) if p.any() else np.zeros_like(p)
    st = skeletonize(t) if t.any() else np.zeros_like(t)
    # clDice: topology precision / sensitivity
    t_prec = (np.sum(sp & t) / (sp.sum() + _EPS)) if sp.any() else 1.0
    t_sens = (np.sum(st & p) / (st.sum() + _EPS)) if st.any() else 1.0
    cld = (2 * t_prec * t_sens) / (t_prec + t_sens + _EPS)
    # Betti-0 error (normalized) and fragmentation index
    _, b0p = ndimage.label(p, structure=_CONN2)
    _, b0t = ndimage.label(t, structure=_CONN2)
    betti = abs(b0p - b0t) / max(b0t, 1)
    frag = max(b0p - b0t, 0) / max(b0t, 1)
    # Component of the Betti-0 error attributable to spurious merges (bridging):
    bridge = max(b0t - b0p, 0) / max(b0t, 1)
    return cld, betti, frag, bridge


def topology_metrics(preds: torch.Tensor, targets: torch.Tensor,
                     per_patch: bool = False):
    """clDice (higher better), normalized Betti-0 error, fragmentation index,
    and the bridging component of the Betti-0 error (all lower better)."""
    p = preds.squeeze(1).cpu().numpy().astype(bool)
    t = targets.squeeze(1).cpu().numpy().astype(bool)
    tasks = [(p[i], t[i]) for i in range(p.shape[0])]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        res = list(ex.map(_topo_single, tasks))
    cld, bet, frg, brd = zip(*res)
    if per_patch:
        return (np.array(cld), np.array(bet), np.array(frg), np.array(brd))
    return (float(np.mean(cld)), float(np.mean(bet)),
            float(np.mean(frg)), float(np.mean(brd)))


# Region overlap and bootstrap
def per_patch_iou_dice(preds: torch.Tensor, targets: torch.Tensor):
    p = preds.squeeze(1).cpu().numpy().astype(bool)
    t = targets.squeeze(1).cpu().numpy().astype(bool)
    inter = np.logical_and(p, t).sum(axis=(1, 2)).astype(np.float64)
    union = np.logical_or(p, t).sum(axis=(1, 2)).astype(np.float64)
    psum = p.sum(axis=(1, 2)) + t.sum(axis=(1, 2))
    iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    dice = np.where(psum > 0, 2 * inter / np.maximum(psum, 1), 1.0)
    return iou, dice


def bootstrap_ci(values, n_boot: int = C.BOOTSTRAP_N,
                 alpha: float = C.BOOTSTRAP_ALPHA, seed: int = 42):
    """Patch-level bootstrap interval for a mean metric."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def seed_ci(values):
    """Mean and normal-approximation 95 % CI half-width over random seeds."""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2:
        return float(v.mean()) if v.size else float("nan"), float("nan"), float("nan")
    sd = v.std(ddof=1)
    half = 1.96 * sd / np.sqrt(v.size)
    return float(v.mean()), float(half), float(sd)


# Self-test on synthetic shapes: a break must raise fragmentation, a spurious
# bridge must raise the Betti-0 error. Run by scripts/selftest.py.
def selftest(verbose: bool = True) -> bool:
    gt = np.zeros((2, 1, 64, 64), dtype=np.uint8)
    gt[0, 0, 30:33, 5:60] = 1
    gt[1, 0, 10:13, 5:60] = 1
    gt[1, 0, 40:43, 5:60] = 1
    pr = gt.copy()
    pr[0, 0, 30:33, 30:35] = 0        # break  -> fragmentation
    pr[1, 0, 13:40, 20:23] = 1        # bridge -> Betti-0 error
    G, P = torch.from_numpy(gt), torch.from_numpy(pr)
    assert abs(boundary_f1(G, G, 2) - 1.0) < 1e-6
    c, b, f, br = topology_metrics(G, G)
    assert c > 0.999 and b < 1e-9 and f < 1e-9
    c, b, f, br = topology_metrics(P, G)
    assert f > 0 and b > 0
    if verbose:
        print(f"[selftest] BF1 r=1 {boundary_f1(P, G, 1):.4f} | r=2 {boundary_f1(P, G, 2):.4f} "
              f"| clDice {c:.4f} | Betti-0 {b:.4f} | Frag {f:.4f} | Bridge {br:.4f} -> OK")
    return True
