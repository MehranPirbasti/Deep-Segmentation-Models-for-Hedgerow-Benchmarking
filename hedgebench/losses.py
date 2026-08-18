"""Loss terms. 0.3 Dice + 0.4 Focal + 0.3 BCE.

The third term is plain BCEWithLogits over the whole mask. Some older comments
called it a boundary loss, which it isn't; boundary quality is measured at
evaluation time with BF1.
"""
from __future__ import annotations
from typing import List, Union

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from . import config as C

_dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
_focal = smp.losses.FocalLoss(mode="binary", gamma=C.FOCAL_GAMMA)
_bce = nn.BCEWithLogitsLoss()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """0.3 Dice + 0.4 Focal + 0.3 BCEWithLogits.

    Note on naming: some earlier code referred to the third term as 'boundary'.
    The implemented term is BCEWithLogits over the full mask, not a
    distance-transform boundary loss. Boundary quality is assessed at evaluation
    time via BF1 (metrics.boundary_f1).
    """
    return (C.LOSS_WEIGHTS["dice"] * _dice(logits, targets) +
            C.LOSS_WEIGHTS["focal"] * _focal(logits, targets) +
            C.LOSS_WEIGHTS["bce"] * _bce(logits, targets))


def supervised_loss(output: Union[torch.Tensor, List[torch.Tensor]],
                    targets: torch.Tensor) -> torch.Tensor:
    """
    Deep-supervision-aware loss.

    If the model returns a list (U-Net++ with deep supervision in training mode),
    the identical combined loss is applied to the main head and to every
    auxiliary head, and the results are averaged with equal weight. If the model
    returns a single tensor, this reduces exactly to combined_loss, so every
    other architecture in the suite is unaffected.
    """
    if isinstance(output, (list, tuple)):
        losses = [combined_loss(o, targets) for o in output]
        w = [1.0] + [C.DS_AUX_WEIGHT] * (len(losses) - 1)
        total = sum(wi * li for wi, li in zip(w, losses))
        return total / sum(w)
    return combined_loss(output, targets)


def main_logits(output: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    """The head used for prediction and for all reported metrics."""
    return output[0] if isinstance(output, (list, tuple)) else output
