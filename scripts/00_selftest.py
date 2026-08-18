"""Sanity checks. Run this before starting a long training job."""
#!/usr/bin/env python3
"""Verify the protocol implementation before spending GPU hours on it."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from hedgebench import config as C
from hedgebench.metrics import selftest
from hedgebench.models import (build_experiment_grid, build_ablation_configs,
                               build_model, build_run_plan, adapt_first_conv_4band)
from hedgebench.losses import supervised_loss, main_logits

ok = True
print("1. metrics"); selftest()

print("2. experiment grid")
grid = build_experiment_grid(); abl = build_ablation_configs(grid)
jobs = build_run_plan(grid, abl)
assert len(grid) == 60, f"expected 60 configurations, got {len(grid)}"
assert all(c["init"] == "imagenet" for c in grid), "all runs must use ImageNet init"
print(f"   {len(grid)} configurations | {len(abl)} ablation(s) | {len(jobs)} runs "
      f"(plan '{C.RUN_PLAN}')")

print("3. four-band adaptation")
import segmentation_models_pytorch as smp
m = smp.Unet("resnet34", encoder_weights=None, in_channels=4, classes=1)
conv = next(x for x in m.modules() if isinstance(x, nn.Conv2d) and x.in_channels == 4)
before = conv.weight.detach().clone(); adapt_first_conv_4band(m); after = conv.weight.detach()
assert torch.allclose(before[:, :3], after[:, :3]), "RGB weights must be copied UNCHANGED"
assert torch.allclose(after[:, 3], after[:, :3].mean(1), atol=1e-7), "NIR must be mean(RGB)"
print("   RGB copied unchanged; NIR = mean(RGB)  [matches manuscript Section IV-E]")

print("4. deep supervision")
cfg = dict(next(c for c in grid if c["name"] == "unetpp-resnet34")); cfg["init"] = None
ds = build_model(cfg)
x = torch.zeros(2, 4, 128, 128); y = torch.zeros(2, 1, 128, 128)
ds.train(); out = ds(x)
assert isinstance(out, list) and len(out) == 4, "expected main + 3 auxiliary heads"
loss = supervised_loss(out, y); assert torch.isfinite(loss)
ds.eval()
with torch.no_grad():
    assert torch.is_tensor(ds(x)), "eval mode must return only the main head"
print("   train -> 4 heads, eval -> 1 head; latency unaffected")

cfg2 = dict(abl[0]); cfg2["init"] = None; cfg2["encoder"] = "resnet34"
nods = build_model(cfg2); nods.train()
assert torch.is_tensor(nods(x)), "ablation variant must be a plain model"
print("   ablation variant has no auxiliary heads")

print("5. protocol constants")
assert C.RUN_PLAN == "paper_table"
assert C.EXTRACT_STRIDE == C.PATCH_SIZE, "tiling must be non-overlapping"
assert C.NUM_EPOCHS == 100
assert not hasattr(C, "EARLY_STOP"), "training runs the full epoch budget"
print(f"   {C.NUM_EPOCHS} epochs, full budget (no early stopping); "
      f"stride {C.EXTRACT_STRIDE} = patch size (non-overlapping); "
      f"run plan '{C.RUN_PLAN}'")
print("\nALL CHECKS PASSED")
