"""Builds the decoder/backbone combinations.

Two things here are easy to get wrong and are worth reading before you change
them:

- 4-band input. smp makes the stem with in_channels=4 and copies the pretrained
  RGB kernels into the first three. We leave those alone and set the NIR kernel
  to their mean. No rescaling. The paper says the same thing in Section IV-E.
- Deep supervision. Only U-Net++ has intermediate nested outputs, so it is the
  only model that gets auxiliary heads. In eval mode the wrapper returns just
  the final head, so latency is unchanged.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from . import config as C


DECODER_LABEL = {
    "unet": "U-Net", "unetpp": "U-Net++", "deeplab": "DeepLabV3",
    "deeplabplus": "DeepLabV3+", "fpn": "FPN", "linknet": "LinkNet",
    "pan": "PAN", "segformer": "SegFormer", "dpt": "DPT", "upernet": "UPerNet",
}
BACKBONE_LABEL = {
    "resnet34": "ResNet-34", "resnet50": "ResNet-50",
    "mobilenetv2": "MobileNet-V2", "mitb0": "MiT-B0",
    "mitb2": "MiT-B2", "mitb4": "MiT-B4",
}

HEADS = [
    (smp.Unet, "unet"), (smp.UnetPlusPlus, "unetpp"),
    (smp.DeepLabV3, "deeplab"), (smp.DeepLabV3Plus, "deeplabplus"),
    (smp.FPN, "fpn"), (smp.Linknet, "linknet"), (smp.PAN, "pan"),
    (smp.Segformer, "segformer"), (smp.DPT, "dpt"), (smp.UPerNet, "upernet"),
]
BACKBONES = [
    ("resnet34", "resnet34"), ("resnet50", "resnet50"),
    ("tu-mobilenetv2_100", "mobilenetv2"), ("mit_b0", "mitb0"),
    ("mit_b2", "mitb2"), ("mit_b4", "mitb4"),
]


# Experiment grid
def build_experiment_grid() -> List[Dict]:
    """The 60 reported configurations, all under a single encoder initialisation."""
    grid = []
    for arch, head in HEADS:
        for encoder, bb in BACKBONES:
            enc = encoder
            if arch is smp.DPT and encoder in ("resnet34", "resnet50"):
                enc = "tu-" + encoder            # DPT requires timm-style encoders
            grid.append({
                "arch": arch, "head": head, "encoder": enc, "backbone": bb,
                "init": C.ENCODER_INIT,
                "deep_supervision": C.DEEP_SUPERVISION_HEADS.get(head, False),
                "name": f"{head}-{bb}",
            })
    return grid


def build_ablation_configs(grid: List[Dict]) -> List[Dict]:
    """Ablation variants declared in config.ABLATION_RUNS (e.g. U-Net++ without DS)."""
    by_name = {c["name"]: c for c in grid}
    out = []
    for spec in C.ABLATION_RUNS:
        base = by_name.get(spec["base"])
        if base is None:
            continue
        cfg = dict(base)
        cfg.update(spec["override"])
        cfg["name"] = spec["name"]
        cfg["is_ablation"] = True
        out.append(cfg)
    return out


# Four-band adaptation
def adapt_first_conv_4band(model: nn.Module, verbose: bool = False) -> nn.Module:
    """
    Channel-mean inflation, exactly as described in the manuscript.

    The stem was created by `segmentation_models_pytorch` with in_channels=4; smp
    populates the first three input channels from the pretrained checkpoint. We
    keep those three copied RGB kernels UNCHANGED and overwrite the fourth (NIR)
    channel with their mean, so the NIR channel starts from an informative,
    unbiased prior instead of from random values.

    No rescaling factor is applied to the RGB kernels. Any rescaling would make
    the description in the paper and the implementation disagree.
    """
    first = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == C.IN_CHANNELS:
            first = m
            break
    if first is None:
        if verbose:
            print("  [warn] no 4-channel stem found; skipping 4-band adaptation")
        return model
    with torch.no_grad():
        w = first.weight.detach().clone()
        rgb = w[:, :3]                          # copied unchanged
        nir = rgb.mean(dim=1, keepdim=True)     # mean of the copied RGB weights
        first.weight.copy_(torch.cat([rgb, nir], dim=1))
    if verbose:
        print(f"  [4-band] channel-mean inflation on stem {tuple(first.weight.shape)}")
    return model


# Deep supervision wrapper for U-Net++
class UnetPlusPlusDeepSupervision(nn.Module):
    """
    Wraps an smp.UnetPlusPlus and attaches auxiliary 1x1 segmentation heads to
    the intermediate nested decoder outputs X^{0,1}, X^{0,2}, X^{0,3}.

    Training  : forward() returns [main_logits, aux1, aux2, aux3], each already
                resized to the input resolution. The combined loss is applied to
                every head and averaged with equal weight.
    Inference : model.eval() -> forward() returns only the main logits, so the
                measured latency is identical to plain U-Net++.

    The channel widths of the intermediate blocks are discovered by a dry run at
    construction time rather than hard-coded, so the wrapper does not depend on
    the internal naming or channel schedule of any particular smp version.
    """

    AUX_BLOCKS = ("x_0_1", "x_0_2", "x_0_3")

    def __init__(self, base: nn.Module, classes: int = 1,
                 in_channels: int = C.IN_CHANNELS, probe_size: int = 128):
        super().__init__()
        self.base = base
        self.classes = classes
        self._captured: Dict[str, torch.Tensor] = {}
        self._hooks = []

        blocks = self._find_blocks(base)
        if blocks is None:
            raise RuntimeError(
                "Deep supervision requested but the nested decoder blocks could "
                "not be located. Deep supervision is only defined for U-Net++.")
        self._tracked = {}
        for key in self.AUX_BLOCKS:
            mod = blocks.get(key)
            if mod is not None:
                self._tracked[key] = mod
        if not self._tracked:
            raise RuntimeError("No intermediate nested decoder blocks found.")

        for key, mod in self._tracked.items():
            self._hooks.append(mod.register_forward_hook(self._make_hook(key)))

        # Dry run to discover intermediate channel widths.
        widths = self._probe(in_channels, probe_size)
        self.aux_heads = nn.ModuleDict({
            key: nn.Conv2d(widths[key], classes, kernel_size=1)
            for key in widths
        })

    # internals
    @staticmethod
    def _find_blocks(base: nn.Module) -> Optional[Dict[str, nn.Module]]:
        decoder = getattr(base, "decoder", None)
        if decoder is None:
            return None
        blocks = getattr(decoder, "blocks", None)
        if blocks is None:
            return None
        return {k: v for k, v in blocks.items()}

    def _make_hook(self, key: str):
        def hook(_module, _inp, out):
            self._captured[key] = out
        return hook

    @torch.no_grad()
    def _probe(self, in_channels: int, size: int) -> Dict[str, int]:
        was_training = self.base.training
        self.base.eval()
        dummy = torch.zeros(1, in_channels, size, size)
        self._captured.clear()
        self.base(dummy)
        widths = {k: int(v.shape[1]) for k, v in self._captured.items()
                  if k in self._tracked}
        self._captured.clear()
        if was_training:
            self.base.train()
        if not widths:
            raise RuntimeError("Dry run did not capture any nested decoder output.")
        return widths

    # forward
    def forward(self, x):
        self._captured.clear()
        main = self.base(x)
        if not self.training:
            self._captured.clear()
            return main
        outs = [main]
        for key, head in self.aux_heads.items():
            feat = self._captured.get(key)
            if feat is None:
                continue
            logit = head(feat)
            if logit.shape[-2:] != main.shape[-2:]:
                logit = F.interpolate(logit, size=main.shape[-2:],
                                      mode="bilinear", align_corners=False)
            outs.append(logit)
        self._captured.clear()
        return outs

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# Public factory
def build_model(cfg: Dict, verbose: bool = False) -> nn.Module:
    kwargs = {
        "encoder_name": cfg["encoder"],
        "encoder_weights": cfg.get("init", C.ENCODER_INIT),
        "in_channels": C.IN_CHANNELS,
        "classes": 1,
    }
    if cfg["head"] in ("unet", "unetpp"):
        kwargs["decoder_attention_type"] = "scse"

    try:
        model = cfg["arch"](**kwargs)
    except KeyError:
        if "tu-" in kwargs["encoder_name"]:
            kwargs["encoder_name"] = kwargs["encoder_name"].replace("tu-", "")
        model = cfg["arch"](**kwargs)

    model = adapt_first_conv_4band(model, verbose=verbose)

    if cfg.get("deep_supervision", False):
        model = UnetPlusPlusDeepSupervision(model, classes=1)
        if verbose:
            print(f"  [deep supervision] auxiliary heads: "
                  f"{list(model.aux_heads.keys())}")
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_complexity(model: nn.Module):
    """Trainable parameters (M) and GFLOPs at the native evaluation size."""
    n_params = count_parameters(model) / 1e6
    gflops = float("nan")
    try:
        from ptflops import get_model_complexity_info
        model.eval()   # DS wrapper returns a single tensor in eval mode
        macs, _ = get_model_complexity_info(
            model, (C.IN_CHANNELS, C.EVAL_SIZE, C.EVAL_SIZE),
            as_strings=False, print_per_layer_stat=False)
        gflops = 2.0 * macs / 1e9
    except Exception as exc:
        print(f"  [warn] GFLOPs measurement failed: {exc}")
    return n_params, gflops


def build_run_plan(grid: List[Dict], ablations: List[Dict]) -> List:
    """Expand the grid into (config, seed) jobs according to config.RUN_PLAN."""
    jobs = []
    for cfg in grid:
        seeds = [C.BASE_SEED]
        if C.RUN_PLAN == "full":
            seeds = seeds + list(C.EXTRA_SEEDS)
        elif C.RUN_PLAN == "paper_table" and cfg["name"] in set(C.MULTI_SEED_MODELS):
            seeds = seeds + list(C.EXTRA_SEEDS)
        for s in seeds:
            jobs.append((cfg, s))
    for cfg in ablations:                    # ablations run at the base seed
        jobs.append((cfg, C.BASE_SEED))
    return jobs
