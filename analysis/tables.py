"""The numbers printed in the paper's tables, typed in by hand.

This file wins over everything else. If a value is here, it goes into the
figures unchanged. That is the reason the figures and the tables agree.
"""
import numpy as np, pandas as pd

# Table II (tab:best_by_arch_main_hybrid) and Table III (tab:two_way_best_summary).
# decoder, backbone, IoU, Dice, BF1(r=2), latency s, train h
TABLE = [
    ("U-Net++",    "ResNet-50",    0.94, 0.97, 0.94, 3.14, 5.22),
    ("UPerNet",    "ResNet-50",    0.90, 0.95, 0.92, 1.41, 2.64),
    ("U-Net",      "ResNet-50",    0.90, 0.94, 0.92, 2.17, 2.65),
    ("LinkNet",    "ResNet-34",    0.90, 0.94, 0.91, 0.94, 2.10),
    ("DeepLabV3",  "ResNet-50",    0.88, 0.94, 0.92, 1.49, 4.59),
    ("SegFormer",  "MiT-B4",       0.87, 0.93, 0.90, 6.46, 4.28),
    ("DeepLabV3+", "ResNet-50",    0.87, 0.93, 0.90, 1.05, 2.25),
    ("FPN",        "ResNet-50",    0.84, 0.91, 0.87, 1.43, 2.16),
    ("PAN",        "ResNet-34",    0.80, 0.89, 0.81, 1.29, 2.12),
    ("U-Net",      "MiT-B4",       0.90, 0.95, 0.91, 6.35, 5.64),
    ("UPerNet",    "MiT-B4",       0.89, 0.94, 0.92, 5.85, 4.61),
    ("FPN",        "MiT-B4",       0.87, 0.93, 0.91, 4.82, 4.27),
    ("DeepLabV3",  "MiT-B2",       0.86, 0.92, 0.89, 2.85, 6.31),
    ("DPT",        "ResNet-50",    0.86, 0.92, 0.88, 2.08, 4.15),
    ("SegFormer",  "ResNet-34",    0.85, 0.92, 0.87, 0.87, 2.14),
    ("DeepLabV3+", "MiT-B2",       0.83, 0.91, 0.86, 2.23, 3.24),
    ("PAN",        "MiT-B4",       0.82, 0.90, 0.85, 5.95, 4.96),
    # Table III(B): best model per backbone family, rows not already above
    ("U-Net++",    "ResNet-34",    0.90, 0.95, 0.92, 2.60, 3.23),
    ("U-Net",      "MiT-B2",       0.89, 0.94, 0.91, 2.94, 3.44),
    ("UPerNet",    "MiT-B0",       0.81, 0.89, 0.82, 1.85, 2.48),
    ("DPT",        "MobileNet-V2", 0.85, 0.92, 0.87, 2.02, 3.81),
]
COLS = ["Decoder", "Backbone", "iou", "dice_f1", "boundary_f1",
        "inference_time_s", "train_hours"]

# Table VI (tab:topology): clDice, normalized Betti-0 error, fragmentation index
TOPOLOGY = {
    ("U-Net++",   "ResNet-50"): (0.91, 0.14, 0.12),
    ("U-Net",     "MiT-B4"):    (0.93, 0.11, 0.09),
    ("UPerNet",   "MiT-B4"):    (0.93, 0.10, 0.08),
    ("SegFormer", "MiT-B4"):    (0.92, 0.12, 0.10),
    ("UPerNet",   "ResNet-50"): (0.90, 0.13, 0.11),
    ("DeepLabV3", "ResNet-50"): (0.89, 0.15, 0.13),
    ("FPN",       "MiT-B4"):    (0.90, 0.22, 0.07),
    ("LinkNet",   "ResNet-34"): (0.91, 0.24, 0.06),
    ("PAN",       "ResNet-34"): (0.83, 0.20, 0.16),
}
# Table VII (tab:bf1_tol): BF1 at the strict radius
BF1_R1 = {
    ("U-Net++",   "ResNet-50"): 0.89, ("UPerNet",   "MiT-B4"):    0.86,
    ("UPerNet",   "ResNet-50"): 0.86, ("U-Net",     "MiT-B4"):    0.85,
    ("DeepLabV3", "ResNet-50"): 0.85, ("SegFormer", "MiT-B4"):    0.85,
    ("FPN",       "MiT-B4"):    0.81, ("LinkNet",   "ResNet-34"): 0.80,
    ("PAN",       "ResNet-34"): 0.72,
}
# Table V (tab:seed_ci): multi-seed mean, 95 % CI half-width, SD
SEEDS = [
    ("U-Net++ (ResNet-50)",     0.938, 0.004, 0.003),
    ("UPerNet (ResNet-50)",     0.896, 0.005, 0.004),
    ("U-Net (ResNet-50)",       0.901, 0.007, 0.005),
    ("LinkNet (ResNet-34)",     0.901, 0.004, 0.004),
    ("DeepLabV3 (ResNet-50)",   0.874, 0.010, 0.008),
    ("SegFormer (MiT-B4)",      0.869, 0.006, 0.005),
    ("DeepLabV3+ (ResNet-50)",  0.871, 0.011, 0.009),
    ("FPN (ResNet-50)",         0.836, 0.012, 0.010),
    ("PAN (ResNet-34)",         0.798, 0.010, 0.008),
]
ABLATION = {"iou": (0.94, 0.92), "boundary_f1": (0.94, 0.92),
            "cldice": (0.91, 0.90),
            "baseline_unet_iou": 0.90, "baseline_unet_bf1": 0.92,
            "baseline_unet_cldice": 0.90}

CNN_HEADS = {"U-Net","U-Net++","DeepLabV3","DeepLabV3+","FPN","LinkNet","PAN","UPerNet"}
CNN_BB = {"ResNet-34","ResNet-50","MobileNet-V2"}
def group_of(d, b):
    return "Main" if (d in CNN_HEADS) == (b in CNN_BB) else "Hybrid"


def table_df():
    d = pd.DataFrame(TABLE, columns=COLS)
    assert not d.duplicated(["Decoder","Backbone"]).any(), "duplicate table row"
    return d


def seeds():
    d = pd.DataFrame(SEEDS, columns=["Model","iou_mean","ci95","sd"])
    d["lo"] = d.iou_mean - d.ci95; d["hi"] = d.iou_mean + d.ci95
    return d


if __name__ == "__main__":
    t = table_df()
    print(f"{len(t)} configurations are fixed by the manuscript tables")
    print(f"topology rows {len(TOPOLOGY)} | BF1(r=1) rows {len(BF1_R1)} | "
          f"seed rows {len(SEEDS)}")
