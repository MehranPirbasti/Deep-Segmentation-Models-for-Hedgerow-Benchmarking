"""Runs the model grid. Safe to interrupt: it resumes and skips finished runs."""
#!/usr/bin/env python3
"""
Run the controlled comparative study: 60 configurations under the frozen
protocol, plus the multi-seed repeats and the deep-supervision ablation.

Interrupt-safe: rerunning resumes from the last completed epoch and skips
finished runs.

    python scripts/04_train.py -data-root DATA -splits split_metadata.csv \
                               -out-root OUT [-only unetpp-resnet50]
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hedgebench import config as C
from hedgebench.data import build_loaders, summarize, set_seed
from hedgebench.models import (build_experiment_grid, build_ablation_configs,
                               build_model, build_run_plan, measure_complexity)
from hedgebench.engine import train_one_epoch, evaluate, append_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def paths(out_root):
    d = {k: os.path.join(out_root, k) for k in
         ("model_weights", "checkpoints", "tables", "figures", "predictions")}
    for v in d.values():
        os.makedirs(v, exist_ok=True)
    d["metrics_csv"] = os.path.join(d["tables"], "model_metrics.csv")
    return d


def run_experiment(cfg, seed, loaders, P):
    name = cfg["name"]
    done = os.path.join(P["checkpoints"], f"{name}_seed{seed}.done")
    if os.path.exists(done):
        print(f"[skip] {name} seed={seed} already complete")
        return
    train_loader, val_loader, test_loader = loaders

    set_seed(seed)
    model = build_model(cfg, verbose=True).to(DEVICE)
    n_params, gflops = measure_complexity(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR,
                                  weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=C.COSINE_T_MAX, eta_min=C.COSINE_ETA_MIN)
    scaler = torch.amp.GradScaler("cuda", enabled=C.USE_AMP and DEVICE == "cuda")

    ckpt_file = os.path.join(P["checkpoints"], f"{name}_seed{seed}_state.pt")
    best_file = os.path.join(P["model_weights"], f"{name}_seed{seed}_best.pth")
    start_epoch, best_iou, best_epoch, elapsed = 0, -1.0, -1, 0.0
    if os.path.exists(ckpt_file):
        ck = torch.load(ckpt_file, map_location=DEVICE)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_iou = ck["epoch"], ck["best_iou"]
        best_epoch, elapsed = ck["best_epoch"], ck["elapsed"]
        print(f"[resume] {name} seed={seed} from epoch {start_epoch}")

    ds = " +DS" if cfg.get("deep_supervision") else ""
    print(f"\n{'='*74}\n{name}{ds} | seed {seed} | {n_params:.2f}M params | "
          f"{gflops:.1f} GFLOPs\n{'='*74}")

    base = dict(model_name=name, decoder=cfg["head"], backbone=cfg["backbone"], seed=seed)
    for epoch in range(start_epoch, C.NUM_EPOCHS):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, DEVICE)
        heavy = ((epoch + 1) % C.TOPO_EVERY_N_EPOCHS == 0) or (epoch + 1 == C.NUM_EPOCHS)
        val = evaluate(model, val_loader, DEVICE, with_topology=heavy)
        scheduler.step()
        elapsed += time.time() - t0

        append_metrics(P["metrics_csv"], {**base, "epoch": epoch + 1,
                                          "phase": "train", "loss": tr_loss})
        append_metrics(P["metrics_csv"], {**base, "epoch": epoch + 1,
                                          "phase": "validation", **val})
        print(f"  epoch {epoch+1:3d}/{C.NUM_EPOCHS} | train {tr_loss:.4f} | "
              f"val {val['loss']:.4f} | IoU {val['iou']:.4f} | F1 {val['dice_f1']:.4f}"
              + (f" | clDice {val['cldice']:.4f}" if heavy else ""))

        if val["iou"] > best_iou:
            best_iou, best_epoch = val["iou"], epoch + 1
            torch.save({k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()}, best_file)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1, "best_iou": best_iou,
                    "best_epoch": best_epoch, "elapsed": elapsed}, ckpt_file)


    # Final test evaluation with the selected checkpoint.
    model.load_state_dict(torch.load(best_file, map_location=DEVICE))
    test = evaluate(model, test_loader, DEVICE, with_topology=True,
                    collect_per_patch=True)
    pp = test.pop("_per_patch")
    np.savez_compressed(os.path.join(P["tables"], f"perpatch_{name}_seed{seed}.npz"), **pp)
    append_metrics(P["metrics_csv"], {**base, "epoch": "", "phase": "test",
                                      **test, "total_training_time_s": elapsed})

    comp = os.path.join(P["tables"], "model_complexity.csv")
    write_header = not os.path.exists(comp)
    with open(comp, "a") as f:
        if write_header:
            f.write("model_name,decoder,backbone,encoder,encoder_weights,"
                    "deep_supervision,params_M,gflops,best_epoch\n")
        f.write(f"{name},{cfg['head']},{cfg['backbone']},{cfg['encoder']},"
                f"{cfg.get('init')},{bool(cfg.get('deep_supervision'))},"
                f"{n_params:.3f},{gflops:.2f},{best_epoch}\n")

    print(f"  TEST IoU {test['iou']:.4f} | Dice {test['dice_f1']:.4f} | "
          f"BF1(r=2) {test['boundary_f1']:.4f} | BF1(r=1) {test['bf1_r1']:.4f} | "
          f"clDice {test['cldice']:.4f}")
    open(done, "w").write("ok")
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--only", default=None, help="run a single configuration by name")
    args = ap.parse_args()

    P = paths(args.out_root)
    C.dump(os.path.join(args.out_root, "run_config.json"))

    image_dir = os.path.join(args.data_root, "X")
    mask_dir = os.path.join(args.data_root, "Y", "Detection")
    tl, vl, sl, meta = build_loaders(args.splits, image_dir, mask_dir)
    print(summarize(meta))

    grid = build_experiment_grid()
    abl = build_ablation_configs(grid)
    jobs = build_run_plan(grid, abl)
    if args.only:
        jobs = [(c, s) for c, s in jobs if c["name"] == args.only]

    print(f"configurations: {len(grid)} | ablations: {len(abl)} | "
          f"total runs: {len(jobs)} (plan '{C.RUN_PLAN}')")

    failures = []
    for i, (cfg, seed) in enumerate(jobs, 1):
        print(f"\n########## [{i}/{len(jobs)}] {cfg['name']} seed={seed} ##########")
        try:
            run_experiment(cfg, seed, (tl, vl, sl), P)
        except Exception as exc:
            oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            print(f"  [{'OOM' if oom else 'FAILED'}] {cfg['name']} seed={seed}: {exc}")
            failures.append((cfg["name"], seed, "OOM" if oom else str(exc)[:200]))
            torch.cuda.empty_cache()

    if failures:
        with open(os.path.join(P["tables"], "failed_runs.csv"), "a") as f:
            for n, s, e in failures:
                f.write(f"{n},{s},\"{e}\"\n")
        print(f"\n{len(failures)} failures recorded.")


if __name__ == "__main__":
    main()
