"""Train one LOW or HIGH DeepLabV3+ specialist on manifest patches.

Invoke once per specialist. Each invocation creates a new, isolated output
directory; this script never loads or writes the baseline checkpoint.
"""

import argparse
import contextlib
import csv
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diagnostic_utils import PROJECT_ROOT
from deeplabv3 import DeepLabV3Plus
from losses import TverskyDiceLoss
from sen1_patch_dataset import Sen1FloodsPatchDataset


SEED = 42
BATCH_SIZE = 1
MAX_EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 3.0
DICE_WEIGHT = 0.5
TVERSKY_ALPHA = 0.7
TVERSKY_BETA = 0.3
THRESHOLD = 0.50
NUM_WORKERS = 0
OUTPUT_ROOT = PROJECT_ROOT / "checkpoints_deeplab_specialists"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics_from_counts(tp, tn, fp, fn):
    eps = 1e-8
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "iou": iou, "dice": dice}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp):
    model.eval()
    loss_sum = 0.0
    scored_patches = 0
    tp = tn = fp = fn = 0
    for images, targets, ignore_mask in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        ignore_mask = ignore_mask.to(device, non_blocking=True)
        if ignore_mask.all():
            continue
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
        loss = criterion(logits.float(), targets, ignore_mask)
        loss_sum += float(loss.item())
        scored_patches += int(images.size(0))
        predicted = torch.sigmoid(logits.float())[:, 0] >= THRESHOLD
        valid = ~ignore_mask
        tp += int(((predicted == 1) & (targets == 1) & valid).sum().item())
        tn += int(((predicted == 0) & (targets == 0) & valid).sum().item())
        fp += int(((predicted == 1) & (targets == 0) & valid).sum().item())
        fn += int(((predicted == 0) & (targets == 1) & valid).sum().item())
    result = metrics_from_counts(tp, tn, fp, fn)
    result["loss"] = loss_sum / max(scored_patches, 1)
    result["scored_patches"] = scored_patches
    return result


def preflight(manifest_path, specialist, minimum_parent_count):
    train = Sen1FloodsPatchDataset(manifest_path, "train", group=specialist, augment=True)
    validation = Sen1FloodsPatchDataset(manifest_path, "validation", augment=False)
    if not len(train):
        raise RuntimeError(f"{specialist} specialist has zero train patches.")
    train_parents = {row["parent_id"] for row in train.rows}
    val_parents = {row["parent_id"] for row in validation.rows}
    if train_parents & val_parents:
        raise RuntimeError("Training specialist and validation share parent IDs.")
    parent_count = len(train_parents)
    if parent_count < minimum_parent_count:
        raise RuntimeError(
            f"{specialist} specialist has only {parent_count} training parents; "
            f"configured minimum is {minimum_parent_count}. Stopping before training."
        )
    for row in train.rows:
        if row["split"] != "train" or row["training_group"] != specialist:
            raise RuntimeError("Non-training or misassigned patch entered specialist dataset.")
    return train, validation, parent_count


def run_training(args):
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    specialist = args.specialist.upper()
    train_dataset, val_dataset, train_parent_count = preflight(
        manifest_path, specialist, args.minimum_parent_count,
    )
    if not len(val_dataset):
        raise RuntimeError("Validation patch dataset is empty.")
    val_parent_count = len({row["parent_id"] for row in val_dataset.rows})

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run-name must be a single directory name.")
    out_dir = OUTPUT_ROOT / run_name / specialist.lower()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    history_path = out_dir / "training_history.csv"
    checkpoint_path = out_dir / "best_model.pth"
    config_path = out_dir / "experiment_config.json"
    log_path = out_dir / "training.log"

    train_counts = Counter(row["training_group"] for row in train_dataset.rows)
    train_parents_by_group = {
        specialist: train_parent_count,
    }
    config = {
        "experiment": f"DeepLabV3+ {specialist} coverage specialist",
        "specialist": specialist,
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)) if manifest_path.is_relative_to(PROJECT_ROOT) else str(manifest_path),
        "training_split": "train only",
        "validation_split": "validation only; all validation patches used without specialist routing",
        "test_data_used": False,
        "seed": SEED,
        "parent_size": 512,
        "patch_size": 256,
        "coverage_threshold_fraction": 0.30,
        "group_rule": "LOW < 0.30; HIGH >= 0.30 among valid labels 0/1; -1 ignored",
        "specialist_training_patch_count": len(train_dataset),
        "specialist_training_parent_count": train_parent_count,
        "validation_patch_count": len(val_dataset),
        "training_group_counts": dict(train_counts),
        "minimum_training_parent_count": args.minimum_parent_count,
        "model": {"name": "DeepLabV3Plus", "in_channels": 2, "num_classes": 1, "pretrained": False},
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": {"name": "TverskyDiceLoss", "pos_weight": POS_WEIGHT,
                 "dice_weight": DICE_WEIGHT, "alpha": TVERSKY_ALPHA,
                 "beta": TVERSKY_BETA, "smooth": 1.0},
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau(mode=max, factor=0.5, patience=2, min_lr=1e-6)",
        "gradient_clip_norm": 1.0,
        "amp": torch.cuda.is_available(),
        "threshold": THRESHOLD,
        "validation_parent_overlap": 0,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    set_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=amp,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=amp,
    )
    model = DeepLabV3Plus(in_channels=2, num_classes=1, pretrained=False).to(device)
    criterion = TverskyDiceLoss(
        pos_weight=POS_WEIGHT, dice_weight=DICE_WEIGHT,
        alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_val_iou = -1.0
    best_epoch = 0
    stale_epochs = 0
    fields = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_precision",
              "val_recall", "val_f1", "val_iou", "val_dice", "learning_rate", "epoch_seconds"]

    with log_path.open("w", encoding="utf-8") as logfile, contextlib.redirect_stdout(Tee(sys.stdout, logfile)):
        print(f"Specialist: {specialist}; device={device}; train patches={len(train_dataset)}; train parents={train_parent_count}")
        print(f"Validation patches={len(val_dataset)}; validation parents={val_parent_count}")
        print(f"Checkpoint: {checkpoint_path}")
        if amp:
            print("GPU:", torch.cuda.get_device_name(0))
            torch.cuda.reset_peak_memory_stats()
        with history_path.open("w", newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, MAX_EPOCHS + 1):
                started = time.time()
                model.train()
                loss_total = 0.0
                patches_seen = 0
                for images, targets, ignore_mask in train_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    ignore_mask = ignore_mask.to(device, non_blocking=True)
                    if ignore_mask.all():
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                        logits = model(images)
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f"Non-finite logits at epoch {epoch}.")
                    loss = criterion(logits.float(), targets, ignore_mask)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite TverskyDiceLoss at epoch {epoch}.")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    loss_total += float(loss.item())
                    patches_seen += int(images.size(0))

                if not patches_seen:
                    raise RuntimeError("No supervised training patches were processed.")
                train_loss = loss_total / patches_seen
                val_metrics = evaluate(model, val_loader, criterion, device, amp)
                scheduler.step(val_metrics["iou"])
                elapsed = time.time() - started
                lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | Train Loss {train_loss:.4f} | "
                      f"Val Loss {val_metrics['loss']:.4f} | Val IoU {val_metrics['iou']:.4f} | "
                      f"Val Dice {val_metrics['dice']:.4f} | LR {lr:.2e} | {elapsed:.1f}s")
                writer.writerow({"epoch": epoch, "train_loss": train_loss,
                                 **{f"val_{key}": value for key, value in val_metrics.items() if key != "scored_patches"},
                                 "learning_rate": lr, "epoch_seconds": elapsed})
                history_file.flush()

                if val_metrics["iou"] > best_val_iou:
                    best_val_iou = val_metrics["iou"]
                    best_epoch = epoch
                    stale_epochs = 0
                    checkpoint_config = {
                        "seed": SEED, "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                        "loss_name": "TverskyDiceLoss", "pos_weight": POS_WEIGHT,
                        "dice_weight": DICE_WEIGHT, "tversky_alpha": TVERSKY_ALPHA,
                        "tversky_beta": TVERSKY_BETA, "smooth": criterion.smooth,
                        "threshold": THRESHOLD, "pretrained": False,
                        "specialist": specialist, "coverage_threshold_fraction": 0.30,
                        "training_patch_count": len(train_dataset),
                        "training_parent_count": train_parent_count,
                        "manifest": config["manifest"],
                    }
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_iou": best_val_iou,
                        "config": checkpoint_config,
                    }, checkpoint_path)
                else:
                    stale_epochs += 1
                if stale_epochs >= PATIENCE:
                    print(f"Early stopping after epoch {epoch}; best epoch={best_epoch}.")
                    break
        print(f"Training complete. Best epoch={best_epoch}, best validation IoU={best_val_iou:.6f}")
        if amp:
            print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        print(f"History: {history_path}")
        print(f"Config: {config_path}")
        print(f"Log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--specialist", required=True, choices=("LOW", "HIGH", "low", "high"))
    parser.add_argument("--run-name", required=True, help="Shared run folder name; each specialist gets its own subdirectory.")
    parser.add_argument("--minimum-parent-count", type=int, required=True,
                        help="Explicit minimum distinct train-parent count approved after inspecting the manifest.")
    args = parser.parse_args()
    if args.minimum_parent_count < 1:
        parser.error("--minimum-parent-count must be positive")
    run_training(args)


if __name__ == "__main__":
    main()
