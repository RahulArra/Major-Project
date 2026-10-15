"""DeepLabV3+ training with modest training-only coverage-aware sampling.

The model, preprocessing, augmentation, loss, optimizer, scheduler, and
validation protocol mirror train_deeplab.py. Only the training image sampler
differs. Run with --smoke-test to validate category mapping and sampler draws
without constructing a model or starting training.
"""

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from sen1_dataset import Sen1FloodsDataset
from deeplabv3 import DeepLabV3Plus
from losses import TverskyDiceLoss


SEED = 42
TRAIN_SPLIT = Path("data/raw/sen1floods11/splits/flood_train_data.csv")
VAL_SPLIT = Path("data/raw/sen1floods11/splits/flood_valid_data.csv")
COVERAGE_CSV = Path(
    "outputs/sen1_distribution/20261010_010822/per_image_distribution.csv"
)
OUTPUT_ROOT = Path("checkpoints_deeplab_coverage_aware")

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
USE_AMP = torch.cuda.is_available()

# Expected exposure multipliers relative to uniform per-image sampling.
# Fully ignored images are excluded (zero weight).
SAMPLING_WEIGHTS = {
    "flood_free": 1.5,
    "low": 1.5,
    "moderate": 1.0,
    "high": 1.0,
}
VALID_CATEGORIES = set(SAMPLING_WEIGHTS)
SMOKE_DRAWS = 20_000


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sampling_metadata(train_split=TRAIN_SPLIT, coverage_csv=COVERAGE_CSV,
                           val_split=VAL_SPLIT):
    """Validate exact split/CSV mapping and return per-row categories/weights."""
    train_df = pd.read_csv(train_split, header=None)
    val_df = pd.read_csv(val_split, header=None)
    coverage_df = pd.read_csv(coverage_csv)

    if train_df.shape[1] < 2 or val_df.shape[1] < 2:
        raise ValueError("Expected split CSVs with image and label columns.")
    if "split" not in coverage_df or "image" not in coverage_df or "coverage_category" not in coverage_df:
        raise ValueError("Coverage CSV is missing required columns.")

    train_names = train_df.iloc[:, 0].astype(str).tolist()
    val_names = val_df.iloc[:, 0].astype(str).tolist()
    if len(set(train_names)) != len(train_names):
        raise ValueError("Training split contains duplicate image identifiers.")
    if len(set(val_names)) != len(val_names):
        raise ValueError("Validation split contains duplicate image identifiers.")
    overlap = set(train_names) & set(val_names)
    if overlap:
        raise ValueError(f"Train/validation image overlap found: {sorted(overlap)[:5]}")

    train_rows = coverage_df[coverage_df["split"].astype(str).str.lower() == "train"].copy()
    if train_rows["image"].astype(str).duplicated().any():
        raise ValueError("Training coverage CSV contains duplicate image identifiers.")
    # The CSV must map exactly to the training split; validation rows are never
    # used to build sampler weights.
    coverage_names = set(train_rows["image"].astype(str))
    if coverage_names != set(train_names):
        missing = sorted(set(train_names) - coverage_names)
        extra = sorted(coverage_names - set(train_names))
        raise ValueError(
            f"Coverage CSV/train split mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    category_by_image = dict(zip(
        train_rows["image"].astype(str),
        train_rows["coverage_category"].astype(str),
    ))
    categories = [category_by_image[name] for name in train_names]
    invalid = sorted(set(categories) - VALID_CATEGORIES - {"fully_ignored"})
    if invalid:
        raise ValueError(f"Invalid training coverage categories: {invalid}")

    # Cross-check fully ignored rows from the CSV's valid-pixel count when
    # available, so they can never distort weighted exposure.
    if "valid_pixels" in train_rows.columns:
        zero_valid = set(train_rows.loc[
            pd.to_numeric(train_rows["valid_pixels"], errors="raise") == 0,
            "image",
        ].astype(str))
        fully_ignored = {name for name, cat in category_by_image.items() if cat == "fully_ignored"}
        if zero_valid != fully_ignored:
            raise ValueError("fully_ignored category does not match valid_pixels == 0.")

    weights = [0.0 if cat == "fully_ignored" else SAMPLING_WEIGHTS[cat]
               for cat in categories]
    eligible_count = sum(weight > 0 for weight in weights)
    if eligible_count == 0:
        raise ValueError("No supervised training images available to sample.")

    # Also assert that the unchanged validation split has no image in sampler map.
    if set(val_names) & coverage_names:
        # The combined per-image CSV includes validation rows; those are fine,
        # but must not enter the training category map or the training sampler.
        if any(name in category_by_image for name in val_names):
            raise AssertionError("Validation image entered training category mapping.")

    return train_df, val_df, categories, torch.as_tensor(weights, dtype=torch.double)


def expected_proportions(categories):
    counts = Counter(cat for cat in categories if cat in VALID_CATEGORIES)
    weighted_total = sum(counts[cat] * SAMPLING_WEIGHTS[cat] for cat in VALID_CATEGORIES)
    return {
        cat: (counts.get(cat, 0) * SAMPLING_WEIGHTS[cat] / weighted_total)
        for cat in SAMPLING_WEIGHTS
    }


def run_sampler_smoke_test():
    train_df, val_df, categories, weights = load_sampling_metadata()
    expected = expected_proportions(categories)
    eligible_count = int((weights > 0).sum().item())
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=eligible_count,
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    distribution_sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=int(SMOKE_DRAWS),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    draws = list(distribution_sampler)
    observed_counts = Counter(categories[int(index)] for index in draws)
    observed = {cat: observed_counts.get(cat, 0) / SMOKE_DRAWS for cat in SAMPLING_WEIGHTS}

    # Validate the sampler itself as well as a large empirical draw. The
    # smoke draw is independent of the epoch sampler's 251-sample length.
    epoch_indices = list(sampler)
    assert len(epoch_indices) == eligible_count
    assert all(categories[index] in VALID_CATEGORIES for index in epoch_indices)
    train_names = set(train_df.iloc[:, 0].astype(str))
    assert all(train_df.iloc[index, 0] in train_names for index in epoch_indices)
    val_names = set(val_df.iloc[:, 0].astype(str))
    assert not any(train_df.iloc[index, 0] in val_names for index in epoch_indices)
    assert all(abs(observed[cat] - expected[cat]) < 0.02 for cat in expected)

    print(f"Training images: {len(train_df)}; supervised eligible: {eligible_count}")
    print(f"Validation images: {len(val_df)}; validation sampler: unchanged sequential order")
    print("Coverage category counts (training only):", dict(Counter(categories)))
    print("Sampling weights:", SAMPLING_WEIGHTS, "; fully_ignored=0.0")
    print("Expected category proportions:")
    for cat, proportion in expected.items():
        print(f"  {cat}: {proportion:.4%}")
    print(f"Observed proportions over {SMOKE_DRAWS:,} smoke draws:")
    for cat, proportion in observed.items():
        print(f"  {cat}: {proportion:.4%}")
    print("PASS: split mapping, exclusion, valid categories, and sampler distribution checks.")


def calculate_metrics(tp, tn, fp, fn):
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
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    tp = tn = fp = fn = 0
    for images, targets, ignore_mask in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        ignore_mask = ignore_mask.to(device, non_blocking=True)
        if ignore_mask.all():
            continue
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
            logits = model(images)
        loss = criterion(logits.float(), targets, ignore_mask)
        total_loss += loss.item()
        total_samples += images.size(0)
        predictions = torch.sigmoid(logits.float())[:, 0] >= THRESHOLD
        valid = ~ignore_mask
        tp += int(((predictions == 1) & (targets == 1) & valid).sum().item())
        tn += int(((predictions == 0) & (targets == 0) & valid).sum().item())
        fp += int(((predictions == 1) & (targets == 0) & valid).sum().item())
        fn += int(((predictions == 0) & (targets == 1) & valid).sum().item())
    metrics = calculate_metrics(tp, tn, fp, fn)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Check mapping and sampler draws; do not train or write artifacts.")
    parser.add_argument("--run-name", default=None,
                        help="Create a fresh run in OUTPUT_ROOT/<run-name>; preserves earlier artifacts.")
    args = parser.parse_args()
    if args.smoke_test:
        run_sampler_smoke_test()
        return

    output_dir = OUTPUT_ROOT / args.run_name if args.run_name else OUTPUT_ROOT
    history_file = output_dir / "training_history.csv"
    best_checkpoint = output_dir / "best_deeplab_coverage_aware.pth"
    config_file = output_dir / "experiment_config.json"

    if history_file.exists() or best_checkpoint.exists() or config_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment artifacts in {output_dir}.")
    set_seed(SEED)
    train_df, val_df, categories, sample_weights = load_sampling_metadata()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = Sen1FloodsDataset(str(TRAIN_SPLIT), augment=True)
    val_dataset = Sen1FloodsDataset(str(VAL_SPLIT), augment=False)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=int((sample_weights > 0).sum().item()),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    # Keep the validation protocol deterministic and unweighted.
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    category_counts = dict(Counter(categories))
    config = {
        "experiment": "DeepLabV3+ coverage-aware image sampling",
        "seed": SEED,
        "train_split": str(TRAIN_SPLIT),
        "validation_split": str(VAL_SPLIT),
        "coverage_csv": str(COVERAGE_CSV),
        "training_images": len(train_df),
        "validation_images": len(val_df),
        "supervised_training_images": int((sample_weights > 0).sum().item()),
        "category_counts_training_only": category_counts,
        "sampling_weights": {**SAMPLING_WEIGHTS, "fully_ignored": 0.0},
        "expected_category_proportions": expected_proportions(categories),
        "sampler": "WeightedRandomSampler(replacement=True, num_samples=251 per epoch)",
        "model": {"name": "DeepLabV3Plus", "in_channels": 2, "num_classes": 1, "pretrained": False},
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": {"name": "TverskyDiceLoss", "pos_weight": POS_WEIGHT,
                 "dice_weight": DICE_WEIGHT, "alpha": TVERSKY_ALPHA,
                 "beta": TVERSKY_BETA, "smooth": 1.0},
        "threshold": THRESHOLD,
        "amp": USE_AMP,
        "validation_sampling": "sequential, shuffle=False, no sampler",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()
    print(f"Device: {device}; train images: {len(train_dataset)}; validation images: {len(val_dataset)}")
    print("Expected sampled proportions:", config["expected_category_proportions"])

    model = DeepLabV3Plus(in_channels=2, num_classes=1, pretrained=False).to(device)
    criterion = TverskyDiceLoss(pos_weight=POS_WEIGHT, dice_weight=DICE_WEIGHT,
                                alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    best_val_iou = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    fields = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_precision",
              "val_recall", "val_f1", "val_iou", "val_dice", "learning_rate", "epoch_seconds"]

    with history_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, MAX_EPOCHS + 1):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            samples_seen = 0
            for images, targets, ignore_mask in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                ignore_mask = ignore_mask.to(device, non_blocking=True)
                if ignore_mask.all():
                    continue
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
                    logits = model(images)
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Non-finite logits at epoch {epoch}.")
                loss = criterion(logits.float(), targets, ignore_mask)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"TverskyDiceLoss produced non-finite loss at epoch {epoch}.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * images.size(0)
                samples_seen += images.size(0)

            train_loss = running_loss / max(samples_seen, 1)
            val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_metrics["iou"])
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | Val IoU: {val_metrics['iou']:.4f} | "
                  f"Val Dice: {val_metrics['dice']:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")
            writer.writerow({"epoch": epoch, "train_loss": train_loss,
                             **{f"val_{k}": v for k, v in val_metrics.items()},
                             "learning_rate": current_lr, "epoch_seconds": elapsed})
            csvfile.flush()

            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                best_epoch = epoch
                epochs_without_improvement = 0
                checkpoint_config = {
                    "seed": SEED, "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                    "loss_name": "TverskyDiceLoss", "pos_weight": POS_WEIGHT,
                    "dice_weight": DICE_WEIGHT, "tversky_alpha": TVERSKY_ALPHA,
                    "tversky_beta": TVERSKY_BETA, "smooth": criterion.smooth,
                    "threshold": THRESHOLD, "pretrained": False,
                    "sampling_weights": {**SAMPLING_WEIGHTS, "fully_ignored": 0.0},
                    "expected_category_proportions": expected_proportions(categories),
                    "coverage_csv": str(COVERAGE_CSV),
                }
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_iou": best_val_iou,
                    "config": checkpoint_config,
                }, best_checkpoint)
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break

    print(f"Training complete. Best epoch={best_epoch}, validation IoU={best_val_iou:.4f}")
    print("Checkpoint:", best_checkpoint)
    print("History:", history_file)
    print("Configuration:", config_file)
    if device.type == "cuda":
        print("Peak GPU memory (MB):", round(torch.cuda.max_memory_allocated() / 1024**2, 1))


if __name__ == "__main__":
    main()
