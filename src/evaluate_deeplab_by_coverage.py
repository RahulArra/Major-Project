"""Evaluate the existing DeepLabV3+ checkpoint by validation scene coverage.

This script is read-only with respect to model weights. It evaluates only the
validation split at a fixed probability threshold of 0.46, excludes ignored
pixels from confusion counts, and takes per-image coverage categories from the
generated distribution CSV. Fully ignored images are listed but not scored.

Category micro metrics pool valid pixels across images in the category. Macro
metrics average the defined per-image metrics equally across evaluable images.
IoU/Dice are undefined (blank in CSV) for images/categories with no positive
ground truth and no positive predictions; such cases are not assigned an
artificial perfect score.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diagnostic_utils import (
    CHECKPOINT_PATH,
    PROJECT_ROOT,
    get_device,
    load_deeplab,
    new_output_dir,
)
from sen1_dataset import Sen1FloodsDataset


VAL_SPLIT = PROJECT_ROOT / "data/raw/sen1floods11/splits/flood_valid_data.csv"
THRESHOLD = 0.46
BATCH_SIZE = 1
CATEGORY_ORDER = ("flood_free", "low", "moderate", "high", "fully_ignored")
PER_IMAGE_FIELDS = (
    "split", "image", "coverage_category", "flood_coverage_percentage",
    "evaluated", "valid_pixels", "ignored_pixels", "tp", "fp", "fn",
    "iou", "dice", "precision", "recall",
)
SUMMARY_FIELDS = (
    "coverage_category", "image_count", "evaluated_images",
    "fully_ignored_images", "valid_pixels", "tp", "fp", "fn",
    "micro_iou", "micro_dice", "micro_precision", "micro_recall",
    "macro_mean_iou", "macro_mean_dice", "macro_mean_precision",
    "macro_mean_recall",
)


def latest_per_image_csv():
    pattern = PROJECT_ROOT / "outputs" / "sen1_distribution"
    candidates = list(pattern.glob("*/per_image_distribution.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No per_image_distribution.csv found under outputs/sen1_distribution. "
            "Run src/analyze_sen1_distribution.py first or pass --coverage-csv."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_validation_categories(csv_path):
    categories = {}
    coverages = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "image", "coverage_category", "flood_coverage_percentage"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Coverage CSV needs columns: {sorted(required)}")
        for row in reader:
            if row["split"] != "validation":
                continue
            image = row["image"]
            if image in categories:
                raise ValueError(f"Duplicate validation image in coverage CSV: {image}")
            category = row["coverage_category"]
            if category not in CATEGORY_ORDER:
                raise ValueError(f"Unknown validation coverage category {category!r} for {image}")
            categories[image] = category
            value = row["flood_coverage_percentage"].strip()
            coverages[image] = float(value) if value else None
    if not categories:
        raise ValueError(f"No validation rows found in {csv_path}")
    return categories, coverages


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics_from_counts(tp, fp, fn):
    return {
        "iou": ratio(tp, tp + fp + fn),
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
    }


def macro_mean(rows, metric):
    values = [row[metric] for row in rows if row[metric] is not None]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-csv",
        type=lambda value: Path(value) if os.path.isabs(value) else PROJECT_ROOT / value,
        default=None,
        help="Per-image coverage CSV; defaults to the latest generated file.",
    )
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)

    coverage_csv = args.coverage_csv or latest_per_image_csv()
    if not coverage_csv.is_file():
        raise FileNotFoundError(f"Coverage CSV not found: {coverage_csv}")
    if not VAL_SPLIT.is_file():
        raise FileNotFoundError(f"Validation split not found: {VAL_SPLIT}")

    category_by_image, coverage_by_image = load_validation_categories(coverage_csv)
    dataset = Sen1FloodsDataset(str(VAL_SPLIT), augment=False)
    dataset_names = [str(dataset.df.iloc[i, 0]) for i in range(len(dataset))]
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("Validation split contains duplicate image names")
    missing = sorted(set(dataset_names) - set(category_by_image))
    extra = sorted(set(category_by_image) - set(dataset_names))
    if missing or extra:
        raise ValueError(
            "Validation split and coverage CSV do not match exactly. "
            f"Missing categories for {len(missing)} images; extra validation rows: {len(extra)}."
        )

    device = get_device()
    model, checkpoint = load_deeplab(device)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    per_image_rows = []
    model.eval()
    with torch.inference_mode():
        for index, (images, targets, ignore_mask) in enumerate(loader):
            image_name = dataset_names[index]
            category = category_by_image[image_name]
            valid = ~ignore_mask[0].bool()
            valid_pixels = int(valid.sum().item())
            ignored_pixels = int(ignore_mask[0].sum().item())

            if valid_pixels == 0:
                counts = {"tp": 0, "fp": 0, "fn": 0}
                metrics = {key: None for key in ("iou", "dice", "precision", "recall")}
                evaluated = False
            else:
                logits = model(images.to(device))
                predicted = torch.sigmoid(logits.float())[0, 0] >= THRESHOLD
                truth = targets[0].to(device).bool()
                valid_device = valid.to(device)
                tp = int((predicted & truth & valid_device).sum().item())
                fp = int((predicted & ~truth & valid_device).sum().item())
                fn = int((~predicted & truth & valid_device).sum().item())
                counts = {"tp": tp, "fp": fp, "fn": fn}
                metrics = metrics_from_counts(tp, fp, fn)
                evaluated = True

            per_image_rows.append({
                "split": "validation",
                "image": image_name,
                "coverage_category": category,
                "flood_coverage_percentage": coverage_by_image[image_name],
                "evaluated": evaluated,
                "valid_pixels": valid_pixels,
                "ignored_pixels": ignored_pixels,
                **counts,
                **metrics,
            })

    summary_rows = []
    for category in CATEGORY_ORDER:
        members = [row for row in per_image_rows if row["coverage_category"] == category]
        scored = [row for row in members if row["evaluated"]]
        tp = sum(row["tp"] for row in scored)
        fp = sum(row["fp"] for row in scored)
        fn = sum(row["fn"] for row in scored)
        micro = metrics_from_counts(tp, fp, fn)
        summary_rows.append({
            "coverage_category": category,
            "image_count": len(members),
            "evaluated_images": len(scored),
            "fully_ignored_images": sum(not row["evaluated"] for row in members),
            "valid_pixels": sum(row["valid_pixels"] for row in members),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "micro_iou": micro["iou"],
            "micro_dice": micro["dice"],
            "micro_precision": micro["precision"],
            "micro_recall": micro["recall"],
            "macro_mean_iou": macro_mean(scored, "iou"),
            "macro_mean_dice": macro_mean(scored, "dice"),
            "macro_mean_precision": macro_mean(scored, "precision"),
            "macro_mean_recall": macro_mean(scored, "recall"),
        })

    out_dir = new_output_dir("deeplab_coverage_evaluation")
    per_image_path = out_dir / "per_image_metrics.csv"
    summary_path = out_dir / "category_summary.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_IMAGE_FIELDS)
        writer.writeheader()
        writer.writerows(per_image_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "threshold": THRESHOLD,
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_best_val_iou": checkpoint.get("best_val_iou") if isinstance(checkpoint, dict) else None,
        "validation_split": str(VAL_SPLIT.relative_to(PROJECT_ROOT)),
        "coverage_csv": str(coverage_csv.relative_to(PROJECT_ROOT)) if coverage_csv.is_relative_to(PROJECT_ROOT) else str(coverage_csv),
        "ignore_handling": "Ignored pixels excluded from TP, FP, FN and all metrics.",
        "category_metric_notes": "Micro metrics pool pixels; macro metrics average defined per-image metrics. Fully ignored images are listed but not scored. Undefined ratios are blank.",
        "test_data_used": False,
        "training_performed": False,
    }
    (out_dir / "evaluation_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Device: {device}; validation split images: {len(dataset)}; threshold: {THRESHOLD:.2f}")
    print("Images by coverage category (evaluated images; fully ignored images shown separately):")
    for row in summary_rows:
        print(
            f"  {row['coverage_category']}: {row['evaluated_images']} evaluated / "
            f"{row['image_count']} total; fully ignored={row['fully_ignored_images']}"
        )
    print(f"Per-image metrics: {per_image_path}")
    print(f"Category summary: {summary_path}")
    print(f"Evaluation metadata: {out_dir / 'evaluation_info.json'}")


if __name__ == "__main__":
    main()
