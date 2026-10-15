"""Compare existing segmentation checkpoints on the Sen1Floods11 validation split.

Read-only evaluation: no model training, checkpoint writes, or test split access.
Every compatible checkpoint is evaluated on the same ordered validation images
using the existing dataset loader and thresholds 0.50 (primary) and 0.46
(sensitivity). Ignored label pixels are excluded from all confusion counts.

Outputs are written under a fresh outputs/checkpoint_validation_comparison/
timestamp directory. Category IoU and Dice are pixel-pooled (micro) scores.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deeplabv3 import DeepLabV3Plus
from sen1_dataset import Sen1FloodsDataset
from unet import UNet


ROOT = Path(__file__).resolve().parents[1]
VAL_SPLIT = ROOT / "data/raw/sen1floods11/splits/flood_valid_data.csv"
COVERAGE_ROOT = ROOT / "outputs/sen1_distribution"
CHECKPOINTS = (
    ("unet_baseline", "UNet", ROOT / "checkpoints/best_unet.pth"),
    (
        "unet_tversky_T2_alpha07_beta03",
        "UNet",
        ROOT / "tversky_checkpoints/T2_alpha07_beta03/best_model.pth",
    ),
    ("deeplabv3plus_baseline", "DeepLabV3Plus", ROOT / "checkpoints_deeplab/best_deeplab.pth"),
)
THRESHOLDS = (0.50, 0.46)
CATEGORIES = ("flood_free", "low", "moderate", "high")
BATCH_SIZE = 1


def latest_coverage_csv():
    candidates = list(COVERAGE_ROOT.glob("*/per_image_distribution.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No per-image coverage CSV found under {COVERAGE_ROOT}. "
            "Run src/analyze_sen1_distribution.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_coverage(path):
    result = {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        needed = {"split", "image", "coverage_category"}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f"Coverage CSV must contain {sorted(needed)}")
        for row in reader:
            if row["split"].strip().lower() != "validation":
                continue
            image = row["image"].strip()
            category = row["coverage_category"].strip()
            if category == "fully_ignored":
                # Kept as an explicit category in the per-image output, but not
                # part of the four requested flood-coverage summaries.
                pass
            elif category not in CATEGORIES:
                raise ValueError(f"Unknown validation category {category!r} for {image}")
            if image in result:
                raise ValueError(f"Duplicate validation coverage row for {image}")
            result[image] = category
    return result


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(tp, tn, fp, fn):
    return {
        "iou": safe_ratio(tp, tp + fp + fn),
        "dice": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "accuracy": safe_ratio(tp + tn, tp + tn + fp + fn),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_model(architecture, checkpoint_path, device):
    # Explicit weights_only=False is required for training checkpoints that
    # include ordinary Python metadata alongside state_dict tensors.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a dictionary")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint has no model_state_dict dictionary")

    if architecture == "UNet":
        model = UNet(in_channels=2, out_channels=1)
        model_config = {"in_channels": 2, "out_channels": 1, "architecture": "src/unet.py::UNet"}
    elif architecture == "DeepLabV3Plus":
        model = DeepLabV3Plus(in_channels=2, num_classes=1, pretrained=False)
        model_config = {
            "in_channels": 2,
            "num_classes": 1,
            "pretrained": False,
            "architecture": "src/deeplabv3.py::DeepLabV3Plus",
        }
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # Strict loading is the compatibility check; do not silently ignore keys.
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint, model_config


def make_output_dir():
    parent = ROOT / "outputs/checkpoint_validation_comparison"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = parent / stamp
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def main():
    os.chdir(ROOT)
    if not VAL_SPLIT.is_file():
        raise FileNotFoundError(f"Validation split missing: {VAL_SPLIT}")
    coverage_path = latest_coverage_csv()
    category_by_image = read_coverage(coverage_path)

    dataset = Sen1FloodsDataset(str(VAL_SPLIT), augment=False)
    image_names = [str(dataset.df.iloc[i, 0]) for i in range(len(dataset))]
    if len(image_names) != len(set(image_names)):
        raise ValueError("Validation split contains duplicate image names")
    missing = sorted(set(image_names) - set(category_by_image))
    extra = sorted(set(category_by_image) - set(image_names))
    if missing or extra:
        raise ValueError(
            "Coverage CSV and validation split do not match exactly: "
            f"{len(missing)} missing categories, {len(extra)} extra validation rows."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    per_image = []
    category_rows = []
    checkpoint_info = []

    for name, architecture, checkpoint_path in CHECKPOINTS:
        record = {
            "name": name,
            "architecture": architecture,
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
            "compatible": False,
        }
        if not checkpoint_path.is_file():
            record["compatibility_issue"] = "Checkpoint file not found"
            compatibility.append(record)
            checkpoint_info.append(record)
            continue

        model = None
        try:
            model, checkpoint, model_config = load_model(architecture, checkpoint_path, device)
            record.update(
                {
                    "compatible": True,
                    "epoch": checkpoint.get("epoch"),
                    "checkpoint_config": json_safe(checkpoint.get("config")),
                    "model_config": model_config,
                    "checkpoint_metrics": json_safe(checkpoint.get("metrics")),
                }
            )
            for threshold in THRESHOLDS:
                global_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
                category_counts = {
                    category: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "images": 0, "valid_pixels": 0}
                    for category in CATEGORIES
                }
                for idx, (images, targets, ignore_mask) in enumerate(loader):
                    image_name = image_names[idx]
                    category = category_by_image[image_name]
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True).bool()
                    valid = ~ignore_mask.to(device, non_blocking=True).bool()
                    with torch.inference_mode():
                        logits = model(images)
                    if logits.ndim != 4 or logits.shape[:2] != (1, 1) or logits.shape[-2:] != targets.shape[-2:]:
                        raise ValueError(f"Unexpected output shape {tuple(logits.shape)} for {image_name}")

                    if category == "fully_ignored":
                        per_image.append(
                            {
                                "checkpoint": name,
                                "threshold": threshold,
                                "image": image_name,
                                "coverage_category": category,
                                "valid_pixels": 0,
                                "ignored_pixels": int(ignore_mask.sum().item()),
                                "tp": 0,
                                "tn": 0,
                                "fp": 0,
                                "fn": 0,
                                **{key: None for key in ("iou", "dice", "precision", "recall", "accuracy")},
                            }
                        )
                        continue

                    pred = torch.sigmoid(logits.float())[:, 0] >= threshold
                    truth = targets[:, 0] if targets.ndim == 4 else targets
                    valid_2d = valid[:, 0] if valid.ndim == 4 else valid
                    truth = truth & valid_2d
                    pred = pred & valid_2d
                    counts = {
                        "tp": int((pred & truth).sum().item()),
                        "tn": int((~pred & ~truth & valid_2d).sum().item()),
                        "fp": int((pred & ~truth & valid_2d).sum().item()),
                        "fn": int((~pred & truth).sum().item()),
                    }
                    valid_pixels = int(valid_2d.sum().item())
                    ignored_pixels = int((~valid_2d).sum().item())
                    for key in global_counts:
                        global_counts[key] += counts[key]
                    bucket = category_counts[category]
                    bucket["images"] += 1
                    bucket["valid_pixels"] += valid_pixels
                    for key in ("tp", "tn", "fp", "fn"):
                        bucket[key] += counts[key]
                    per_image.append(
                        {
                            "checkpoint": name,
                            "threshold": threshold,
                            "image": image_name,
                            "coverage_category": category,
                            "valid_pixels": valid_pixels,
                            "ignored_pixels": ignored_pixels,
                            **counts,
                            **metrics(**counts),
                        }
                    )

                summary = {
                    "checkpoint": name,
                    "architecture": architecture,
                    "threshold": threshold,
                    "validation_images_in_split": len(dataset),
                    "evaluated_images": len(dataset),
                    "scored_images_with_valid_labels": sum(v["images"] for v in category_counts.values()),
                    "valid_pixels": sum(global_counts[k] for k in ("tp", "tn", "fp", "fn")),
                    **metrics(**global_counts),
                    "tp": global_counts["tp"],
                    "tn": global_counts["tn"],
                    "fp": global_counts["fp"],
                    "fn": global_counts["fn"],
                    "flood_free_false_positive_pixels": category_counts["flood_free"]["fp"],
                }
                checkpoint_info.append({**record, "threshold": threshold, "global_summary": summary})
                for category in CATEGORIES:
                    bucket = category_counts[category]
                    category_summary = metrics(bucket["tp"], bucket["tn"], bucket["fp"], bucket["fn"])
                    category_rows.append(
                        {
                            "checkpoint": name,
                            "threshold": threshold,
                            "coverage_category": category,
                            "evaluated_images": bucket["images"],
                            "valid_pixels": bucket["valid_pixels"],
                            "tp": bucket["tp"],
                            "tn": bucket["tn"],
                            "fp": bucket["fp"],
                            "fn": bucket["fn"],
                            "iou": category_summary["iou"],
                            "dice": category_summary["dice"],
                            "false_positive_pixels": bucket["fp"] if category == "flood_free" else None,
                        }
                    )
        except Exception as exc:
            record["compatible"] = False
            record["compatibility_issue"] = f"{type(exc).__name__}: {exc}"
            checkpoint_info.append(record)
        finally:
            if model is not None:
                del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out_dir = make_output_dir()
    if checkpoint_info:
        with (out_dir / "checkpoint_summary.json").open("w", encoding="utf-8") as stream:
            json.dump(json_safe(checkpoint_info), stream, indent=2)
    with (out_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["checkpoint", "threshold", "image", "coverage_category", "valid_pixels", "ignored_pixels", "tp", "tn", "fp", "fn", "iou", "dice", "precision", "recall", "accuracy"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_image)
    with (out_dir / "category_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["checkpoint", "threshold", "coverage_category", "evaluated_images", "valid_pixels", "tp", "tn", "fp", "fn", "iou", "dice", "false_positive_pixels"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(category_rows)

    global_rows = []
    for entry in checkpoint_info:
        if "global_summary" in entry:
            global_rows.append(entry["global_summary"])
    with (out_dir / "global_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["checkpoint", "architecture", "threshold", "validation_images_in_split", "evaluated_images", "scored_images_with_valid_labels", "valid_pixels", "tp", "tn", "fp", "fn", "iou", "dice", "precision", "recall", "accuracy", "flood_free_false_positive_pixels"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(global_rows)

    metadata = {
        "validation_split": str(VAL_SPLIT.relative_to(ROOT)),
        "coverage_csv": str(coverage_path.relative_to(ROOT)),
        "thresholds": list(THRESHOLDS),
        "primary_threshold": 0.50,
        "secondary_sensitivity_threshold": 0.46,
        "device": str(device),
        "validation_images_inference_per_compatible_model": len(dataset),
        "validation_images_with_valid_labels_scored": sum(
            category_by_image[name] != "fully_ignored" for name in image_names
        ),
        "model_training_performed": False,
        "test_split_accessed": False,
        "ignore_handling": "Label -1 pixels excluded from TP/TN/FP/FN and all metrics.",
        "category_metric_definition": "Pixel-pooled (micro) IoU and Dice within each coverage category.",
        "checkpoint_compatibility": [
            {key: val for key, val in entry.items() if key != "global_summary"}
            for entry in checkpoint_info
        ],
    }
    (out_dir / "evaluation_info.json").write_text(json.dumps(json_safe(metadata), indent=2), encoding="utf-8")
    print(f"Device: {device}; validation images: {len(dataset)}")
    print(f"Outputs: {out_dir}")
    print("Global metrics:")
    for row in global_rows:
        print(
            f"  {row['checkpoint']} @ {row['threshold']:.2f}: "
            f"IoU={row['iou']:.6f}, Dice={row['dice']:.6f}, "
            f"P={row['precision']:.6f}, R={row['recall']:.6f}, "
            f"Acc={row['accuracy']:.6f}; flood-free FP={row['flood_free_false_positive_pixels']}"
        )
    for entry in checkpoint_info:
        if entry.get("compatibility_issue"):
            print(f"Compatibility issue: {entry['name']}: {entry['compatibility_issue']}")


if __name__ == "__main__":
    main()
