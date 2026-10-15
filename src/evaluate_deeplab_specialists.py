"""Validation-only comparison of baseline, specialists, and binary-OR fusion."""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from torch.utils.data import DataLoader

from diagnostic_utils import PROJECT_ROOT, get_device, load_deeplab, new_output_dir
from deeplabv3 import DeepLabV3Plus
from sen1_dataset import Sen1FloodsDataset
from sen1_patch_dataset import Sen1FloodsPatchDataset
from coverage_patch_utils import (
    IMAGE_SIZE,
    PATCH_SIZE,
    binary_or,
    confusion_counts,
    metrics_from_counts,
    reassemble_patches,
    scene_coverage_category,
)


VAL_SPLIT = PROJECT_ROOT / "data/raw/sen1floods11/splits/flood_valid_data.csv"
THRESHOLD = 0.50
BATCH_SIZE = 1
METHODS = ("baseline_deeplab", "low_specialist", "high_specialist", "binary_or")
CATEGORIES = ("flood_free", "low", "moderate", "high", "fully_ignored")
SAR_MEAN = np.array([-10.3929, -17.2411], dtype=np.float32)
SAR_STD = np.array([4.0388, 4.7540], dtype=np.float32)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_specialist(checkpoint_path, device, expected_group):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unexpected specialist checkpoint format: {checkpoint_path}")
    config = checkpoint.get("config", {})
    if config.get("specialist") != expected_group:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is tagged {config.get('specialist')!r}, "
            f"expected {expected_group!r}."
        )
    model = DeepLabV3Plus(in_channels=2, num_classes=1, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def collect_predictions(model, dataset, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    predictions = {}
    targets = {}
    ignored = {}
    sar_patches = {}
    patch_meta = {}
    for batch in loader:
        images, target, ignore = batch[:3]
        patch_ids = batch[3]
        parent_ids = batch[4]
        row_offsets = batch[5]
        col_offsets = batch[6]
        logits = model(images.to(device))
        probs = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
        for index, patch_id in enumerate(patch_ids):
            patch_id = str(patch_id)
            predictions[patch_id] = probs[index] >= THRESHOLD
            targets[patch_id] = target[index].numpy() >= 0.5
            ignored[patch_id] = ignore[index].numpy().astype(bool)
            sar_patches[patch_id] = images[index].numpy()
            patch_meta[patch_id] = (
                str(parent_ids[index]), int(row_offsets[index]), int(col_offsets[index])
            )
    return predictions, targets, ignored, sar_patches, patch_meta


def assemble_parent_arrays(dataset, prediction_map, targets, ignored, sar_patches, patch_meta):
    patches_by_parent = defaultdict(list)
    for row in dataset.rows:
        patches_by_parent[row["parent_id"]].append(row)
    assembled = {}
    for parent_id, rows in patches_by_parent.items():
        pred_items = []
        target_items = []
        ignore_items = []
        sar_items = []
        for row in rows:
            patch_id = row["patch_id"]
            _, row_off, col_off = patch_meta[patch_id]
            pred_items.append((row_off, col_off, prediction_map[patch_id]))
            target_items.append((row_off, col_off, targets[patch_id]))
            ignore_items.append((row_off, col_off, ignored[patch_id]))
            sar_items.append((row_off, col_off, sar_patches[patch_id]))
        # Helper checks each pixel is placed exactly once.
        pred = reassemble_patches(pred_items)
        target = reassemble_patches(target_items)
        ignore = reassemble_patches(ignore_items).astype(bool)
        sar = reassemble_patches(sar_items)
        counts = confusion_counts(np.zeros_like(target, dtype=bool), target, ignore)
        category = scene_coverage_category(counts["valid_pixels"], int(np.count_nonzero(target & ~ignore)))
        assembled[parent_id] = {
            "prediction": pred.astype(bool), "target": target.astype(bool),
            "ignore": ignore, "sar": sar.astype(np.float32), "category": category,
            "image_name": rows[0]["image_name"],
        }
    return assembled


def all_methods_parent_arrays(dataset, baseline_outputs, low_outputs, high_outputs):
    base_pred, target_map, ignore_map, sar_map, meta = baseline_outputs
    baseline = assemble_parent_arrays(dataset, base_pred, target_map, ignore_map, sar_map, meta)
    low = assemble_parent_arrays(dataset, low_outputs[0], target_map, ignore_map, sar_map, meta)
    high = assemble_parent_arrays(dataset, high_outputs[0], target_map, ignore_map, sar_map, meta)
    result = {}
    for parent_id in baseline:
        if not (np.array_equal(baseline[parent_id]["target"], low[parent_id]["target"])
                and np.array_equal(baseline[parent_id]["ignore"], low[parent_id]["ignore"])):
            raise AssertionError("Validation targets differ across evaluation methods.")
        ref = baseline[parent_id]
        result[parent_id] = {
            "baseline_deeplab": ref,
            "low_specialist": {**ref, "prediction": low[parent_id]["prediction"]},
            "high_specialist": {**ref, "prediction": high[parent_id]["prediction"]},
            "binary_or": {**ref, "prediction": binary_or(low[parent_id]["prediction"], high[parent_id]["prediction"])},
        }
    return result


def compute_metrics_by_method(parent_predictions):
    global_rows = []
    category_rows = []
    per_image_rows = []
    for method in METHODS:
        overall = {key: 0 for key in ("tp", "tn", "fp", "fn", "valid_pixels")}
        grouped = {category: {key: 0 for key in overall} for category in CATEGORIES}
        for parent_id, methods in parent_predictions.items():
            sample = methods[method]
            counts = confusion_counts(sample["prediction"], sample["target"], sample["ignore"])
            overall = {key: overall[key] + counts[key] for key in overall}
            category = sample["category"]
            grouped[category] = {key: grouped[category][key] + counts[key] for key in grouped[category]}
            metrics = metrics_from_counts(counts)
            per_image_rows.append({
                "method": method, "parent_id": parent_id,
                "image": sample["image_name"], "coverage_category": category,
                **counts, **metrics,
            })
        global_rows.append({"method": method, **overall, **metrics_from_counts(overall),
                            "flood_free_false_positive_pixels": grouped["flood_free"]["fp"]})
        for category in CATEGORIES:
            counts = grouped[category]
            category_rows.append({"method": method, "coverage_category": category,
                                  "parent_images": sum(
                                      1 for methods in parent_predictions.values()
                                      if methods[method]["category"] == category
                                  ), **counts, **metrics_from_counts(counts),
                                  "false_positive_pixels": counts["fp"]})
    return global_rows, category_rows, per_image_rows


def sar_display(image_chw):
    raw = image_chw * SAR_STD[:, None, None] + SAR_MEAN[:, None, None]
    channels = [raw[0], raw[1], (raw[0] + raw[1]) / 2.0]
    scaled = []
    for channel in channels:
        low, high = np.percentile(channel, (2, 98))
        scaled.append(np.clip((channel - low) / max(float(high - low), 1e-6), 0, 1))
    return np.stack(scaled, axis=-1)


def save_visualizations(parent_predictions, output_dir):
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir()
    for parent_id, methods in parent_predictions.items():
        ref = methods["baseline_deeplab"]
        gt = ref["target"].astype(np.uint8)
        gt[ref["ignore"]] = 2
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        axes = axes.ravel()
        axes[0].imshow(sar_display(ref["sar"]))
        axes[0].set_title("VV/VH SAR (display stretch)")
        axes[1].imshow(gt, cmap=ListedColormap(["#222222", "#00d4ff", "#ff00a8"]),
                       vmin=0, vmax=2, interpolation="nearest")
        axes[1].set_title("Ground truth (gray=background, cyan=flood, magenta=ignore)")
        for axis, method in zip(axes[2:], METHODS):
            axis.imshow(methods[method]["prediction"], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axis.set_title(method.replace("_", " "))
        for axis in axes:
            axis.set_axis_off()
        fig.suptitle(f"{parent_id} — {ref['category']} coverage; threshold={THRESHOLD:.2f}")
        safe_name = parent_id.replace(":", "_").replace("/", "_").replace("\\", "_")
        fig.savefig(vis_dir / f"{safe_name}.png", dpi=120)
        plt.close(fig)
    return vis_dir


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--low-checkpoint", required=True)
    parser.add_argument("--high-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", default=None,
                        help="Defaults to checkpoints_deeplab/best_deeplab.pth")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    manifest = resolve_path(args.manifest)
    low_path = resolve_path(args.low_checkpoint)
    high_path = resolve_path(args.high_checkpoint)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    dataset = Sen1FloodsPatchDataset(manifest, "validation", augment=False, return_metadata=True)
    validation_parent_dataset = Sen1FloodsDataset(str(VAL_SPLIT), augment=False)
    val_names = [str(name) for name in validation_parent_dataset.df.iloc[:, 0].tolist()]
    manifest_names = {row["image_name"] for row in dataset.rows}
    if manifest_names != set(val_names):
        raise ValueError("Validation manifest does not exactly match the existing validation parent split.")
    if len(dataset) != len(val_names) * 4:
        raise ValueError(f"Expected four patches for each validation parent; got {len(dataset)} patches for {len(val_names)} parents.")

    device = get_device()
    print(f"Validation parents: {len(val_names)}; patches: {len(dataset)}; device: {device}; threshold: {THRESHOLD:.2f}")
    baseline_outputs = load_deeplab(device)
    baseline_model = baseline_outputs[0]
    baseline_checkpoint = baseline_outputs[1]
    baseline_predictions = collect_predictions(baseline_model, dataset, device)
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    low_model, low_checkpoint = load_specialist(low_path, device, "LOW")
    low_predictions = collect_predictions(low_model, dataset, device)
    del low_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    high_model, high_checkpoint = load_specialist(high_path, device, "HIGH")
    high_predictions = collect_predictions(high_model, dataset, device)
    del high_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    parent_predictions = all_methods_parent_arrays(
        dataset, baseline_predictions, low_predictions, high_predictions,
    )
    global_rows, category_rows, per_image_rows = compute_metrics_by_method(parent_predictions)
    output_dir = new_output_dir("deeplab_specialist_evaluation")
    write_csv(output_dir / "global_metrics.csv", global_rows)
    write_csv(output_dir / "category_metrics.csv", category_rows)
    write_csv(output_dir / "per_image_metrics.csv", per_image_rows)
    vis_dir = save_visualizations(parent_predictions, output_dir)

    baseline_value = baseline_checkpoint.get("best_val_iou") if isinstance(baseline_checkpoint, dict) else None
    info = {
        "validation_split": str(VAL_SPLIT.relative_to(PROJECT_ROOT)),
        "patch_manifest": str(manifest.relative_to(PROJECT_ROOT)) if manifest.is_relative_to(PROJECT_ROOT) else str(manifest),
        "threshold": THRESHOLD,
        "validation_parent_count": len(val_names),
        "validation_patch_count": len(dataset),
        "methods": {
            "baseline_deeplab": {"checkpoint": "checkpoints_deeplab/best_deeplab.pth",
                                 "checkpoint_epoch": baseline_checkpoint.get("epoch") if isinstance(baseline_checkpoint, dict) else None,
                                 "historical_best_val_iou": baseline_value,
                                 "inference_context": "256x256 windows stitched to parent scenes"},
            "low_specialist": {"checkpoint": str(low_path.relative_to(PROJECT_ROOT)) if low_path.is_relative_to(PROJECT_ROOT) else str(low_path),
                                "checkpoint_epoch": low_checkpoint.get("epoch"), "best_val_iou": low_checkpoint.get("best_val_iou")},
            "high_specialist": {"checkpoint": str(high_path.relative_to(PROJECT_ROOT)) if high_path.is_relative_to(PROJECT_ROOT) else str(high_path),
                                 "checkpoint_epoch": high_checkpoint.get("epoch"), "best_val_iou": high_checkpoint.get("best_val_iou")},
            "binary_or": {"rule": "(low_probability >= 0.50) OR (high_probability >= 0.50) per pixel"},
        },
        "ignore_handling": "Label -1 excluded from every confusion count and metric.",
        "validation_coverage_categories": "Computed from each validation parent label for post-hoc metrics only; never used to route predictions or create training groups.",
        "test_data_used": False,
        "training_performed": False,
        "baseline_context_note": "Baseline is evaluated on the same 256x256 windows as the specialists; this differs from its historical full-512px validation context.",
        "visualization_directory": str(vis_dir.relative_to(PROJECT_ROOT)),
    }
    (output_dir / "evaluation_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Output directory: {output_dir}")
    print("Global pooled validation metrics:")
    for row in global_rows:
        print(f"  {row['method']}: IoU={row['iou']:.6f}, Dice={row['dice']:.6f}, "
              f"P={row['precision']:.6f}, R={row['recall']:.6f}, Acc={row['accuracy']:.6f}, "
              f"flood-free FP={row['flood_free_false_positive_pixels']:,}")
    print(f"Category metrics: {output_dir / 'category_metrics.csv'}")
    print(f"Per-image metrics: {output_dir / 'per_image_metrics.csv'}")
    print(f"Visualizations: {vis_dir}")


if __name__ == "__main__":
    main()
