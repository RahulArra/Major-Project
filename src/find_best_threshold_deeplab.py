"""Select a DeepLabV3+ operating threshold on validation data only."""

import csv
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diagnostic_utils import CHECKPOINT_PATH, PROJECT_ROOT, get_device, load_deeplab, new_output_dir
from sen1_dataset import Sen1FloodsDataset


VAL_SPLIT = PROJECT_ROOT / "data/raw/sen1floods11/splits/flood_valid_data.csv"
BATCH_SIZE = 1
THRESHOLDS = [round(i / 100, 2) for i in range(10, 91)]


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main():
    os.chdir(PROJECT_ROOT)
    if not VAL_SPLIT.is_file():
        raise FileNotFoundError(f"Validation split not found: {VAL_SPLIT}")
    device = get_device()
    model, checkpoint = load_deeplab(device)
    dataset = Sen1FloodsDataset(str(VAL_SPLIT), augment=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    probabilities_all = []
    targets_all = []
    model.eval()
    with torch.inference_mode():
        for images, targets, ignore_mask in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits.float())[:, 0]
            valid = ~ignore_mask.bool().to(device)
            probabilities_all.append(probs[valid].cpu())
            targets_all.append(targets.to(device)[valid].cpu().bool())

    probabilities = torch.cat(probabilities_all) if probabilities_all else torch.empty(0)
    targets = torch.cat(targets_all) if targets_all else torch.empty(0, dtype=torch.bool)
    rows = []
    best_threshold = None
    best_iou = -1.0
    for threshold in THRESHOLDS:
        predicted = probabilities >= threshold
        tp = int((predicted & targets).sum().item())
        fp = int((predicted & ~targets).sum().item())
        fn = int((~predicted & targets).sum().item())
        iou = safe_ratio(tp, tp + fp + fn)
        dice = safe_ratio(2 * tp, 2 * tp + fp + fn)
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        rows.append({
            "threshold": f"{threshold:.2f}", "iou": iou, "dice": dice,
            "precision": precision, "recall": recall,
            "false_positive_count": fp, "false_negative_count": fn,
        })
        if probabilities.numel() and iou > best_iou:
            best_iou, best_threshold = iou, threshold

    output_dir = new_output_dir("deeplab_threshold_search")
    table_path = output_dir / "threshold_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "selected_threshold": best_threshold,
        "selected_validation_iou": best_iou if best_threshold is not None else None,
        "validation_split": str(VAL_SPLIT.relative_to(PROJECT_ROOT)),
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_best_val_iou": checkpoint.get("best_val_iou") if isinstance(checkpoint, dict) else None,
        "valid_pixel_count": int(probabilities.numel()),
        "selection_note": "Validation-selected operating point only. Never select or tune this threshold using test data.",
    }
    json_path = output_dir / "selected_threshold.json"
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Device: {device}; validation images: {len(dataset)}; valid pixels: {probabilities.numel()}")
    print(f"Selected threshold: {best_threshold} (validation IoU: {metadata['selected_validation_iou']})")
    print(f"Threshold table: {table_path}")
    print(f"Selection metadata: {json_path}")
    if best_threshold is None:
        print("Warning: no valid pixels were found; no threshold could be selected.")


if __name__ == "__main__":
    main()
