"""Create validation-only DeepLabV3+ prediction panels."""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from diagnostic_utils import PROJECT_ROOT, get_device, load_deeplab, new_output_dir
from sen1_dataset import Sen1FloodsDataset


VAL_SPLIT = PROJECT_ROOT / "data/raw/sen1floods11/splits/flood_valid_data.csv"
THRESHOLD_ROOT = PROJECT_ROOT / "outputs/deeplab_threshold_search"
DISPLAY_MEAN = np.array([-10.3929, -17.2411], dtype=np.float32)
DISPLAY_STD = np.array([4.0388, 4.7540], dtype=np.float32)


def get_selected_threshold(path_arg):
    path = path_arg
    if path is None:
        candidates = list(THRESHOLD_ROOT.glob("*/selected_threshold.json")) if THRESHOLD_ROOT.exists() else []
        path = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
    if path is None or not path.is_file():
        print("WARNING: threshold-search JSON is absent; using 0.50 for the selected-threshold panel.")
        return 0.50
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        value = data.get("selected_threshold")
        if value is None or not 0.0 <= float(value) <= 1.0:
            raise ValueError("selected_threshold is missing or outside [0, 1]")
        print(f"Using validation-selected threshold {float(value):.2f} from {path}")
        return float(value)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"WARNING: cannot read a selected threshold from {path} ({exc}); using 0.50.")
        return 0.50


def sar_rgb(image_chw):
    """Undo training standardization for display only and robustly stretch channels."""
    raw = image_chw * DISPLAY_STD[:, None, None] + DISPLAY_MEAN[:, None, None]
    channels = [raw[0], raw[1], (raw[0] + raw[1]) / 2.0]
    scaled = []
    for channel in channels:
        lo, hi = np.percentile(channel, (2, 98))
        scaled.append(np.clip((channel - lo) / max(float(hi - lo), 1e-6), 0, 1))
    return np.stack(scaled, axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold-json",
        type=lambda s: (PROJECT_ROOT / s) if not os.path.isabs(s) else Path(s),
    )
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    threshold = get_selected_threshold(args.threshold_json)
    device = get_device()
    model, _ = load_deeplab(device)
    dataset = Sen1FloodsDataset(str(VAL_SPLIT), augment=False)
    if not len(dataset):
        raise ValueError("Validation split contains no samples")
    count = min(max(args.samples, 1), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    out_dir = new_output_dir("deeplab_validation_visualizations")
    gt_cmap = ListedColormap(["#202020", "#00d4ff", "#ff00a8"])
    pred_cmap = ListedColormap(["#202020", "#ffe600"])

    model.eval()
    for index in indices:
        image, target, ignore = dataset[int(index)]
        with torch.inference_mode():
            logits = model(image.unsqueeze(0).to(device))
            probability = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        truth = target.numpy().astype(np.uint8)
        ignored = ignore.numpy().astype(bool)
        gt_display = truth.copy()
        gt_display[ignored] = 2  # Magenta is ignore; black is valid background.
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        axes = axes.ravel()
        axes[0].imshow(sar_rgb(image.numpy()))
        axes[0].set_title("SAR VV/VH false color (display stretch)")
        axes[1].imshow(gt_display, cmap=gt_cmap, vmin=0, vmax=2, interpolation="nearest")
        axes[1].set_title("Ground truth (black=background, cyan=flood, magenta=ignore)")
        im = axes[2].imshow(probability, cmap="magma", vmin=0, vmax=1)
        axes[2].set_title("Predicted flood probability")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        axes[3].imshow((probability >= 0.50).astype(np.uint8), cmap=pred_cmap, vmin=0, vmax=1, interpolation="nearest")
        axes[3].set_title("Binary prediction (threshold 0.50)")
        axes[4].imshow((probability >= threshold).astype(np.uint8), cmap=pred_cmap, vmin=0, vmax=1, interpolation="nearest")
        axes[4].set_title(f"Binary prediction (validation threshold {threshold:.2f})")
        axes[5].axis("off")
        for ax in axes[:5]:
            ax.set_axis_off()
        image_name = str(dataset.df.iloc[int(index), 0])
        fig.suptitle(
            f"Sen1Floods11 validation: {image_name} — ignore pixels marked in ground truth",
            fontsize=14,
        )
        fig.savefig(out_dir / f"validation_{int(index):03d}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {len(indices)} validation figures to {out_dir}")


if __name__ == "__main__":
    main()
