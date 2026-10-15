"""Validate and smoke-test deterministic manifest-backed 256px patch reading.

No derived TIFFs are written: ``Sen1FloodsPatchDataset`` reads four windows
from each original 512px parent raster, so every child inherits its parent's
split and no duplicate patch files can drift out of alignment.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diagnostic_utils import PROJECT_ROOT
from deeplabv3 import DeepLabV3Plus
from losses import TverskyDiceLoss
from sen1_patch_dataset import Sen1FloodsPatchDataset


def get_group_parent_counts(dataset):
    return len({row["parent_id"] for row in dataset.rows})


def smoke_test(manifest_path, gpu_smoke_test=False):
    train_low = Sen1FloodsPatchDataset(manifest_path, "train", group="LOW", augment=False)
    train_high = Sen1FloodsPatchDataset(manifest_path, "train", group="HIGH", augment=False)
    validation = Sen1FloodsPatchDataset(
        manifest_path, "validation", augment=False, return_metadata=True,
    )
    if not len(train_low) or not len(train_high):
        raise RuntimeError(
            f"Cannot smoke-test both specialists: LOW={len(train_low)}, HIGH={len(train_high)} patches."
        )
    if not len(validation):
        raise RuntimeError("Validation patch dataset is empty.")

    print(f"LOW train patches={len(train_low)}, parents={get_group_parent_counts(train_low)}")
    print(f"HIGH train patches={len(train_high)}, parents={get_group_parent_counts(train_high)}")
    print(f"Validation patches={len(validation)}, parents={get_group_parent_counts(validation)}")

    samples = {}
    for name, dataset in (("LOW", train_low), ("HIGH", train_high)):
        image, target, ignore = dataset[0]
        if image.shape != (2, 256, 256) or target.shape != (256, 256) or ignore.shape != (256, 256):
            raise AssertionError(f"Unexpected {name} sample shapes: {image.shape}, {target.shape}, {ignore.shape}")
        if image.dtype != torch.float32 or target.dtype != torch.float32 or ignore.dtype != torch.bool:
            raise AssertionError("Unexpected patch tensor dtypes.")
        if not torch.isfinite(image).all() or not torch.isin(target, torch.tensor([0.0, 1.0])).all():
            raise AssertionError("Non-finite image values or invalid target values.")
        samples[name] = (image.unsqueeze(0), target.unsqueeze(0), ignore.unsqueeze(0))

    first = validation[0]
    second = validation[0]
    if not all(torch.equal(a, b) for a, b in zip(first[:3], second[:3])):
        raise AssertionError("Validation patch read is not deterministic.")
    for batch in DataLoader(validation, batch_size=1, shuffle=False, num_workers=0):
        if batch[0].shape != (1, 2, 256, 256) or batch[1].shape != (1, 256, 256):
            raise AssertionError("Unexpected validation batch shapes.")
        break
    print("Data-loader shape/value/alignment smoke test: PASSED")
    print("Validation determinism and sequential sampling: PASSED")

    if gpu_smoke_test:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu-smoke-test requested but CUDA is unavailable.")
        device = torch.device("cuda")
        criterion = TverskyDiceLoss(pos_weight=3.0, dice_weight=0.5, alpha=0.7, beta=0.3)
        for name in ("LOW", "HIGH"):
            model = DeepLabV3Plus(in_channels=2, num_classes=1, pretrained=False).to(device)
            model.train()
            torch.cuda.reset_peak_memory_stats()
            image, target, ignore = (x.to(device) for x in samples[name])
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(image)
            loss = criterion(logits.float(), target, ignore)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite {name} smoke loss.")
            loss.backward()
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"{name} 256px DeepLab forward/backward: PASSED; peak allocated={peak_mb:.1f} MB")
            del model, logits, loss, image, target, ignore
            torch.cuda.empty_cache()
        print("GPU memory feasibility smoke test: PASSED (one synthetic optimizer-free batch per specialist).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to patch_manifest.csv")
    parser.add_argument("--smoke-test", action="store_true", help="Read patches and validate shapes and values.")
    parser.add_argument("--gpu-smoke-test", action="store_true", help="Also run one 256px forward/backward per specialist.")
    args = parser.parse_args()
    path = Path(args.manifest)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    if not args.smoke_test and not args.gpu_smoke_test:
        parser.error("Specify --smoke-test and/or --gpu-smoke-test.")
    smoke_test(path, gpu_smoke_test=args.gpu_smoke_test)


if __name__ == "__main__":
    main()
