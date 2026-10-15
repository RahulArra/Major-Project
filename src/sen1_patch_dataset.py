"""Deterministic window reader for the 256x256 coverage-patch manifest."""

import csv
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from diagnostic_utils import PROJECT_ROOT
from sen1_dataset import Sen1FloodsDataset
from coverage_patch_utils import PATCH_SIZE, raster_pixel_dimensions_match, transform_aligned, validate_patch_grid


class Sen1FloodsPatchDataset(Dataset):
    """Read source rasters by manifest window; no patch TIFF duplication."""

    def __init__(self, manifest_path, split, group=None, augment=False,
                 return_metadata=False):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_absolute():
            self.manifest_path = PROJECT_ROOT / self.manifest_path
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split {split!r}.")
        if group is not None and split != "train":
            raise ValueError("Specialist filtering is allowed only on train patches.")
        self.split = split
        self.group = group.upper() if group else None
        self.augment = bool(augment)
        self.return_metadata = bool(return_metadata)
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        required = {
            "patch_id", "parent_id", "image_name", "label_name", "split",
            "row_off", "col_off", "height", "width", "group", "training_group",
        }
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"Manifest missing required columns: {sorted(required)}")
        rows = [row for row in rows if row["split"] == split]
        if self.group:
            if self.group not in {"LOW", "HIGH"}:
                raise ValueError("group must be LOW or HIGH")
            rows = [row for row in rows if row["training_group"] == self.group]
        self.rows = rows
        if len({row["patch_id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate patch IDs in {split} dataset.")
        self.image_dir = PROJECT_ROOT / "data/raw/sen1floods11/S1Hand"
        self.label_dir = PROJECT_ROOT / "data/raw/sen1floods11/LabelHand"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        row_off = int(row["row_off"])
        col_off = int(row["col_off"])
        height = int(row["height"])
        width = int(row["width"])
        validate_patch_grid(width, height, row_off, col_off, PATCH_SIZE)
        window = Window(col_off, row_off, width, height)
        image_path = self.image_dir / row["image_name"]
        label_path = self.label_dir / row["label_name"]
        with rasterio.open(image_path) as image_src, rasterio.open(label_path) as label_src:
            if image_src.count != 2:
                raise ValueError(f"Expected two SAR bands in {image_path}")
            if not raster_pixel_dimensions_match(image_src, label_src):
                raise ValueError(f"Image/label raster pixel dimensions differ: {image_path}, {label_path}")
            image = image_src.read(window=window).astype(np.float32, copy=False)
            label = label_src.read(1, window=window)
        if image.shape != (2, PATCH_SIZE, PATCH_SIZE) or label.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Unexpected patch shapes: {image.shape} and {label.shape}")
        if not np.isin(label, (-1, 0, 1)).all():
            raise ValueError(f"Unexpected label value in {label_path} patch {row['patch_id']}")

        # Match the existing dataset's non-finite replacement and z-score stats.
        for band in range(2):
            invalid = ~np.isfinite(image[band])
            image[band][invalid] = Sen1FloodsDataset.MEAN[band]
        image = (image - Sen1FloodsDataset.MEAN[:, None, None]) / Sen1FloodsDataset.STD[:, None, None]
        ignore_mask = label == -1
        target = (label == 1).astype(np.float32)
        target[ignore_mask] = 0.0

        if self.augment:
            rotate = random.random() < 0.5
            image, target, ignore_mask = transform_aligned(
                image, target, ignore_mask,
                horizontal=(random.random() < 0.5),
                vertical=(random.random() < 0.5),
                rotations=(random.randint(1, 3) if rotate else 0),
            )

        result = (
            torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(ignore_mask, dtype=bool)),
        )
        if self.return_metadata:
            return (*result, row["patch_id"], row["parent_id"], row_off, col_off)
        return result
