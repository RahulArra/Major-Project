"""Shared pure helpers for deterministic Sen1Floods11 patch experiments."""

from collections import Counter

import numpy as np


PATCH_SIZE = 256
IMAGE_SIZE = 512
LOW_HIGH_THRESHOLD = 0.30


def patch_windows(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    """Return (row_offset, col_offset, height, width) for a complete grid."""
    if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
        raise ValueError("Image dimensions must be positive and divisible by patch size.")
    return [
        (row * patch_size, col * patch_size, patch_size, patch_size)
        for row in range(image_size // patch_size)
        for col in range(image_size // patch_size)
    ]


def label_counts(label):
    """Count Sen1Floods11 valid/ignored/flood pixels and coverage fraction."""
    label = np.asarray(label)
    allowed = np.isin(label, (-1, 0, 1))
    if not bool(allowed.all()):
        values = np.unique(label[~allowed]).tolist()
        raise ValueError(f"Unexpected label values: {values}")
    ignored = int(np.count_nonzero(label == -1))
    flood = int(np.count_nonzero(label == 1))
    non_flood = int(np.count_nonzero(label == 0))
    valid = flood + non_flood
    coverage = flood / valid if valid else None
    return {
        "total_pixels": int(label.size),
        "valid_pixels": valid,
        "flood_pixels": flood,
        "non_flood_pixels": non_flood,
        "ignored_pixels": ignored,
        "coverage": coverage,
    }


def coverage_group(valid_pixels, flood_pixels, threshold=LOW_HIGH_THRESHOLD):
    """Return IGNORE, LOW (< threshold), or HIGH (>= threshold)."""
    if valid_pixels < 0 or flood_pixels < 0 or flood_pixels > valid_pixels:
        raise ValueError("Invalid valid/flood pixel counts.")
    if valid_pixels == 0:
        return "IGNORE"
    return "HIGH" if (flood_pixels / valid_pixels) >= threshold else "LOW"


def scene_coverage_category(valid_pixels, flood_pixels):
    """Match existing Sen1 distribution bins for parent-scene reporting."""
    if valid_pixels <= 0:
        return "fully_ignored"
    percentage = 100.0 * flood_pixels / valid_pixels
    if flood_pixels == 0:
        return "flood_free"
    if percentage <= 1.0:
        return "low"
    if percentage <= 10.0:
        return "moderate"
    return "high"


def validate_parent_splits(parent_to_split):
    """Reject a parent ID assigned to more than one original split."""
    seen = {}
    for parent_id, split in parent_to_split:
        previous = seen.setdefault(parent_id, split)
        if previous != split:
            raise ValueError(
                f"Parent {parent_id!r} appears in both {previous!r} and {split!r}."
            )
    return seen


def raster_pixel_dimensions_match(image_src, label_src):
    """Check index-grid dimensions; this pipeline pairs rasters by pixel indices."""
    return image_src.width == label_src.width and image_src.height == label_src.height


def georeferencing_matches(image_src, label_src):
    """Compare CRS and affine transforms while tolerating serialization noise.

    A 1e-12 absolute tolerance is far below a pixel for the Sen1Floods11
    rasters, whose angular pixel size is about 9e-5 degrees.
    """
    return (
        image_src.crs == label_src.crs
        and np.allclose(
            tuple(image_src.transform), tuple(label_src.transform),
            rtol=1e-12, atol=1e-12,
        )
    )


def validate_patch_grid(width, height, row_off, col_off, patch_size=PATCH_SIZE):
    if width != patch_size or height != patch_size:
        raise ValueError(f"Patch must be {patch_size}x{patch_size}, got {width}x{height}.")
    if row_off < 0 or col_off < 0 or row_off + height > IMAGE_SIZE or col_off + width > IMAGE_SIZE:
        raise ValueError("Patch window crosses its parent image boundary.")


def transform_aligned(image, target, ignore_mask, horizontal=False, vertical=False, rotations=0):
    """Apply a fixed spatial transform identically to image, target, and ignore mask."""
    image = np.asarray(image)
    target = np.asarray(target)
    ignore_mask = np.asarray(ignore_mask)
    if image.ndim != 3 or target.ndim != 2 or ignore_mask.ndim != 2:
        raise ValueError("Expected image [C,H,W], target [H,W], and ignore mask [H,W].")
    if image.shape[-2:] != target.shape or target.shape != ignore_mask.shape:
        raise ValueError("Image, target, and ignore mask spatial shapes differ.")
    rotations = int(rotations) % 4
    if horizontal:
        image = np.flip(image, axis=2)
        target = np.flip(target, axis=1)
        ignore_mask = np.flip(ignore_mask, axis=1)
    if vertical:
        image = np.flip(image, axis=1)
        target = np.flip(target, axis=0)
        ignore_mask = np.flip(ignore_mask, axis=0)
    if rotations:
        image = np.rot90(image, k=rotations, axes=(1, 2))
        target = np.rot90(target, k=rotations, axes=(0, 1))
        ignore_mask = np.rot90(ignore_mask, k=rotations, axes=(0, 1))
    return image.copy(), target.copy(), ignore_mask.copy()


def reassemble_patches(patches, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    """Place (row, col, HxW-array) patches into one parent image exactly once."""
    canvas = None
    occupied = np.zeros((image_size, image_size), dtype=bool)
    for row_off, col_off, patch in patches:
        patch = np.asarray(patch)
        height, width = patch.shape[-2:]
        validate_patch_grid(width, height, row_off, col_off, patch_size)
        if canvas is None:
            canvas = np.zeros((*patch.shape[:-2], image_size, image_size), dtype=patch.dtype)
        if canvas.shape[:-2] != patch.shape[:-2]:
            raise ValueError("Patch leading dimensions are inconsistent.")
        region = occupied[row_off:row_off + height, col_off:col_off + width]
        if region.any():
            raise ValueError("Patch windows overlap during reassembly.")
        canvas[..., row_off:row_off + height, col_off:col_off + width] = patch
        region[:] = True
    if canvas is None or not occupied.all():
        raise ValueError("Reassembly did not cover the complete parent image.")
    return canvas


def confusion_counts(prediction, target, ignore_mask):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    ignore_mask = np.asarray(ignore_mask, dtype=bool)
    if prediction.shape != target.shape or target.shape != ignore_mask.shape:
        raise ValueError("Prediction, target, and ignore mask shapes must match.")
    valid = ~ignore_mask
    return {
        "tp": int(np.count_nonzero(prediction & target & valid)),
        "tn": int(np.count_nonzero(~prediction & ~target & valid)),
        "fp": int(np.count_nonzero(prediction & ~target & valid)),
        "fn": int(np.count_nonzero(~prediction & target & valid)),
        "valid_pixels": int(np.count_nonzero(valid)),
    }


def binary_or(prediction_a, prediction_b):
    a = np.asarray(prediction_a, dtype=bool)
    b = np.asarray(prediction_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Binary predictions must have matching shapes.")
    return np.logical_or(a, b)


def metrics_from_counts(counts):
    tp, tn, fp, fn = (counts[k] for k in ("tp", "tn", "fp", "fn"))

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    dice = ratio(2 * tp, 2 * tp + fp + fn)
    return {
        "iou": ratio(tp, tp + fp + fn),
        "dice": dice,
        "f1": dice,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "accuracy": ratio(tp + tn, tp + tn + fp + fn),
    }


def category_counts(rows, key="group"):
    return dict(Counter(row[key] for row in rows))
