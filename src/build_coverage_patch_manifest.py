"""Build split-preserving 256x256 Sen1Floods11 coverage manifests.

All label reads used to define LOW/HIGH training groups come only from the
existing training split. Validation/test groups are descriptive metadata only;
their ``training_group`` field is deliberately blank.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from diagnostic_utils import PROJECT_ROOT, new_output_dir
from coverage_patch_utils import (
    IMAGE_SIZE,
    PATCH_SIZE,
    coverage_group,
    label_counts,
    patch_windows,
    georeferencing_matches,
    raster_pixel_dimensions_match,
    validate_parent_splits,
    validate_patch_grid,
)


SPLITS = {
    "train": PROJECT_ROOT / "data/raw/sen1floods11/splits/flood_train_data.csv",
    "validation": PROJECT_ROOT / "data/raw/sen1floods11/splits/flood_valid_data.csv",
    "test": PROJECT_ROOT / "data/raw/sen1floods11/splits/flood_test_data.csv",
}
IMAGE_DIR = PROJECT_ROOT / "data/raw/sen1floods11/S1Hand"
LABEL_DIR = PROJECT_ROOT / "data/raw/sen1floods11/LabelHand"
MANIFEST_FIELDS = [
    "patch_id", "parent_id", "image_name", "label_name", "split",
    "row_off", "col_off", "height", "width", "total_pixels",
    "valid_pixels", "flood_pixels", "non_flood_pixels", "ignored_pixels",
    "coverage", "group", "training_group", "parent_crs",
    "parent_left", "parent_bottom", "parent_right", "parent_top",
    "image_crs", "label_crs", "georeferencing_matches",
    "image_transform", "label_transform",
]


def read_split_rows():
    rows = []
    parent_splits = []
    for split, path in SPLITS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Split file missing: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for line_number, row in enumerate(reader, 1):
                if len(row) < 2 or not row[0] or not row[1]:
                    raise ValueError(f"Malformed row {line_number} in {path}: {row}")
                image_name, label_name = row[0], row[1]
                parent_id = Path(image_name).name
                rows.append((split, parent_id, image_name, label_name))
                parent_splits.append((parent_id, split))
    validate_parent_splits(parent_splits)
    ids = [(split, image) for split, _, image, _ in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("A split contains duplicate image/label rows.")
    return rows


def inspect_parent(split, parent_id, image_name, label_name):
    image_path = IMAGE_DIR / image_name
    label_path = LABEL_DIR / label_name
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing paired raster: {image_path} / {label_path}")
    with rasterio.open(image_path) as image_src, rasterio.open(label_path) as label_src:
        if image_src.count != 2:
            raise ValueError(f"Expected two SAR bands, got {image_src.count}: {image_path}")
        if (image_src.width, image_src.height) != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}: {image_path}")
        if (label_src.width, label_src.height) != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}: {label_path}")
        if not raster_pixel_dimensions_match(image_src, label_src):
            raise ValueError(f"Image and label pixel dimensions differ: {image_path} / {label_path}")
        georef_match = georeferencing_matches(image_src, label_src)
        label = label_src.read(1)
        if label.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Unexpected label shape {label.shape}: {label_path}")
        bounds = image_src.bounds
        crs = image_src.crs.to_string() if image_src.crs else ""
        label_crs = label_src.crs.to_string() if label_src.crs else ""
        image_transform = tuple(image_src.transform)
        label_transform = tuple(label_src.transform)

    records = []
    for patch_index, (row_off, col_off, height, width) in enumerate(patch_windows()):
        validate_patch_grid(width, height, row_off, col_off)
        window = Window(col_off, row_off, width, height)
        patch_label = label[row_off:row_off + height, col_off:col_off + width]
        if patch_label.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Bad patch dimensions for {parent_id} at {row_off},{col_off}")
        counts = label_counts(patch_label)
        group = coverage_group(counts["valid_pixels"], counts["flood_pixels"])
        patch_id = f"{split}:{parent_id}:p{patch_index:02d}"
        records.append({
            "patch_id": patch_id,
            "parent_id": parent_id,
            "image_name": image_name,
            "label_name": label_name,
            "split": split,
            "row_off": row_off,
            "col_off": col_off,
            "height": height,
            "width": width,
            "total_pixels": counts["total_pixels"],
            "valid_pixels": counts["valid_pixels"],
            "flood_pixels": counts["flood_pixels"],
            "non_flood_pixels": counts["non_flood_pixels"],
            "ignored_pixels": counts["ignored_pixels"],
            "coverage": "" if counts["coverage"] is None else counts["coverage"],
            "group": group,
            "training_group": group if split == "train" and group != "IGNORE" else "",
            "parent_crs": crs,
            "parent_left": bounds.left,
            "parent_bottom": bounds.bottom,
            "parent_right": bounds.right,
            "parent_top": bounds.top,
            "image_crs": crs,
            "label_crs": label_crs,
            "georeferencing_matches": georef_match,
            "image_transform": repr(image_transform),
            "label_transform": repr(label_transform),
        })
    return records


def find_parent_bbox_overlap(parent_records):
    """Report cross-split bbox overlaps only where CRS values match."""
    unique = {}
    for row in parent_records:
        unique.setdefault(row["parent_id"], row)
    parents = list(unique.values())
    overlaps = []
    incomparable_crs_pairs = 0
    for index, left in enumerate(parents):
        for right in parents[index + 1:]:
            if left["split"] == right["split"]:
                continue
            if not left["parent_crs"] or left["parent_crs"] != right["parent_crs"]:
                incomparable_crs_pairs += 1
                continue
            width = min(float(left["parent_right"]), float(right["parent_right"])) - max(
                float(left["parent_left"]), float(right["parent_left"])
            )
            height = min(float(left["parent_top"]), float(right["parent_top"])) - max(
                float(left["parent_bottom"]), float(right["parent_bottom"])
            )
            if width > 0 and height > 0:
                overlaps.append({
                    "parent_a": left["parent_id"], "split_a": left["split"],
                    "parent_b": right["parent_id"], "split_b": right["split"],
                    "bbox_overlap_area_crs_units2": width * height,
                    "crs": left["parent_crs"],
                    "note": "Bounding-box intersection candidate; not proof of pixel overlap.",
                })
    return overlaps, incomparable_crs_pairs


def main():
    split_rows = read_split_rows()
    records = []
    parent_rows = []
    for split, parent_id, image_name, label_name in split_rows:
        parent_records = inspect_parent(split, parent_id, image_name, label_name)
        records.extend(parent_records)
        parent_rows.append(parent_records[0])

    if len({row["patch_id"] for row in records}) != len(records):
        raise ValueError("Generated patch IDs are not unique.")
    by_parent = defaultdict(set)
    for row in records:
        by_parent[row["parent_id"]].add(row["split"])
    mixed = {parent: splits for parent, splits in by_parent.items() if len(splits) != 1}
    if mixed:
        raise ValueError(f"Patches cross split boundaries: {mixed}")

    overlaps, incomparable_pairs = find_parent_bbox_overlap(parent_rows)
    out_dir = new_output_dir("coverage_specialists")
    manifest_path = out_dir / "patch_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    report_rows = []
    for split in SPLITS:
        split_records = [row for row in records if row["split"] == split]
        for group in ("LOW", "HIGH", "IGNORE"):
            members = [row for row in split_records if row["group"] == group]
            report_rows.append({
                "split": split,
                "group": group,
                "patch_count": len(members),
                "parent_count": len({row["parent_id"] for row in members}),
                "valid_pixels": sum(int(row["valid_pixels"]) for row in members),
                "flood_pixels": sum(int(row["flood_pixels"]) for row in members),
            })
    report_path = out_dir / "coverage_distribution.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    overlap_path = out_dir / "parent_bbox_overlap_candidates.csv"
    with overlap_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["parent_a", "split_a", "parent_b", "split_b", "bbox_overlap_area_crs_units2", "crs", "note"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(overlaps)
    metadata = {
        "patch_size": PATCH_SIZE,
        "parent_size": IMAGE_SIZE,
        "patches_per_parent": len(patch_windows()),
        "group_threshold_fraction": 0.30,
        "group_rule": "LOW if valid coverage < 0.30; HIGH if >= 0.30; IGNORE if no valid pixels",
        "coverage_denominator": "label values 0 and 1 only; -1 excluded",
        "train_groups_used_for_training_only": True,
        "validation_test_training_group_fields": "blank",
        "test_labels_used_to_construct_training_groups": False,
        "test_split_read_for_manifest_diagnostics_only": True,
        "train_parent_count": sum(row["split"] == "train" for row in parent_rows),
        "validation_parent_count": sum(row["split"] == "validation" for row in parent_rows),
        "test_parent_count": sum(row["split"] == "test" for row in parent_rows),
        "patch_count": len(records),
        "cross_split_bbox_overlap_candidate_count": len(overlaps),
        "cross_split_pairs_not_compared_due_to_crs": incomparable_pairs,
        "bbox_overlap_caveat": "Bounding-box intersections are candidates, not proof of pixel/geographic overlap; different CRSs are not compared.",
        "raster_pairing_policy": "Images and labels use the existing Sen1Floods11 filename pairing and same pixel indices. Dimensions must match. CRS/affine metadata differences are recorded, not silently resampled; review them before interpreting geographic overlap.",
        "parents_with_georeferencing_mismatch": len({row["parent_id"] for row in parent_rows if not row["georeferencing_matches"]}),
    }
    (out_dir / "manifest_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Manifest: {manifest_path}")
    print(f"Coverage report: {report_path}")
    print(f"Parent bbox overlap candidates: {len(overlaps)}")
    print(f"Cross-CRS pairs not compared: {incomparable_pairs}")
    print("Training-only group counts:")
    train_counts = Counter(row["training_group"] for row in records if row["split"] == "train")
    for key in ("LOW", "HIGH", ""):
        if key in train_counts:
            print(f"  {key or 'fully ignored'}: {train_counts[key]} patches")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
