"""Read-only raster metadata and pixel-grid audit for Sen1Floods11.

Reads the existing train/validation/test split CSVs and their paired TIFFs.
Writes timestamped CSV/JSON reports under outputs/gis_audit/. No source data
is modified, reprojected, or resampled.
"""

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "sen1floods11"
IMAGE_DIR = DATA_ROOT / "S1Hand"
LABEL_DIR = DATA_ROOT / "LabelHand"
SPLITS = {
    "train": DATA_ROOT / "splits" / "flood_train_data.csv",
    "validation": DATA_ROOT / "splits" / "flood_valid_data.csv",
    "test": DATA_ROOT / "splits" / "flood_test_data.csv",
}
MEANINGFUL_SHIFT_PIXELS = 0.01


def make_output_dir():
    base = PROJECT_ROOT / "outputs" / "gis_audit"
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / stamp
    suffix = 1
    while candidate.exists():
        candidate = base / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def read_split_rows():
    rows = []
    for split, path in SPLITS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Split CSV not found: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for line_no, row in enumerate(reader, 1):
                if len(row) < 2 or not row[0].strip() or not row[1].strip():
                    raise ValueError(f"Malformed split row {line_no} in {path}: {row}")
                rows.append({
                    "split": split,
                    "split_csv": str(path.relative_to(PROJECT_ROOT)),
                    "split_row": line_no,
                    "image_name": row[0].strip(),
                    "label_name": row[1].strip(),
                })
    return rows


def transform_discrepancy_pixels(image_transform, label_transform, width, height):
    """Max corner displacement after mapping label grid into image pixel space."""
    image_transform = Affine(*tuple(image_transform)[:6])
    label_transform = Affine(*tuple(label_transform)[:6])
    image_inverse = ~image_transform
    corners = ((0.0, 0.0), (float(width), 0.0),
               (0.0, float(height)), (float(width), float(height)))
    displacements = []
    for col, row in corners:
        world_x, world_y = label_transform * (col, row)
        image_col, image_row = image_inverse * (world_x, world_y)
        displacements.append(math.hypot(image_col - col, image_row - row))
    return max(displacements)


def json_value(value):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def audit_pair(row):
    image_path = IMAGE_DIR / row["image_name"]
    label_path = LABEL_DIR / row["label_name"]
    result = dict(row)
    result.update({
        "image_path": str(image_path), "label_path": str(label_path),
        "image_exists": image_path.is_file(), "label_exists": label_path.is_file(),
        "pair_status": "ok", "grid_status": "not_checked",
    })
    if not image_path.is_file() or not label_path.is_file():
        result["pair_status"] = "missing_image" if not image_path.is_file() else "missing_label"
        if not image_path.is_file() and not label_path.is_file():
            result["pair_status"] = "missing_both"
        return result, [], []

    band_rows = []
    label_value_rows = []
    with rasterio.open(image_path) as image_src, rasterio.open(label_path) as label_src:
        image_transform = image_src.transform
        label_transform = label_src.transform
        image_crs = image_src.crs.to_string() if image_src.crs else None
        label_crs = label_src.crs.to_string() if label_src.crs else None
        dimensions_match = (image_src.width, image_src.height) == (label_src.width, label_src.height)
        crs_match = image_src.crs == label_src.crs
        result.update({
            "image_crs": image_crs,
            "image_width": image_src.width,
            "image_height": image_src.height,
            "image_transform": repr(tuple(image_transform)),
            "image_bounds": repr(tuple(image_src.bounds)),
            "image_resolution": repr((image_transform.a, image_transform.e)),
            "image_band_count": image_src.count,
            "image_dtypes": repr(tuple(image_src.dtypes)),
            "image_nodatavals": repr(tuple(json_value(x) for x in image_src.nodatavals)),
            "image_bounds_left": image_src.bounds.left,
            "image_bounds_bottom": image_src.bounds.bottom,
            "image_bounds_right": image_src.bounds.right,
            "image_bounds_top": image_src.bounds.top,
            "label_crs": label_crs,
            "label_width": label_src.width,
            "label_height": label_src.height,
            "label_transform": repr(tuple(label_transform)),
            "label_bounds": repr(tuple(label_src.bounds)),
            "label_resolution": repr((label_transform.a, label_transform.e)),
            "label_band_count": label_src.count,
            "label_dtypes": repr(tuple(label_src.dtypes)),
            "label_nodatavals": repr(tuple(json_value(x) for x in label_src.nodatavals)),
            "crs_match": crs_match,
            "dimensions_match": dimensions_match,
            "transform_exactly_equal": tuple(image_transform) == tuple(label_transform),
        })

        if crs_match and dimensions_match:
            pixel_discrepancy = transform_discrepancy_pixels(
                image_transform, label_transform, image_src.width, image_src.height,
            )
            result["max_transform_discrepancy_pixels"] = pixel_discrepancy
            result["meaningful_misalignment"] = pixel_discrepancy > MEANINGFUL_SHIFT_PIXELS
            result["grid_status"] = (
                "meaningful_misalignment" if pixel_discrepancy > MEANINGFUL_SHIFT_PIXELS
                else "aligned_within_tolerance"
            )
        else:
            result["max_transform_discrepancy_pixels"] = None
            result["meaningful_misalignment"] = None
            result["grid_status"] = "crs_mismatch" if not crs_match else "dimension_mismatch"

        image = image_src.read()
        for band_index in range(image_src.count):
            values = image[band_index]
            finite = np.isfinite(values)
            nodata = image_src.nodatavals[band_index]
            if nodata is None:
                nodata_mask = np.zeros(values.shape, dtype=bool)
            elif isinstance(nodata, float) and math.isnan(nodata):
                nodata_mask = np.isnan(values)
            else:
                nodata_mask = values == nodata
            usable = finite & ~nodata_mask
            finite_values = values[finite]
            usable_values = values[usable]
            band_rows.append({
                "split": row["split"], "image_name": row["image_name"],
                "band_index_1based": band_index + 1,
                "stored_dtype": image_src.dtypes[band_index],
                "nodata": json_value(nodata),
                "finite_pixel_count": int(finite.sum()),
                "nan_count": int(np.isnan(values).sum()),
                "positive_infinity_count": int(np.isposinf(values).sum()),
                "negative_infinity_count": int(np.isneginf(values).sum()),
                "nodata_pixel_count": int(nodata_mask.sum()),
                "finite_min": json_value(float(finite_values.min())) if finite_values.size else None,
                "finite_max": json_value(float(finite_values.max())) if finite_values.size else None,
                "valid_non_nodata_min": json_value(float(usable_values.min())) if usable_values.size else None,
                "valid_non_nodata_max": json_value(float(usable_values.max())) if usable_values.size else None,
            })

        label = label_src.read(1)
        values, counts = np.unique(label, return_counts=True)
        count_by_value = {str(json_value(v.item() if hasattr(v, "item") else v)): int(c)
                          for v, c in zip(values, counts)}
        ignored_count = int(np.count_nonzero(label == -1))
        background_count = int(np.count_nonzero(label == 0))
        flood_count = int(np.count_nonzero(label == 1))
        unexpected = {key: value for key, value in count_by_value.items()
                      if key not in {"-1", "0", "1"}}
        result.update({
            "label_unique_values_and_counts": json.dumps(count_by_value, sort_keys=True),
            "label_ignore_count": ignored_count,
            "label_background_count": background_count,
            "label_flood_count": flood_count,
            "label_unexpected_values_and_counts": json.dumps(unexpected, sort_keys=True),
            "label_fully_ignored": ignored_count == label.size,
            "label_has_valid_pixels": background_count + flood_count > 0,
            "label_has_flood": flood_count > 0,
            "label_has_background": background_count > 0,
        })
        for value, count in zip(values, counts):
            label_value_rows.append({
                "split": row["split"], "image_name": row["image_name"],
                "label_name": row["label_name"],
                "label_value": json_value(value.item() if hasattr(value, "item") else value),
                "pixel_count": int(count),
                "is_expected_value": bool(value in (-1, 0, 1)),
            })
    return result, band_rows, label_value_rows


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    pairs = read_split_rows()
    pair_rows, band_rows, label_value_rows = [], [], []
    for index, pair in enumerate(pairs, 1):
        result, bands, label_values = audit_pair(pair)
        pair_rows.append(result)
        band_rows.extend(bands)
        label_value_rows.extend(label_values)
        if index % 25 == 0 or index == len(pairs):
            print(f"Audited {index}/{len(pairs)} image/label pairs")

    output_dir = make_output_dir()
    write_csv(output_dir / "pair_audit.csv", pair_rows)
    write_csv(output_dir / "sar_band_audit.csv", band_rows)
    write_csv(output_dir / "label_value_counts.csv", label_value_rows)

    successful = [row for row in pair_rows if row["pair_status"] == "ok"]
    bounds_by_crs = {}
    for row in successful:
        crs = row.get("image_crs") or "<missing CRS>"
        values = bounds_by_crs.setdefault(crs, {
            "left": [], "bottom": [], "right": [], "top": [],
        })
        for key in values:
            bound_key = f"image_bounds_{key}"
            if row.get(bound_key) is not None:
                values[key].append(float(row[bound_key]))
    bounds_summary = {}
    for crs, values in bounds_by_crs.items():
        bounds_summary[crs] = {
            "left_min": min(values["left"]) if values["left"] else None,
            "bottom_min": min(values["bottom"]) if values["bottom"] else None,
            "right_max": max(values["right"]) if values["right"] else None,
            "top_max": max(values["top"]) if values["top"] else None,
            "scene_count": len(values["left"]),
            "interpretation": "bounding envelope of separate scene chips; not continuous coverage",
        }
    summary = {
        "audit_timestamp": datetime.now().isoformat(timespec="seconds"),
        "read_only": True,
        "threshold_for_meaningful_transform_shift_pixels": MEANINGFUL_SHIFT_PIXELS,
        "split_pair_counts": {
            split: sum(row["split"] == split for row in pairs) for split in SPLITS
        },
        "audited_pair_count": len(pairs),
        "successful_pair_count": len(successful),
        "missing_or_failed_pairs": [
            {"split": row["split"], "image_name": row["image_name"],
             "label_name": row["label_name"], "pair_status": row["pair_status"]}
            for row in pair_rows if row["pair_status"] != "ok"
        ],
        "crs_mismatch_count": sum(row.get("grid_status") == "crs_mismatch" for row in pair_rows),
        "dimension_mismatch_count": sum(row.get("grid_status") == "dimension_mismatch" for row in pair_rows),
        "meaningful_misalignment_count": sum(row.get("meaningful_misalignment") is True for row in pair_rows),
        "exact_transform_inequality_count": sum(
            row.get("transform_exactly_equal") is False for row in pair_rows
        ),
        "fully_ignored_scene_count": sum(row.get("label_fully_ignored") is True for row in pair_rows),
        "scenes_with_flood_count": sum(row.get("label_has_flood") is True for row in pair_rows),
        "scenes_with_background_count": sum(row.get("label_has_background") is True for row in pair_rows),
        "scenes_with_unexpected_label_values": [
            {"split": row["split"], "image_name": row["image_name"],
             "unexpected_values": row.get("label_unexpected_values_and_counts")}
            for row in pair_rows if row.get("label_unexpected_values_and_counts") not in (None, "{}")
        ],
        "bounds_by_crs": bounds_summary,
        "coverage_note": "The overall bounds summarize the envelope around listed scene chips and do not imply continuous imagery between scenes.",
        "outputs": ["pair_audit.csv", "sar_band_audit.csv", "label_value_counts.csv"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Output directory: {output_dir}")
    print(f"Audited pairs: {len(pairs)}; successful: {len(successful)}")
    print(f"CRS mismatches: {summary['crs_mismatch_count']}; dimension mismatches: {summary['dimension_mismatch_count']}")
    print(f"Meaningful transform shifts (> {MEANINGFUL_SHIFT_PIXELS} px): {summary['meaningful_misalignment_count']}")
    print(f"Exact transform inequalities: {summary['exact_transform_inequality_count']}")
    print(f"Fully ignored scenes: {summary['fully_ignored_scene_count']}; scenes with flood: {summary['scenes_with_flood_count']}; scenes with background: {summary['scenes_with_background_count']}")


if __name__ == "__main__":
    main()
