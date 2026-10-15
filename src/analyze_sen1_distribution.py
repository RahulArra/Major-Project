"""Read-only per-image and split-level Sen1Floods11 label diagnostics.

Coverage bins are based on flood pixels divided by valid pixels:
* flood_free: valid pixels exist and flood coverage is exactly 0%.
* low: 0% < coverage <= 1%.
* moderate: 1% < coverage <= 10%.
* high: coverage > 10%.
* fully_ignored: the image has no valid pixels; reported separately because
  flood coverage is undefined for it.

The pixel-weighted split percentage is total flood pixels / total valid pixels.
The unweighted per-image mean gives each image with valid pixels equal weight,
regardless of how many valid pixels it contains. These answer different
questions and are both reported explicitly.
"""

import csv
import math

import numpy as np
import rasterio

from diagnostic_utils import PROJECT_ROOT, new_output_dir


SPLITS = {
    "train": PROJECT_ROOT / "data/raw/sen1floods11/splits/flood_train_data.csv",
    "validation": PROJECT_ROOT / "data/raw/sen1floods11/splits/flood_valid_data.csv",
}
LABEL_DIR = PROJECT_ROOT / "data/raw/sen1floods11/LabelHand"
LOW_COVERAGE_MAX_PERCENT = 1.0
MODERATE_COVERAGE_MAX_PERCENT = 10.0

PER_IMAGE_FIELDS = [
    "split", "image", "total_pixels", "valid_pixels", "flood_pixels",
    "non_flood_pixels", "ignored_pixels", "flood_coverage_percentage",
    "coverage_category",
]


def classify_coverage(valid_pixels, flood_pixels, coverage_percentage):
    if valid_pixels == 0:
        return "fully_ignored"
    if flood_pixels == 0:
        return "flood_free"
    if coverage_percentage <= LOW_COVERAGE_MAX_PERCENT:
        return "low"
    if coverage_percentage <= MODERATE_COVERAGE_MAX_PERCENT:
        return "moderate"
    return "high"


def analyze_split(name, split_path):
    if not split_path.is_file():
        raise FileNotFoundError(f"Split not found: {split_path}")

    image_rows = []
    total_pixels = valid_pixels = ignored_pixels = flood_pixels = non_flood_pixels = 0
    coverage_values = []
    category_counts = {
        "flood_free_image_count": 0,
        "low_flood_coverage_image_count": 0,
        "moderate_flood_coverage_image_count": 0,
        "high_flood_coverage_image_count": 0,
        "fully_ignored_image_count": 0,
    }

    with split_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                raise ValueError(f"Malformed split row in {split_path}: {row}")
            label_path = LABEL_DIR / row[1]
            with rasterio.open(label_path) as src:
                label = src.read(1)

            total = int(label.size)
            ignored = int(np.count_nonzero(label == -1))
            flood = int(np.count_nonzero(label == 1))
            non_flood = int(np.count_nonzero(label == 0))
            valid = total - ignored
            unknown = valid - flood - non_flood
            if unknown:
                raise ValueError(
                    f"Unexpected label values in {label_path}; expected only -1, 0, 1"
                )

            coverage = 100.0 * flood / valid if valid else None
            category = classify_coverage(valid, flood, coverage)
            if category == "fully_ignored":
                category_counts["fully_ignored_image_count"] += 1
            elif category == "flood_free":
                category_counts["flood_free_image_count"] += 1
                coverage_values.append(coverage)
            elif category == "low":
                category_counts["low_flood_coverage_image_count"] += 1
                coverage_values.append(coverage)
            elif category == "moderate":
                category_counts["moderate_flood_coverage_image_count"] += 1
                coverage_values.append(coverage)
            else:
                category_counts["high_flood_coverage_image_count"] += 1
                coverage_values.append(coverage)

            image_rows.append({
                "split": name,
                "image": row[0],
                "total_pixels": total,
                "valid_pixels": valid,
                "flood_pixels": flood,
                "non_flood_pixels": non_flood,
                "ignored_pixels": ignored,
                "flood_coverage_percentage": coverage,
                "coverage_category": category,
            })
            total_pixels += total
            valid_pixels += valid
            ignored_pixels += ignored
            flood_pixels += flood
            non_flood_pixels += non_flood

    if not image_rows:
        raise ValueError(f"No samples listed in {split_path}")

    coverage_array = np.asarray(coverage_values, dtype=np.float64)
    stats = {
        "split": name,
        "number_of_images": len(image_rows),
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "ignored_pixels": ignored_pixels,
        "flood_pixels": flood_pixels,
        "non_flood_pixels": non_flood_pixels,
        "pixel_weighted_flood_percentage_valid": (
            100.0 * flood_pixels / valid_pixels if valid_pixels else 0.0
        ),
        "unweighted_mean_per_image_flood_percentage": (
            float(coverage_array.mean()) if coverage_array.size else math.nan
        ),
        "images_in_unweighted_mean": int(coverage_array.size),
        "ignore_percentage_all": (
            100.0 * ignored_pixels / total_pixels if total_pixels else 0.0
        ),
        "per_image_flood_percentage_min": (
            float(coverage_array.min()) if coverage_array.size else math.nan
        ),
        "per_image_flood_percentage_median": (
            float(np.median(coverage_array)) if coverage_array.size else math.nan
        ),
        "per_image_flood_percentage_max": (
            float(coverage_array.max()) if coverage_array.size else math.nan
        ),
        "coverage_bin_low_max_percent_inclusive": LOW_COVERAGE_MAX_PERCENT,
        "coverage_bin_moderate_max_percent_inclusive": MODERATE_COVERAGE_MAX_PERCENT,
        **category_counts,
    }
    return stats, image_rows


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    results = []
    per_image_rows = []
    for name, path in SPLITS.items():
        summary, images = analyze_split(name, path)
        results.append(summary)
        per_image_rows.extend(images)

    out_dir = new_output_dir("sen1_distribution")
    per_image_path = out_dir / "per_image_distribution.csv"
    summary_path = out_dir / "split_distribution.csv"
    write_csv(per_image_path, per_image_rows, PER_IMAGE_FIELDS)
    write_csv(summary_path, results, list(results[0]))

    print("Coverage bins: flood-free = 0%; low = (0%, 1%]; moderate = (1%, 10%]; high = >10%.")
    print("Fully ignored images (0 valid pixels) are counted separately; their coverage is blank in the per-image CSV.")
    print("Pixel-weighted flood % = total flood / total valid pixels. Unweighted per-image mean gives equal weight to each image with valid pixels.")
    for result in results:
        print(f"\n{result['split'].upper()} — {result['number_of_images']} images")
        print(f"  Pixels: total={result['total_pixels']:,}, valid={result['valid_pixels']:,}, ignored={result['ignored_pixels']:,}")
        print(f"  Valid labels: flood={result['flood_pixels']:,}, non-flood={result['non_flood_pixels']:,}")
        print(f"  Pixel-weighted flood: {result['pixel_weighted_flood_percentage_valid']:.3f}%")
        print(f"  Unweighted mean per-image flood coverage: {result['unweighted_mean_per_image_flood_percentage']:.3f}% ({result['images_in_unweighted_mean']} images)")
        print(f"  Image counts: flood-free={result['flood_free_image_count']}, low={result['low_flood_coverage_image_count']}, moderate={result['moderate_flood_coverage_image_count']}, high={result['high_flood_coverage_image_count']}, fully ignored={result['fully_ignored_image_count']}")
    print(f"\nPer-image CSV: {per_image_path}")
    print(f"Split summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
