"""Read-only scan of Sen1Floods11 SAR TIFF metadata for acquisition dates.

Reads only TIFF metadata from the existing train/validation/test scenes; it
does not read pixel arrays or modify dataset files. Full dataset and per-band
tags are retained so date candidates can be reviewed rather than assumed.
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "sen1floods11"
IMAGE_DIR = DATA_ROOT / "S1Hand"
SPLITS = {
    "train": DATA_ROOT / "splits" / "flood_train_data.csv",
    "validation": DATA_ROOT / "splits" / "flood_valid_data.csv",
    "test": DATA_ROOT / "splits" / "flood_test_data.csv",
}
DATE_PATTERN = re.compile(
    r"(?<!\d)(?:19|20)\d{2}[-_/.:]?(?:0[1-9]|1[0-2])[-_/.:]?"
    r"(?:0[1-9]|[12]\d|3[01])(?:[Tt _-]?[0-2]\d[:.]?[0-5]\d[:.]?[0-5]\d)?(?!\d)"
)


def read_scenes():
    scenes = []
    for split, path in SPLITS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Split CSV not found: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.reader(handle), start=1):
                if len(row) < 2 or not row[0].strip():
                    raise ValueError(f"Malformed row {row_number} in {path}: {row}")
                scenes.append({
                    "split": split,
                    "split_row": row_number,
                    "image_name": row[0].strip(),
                })
    return scenes


def scan_one(scene):
    path = IMAGE_DIR / scene["image_name"]
    row = dict(scene)
    row["image_path"] = str(path)
    row["metadata_status"] = "ok"
    row["dataset_tags_json"] = "{}"
    row["band_tags_json"] = "[]"
    row["date_like_metadata_json"] = "[]"
    if not path.is_file():
        row["metadata_status"] = "missing_image"
        return row

    with rasterio.open(path) as src:
        dataset_tags = src.tags()
        band_tags = [src.tags(index) for index in range(1, src.count + 1)]
        row.update({
            "width": src.width,
            "height": src.height,
            "crs": src.crs.to_string() if src.crs else "",
            "band_count": src.count,
            "dataset_tags_json": json.dumps(dataset_tags, sort_keys=True),
            "band_tags_json": json.dumps(band_tags, sort_keys=True),
        })

    date_candidates = []
    for scope, tags in [("dataset", dataset_tags)] + [
        (f"band_{index}", tags) for index, tags in enumerate(band_tags, 1)
    ]:
        for key, value in tags.items():
            matches = DATE_PATTERN.findall(str(value))
            if matches or any(term in key.lower() for term in ("date", "time", "acq")):
                date_candidates.append({
                    "scope": scope,
                    "key": key,
                    "value": value,
                    "date_patterns": matches,
                })
    row["date_like_metadata_json"] = json.dumps(date_candidates, sort_keys=True)
    return row


def main():
    scenes = read_scenes()
    rows = [scan_one(scene) for scene in scenes]
    output_root = PROJECT_ROOT / "outputs" / "gis_tile_planning"
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"scene_metadata_{stamp}"
    suffix = 1
    while output_dir.exists():
        output_dir = output_root / f"scene_metadata_{stamp}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir()

    csv_path = output_dir / "sar_tiff_metadata.csv"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    successful = [row for row in rows if row["metadata_status"] == "ok"]
    candidates = sum(row["date_like_metadata_json"] != "[]" for row in successful)
    summary = {
        "read_only": True,
        "pixel_data_read": False,
        "scene_count": len(rows),
        "successful_metadata_reads": len(successful),
        "missing_or_failed_reads": len(rows) - len(successful),
        "scenes_with_date_like_tiff_metadata": candidates,
        "warning": "Date-like tags are candidates only; verify their meaning and timezone against Sen1Floods11 source metadata before temporal use.",
        "csv": str(csv_path),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Metadata scenes: {len(rows)}; successful: {len(successful)}")
    print(f"Scenes with date-like TIFF metadata: {candidates}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
