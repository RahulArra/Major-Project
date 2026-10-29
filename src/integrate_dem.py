"""Download required Copernicus DEM GLO-30 products and align them to Sen1Floods11 grids.

The script intentionally performs a strict disk-space and authentication preflight before
creating output directories or downloading anything. Set CDSE_ACCESS_TOKEN to a valid
Copernicus Data Space bearer token. Products are fetched once by product UUID and cached.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import getpass
import json
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject


ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "outputs" / "gis_tile_planning" / "20261014_101626"
AUDIT_DIR = ROOT / "outputs" / "gis_audit" / "20261014_095659"
DEM_MANIFEST = PLAN_DIR / "dem_tile_manifest.csv"
PAIR_AUDIT = AUDIT_DIR / "pair_audit.csv"
DATA_DIR = ROOT / "data" / "raw" / "sen1floods11" / "S1Hand"
DEM_ROOT = ROOT / "data" / "processed" / "dem"
SOURCE_DIR = DEM_ROOT / "source_tiles"
ALIGNED_DIR = DEM_ROOT / "aligned"
REPORT_ROOT = ROOT / "outputs" / "dem_integration"
VERSION = "COP-DEM_GLO-30-DGED/2024_1"
DOWNLOAD_BASE = "https://download.dataspace.copernicus.eu/odata/v1/Products"
NODATA_OUT = -9999.0
RETRIES = 5
CHUNK = 1024 * 1024


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def unique_products(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    products: dict[str, dict[str, str]] = {}
    for row in rows:
        tile = row["source_tile_id"].strip()
        if row.get("catalog_status") != "verified_available":
            raise RuntimeError(f"Tile {tile} is not verified available in the saved official catalog.")
        product_id = row.get("catalog_product_id", "").strip()
        size_text = row.get("source_file_size_bytes", "").strip()
        if not product_id or not size_text:
            raise RuntimeError(f"Missing catalog product ID or size for tile {tile}.")
        item = {"tile_id": tile, "product_id": product_id, "size": int(float(size_text)),
                "product_name": row.get("catalog_product_name", ""),
                "catalog_s3_path": row.get("catalog_s3_path", "")}
        if tile in products and products[tile]["product_id"] != product_id:
            raise RuntimeError(f"Conflicting product IDs for tile {tile}.")
        products[tile] = item
    return products


def check_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            bad_member = zf.testzip()
            return bad_member is None
    except (OSError, zipfile.BadZipFile):
        return False


def verify_tile_raster(path: Path, tile_id: str) -> None:
    match = re.fullmatch(r"([NS])(\d{2})_([EW])(\d{3})", tile_id)
    if not match:
        raise RuntimeError(f"Unexpected Copernicus DEM geocell ID: {tile_id}")
    lat = int(match.group(2)) * (1 if match.group(1) == "N" else -1)
    lon = int(match.group(4)) * (1 if match.group(3) == "E" else -1)
    expected = (lon, lat, lon + 1, lat + 1)
    with rasterio.open(path) as src:
        if src.count < 1 or src.width < 2 or src.height < 2 or not src.crs:
            raise RuntimeError(f"Invalid/unreferenced cached DEM tile: {path}")
        if src.crs.to_epsg() != 4326:
            raise RuntimeError(f"Expected EPSG:4326 source tile {tile_id}, got {src.crs} in {path}")
        actual = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        if any(abs(a - b) > 0.02 for a, b in zip(actual, expected)):
            raise RuntimeError(f"Raster bounds do not match tile {tile_id}: expected {expected}, got {actual}")


def check_cdse_access(token: str, product_id: str) -> None:
    url = f"{DOWNLOAD_BASE}({product_id})/Nodes"
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=45)
    if response.status_code in (401, 403):
        raise RuntimeError(f"CDSE rejected the token (HTTP {response.status_code}); no download started.")
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("CDSE access probe returned non-JSON content; no download started.") from exc
    if not isinstance(payload.get("result"), list):
        raise RuntimeError("CDSE access probe response had no product node listing; no download started.")
    # Verify the protected product-content endpoint itself, without consuming the product.
    probe_url = f"{DOWNLOAD_BASE}({product_id})/$value"
    with requests.get(probe_url, headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-0"},
                      stream=True, timeout=(30, 60), allow_redirects=True) as probe:
        if probe.status_code in (401, 403):
            raise RuntimeError(f"CDSE product download endpoint rejected the token (HTTP {probe.status_code}); no download started.")
        if probe.status_code not in (200, 206):
            probe.raise_for_status()
            raise RuntimeError(f"Unexpected CDSE product download probe status {probe.status_code}; no download started.")


def download_product(product: dict[str, str | int], token: str, destination: Path) -> str:
    expected = int(product["size"])
    if destination.exists():
        if destination.stat().st_size == expected and check_zip(destination):
            return "cached_archive_validated"
        raise RuntimeError(f"Existing archive is not a valid expected-size ZIP; preserved without overwrite: {destination}")

    partial = destination.with_suffix(destination.suffix + ".part")
    url = f"{DOWNLOAD_BASE}({product['product_id']})/$value"
    for attempt in range(1, RETRIES + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Authorization": f"Bearer {token}"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 180), allow_redirects=True) as response:
                if response.status_code in (401, 403):
                    raise RuntimeError(f"CDSE download authorization failed (HTTP {response.status_code}).")
                if offset and response.status_code == 200:
                    # Server ignored Range. Restart the partial safely rather than append a full response.
                    offset = 0
                elif offset and response.status_code != 206:
                    response.raise_for_status()
                    raise RuntimeError(f"Resume expected HTTP 206, got {response.status_code}.")
                else:
                    response.raise_for_status()
                mode = "ab" if offset and response.status_code == 206 else "wb"
                with partial.open(mode) as target:
                    for chunk in response.iter_content(CHUNK):
                        if chunk:
                            target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
            if partial.stat().st_size != expected:
                raise IOError(f"Size mismatch for {product['tile_id']}: expected {expected}, got {partial.stat().st_size}.")
            if not check_zip(partial):
                raise IOError(f"Downloaded product ZIP failed integrity validation: {product['tile_id']}.")
            # Exclusive creation semantics: do not replace an archive that appeared meanwhile.
            if destination.exists():
                raise FileExistsError(f"Archive appeared while downloading; leaving both files untouched: {destination}")
            partial.replace(destination)
            return "downloaded_and_validated"
        except RuntimeError:
            raise
        except Exception:
            if attempt == RETRIES:
                raise
            time.sleep(min(2 ** (attempt - 1), 30))
    raise RuntimeError(f"Download retries exhausted: {product['tile_id']}")


def extract_dem(archive: Path, tile: str, product_name: str) -> Path:
    destination = SOURCE_DIR / f"{safe_name(tile)}.tif"
    if destination.exists():
        try:
            verify_tile_raster(destination, tile)
            return destination
        except Exception:
            pass
        raise RuntimeError(f"Existing extracted tile is unreadable; preserved without overwrite: {destination}")

    with zipfile.ZipFile(archive) as zf:
        members = [name for name in zf.namelist()
                   if name.lower().endswith((".tif", ".tiff")) and not name.endswith("/")]
        if not members:
            raise RuntimeError(f"No GeoTIFF found in catalog product archive {archive.name}.")
        # Prefer a filename containing DEM, excluding metadata/quality companion layers if present.
        dem_members = [n for n in members if "dem" in Path(n).name.lower() and
                       not any(tag in Path(n).name.lower() for tag in ("edm", "flm", "hem", "wbm"))]
        member = sorted(dem_members or members, key=lambda n: (len(Path(n).name), n))[0]
        tmp = destination.with_suffix(".tif.tmp")
        if tmp.exists():
            raise RuntimeError(f"Interrupted extracted temporary file exists; inspect and remove manually: {tmp}")
        with zf.open(member) as source, tmp.open("xb") as target:
            shutil.copyfileobj(source, target, length=CHUNK)
        try:
            verify_tile_raster(tmp, tile)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        if destination.exists():
            raise FileExistsError(f"Tile appeared during extraction; leaving temp and existing file: {destination}")
        tmp.replace(destination)
    return destination


def output_paths(split: str, image_name: str) -> tuple[Path, Path]:
    stem = Path(image_name).stem
    folder = ALIGNED_DIR / split
    return folder / f"{stem}_dem.tif", folder / f"{stem}_dem_valid.tif"


def validate_existing_pair(dem_path: Path, mask_path: Path, sar: rasterio.DatasetReader) -> tuple[bool, str]:
    if not dem_path.exists() and not mask_path.exists():
        return False, ""
    if not dem_path.exists() or not mask_path.exists():
        raise RuntimeError(f"Only one existing output found; preserving both and refusing overwrite: {dem_path}, {mask_path}")
    with rasterio.open(dem_path) as dem, rasterio.open(mask_path) as mask:
        same = (dem.crs == sar.crs and dem.width == sar.width and dem.height == sar.height and
                dem.transform == sar.transform and mask.crs == sar.crs and mask.width == sar.width and
                mask.height == sar.height and mask.transform == sar.transform and dem.nodata == NODATA_OUT)
        if not same:
            raise RuntimeError(f"Existing outputs do not exactly match SAR grid; preserved: {dem_path}")
        valid = mask.read(1) > 0
        return True, float(valid.mean() * 100.0)


def align_scene(scene: dict[str, str], tile_paths: list[Path], report_dir: Path) -> dict[str, object]:
    split, image_name = scene["split"], scene["image_name"]
    sar_path = DATA_DIR / image_name
    dem_path, mask_path = output_paths(split, image_name)
    dem_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(sar_path) as sar:
        if not sar.crs:
            raise RuntimeError(f"SAR image has no CRS: {sar_path}")
        exists, existing_result = validate_existing_pair(dem_path, mask_path, sar)
        if exists:
            valid_pct = float(existing_result)
            return result_row(scene, dem_path, mask_path, tile_paths, "existing_outputs_validated", valid_pct,
                              "", sar, elevation_min=None, elevation_max=None)

        opened = [rasterio.open(p) for p in tile_paths]
        try:
            crs_set = {src.crs for src in opened}
            if len(crs_set) != 1:
                raise RuntimeError(f"Source DEM tiles have different CRSs for {image_name}: {crs_set}")
            # Use the common explicit nodata value when available; masks are also honored by masked merge.
            nodata_values = {src.nodata for src in opened}
            src_nodata = next(iter(nodata_values)) if len(nodata_values) == 1 else None
            mosaic, mosaic_transform = merge(opened, masked=True, nodata=src_nodata)
            mosaic_band = np.ma.asarray(mosaic[0])
            source = mosaic_band.filled(NODATA_OUT).astype(np.float32, copy=False)
            source_mask = np.ma.getmaskarray(mosaic_band) | ~np.isfinite(source)
            source[source_mask] = NODATA_OUT
            destination = np.full((sar.height, sar.width), NODATA_OUT, dtype=np.float32)
            reproject(source=source, destination=destination,
                      src_transform=mosaic_transform, src_crs=opened[0].crs, src_nodata=NODATA_OUT,
                      dst_transform=sar.transform, dst_crs=sar.crs, dst_nodata=NODATA_OUT,
                      resampling=Resampling.bilinear, init_dest_nodata=True)
            valid = np.isfinite(destination) & (destination != NODATA_OUT)
            mask_array = valid.astype(np.uint8)
            valid_pct = float(valid.mean() * 100.0)
            if valid.any():
                elev_min = float(destination[valid].min())
                elev_max = float(destination[valid].max())
            else:
                elev_min = elev_max = None

            dem_tmp = dem_path.with_suffix(".tmp.tif")
            mask_tmp = mask_path.with_suffix(".tmp.tif")
            if dem_tmp.exists() or mask_tmp.exists():
                raise RuntimeError(f"Temporary output already exists; inspect before rerun: {dem_tmp} / {mask_tmp}")
            profile = {"driver": "GTiff", "height": sar.height, "width": sar.width, "count": 1,
                       "crs": sar.crs, "transform": sar.transform, "dtype": "float32",
                       "nodata": NODATA_OUT, "compress": "deflate", "predictor": 3}
            with rasterio.open(dem_tmp, "w", **profile) as dst:
                dst.write(destination, 1)
                dst.set_band_description(1, "Copernicus DEM GLO-30 elevation (bilinear aligned)")
                dst.update_tags(source_version=VERSION, resampling="bilinear", nodata_policy="-9999 and separate mask")
            mask_profile = dict(profile, dtype="uint8", nodata=0, predictor=2)
            with rasterio.open(mask_tmp, "w", **mask_profile) as dst:
                dst.write(mask_array, 1)
                dst.set_band_description(1, "1=valid elevation, 0=invalid or no source coverage")
            # Refuse to replace outputs if another process created them meanwhile.
            if dem_path.exists() or mask_path.exists():
                raise FileExistsError(f"Output appeared during alignment; temporary outputs preserved: {dem_path}")
            dem_tmp.replace(dem_path)
            mask_tmp.replace(mask_path)
            suspicious = valid.any() and (elev_min < -500 or elev_max > 9000)
            status = "aligned_suspicious_elevation_range" if suspicious else "aligned"
            note = f"elevation range [{elev_min:.3f}, {elev_max:.3f}] m outside review bounds [-500, 9000]" if suspicious else ""
            return result_row(scene, dem_path, mask_path, tile_paths, status, valid_pct, note, sar,
                              elevation_min=elev_min, elevation_max=elev_max)
        finally:
            for src in opened:
                src.close()


def result_row(scene, dem_path, mask_path, tile_paths, status, valid_pct, note, sar,
               elevation_min=None, elevation_max=None):
    with rasterio.open(dem_path) as dem, rasterio.open(mask_path) as mask:
        grid_exact = (dem.crs == sar.crs and dem.transform == sar.transform and
                      dem.width == sar.width and dem.height == sar.height and
                      mask.crs == sar.crs and mask.transform == sar.transform and
                      mask.width == sar.width and mask.height == sar.height)
        mask_values = np.unique(mask.read(1))
        return {"split": scene["split"], "scene_id": Path(scene["image_name"]).stem,
                "image_name": scene["image_name"], "dem_path": str(dem_path.relative_to(ROOT)),
                "validity_mask_path": str(mask_path.relative_to(ROOT)),
                "source_tile_ids": ";".join(p.stem for p in tile_paths), "source_version": VERSION,
                "alignment_status": status, "crs": dem.crs.to_string(), "width": dem.width, "height": dem.height,
                "transform": str(tuple(dem.transform)), "bounds": str(tuple(dem.bounds)),
                "sar_grid_exact_match": grid_exact, "resampling": "bilinear", "nodata": dem.nodata,
                "mask_values": ";".join(str(int(v)) for v in mask_values),
                "valid_pixel_percentage": valid_pct, "elevation_min_m": elevation_min,
                "elevation_max_m": elevation_max, "issue": note}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true", help="Check manifests, token, access, and free space; download nothing.")
    parser.add_argument("--minimum-free-gib", type=float, default=None,
                        help="Optional manual safety threshold in GiB; default uses 2x download size + output/buffer allowance.")
    args = parser.parse_args()

    if not DEM_MANIFEST.is_file() or not PAIR_AUDIT.is_file():
        raise FileNotFoundError(f"Required saved plan/audit not found: {DEM_MANIFEST} / {PAIR_AUDIT}")
    products = unique_products(read_csv(DEM_MANIFEST))
    scenes_rows = read_csv(PAIR_AUDIT)
    scenes = [r for r in scenes_rows if r.get("pair_status") == "ok" and r.get("image_exists", "").lower() == "true"]
    if len(scenes) != 431:
        raise RuntimeError(f"Expected 431 successful audited scenes, found {len(scenes)}.")
    scene_tiles: dict[str, list[str]] = {}
    for row in read_csv(DEM_MANIFEST):
        scene_tiles.setdefault(row["image_name"], [])
        if row["source_tile_id"] not in scene_tiles[row["image_name"]]:
            scene_tiles[row["image_name"]].append(row["source_tile_id"])
    for scene in scenes:
        if scene["image_name"] not in scene_tiles or not scene_tiles[scene["image_name"]]:
            raise RuntimeError(f"Scene has no planned DEM source tile: {scene['image_name']}")

    expected_bytes = sum(int(item["size"]) for item in products.values())
    aligned_estimate = len(scenes) * 2 * 512 * 512 * 4  # raw-array upper estimate before compression
    required_bytes = 2 * expected_bytes + aligned_estimate + 2 * 1024**3
    if args.minimum_free_gib is not None:
        required_bytes = max(required_bytes, int(args.minimum_free_gib * 1024**3))
    try:
        disk_free = shutil.disk_usage(ROOT).free
    except OSError as exc:
        raise RuntimeError(f"Cannot verify available disk space on project volume; refusing downloads: {exc}") from exc
    print(f"Verified saved catalog products: {len(products)}")
    print(f"Catalog download size estimate: {expected_bytes:,} bytes ({expected_bytes / 1024**3:.2f} GiB)")
    print(f"Available project-volume space: {disk_free:,} bytes ({disk_free / 1024**3:.2f} GiB)")
    print(f"Conservative minimum required free space: {required_bytes:,} bytes ({required_bytes / 1024**3:.2f} GiB)")
    if disk_free < required_bytes:
        raise RuntimeError("Insufficient free space for conservative download/extraction staging; no download started.")
    token = os.environ.get("CDSE_ACCESS_TOKEN", "").strip()
    if not token:
        token = getpass.getpass("CDSE bearer access token (input hidden): ").strip()
    if not token:
        raise RuntimeError("No CDSE bearer token supplied; no download started.")
    first_product = next(iter(products.values()))
    check_cdse_access(token, str(first_product["product_id"]))
    print("CDSE bearer-token download-endpoint probe: passed")
    if args.preflight_only:
        return 0

    run_dir = REPORT_ROOT / timestamp()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing report directory: {run_dir}")
    run_dir.mkdir(parents=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, object]] = {}
    failures: list[dict[str, str]] = []
    source_status: dict[str, str] = {}
    for index, (tile_id, product) in enumerate(products.items(), start=1):
        archive = SOURCE_DIR / f"{safe_name(tile_id)}.zip"
        try:
            raster_path = SOURCE_DIR / f"{safe_name(tile_id)}.tif"
            if raster_path.exists():
                verify_tile_raster(raster_path, tile_id)
                source_status[tile_id] = "cached_raster_validated"
            else:
                source_status[tile_id] = download_product(product, token, archive)
                extract_dem(archive, tile_id, str(product["product_name"]))
                if archive.exists():
                    archive.unlink()
            print(f"[{index}/{len(products)}] cached source tile {tile_id}")
        except Exception as exc:
            failures.append({"source_tile_id": tile_id, "error": str(exc)})
            print(f"FAILED source tile {tile_id}: {exc}", file=sys.stderr)

    available_paths = {tile: SOURCE_DIR / f"{safe_name(tile)}.tif" for tile in products}
    for index, scene in enumerate(scenes, start=1):
        required_tiles = scene_tiles[scene["image_name"]]
        missing = [tile for tile in required_tiles if not available_paths[tile].is_file()]
        if missing:
            results[scene["image_name"]] = {"split": scene["split"], "scene_id": Path(scene["image_name"]).stem,
                                             "image_name": scene["image_name"], "alignment_status": "failed_missing_source_tiles",
                                             "source_tile_ids": ";".join(required_tiles), "issue": ";".join(missing)}
            continue
        try:
            aligned = align_scene(scene, [available_paths[t] for t in required_tiles], run_dir)
            results[scene["image_name"]] = aligned
            if index % 25 == 0:
                print(f"Aligned {index}/{len(scenes)} scenes")
        except Exception as exc:
            results[scene["image_name"]] = {"split": scene["split"], "scene_id": Path(scene["image_name"]).stem,
                                             "image_name": scene["image_name"], "alignment_status": "failed",
                                             "source_tile_ids": ";".join(required_tiles), "issue": str(exc)}

    fields = sorted({key for row in results.values() for key in row})
    with (run_dir / "dem_alignment_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results.values())
    with (run_dir / "source_tile_status.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_tile_id", "product_id", "product_name", "expected_bytes", "download_status", "cached_raster"])
        writer.writeheader()
        for tile, item in products.items():
            raster_path = available_paths[tile]
            writer.writerow({"source_tile_id": tile, "product_id": item["product_id"], "product_name": item["product_name"],
                             "expected_bytes": item["size"], "download_status": source_status.get(tile, "failed"),
                             "cached_raster": raster_path.exists()})
    summary = {"source_version": VERSION, "scene_count": len(scenes), "unique_source_tiles": len(products),
               "download_size_estimate_bytes": expected_bytes, "downloaded_or_cached_tiles": sum(p.exists() for p in available_paths.values()),
               "tiles_downloaded_this_run": sum(v == "downloaded_and_validated" for v in source_status.values()),
               "tiles_reused_from_cache": sum(v == "cached_raster_validated" for v in source_status.values()),
               "failed_source_tiles": failures, "aligned_scenes": sum(str(r.get("alignment_status", "")).startswith(("aligned", "existing_outputs_validated")) for r in results.values()),
               "failed_scenes": sum(not str(r.get("alignment_status", "")).startswith(("aligned", "existing_outputs_validated")) for r in results.values()),
               "output_directory": str(run_dir), "data_output_directory": str(DEM_ROOT),
               "license_attribution": "Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA.",
               "official_download_method": "CDSE OData product download endpoint with bearer token; registered CCM access required.",
               "validation_regions": ["Ghana", "India", "Paraguay"]}
    representative_validation = {}
    for region in ("Ghana", "India", "Paraguay"):
        match = next((r for r in results.values() if str(r.get("image_name", "")).startswith(region + "_")), None)
        if match is None:
            representative_validation[region] = {"status": "not_found"}
            continue
        representative_validation[region] = {
            "scene_id": match.get("scene_id"),
            "status": match.get("alignment_status"),
            "crs": match.get("crs"), "width": match.get("width"), "height": match.get("height"),
            "transform": match.get("transform"), "bounds": match.get("bounds"),
            "valid_pixel_percentage": match.get("valid_pixel_percentage"),
            "nodata": match.get("nodata"), "sar_grid_exact_match": match.get("sar_grid_exact_match"),
            "elevation_min_m": match.get("elevation_min_m"), "elevation_max_m": match.get("elevation_max_m"),
            "issue": match.get("issue", "")}
        coverage = match.get("valid_pixel_percentage")
        if coverage is not None and float(coverage) < 99.0:
            representative_validation[region]["coverage_warning"] = "less than 99% valid pixels; inspect nodata/coverage"
    summary["representative_scene_validation"] = representative_validation
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Aligned scenes: {summary['aligned_scenes']}; failed scenes: {summary['failed_scenes']}")
    print(f"Output: {run_dir}")
    return 0 if not failures and summary["failed_scenes"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
