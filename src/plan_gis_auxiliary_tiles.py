"""Read-only scene-to-tile planner for Copernicus DEM GLO-30 and JRC GSW v1.5.

Reads the existing Sen1Floods11 pair audit and queries official product catalogs
for metadata only. It never downloads raster data. If catalog networking is
unavailable, tile intersections are still listed but catalog status remains
unverified (never inferred to mean available or missing).
"""

import argparse
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "outputs" / "gis_audit" / "20261014_095659"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "gis_tile_planning"
DEM_RELEASE = "COP-DEM_GLO-30-DGED/2024_1"
DEM_OWS = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DEM_DOC = "https://documentation.dataspace.copernicus.eu/Data/Others/CCM.html"
DEM_INDEX_DOC = (
    "https://dataspace.copernicus.eu/explore-data/data-collections/"
    "copernicus-contributing-missions/collections-description/COP-DEM"
)
JRC_LISTING = "https://s3.waw4-1.cloudferro.com/swift/v1/global-surface-water/"
JRC_TILE_ROOT = JRC_LISTING + "download2024/Aggregated/VER1-5/occurrence/"
JRC_DOC = "https://global-surface-water.appspot.com/download"
JRC_VERSION = "GSW v1.5 occurrence, aggregated 1984-2024"
USER_AGENT = "Sen1Floods11-read-only-GIS-tile-planner/1.0"


def get_url(url, timeout=15):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read(), response.headers


def read_scene_rows(audit_dir):
    path = audit_dir / "pair_audit.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Audit pair report not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("pair_status") != "ok" or row.get("image_crs") != "EPSG:4326":
                continue
            rows.append({
                "scene_id": Path(row["image_name"]).stem,
                "image_name": row["image_name"],
                "split": row["split"],
                "left": float(row["image_bounds_left"]),
                "bottom": float(row["image_bounds_bottom"]),
                "right": float(row["image_bounds_right"]),
                "top": float(row["image_bounds_top"]),
            })
    return rows


def intervals(minimum, maximum, size, grid_min=None, grid_max=None):
    """Positive-area tile intervals intersecting [minimum, maximum]."""
    lo = minimum if grid_min is None else max(minimum, grid_min)
    hi = maximum if grid_max is None else min(maximum, grid_max)
    if hi <= lo:
        return []
    first = math.floor(lo / size)
    # Subtract a tiny relative amount so an edge exactly on a tile boundary
    # does not claim the adjacent tile with zero-area contact.
    last = math.floor(math.nextafter(hi, -math.inf) / size)
    return [(index * size, (index + 1) * size) for index in range(first, last + 1)]


def signed_grid(value, width):
    if value < 0:
        return f"{abs(value)}W" if width == 3 else f"{abs(value)}S"
    if value > 0:
        return f"{value}E" if width == 3 else f"{value}N"
    return f"0{'E' if width == 3 else 'N'}"


def dem_tile_id(lon_min, lat_min):
    # CDSE COP-DEM gridId names the 1-degree geocell by its integer south/west edge.
    lon = int(lon_min)
    lat = int(lat_min)
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}_{'E' if lon >= 0 else 'W'}{abs(lon):03d}"


def jrc_tile_id(lon_west, lat_north):
    # Official JRC 10-degree IDs use the west edge and north edge (e.g. 100W_40N).
    lon = int(lon_west)
    lat = int(lat_north)
    return f"{signed_grid(lon, 3)}_{signed_grid(lat, 2)}"


def make_intersections(scenes, tile_size, lat_min=None, lat_max=None, jrc=False):
    entries = []
    scene_tiles = {scene["scene_id"]: set() for scene in scenes}
    tile_scenes = {}
    for scene in scenes:
        xcells = intervals(scene["left"], scene["right"], tile_size)
        ycells = intervals(scene["bottom"], scene["top"], tile_size, lat_min, lat_max)
        for west, east in xcells:
            for south, north in ycells:
                tile_id = jrc_tile_id(west, north) if jrc else dem_tile_id(west, south)
                bounds = (max(scene["left"], west), max(scene["bottom"], south),
                          min(scene["right"], east), min(scene["top"], north))
                if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                    continue
                entries.append({
                    **scene,
                    "source_tile_id": tile_id,
                    "scene_left": scene["left"], "scene_bottom": scene["bottom"],
                    "scene_right": scene["right"], "scene_top": scene["top"],
                    "intersection_left": bounds[0], "intersection_bottom": bounds[1],
                    "intersection_right": bounds[2], "intersection_top": bounds[3],
                    "source_tile_left": west, "source_tile_bottom": south,
                    "source_tile_right": east, "source_tile_top": north,
                })
                scene_tiles[scene["scene_id"]].add(tile_id)
                tile_scenes.setdefault(tile_id, set()).add(scene["scene_id"])
    return entries, scene_tiles, tile_scenes


def check_catalog_reachability(url):
    try:
        data, _ = get_url(url)
        return True, None
    except Exception as exc:  # record the network/catalog issue in the report
        return False, f"{type(exc).__name__}: {exc}"


def dem_query_url(tile_id):
    filter_text = (
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'gridId' "
        f"and att/OData.CSC.StringAttribute/Value eq '{tile_id}') and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'dataset' "
        f"and att/OData.CSC.StringAttribute/Value eq '{DEM_RELEASE}')"
    )
    query = urlencode({"$filter": filter_text, "$top": "100", "$expand": "Attributes"})
    return f"{DEM_OWS}?{query}"


def parse_dem_metadata(payload):
    data = json.loads(payload.decode("utf-8"))
    products = data.get("value", [])
    results = []
    for product in products:
        results.append({
            "catalog_product_name": product.get("Name", ""),
            "catalog_product_id": product.get("Id", ""),
            "catalog_s3_path": product.get("S3Path", ""),
            "catalog_file_size_bytes": (
                product.get("ContentLength") or product.get("ContentLengthBytes")
                or product.get("Size") or ""
            ),
        })
    return results


def query_one_dem_tile(tile_id):
    url = dem_query_url(tile_id)
    try:
        payload, _ = get_url(url)
        return tile_id, "verified_available" if parse_dem_metadata(payload) else "verified_no_product", parse_dem_metadata(payload), ""
    except HTTPError as exc:
        return tile_id, "catalog_query_failed", [], f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:
        return tile_id, "catalog_query_failed", [], f"{type(exc).__name__}: {exc}"


def fetch_dem_catalog(tile_ids):
    if not tile_ids:
        return {}, "no DEM tile intersections"
    # A single short probe avoids hundreds of failed requests when offline.
    first = sorted(tile_ids)[0]
    ok, error = check_catalog_reachability(dem_query_url(first))
    if not ok:
        return {tile: ("unverified_catalog_unreachable", [], error) for tile in tile_ids}, error
    found = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(query_one_dem_tile, tile): tile for tile in sorted(tile_ids)}
        for future in as_completed(futures):
            tile, status, products, error = future.result()
            found[tile] = (status, products, error)
    errors = [value[2] for value in found.values() if value[2]]
    return found, ("; ".join(errors[:5]) if errors else None)


def fetch_jrc_catalog(tile_ids):
    try:
        payload, headers = get_url(JRC_LISTING, timeout=30)
        # This endpoint returns a text file listing, not raster payloads.
        paths = set(line.strip().lstrip("/") for line in payload.decode("utf-8", "replace").splitlines())
        result = {}
        for tile in tile_ids:
            filename = f"occurrence_{tile}_v1_5_2024.tif"
            path = f"download2024/Aggregated/VER1-5/occurrence/{filename}"
            result[tile] = ("verified_available" if path in paths else "verified_not_listed", path if path in paths else "", "")
        return result, None, {"catalog_bytes": len(payload), "catalog_content_length": headers.get("Content-Length", "")}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return {tile: ("unverified_catalog_unreachable", "", error) for tile in tile_ids}, error, {}


def mark_download_dedup(rows, scenes_per_tile):
    seen = set()
    for row in sorted(rows, key=lambda r: (r["source_tile_id"], r["scene_id"])):
        tile = row["source_tile_id"]
        row["scenes_intersecting_tile"] = len(scenes_per_tile[tile])
        row["unique_tile_download"] = tile not in seen
        seen.add(tile)


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def new_output_dir():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_ROOT / stamp
    suffix = 1
    while path.exists():
        path = OUTPUT_ROOT / f"{stamp}_{suffix:02d}"
        suffix += 1
    path.mkdir()
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    args = parser.parse_args()
    scenes = read_scene_rows(args.audit_dir)
    if len(scenes) != 431:
        print(f"Note: audit contains {len(scenes)} usable scenes (expected 431).")

    dem_rows, dem_scene_tiles, dem_tile_scenes = make_intersections(scenes, 1)
    jrc_rows, jrc_scene_tiles, jrc_tile_scenes = make_intersections(
        scenes, 10, lat_min=-50, lat_max=80, jrc=True,
    )
    dem_catalog, dem_error = fetch_dem_catalog(set(dem_tile_scenes))
    jrc_catalog, jrc_error, jrc_catalog_stats = fetch_jrc_catalog(set(jrc_tile_scenes))

    # Keep exact official query/access references in every row; a catalog status
    # is never replaced with an inferred availability status.
    dem_manifest = []
    for row in dem_rows:
        status, products, error = dem_catalog[row["source_tile_id"]]
        product = products[0] if products else {}
        row.update({
            "dataset_version": DEM_RELEASE,
            "catalog_status": status,
            "coverage_status": status,
            "source_url_or_access_method": dem_query_url(row["source_tile_id"]),
            "catalog_source": DEM_DOC,
            "catalog_product_name": product.get("catalog_product_name", ""),
            "catalog_product_id": product.get("catalog_product_id", ""),
            "catalog_s3_path": product.get("catalog_s3_path", ""),
            "source_file_size_bytes": product.get("catalog_file_size_bytes", ""),
            "catalog_error": error,
            "license_access": "CDSE account/access eligibility and license terms apply; attribution required.",
        })
        dem_manifest.append(row)

    jrc_manifest = []
    for row in jrc_rows:
        status, catalog_path, error = jrc_catalog[row["source_tile_id"]]
        row.update({
            "dataset_version": JRC_VERSION,
            "catalog_status": status,
            "coverage_status": status,
            "source_url_or_access_method": JRC_TILE_ROOT + Path(catalog_path).name if catalog_path else JRC_TILE_ROOT,
            "catalog_source": JRC_LISTING,
            "catalog_path": catalog_path,
            "source_file_size_bytes": "",  # listing metadata does not provide size
            "catalog_error": error,
            "license_access": "Free JRC data; cite/attribute EC JRC and Google as specified by JRC.",
            "temporal_warning": "1984-2024 aggregate may include observations after the SAR scene acquisition date.",
        })
        jrc_manifest.append(row)

    mark_download_dedup(dem_manifest, dem_tile_scenes)
    mark_download_dedup(jrc_manifest, jrc_tile_scenes)

    summary_rows = []
    dem_status_counts = {}
    jrc_status_counts = {}
    for tile in dem_tile_scenes:
        status = dem_catalog[tile][0]
        dem_status_counts[status] = dem_status_counts.get(status, 0) + 1
    for tile in jrc_tile_scenes:
        status = jrc_catalog[tile][0]
        jrc_status_counts[status] = jrc_status_counts.get(status, 0) + 1
    for scene in scenes:
        dids = dem_scene_tiles[scene["scene_id"]]
        jids = jrc_scene_tiles[scene["scene_id"]]
        dstatuses = [dem_catalog[t][0] for t in dids]
        jstatuses = [jrc_catalog[t][0] for t in jids]
        summary_rows.append({
            "scene_id": scene["scene_id"], "image_name": scene["image_name"], "split": scene["split"],
            "left": scene["left"], "bottom": scene["bottom"], "right": scene["right"], "top": scene["top"],
            "dem_tile_count": len(dids), "dem_tile_ids": ";".join(sorted(dids)),
            "dem_coverage_status": combine_statuses(dstatuses),
            "jrc_tile_count": len(jids), "jrc_tile_ids": ";".join(sorted(jids)),
            "jrc_coverage_status": combine_statuses(jstatuses),
        })

    output_dir = new_output_dir()
    csv_write(output_dir / "dem_tile_manifest.csv", dem_manifest)
    csv_write(output_dir / "jrc_tile_manifest.csv", jrc_manifest)
    csv_write(output_dir / "scene_coverage_summary.csv", summary_rows)
    summary = {
        "created_local": datetime.now().astimezone().isoformat(),
        "read_only": True,
        "raster_data_downloaded": False,
        "audit_directory": str(args.audit_dir),
        "scene_count": len(scenes),
        "split_counts": {split: sum(row["split"] == split for row in scenes) for split in ("train", "validation", "test")},
        "source_versions": {
            "dem": DEM_RELEASE,
            "jrc": JRC_VERSION,
        },
        "official_catalogs": {
            "dem_tile_index": DEM_INDEX_DOC + " (GLO-30 geocell shapefile); per-cell availability queried via CDSE OData",
            "dem_odata_query_docs": DEM_DOC,
            "jrc_file_listing": JRC_LISTING,
            "jrc_data_access_docs": JRC_DOC,
        },
        "dem_unique_intersecting_tile_count": len(dem_tile_scenes),
        "dem_scene_tile_intersection_count": len(dem_manifest),
        "dem_catalog_status_by_unique_tile": dem_status_counts,
        "dem_catalog_error": dem_error,
        "jrc_unique_intersecting_tile_count": len(jrc_tile_scenes),
        "jrc_scene_tile_intersection_count": len(jrc_manifest),
        "jrc_catalog_status_by_unique_tile": jrc_status_counts,
        "jrc_catalog_error": jrc_error,
        "jrc_catalog_metadata": jrc_catalog_stats,
        "download_size_estimate": {
            "dem": "Not estimated: official catalog response did not expose a verified size for every selected product.",
            "jrc": "Not estimated: official file listing supplies paths but not file sizes.",
        },
        "deduplication": "Manifest rows are per scene/tile intersection; unique_tile_download is true only on the first row for each source_tile_id, so each source tile should be fetched once and reused.",
        "coverage_interpretation": "Tile intersections are computed geometrically from audited scene bounds. verified_available means the tile/package was present in the official catalog response; it does not guarantee valid data at every pixel. unverified statuses are not claims of unavailability.",
        "licensing_access": {
            "dem": "Copernicus DEM free license; CDSE registration/access eligibility may apply; preserve required attribution/citation.",
            "jrc": "JRC states data are freely available; retain required source attribution/citation.",
        },
        "jrc_temporal_warning": "GSW v1.5 aggregate occurrence spans 1984-2024. It may include observations later than the corresponding SAR scene and could create temporal leakage/look-ahead if used as a predictive feature.",
        "output_directory": str(output_dir),
    }
    with (output_dir / "planning_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Scenes: {len(scenes)}; output: {output_dir}")
    print(f"DEM tiles: {len(dem_tile_scenes)} unique; scene intersections: {len(dem_manifest)}; status: {dem_status_counts}")
    print(f"JRC tiles: {len(jrc_tile_scenes)} unique; scene intersections: {len(jrc_manifest)}; status: {jrc_status_counts}")
    if dem_error:
        print(f"DEM catalog limitation: {dem_error}")
    if jrc_error:
        print(f"JRC catalog limitation: {jrc_error}")
    print("No raster data downloaded.")


def combine_statuses(statuses):
    if not statuses:
        return "no_tile_intersection"
    if all(status == "verified_available" for status in statuses):
        return "all_intersecting_tiles_verified_available"
    if any(status == "verified_no_product" or status == "verified_not_listed" for status in statuses):
        return "one_or_more_intersecting_tiles_not_listed"
    if any("unverified" in status or "failed" in status for status in statuses):
        return "catalog_unverified"
    return "mixed_catalog_status"


if __name__ == "__main__":
    main()
