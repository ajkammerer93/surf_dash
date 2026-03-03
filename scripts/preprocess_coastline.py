#!/usr/bin/env python3
"""
Offline preprocessing script for Natural Earth coastline + land data.

Downloads Natural Earth 1:50m shapefiles and converts them into a compact
numpy .npz file consumed by app.py at runtime.

Requirements (dev-only, not in requirements.txt):
    pip install pyshp

Usage:
    python scripts/preprocess_coastline.py

Outputs:
    coastline_data.npz  (~2-4 MB)
"""

import os
import io
import zipfile
import urllib.request
import numpy as np

try:
    import shapefile
except ImportError:
    raise SystemExit(
        "pyshp is required: pip install pyshp"
    )

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(SCRIPT_DIR, "ne_50m_cache")

COASTLINE_URL = "https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip"
LAND_URL = "https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip"


def download_and_extract(url, label):
    """Download a zip from url, extract shapefiles into CACHE_DIR, return shapefile path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(url))[0]  # e.g. ne_50m_coastline
    shp_path = os.path.join(CACHE_DIR, basename + ".shp")

    if os.path.exists(shp_path):
        print(f"  {label}: using cached {shp_path}")
        return shp_path

    print(f"  {label}: downloading {url} ...")
    resp = urllib.request.urlopen(url)
    data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(CACHE_DIR)
    print(f"  {label}: extracted to {CACHE_DIR}")
    return shp_path


def process_coastline(shp_path):
    """Extract line segments from coastline shapefile.

    Returns (seg_start, seg_end, seg_mid) each as float32 arrays of shape (N, 2)
    where columns are [lat, lon].
    """
    print("Processing coastline segments...")
    sf = shapefile.Reader(shp_path)
    starts = []
    ends = []
    for shape in sf.shapes():
        pts = shape.points
        for i in range(len(pts) - 1):
            lon0, lat0 = pts[i]
            lon1, lat1 = pts[i + 1]
            # Skip segments that cross the antimeridian (huge lon jump)
            if abs(lon1 - lon0) > 90:
                continue
            starts.append([lat0, lon0])
            ends.append([lat1, lon1])

    seg_start = np.array(starts, dtype=np.float32)
    seg_end = np.array(ends, dtype=np.float32)
    seg_mid = (seg_start + seg_end) / 2.0

    print(f"  {len(starts)} coastline segments extracted")
    return seg_start, seg_end, seg_mid


def process_land(shp_path):
    """Extract land polygons from land shapefile.

    Returns:
        land_vertices: float32 (M, 2) array [lat, lon] of all polygon vertices concatenated
        land_parts_offsets: int32 (P+1,) — start index of each polygon ring in land_vertices
            (last element is total vertex count)
        land_bboxes: float32 (P, 4) — [lat_min, lat_max, lon_min, lon_max] per polygon ring
    """
    print("Processing land polygons...")
    sf = shapefile.Reader(shp_path)
    all_verts = []
    offsets = [0]
    bboxes = []

    for shape in sf.shapes():
        pts = shape.points
        # shapefile parts mark the start index of each ring
        parts = list(shape.parts) + [len(pts)]
        for ring_idx in range(len(parts) - 1):
            ring_start = parts[ring_idx]
            ring_end = parts[ring_idx + 1]
            ring_pts = pts[ring_start:ring_end]
            if len(ring_pts) < 3:
                continue
            lats = [p[1] for p in ring_pts]
            lons = [p[0] for p in ring_pts]
            # Skip polygons that span the antimeridian
            if max(lons) - min(lons) > 180:
                continue
            for p in ring_pts:
                all_verts.append([p[1], p[0]])  # [lat, lon]
            offsets.append(len(all_verts))
            bboxes.append([min(lats), max(lats), min(lons), max(lons)])

    land_vertices = np.array(all_verts, dtype=np.float32)
    land_parts_offsets = np.array(offsets, dtype=np.int32)
    land_bboxes = np.array(bboxes, dtype=np.float32)

    print(f"  {len(bboxes)} land polygon rings, {len(all_verts)} total vertices")
    return land_vertices, land_parts_offsets, land_bboxes


def main():
    print("=== Coastline data preprocessor ===\n")

    coast_shp = download_and_extract(COASTLINE_URL, "Coastline")
    land_shp = download_and_extract(LAND_URL, "Land")

    seg_start, seg_end, seg_mid = process_coastline(coast_shp)
    land_vertices, land_parts_offsets, land_bboxes = process_land(land_shp)

    out_path = os.path.join(PROJECT_ROOT, "coastline_data.npz")
    np.savez_compressed(
        out_path,
        seg_start=seg_start,
        seg_end=seg_end,
        seg_mid=seg_mid,
        land_vertices=land_vertices,
        land_parts_offsets=land_parts_offsets,
        land_bboxes=land_bboxes,
    )
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nWrote {out_path} ({size_mb:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
