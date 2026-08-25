#!/usr/bin/env python3
"""Build wave-data artifacts from NOAA gfswave grib2 (NODD S3 / NOMADS).

Why this exists: on 2026-08-25 NOAA consolidated the West Coast ERDDAP family
into redirect shims and every griddap query the site depends on landed on one
degraded server that needed ~55s for a 1.5 KB answer. The site 502'd. The
same model runs — gfswave.global.0p25 — publish to NODD S3 within hours of
each cycle, download in fractions of a second, and carry MORE than ERDDAP
ever served us: partitioned swell trains (SWELL/SWPER/SWDIR x3) and surface
wind in the same files. This script turns one model cycle into two compact
npz artifacts the web process can serve from RAM with zero grib dependencies:

  basin.npz  global 2-degree, 3-hourly f000-f384 (129 steps), 14 float16
             fields. Frames 0..56 (f000-f168) are the client's swell-map
             payload; the 16-day tail exists for server-side storm watch.
  tiles/tile_{lat0}_{lon0}.npz
             0.25-degree, hourly f000-f120 (121 steps), 5 float16 fields,
             20-degree core + 2.5-degree halo (101x101 cells) — any
             +-1.5/+-2.0-degree map-forecast bbox around a covered spot fits
             inside exactly one tile. Only tiles containing a shipped spot
             are built.

Decoding happens HERE (CI or a dev machine), never in the web process:
eccodes lives in requirements-dev.txt and the workflow, not requirements.txt.

Grid facts, read from live files rather than assumed: 1440x1440/721, row 0 is
lat +90 (jScansPositively=0), lon 0..359.75, packingType grid_jpeg, missing
value 9999, ~41% of cells missing (land and ice).

Usage:
    python scripts/build_wave_artifacts.py --out wdata          # newest complete cycle
    python scripts/build_wave_artifacts.py --out wdata --cycle 20260825T06Z
"""
import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

S3_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
# Same directory layout, NCEP's own box. The fallback for an S3 outage, not
# for routine use — NOMADS asks for restraint and S3 asks for nothing.
NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"

# (.idx variable, .idx level, artifact key)
BASIN_BANDS = [
    ("HTSGW", "surface", "htsgw"),
    ("PERPW", "surface", "perpw"),
    ("DIRPW", "surface", "dirpw"),
    ("WIND", "surface", "wind"),
    ("WDIR", "surface", "wdir"),
    ("SWELL", "1 in sequence", "sw1h"),
    ("SWPER", "1 in sequence", "sw1p"),
    ("SWDIR", "1 in sequence", "sw1d"),
    ("SWELL", "2 in sequence", "sw2h"),
    ("SWPER", "2 in sequence", "sw2p"),
    ("SWDIR", "2 in sequence", "sw2d"),
    ("SWELL", "3 in sequence", "sw3h"),
    ("SWPER", "3 in sequence", "sw3p"),
    ("SWDIR", "3 in sequence", "sw3d"),
]
TILE_BANDS = [
    ("HTSGW", "surface", "htsgw"),
    ("PERPW", "surface", "perpw"),
    ("DIRPW", "surface", "dirpw"),
    ("WIND", "surface", "wind"),
    ("WDIR", "surface", "wdir"),
]
DIRECTION_KEYS = {"wdir", "dirpw", "sw1d", "sw2d", "sw3d"}
# Wind ships in km/h because that is the client contract (buildVelocityData
# divides by 3.6); converting here keeps the web process arithmetic-free.
KMH_KEYS = {"wind"}

BASIN_STEPS = list(range(0, 385, 3))          # 129 frames
TILE_STEPS = list(range(0, 121))              # 121 frames
# The wire grid is coarser than the analysis grid on purpose. 2 degrees was
# measured at 3.0 MB brotli for the default response against today's 866 KB
# -- and ~28 MB parsed in the browser, which is the exact regression the
# v0.11.69 payload work existed to kill. Storm detection and partition
# sampling keep the 2-degree fields; the client raster gets 3 degrees, the
# resolution the payload budget was proven at.
CLIENT_POOL = 12                              # 0.25 deg x 12 = 3 deg
CLIENT_FRAMES = 57                            # f000-f168, 3-hourly
NI, NJ = 1440, 721
MISSING = 9999.0
POOL = 8                                       # 0.25 deg x 8 = 2 deg blocks
MIN_OCEAN_FRACTION = 0.25
TILE_SIZE_DEG = 20
TILE_HALO_DEG = 2.5


def _http(url, rng=None, timeout=60):
    req = urllib.request.Request(url)
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _exists(url):
    try:
        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=20)
        return True
    except Exception:
        return False


def parse_idx(text):
    """NOAA .idx sidecar -> [(var, level, start, end_exclusive_or_None)].

    Format per line: 'n:start_byte:d=YYYYMMDDHH:VAR:LEVEL:fcst:'.
    The end offset of band N is the start of band N+1; the final band runs to
    end of file, expressed here as None.
    """
    rows = []
    for line in text.strip().splitlines():
        parts = line.split(":")
        if len(parts) < 5:
            continue
        rows.append((parts[3], parts[4], int(parts[1])))
    out = []
    for i, (var, level, start) in enumerate(rows):
        end = rows[i + 1][2] - 1 if i + 1 < len(rows) else None
        out.append((var, level, start, end))
    return out


def wanted_ranges(idx_entries, bands):
    """Byte ranges for the requested bands, keyed by artifact name."""
    out = {}
    for var, level, key in bands:
        for ivar, ilevel, start, end in idx_entries:
            if ivar == var and ilevel == level:
                out[key] = (start, end)
                break
    return out


def decode_band(buf):
    """One grib message -> float32 [720, 1440], SOUTH-up, NaN for missing.

    Row 0 of the file is lat +90; we flip to ascending latitude (the client
    contract) and drop the +90 pole row so the row count pools evenly by 8.
    """
    import eccodes
    h = eccodes.codes_new_from_message(bytes(buf))
    try:
        eccodes.codes_set(h, "missingValue", MISSING)
        values = eccodes.codes_get_values(h).astype(np.float32)
    finally:
        eccodes.codes_release(h)
    grid = values.reshape(NJ, NI)
    grid[grid == MISSING] = np.nan
    grid = grid[::-1]          # ascending latitude: row 0 = -90
    return grid[:-1]           # drop the +90 pole row -> 720 rows


# Full-resolution axes after decode_band's orientation.
FULL_LATS = np.arange(-90.0, 90.0, 0.25, dtype=np.float32)      # 720
FULL_LONS = np.arange(0.0, 360.0, 0.25, dtype=np.float32)       # 1440
BASIN_LATS = FULL_LATS.reshape(-1, POOL).mean(axis=1)           # 90, step 2
BASIN_LONS = FULL_LONS.reshape(-1, POOL).mean(axis=1)           # 180, step 2
CLIENT_LATS = FULL_LATS.reshape(-1, CLIENT_POOL).mean(axis=1)   # 60, step 3
CLIENT_LONS = FULL_LONS.reshape(-1, CLIENT_POOL).mean(axis=1)   # 120, step 3


def pool_basin(grid, key, pool=POOL):
    """Block pooling (8x8 -> 2 deg analysis, 12x12 -> 3 deg client); circular
    mean for directions; blocks that are mostly land pool to NaN rather than
    to the average of their two wet corners."""
    ny, nx = 720 // pool, 1440 // pool
    blocks = grid.reshape(ny, pool, nx, pool)
    ocean = np.isfinite(blocks).mean(axis=(1, 3))
    # All-land blocks make nanmean warn about empty slices; they are about to
    # become NaN via the ocean-fraction mask anyway, so silence is correct.
    import warnings
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if key in DIRECTION_KEYS:
            rad = np.deg2rad(blocks)
            s = np.nanmean(np.sin(rad), axis=(1, 3))
            c = np.nanmean(np.cos(rad), axis=(1, 3))
            pooled = np.rad2deg(np.arctan2(s, c)) % 360.0
        else:
            pooled = np.nanmean(blocks, axis=(1, 3))
    pooled = np.where(ocean >= MIN_OCEAN_FRACTION, pooled, np.nan)
    return pooled.astype(np.float16)


def spot_tiles():
    """Lower-left corners (lat0, lon0, multiples of 20) of every tile that
    contains a shipped spot. lon0 in -180..180."""
    with open(os.path.join(ROOT, "surf_cameras.json")) as f:
        raw = json.load(f)
    entries = raw if isinstance(raw, list) else list(raw.values())[0]
    tiles = set()
    for c in entries:
        lat, lon = float(c["lat"]), float(c["lon"])
        lat0 = int(np.floor(lat / TILE_SIZE_DEG) * TILE_SIZE_DEG)
        lon0 = int(np.floor(lon / TILE_SIZE_DEG) * TILE_SIZE_DEG)
        tiles.add((lat0, lon0))
    return sorted(tiles)


def tile_axes(lat0, lon0):
    lats = np.arange(lat0 - TILE_HALO_DEG,
                     lat0 + TILE_SIZE_DEG + TILE_HALO_DEG + 0.001, 0.25,
                     dtype=np.float32)                     # 101
    lons = np.arange(lon0 - TILE_HALO_DEG,
                     lon0 + TILE_SIZE_DEG + TILE_HALO_DEG + 0.001, 0.25,
                     dtype=np.float32)                     # 101
    return lats, lons


def slice_tile(grid, lat0, lon0):
    """Full-res [720,1440] south-up 0..360 grid -> [101,101] tile slice.

    Longitude columns wrap through 360; latitudes clamp at the poles (a halo
    row past +-90 repeats the edge row rather than inventing data).
    """
    lats, lons = tile_axes(lat0, lon0)
    row_idx = np.clip(np.round((lats + 90.0) / 0.25).astype(int), 0, 719)
    col_idx = (np.round((lons % 360.0) / 0.25).astype(int)) % 1440
    return grid[np.ix_(row_idx, col_idx)]


def find_complete_cycle(base=S3_BASE):
    """Newest cycle whose final file (f384) has landed."""
    now = datetime.now(timezone.utc)
    for hours_back in range(0, 49, 6):
        t = now - timedelta(hours=hours_back)
        cyc = (t.hour // 6) * 6
        day = t.strftime("%Y%m%d")
        url = (f"{base}/gfs.{day}/{cyc:02d}/wave/gridded/"
               f"gfswave.t{cyc:02d}z.global.0p25.f384.grib2.idx")
        if _exists(url):
            return day, cyc
    raise RuntimeError("no complete gfswave cycle found in the last 48h")


def fetch_file_bands(base, day, cyc, fh, bands):
    """One forecast file -> {key: decoded [720,1440] float32 grid}."""
    stem = (f"{base}/gfs.{day}/{cyc:02d}/wave/gridded/"
            f"gfswave.t{cyc:02d}z.global.0p25.f{fh:03d}.grib2")
    idx_entries = parse_idx(_http(stem + ".idx", timeout=45).decode("latin-1"))
    ranges = wanted_ranges(idx_entries, bands)
    missing = [k for _, _, k in bands if k not in ranges]
    if missing:
        raise RuntimeError(f"f{fh:03d}: bands missing from idx: {missing}")
    # One coalesced range from the first wanted byte to the last: the bands
    # sit close together and one request beats a dozen (S3 rounds ~0.35s per
    # request; the extra bytes between bands cost less than the round trips).
    starts = [r[0] for r in ranges.values()]
    ends = [r[1] for r in ranges.values() if r[1] is not None]
    lo = min(starts)
    hi = max(ends) if len(ends) == len(ranges) else None
    blob = _http(stem, rng=(lo, hi if hi is not None else lo + 32 * 1024 * 1024),
                 timeout=90)
    out = {}
    for key, (start, end) in ranges.items():
        seg = blob[start - lo: (end + 1 - lo) if end is not None else len(blob)]
        out[key] = decode_band(seg)
    return out


def assert_float16_tolerance(sample):
    """float16 quantization must stay under the display rounding the client
    receives (GRID_ROUNDING: heights 2dp, periods 1dp, directions integer).
    A representable-value check, not a hope."""
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return
    err = np.abs(finite.astype(np.float16).astype(np.float32) - finite)
    assert float(err.max()) < 0.25, f"float16 error {err.max():.3f} too coarse"


def build(out_dir, day, cyc, base=S3_BASE):
    cycle_iso = f"{day[:4]}-{day[4:6]}-{day[6:]}T{cyc:02d}:00Z"
    cycle_epoch = int(datetime(int(day[:4]), int(day[4:6]), int(day[6:]),
                               cyc, tzinfo=timezone.utc).timestamp())
    tiles = spot_tiles()
    print(f"cycle {cycle_iso}: {len(BASIN_STEPS)} basin frames, "
          f"{len(TILE_STEPS)} tile frames, {len(tiles)} tiles")

    basin = {key: np.full((len(BASIN_STEPS), 90, 180), np.nan, np.float16)
             for _, _, key in BASIN_BANDS}
    client = {f"client_{key}": np.full((CLIENT_FRAMES, 60, 120), np.nan, np.float16)
              for _, _, key in TILE_BANDS}
    tile_data = {t: {key: np.full((len(TILE_STEPS), 101, 101), np.nan, np.float16)
                     for _, _, key in TILE_BANDS} for t in tiles}

    all_steps = sorted(set(BASIN_STEPS) | set(TILE_STEPS))
    for n, fh in enumerate(all_steps):
        in_basin = fh in BASIN_STEPS
        in_tiles = fh in TILE_STEPS
        bands = BASIN_BANDS if in_basin else TILE_BANDS
        grids = fetch_file_bands(base, day, cyc, fh, bands)
        for key, grid in grids.items():
            if key in KMH_KEYS:
                grid = grid * 3.6
            if in_basin and key in basin:
                b_idx = BASIN_STEPS.index(fh)
                basin[key][b_idx] = pool_basin(grid, key)
                if b_idx < CLIENT_FRAMES and f"client_{key}" in client:
                    client[f"client_{key}"][b_idx] = \
                        pool_basin(grid, key, pool=CLIENT_POOL)
            if in_tiles and key in dict((k, 1) for _, _, k in TILE_BANDS):
                for t in tiles:
                    tile_data[t][key][TILE_STEPS.index(fh)] = \
                        slice_tile(grid, *t).astype(np.float16)
        if n % 20 == 0:
            print(f"  f{fh:03d} ({n + 1}/{len(all_steps)})")

    assert_float16_tolerance(basin["htsgw"][0].astype(np.float32))

    os.makedirs(os.path.join(out_dir, "tiles"), exist_ok=True)
    manifest = {"cycle": cycle_iso, "cycle_epoch": cycle_epoch,
                "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "files": {}}

    def _write(rel, arrays):
        path = os.path.join(out_dir, rel)
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        data = buf.getvalue()
        with open(path, "wb") as f:
            f.write(data)
        manifest["files"][rel] = {
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        print(f"  wrote {rel}: {len(data) / 1e6:.1f} MB")

    _write("basin.npz", dict(
        times=(cycle_epoch + np.array(BASIN_STEPS, np.int64) * 3600),
        lats=BASIN_LATS, lons=BASIN_LONS,
        client_lats=CLIENT_LATS, client_lons=CLIENT_LONS,
        **basin, **client))
    for (lat0, lon0) in tiles:
        lats, lons = tile_axes(lat0, lon0)
        _write(f"tiles/tile_{lat0}_{lon0}.npz", dict(
            times=(cycle_epoch + np.array(TILE_STEPS, np.int64) * 3600),
            lats=lats, lons=lons, **tile_data[(lat0, lon0)]))

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest: cycle {cycle_iso}, {len(manifest['files'])} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cycle", help="YYYYMMDDTHHZ; default newest complete")
    ap.add_argument("--base", default=S3_BASE)
    args = ap.parse_args()

    if args.cycle:
        day, cyc = args.cycle[:8], int(args.cycle[9:11])
    else:
        day, cyc = find_complete_cycle(args.base)

    cycle_iso = f"{day[:4]}-{day[4:6]}-{day[6:]}T{cyc:02d}:00Z"
    manifest_path = os.path.join(args.out, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            if json.load(f).get("cycle") == cycle_iso:
                print(f"cycle {cycle_iso} already built — nothing to do")
                return 0
    build(args.out, day, cyc, args.base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
