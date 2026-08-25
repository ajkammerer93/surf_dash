#!/usr/bin/env python3
"""Rebuild ndbc_stations.json from NDBC's active station index.

Why this exists: the station list was a static file, hand-made, and by
2026-08-25 it was six months old. It knew about 74 of the 146 wave-reporting
stations within 120 km of a covered spot -- roughly half. The consequence was
not a slightly worse buoy, it was no buoy at all: BUOY_MAX_DISTANCE_KM is 320
and every Great Lakes spot's nearest KNOWN station was 400-600 km away across
land, so Cleveland, Oswego and Buffalo returned an empty list while live
nearshore buoys sat 1.6 to 4 km offshore.

A list that must be refreshed by hand is a list that goes stale, so this is
wired to a schedule and opens a pull request when the set changes.

Two things are deliberately NOT done here:

  * Stations are not filtered to "reporting waves right now". A buoy pulled for
    winter servicing would be deleted from the list and then not come back on
    its own. The test is whether the realtime2 feed HAS a WVHT column and has
    published at least one real value in the ~45 days it covers -- that is
    capability, which is stable, rather than current state, which is not.

  * activestations.xml is not trusted on its own. It carries no wave flag, and
    709 of its 1,351 entries are 'fixed' land and C-MAN sites. Selecting on
    proximity alone pulls in stations like rprn6 and bufn6, which sit right on
    the beach and have no WVHT column at all -- they would look like a superb
    result and serve nothing. Capability is probed, never inferred.

Usage:
    python scripts/ndbc_station_scan.py            # report the diff, write nothing
    python scripts/ndbc_station_scan.py --write    # rewrite ndbc_stations.json
"""
import argparse
import json
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

ACTIVE_URL = "https://www.ndbc.noaa.gov/activestations.xml"
REALTIME_URL = "https://www.ndbc.noaa.gov/data/realtime2/{}.txt"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIONS_PATH = os.path.join(ROOT, "ndbc_stations.json")

# Every active station is probed, not just those near a curated spot. Scoping
# the probe to the 145 curated spots was the first version and it was wrong:
# users can drop a pin anywhere, and pruning to 400 km of a curated spot
# deleted the open-ocean moorings a custom pin is the only thing that would
# ever reach. ~1,350 requests, once a month, is the cheaper mistake.
MISSING = {"MM", "MM.M", "999.0", "99.0"}


def haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def fetch_active_stations(opener=urllib.request.urlopen):
    root = ET.fromstring(opener(ACTIVE_URL, timeout=60).read())
    out = {}
    for s in root.findall("station"):
        try:
            out[s.get("id")] = {
                "id": s.get("id"),
                "lat": float(s.get("lat")),
                "lon": float(s.get("lon")),
                "name": s.get("name") or s.get("id"),
            }
        except (TypeError, ValueError):
            continue          # entries without usable coordinates
    return out


def reports_waves(station_id, opener=urllib.request.urlopen):
    """Does this station's realtime2 feed carry real wave heights?

    Capability, not current state: any real value anywhere in the file counts,
    so a buoy that is temporarily flatlining is kept.
    """
    try:
        text = opener(REALTIME_URL.format(station_id), timeout=25).read().decode("latin-1")
    except Exception:
        return False
    lines = text.splitlines()
    if len(lines) < 3 or "WVHT" not in lines[0]:
        return False
    try:
        idx = lines[0].replace("#", "").split().index("WVHT")
    except ValueError:
        return False
    for row in lines[2:]:
        fields = row.split()
        if len(fields) > idx and fields[idx] not in MISSING:
            return True
    return False


def build(active, probe=reports_waves, workers=16):
    stations = list(active.values())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        flags = list(pool.map(lambda st: probe(st["id"]), stations))
    keep = [st for st, ok in zip(stations, flags) if ok]
    keep.sort(key=lambda st: st["id"])
    return keep, len(stations)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="rewrite ndbc_stations.json")
    args = ap.parse_args()

    with open(STATIONS_PATH) as f:
        current = {str(s["id"]): s for s in json.load(f)}

    active = fetch_active_stations()
    keep, probed = build(active)
    new = {s["id"]: s for s in keep}

    added = sorted(set(new) - set(current))
    removed = sorted(set(current) - set(new))
    print(f"active stations      : {len(active)}")
    print(f"probed               : {probed}")
    print(f"wave-capable         : {len(new)}")
    print(f"current file         : {len(current)}")
    print(f"added                : {len(added)} {added[:15]}")
    print(f"removed              : {len(removed)} {removed[:15]}")

    if not args.write:
        print("\n(dry run — pass --write to update the file)")
        return 0
    if not new:
        print("refusing to write an empty station list", file=sys.stderr)
        return 1
    # A collapse to a fraction of the previous list means the probe failed
    # (NDBC unreachable, a feed format change), not that the buoys vanished.
    if len(new) < len(current) * 0.5:
        print(f"refusing to write: {len(new)} stations is less than half of "
              f"the current {len(current)} — probe likely failed", file=sys.stderr)
        return 1
    with open(STATIONS_PATH, "w") as f:
        json.dump(keep, f, indent=2)
        f.write("\n")
    print(f"\nwrote {len(keep)} stations to {STATIONS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
