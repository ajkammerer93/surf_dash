#!/usr/bin/env python3
"""Forecast verification pipeline.

Snapshots what the live site actually forecasts at NDBC buoy locations, then
scores those snapshots against the buoy's observed wave height/period once the
valid time has passed. Rolling accuracy stats (bias / MAE / RMSE by station and
lead-time bin) are written to stats.json, which the app reads to render the
public /accuracy page and to bias-correct /api/forecast wave heights.

Data files live on the `verification-data` branch (master is protected, so the
scheduled workflow commits there):
  snapshots.jsonl  one line per (run, station): forecast wave series out to 72h
  pairs.jsonl      one line per scored (forecast, observation) pair
  stats.json       rolling 30-day aggregates consumed by the app

Usage:
  python scripts/forecast_verification.py snapshot --data-dir vdata
  python scripts/forecast_verification.py score --data-dir vdata
  python scripts/forecast_verification.py run --data-dir vdata   # both

The station list comes from data/verification/buoy_pairs.json on master.
Snapshots record the *uncorrected* model output (wave_height_raw when the API
has applied bias correction) so the feedback loop measures raw model error and
the correction stays stable instead of chasing its own tail.
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests

DEFAULT_BASE_URL = "https://freesurfforecast.com"
DEFAULT_PAIRS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "verification", "buoy_pairs.json")

SNAPSHOT_MAX_LEAD_H = 72          # keep forecast points out to 72h lead
SNAPSHOT_STEP_H = 3               # subsample hourly forecast to every 3h
RETENTION_DAYS = 40               # prune snapshots/pairs older than this
STATS_WINDOW_DAYS = 30            # rolling window for stats.json
PAIR_TOLERANCE_MIN = 45           # max |obs time - forecast valid time|
LEAD_BINS = [(0, 24, "0-24"), (24, 48, "24-48"), (48, 72, "48-72")]


def _utcnow():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_station_list(pairs_file):
    with open(pairs_file) as f:
        return json.load(f)["stations"]


def _read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- snapshot

def fetch_forecast_snapshot(base_url, station, timeout=150):
    """Fetch /api/forecast at the buoy's coordinates and reduce it to a
    compact wave series (3-hourly, 0-72h lead)."""
    url = f"{base_url}/api/forecast"
    params = {"lat": station["lat"], "lon": station["lon"]}
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.ok:
                break
        except requests.RequestException as e:
            print(f"  {station['id']}: attempt {attempt + 1} failed: {e}")
        time.sleep(10 * (attempt + 1))
    if resp is None or not resp.ok:
        print(f"  {station['id']}: forecast fetch failed")
        return None

    data = resp.json()
    entries = data.get("forecast") or []
    if not entries or data.get("stale"):
        # A stale-served forecast was issued up to 24h ago; its lead times
        # are unknowable from here, so don't score it.
        print(f"  {station['id']}: no usable forecast (stale={data.get('stale')})")
        return None

    issued = _utcnow()
    times, wh, tp = [], [], []
    for e in entries:
        try:
            t = _parse_iso(e["time"])
        except (KeyError, ValueError):
            continue
        lead_h = (t - issued).total_seconds() / 3600.0
        if lead_h < -0.5 or lead_h > SNAPSHOT_MAX_LEAD_H:
            continue
        if t.minute != 0 or t.hour % SNAPSHOT_STEP_H != 0:
            continue
        # Score the raw model, not the site's bias-corrected value (see
        # module docstring).
        height = e.get("wave_height_raw", e.get("wave_height"))
        if height is None:
            continue
        times.append(_iso(t))
        wh.append(height)
        tp.append(e.get("wave_period"))

    if not times:
        print(f"  {station['id']}: forecast had no wave data at buoy location")
        return None

    return {
        "issued": _iso(issued),
        "station": station["id"],
        "source": data.get("source", "unknown"),
        "times": times,
        "wave_height": wh,
        "wave_period": tp,
    }


def cmd_snapshot(args):
    stations = load_station_list(args.pairs)
    path = os.path.join(args.data_dir, "snapshots.jsonl")
    snapshots = _read_jsonl(path)

    print(f"Snapshotting {len(stations)} stations from {args.base_url}")
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(
            lambda s: fetch_forecast_snapshot(args.base_url, s), stations))
    new = [r for r in results if r]
    print(f"Captured {len(new)}/{len(stations)} snapshots")

    cutoff = _utcnow() - timedelta(days=RETENTION_DAYS)
    kept = [s for s in snapshots
            if _parse_iso(s["issued"]) >= cutoff] + new
    _write_jsonl(path, kept)
    print(f"snapshots.jsonl: {len(kept)} rows")
    return 0 if new else 1


# ------------------------------------------------------------------ score

def fetch_ndbc_series(station_id, timeout=30):
    """Parse an NDBC realtime2 stdmet file into [(utc datetime, wvht_m,
    dpd_s), ...]. Files hold ~45 days, newest first; missing values are MM."""
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        print(f"  {station_id}: NDBC fetch failed: {e}")
        return []
    if not resp.ok:
        print(f"  {station_id}: NDBC HTTP {resp.status_code}")
        return []
    return parse_ndbc_stdmet(resp.text)


def parse_ndbc_stdmet(text):
    lines = text.strip().split("\n")
    if len(lines) < 3:
        return []
    headers = lines[0].replace("#", "").split()
    try:
        idx = {name: headers.index(name)
               for name in ("YY", "MM", "DD", "hh", "mm")}
        wvht_i = headers.index("WVHT")
        dpd_i = headers.index("DPD") if "DPD" in headers else None
    except ValueError:
        return []

    def num(parts, i):
        if i is None or i >= len(parts):
            return None
        v = parts[i]
        if v in ("MM", "MM.M"):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    series = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < len(headers) - 2:
            continue
        try:
            year = int(parts[idx["YY"]])
            year = year + 2000 if year < 100 else year
            dt = datetime(year, int(parts[idx["MM"]]), int(parts[idx["DD"]]),
                          int(parts[idx["hh"]]), int(parts[idx["mm"]]),
                          tzinfo=timezone.utc)
        except ValueError:
            continue
        wvht = num(parts, wvht_i)
        if wvht is None:
            continue
        series.append((dt, wvht, num(parts, dpd_i)))
    return series


def match_observation(series, valid_dt, tolerance_min=PAIR_TOLERANCE_MIN):
    """Nearest observation to valid_dt within tolerance, or None."""
    best, best_diff = None, tolerance_min * 60 + 1
    for dt, wvht, dpd in series:
        diff = abs((dt - valid_dt).total_seconds())
        if diff < best_diff:
            best, best_diff = (wvht, dpd), diff
    return best


def build_new_pairs(snapshots, obs_by_station, existing_keys, now=None):
    """Score every past-valid snapshot point that hasn't been paired yet."""
    now = now or _utcnow()
    new_pairs = []
    for snap in snapshots:
        series = obs_by_station.get(snap["station"])
        if not series:
            continue
        issued = _parse_iso(snap["issued"])
        for i, t_str in enumerate(snap["times"]):
            valid = _parse_iso(t_str)
            if valid > now:
                continue
            key = f"{snap['station']}|{snap['issued']}|{t_str}"
            if key in existing_keys:
                continue
            obs = match_observation(series, valid)
            if obs is None:
                continue
            ob_wh, ob_tp = obs
            lead_h = (valid - issued).total_seconds() / 3600.0
            if lead_h < 0:
                continue
            fc_tp = snap["wave_period"][i] if i < len(snap["wave_period"]) else None
            new_pairs.append({
                "station": snap["station"],
                "issued": snap["issued"],
                "valid": t_str,
                "lead_h": round(lead_h, 1),
                "source": snap.get("source", "unknown"),
                "fc_wh": snap["wave_height"][i],
                "ob_wh": ob_wh,
                "fc_tp": fc_tp,
                "ob_tp": ob_tp,
            })
            existing_keys.add(key)
    return new_pairs


def _bin_label(lead_h):
    for lo, hi, label in LEAD_BINS:
        if lo <= lead_h <= hi and (lead_h < hi or hi == LEAD_BINS[-1][1]):
            return label
    return None


def _aggregate(pairs):
    """bias/MAE/RMSE for wave height plus period bias over a pair list."""
    if not pairs:
        return None
    errs = [p["fc_wh"] - p["ob_wh"] for p in pairs]
    tp_errs = [p["fc_tp"] - p["ob_tp"] for p in pairs
               if p.get("fc_tp") is not None and p.get("ob_tp") is not None]
    agg = {
        "n": len(errs),
        "bias_m": round(sum(errs) / len(errs), 3),
        "mae_m": round(sum(abs(e) for e in errs) / len(errs), 3),
        "rmse_m": round(math.sqrt(sum(e * e for e in errs) / len(errs)), 3),
    }
    if tp_errs:
        agg["period_bias_s"] = round(sum(tp_errs) / len(tp_errs), 2)
        agg["period_n"] = len(tp_errs)
    return agg


def compute_stats(pairs, stations, now=None, window_days=STATS_WINDOW_DAYS):
    now = now or _utcnow()
    cutoff = now - timedelta(days=window_days)
    windowed = [p for p in pairs if _parse_iso(p["valid"]) >= cutoff]

    station_meta = {s["id"]: s for s in stations}
    by_station = {}
    for p in windowed:
        by_station.setdefault(p["station"], []).append(p)

    stations_out = {}
    for sid, plist in sorted(by_station.items()):
        meta = station_meta.get(sid, {})
        bins = {}
        for lo, hi, label in LEAD_BINS:
            agg = _aggregate([p for p in plist if _bin_label(p["lead_h"]) == label])
            if agg:
                bins[label] = agg
        stations_out[sid] = {
            "name": meta.get("name", sid),
            "lat": meta.get("lat"),
            "lon": meta.get("lon"),
            "region": meta.get("region", ""),
            "all": _aggregate(plist),
            "bins": bins,
        }

    overall_bins = {}
    for lo, hi, label in LEAD_BINS:
        agg = _aggregate([p for p in windowed if _bin_label(p["lead_h"]) == label])
        if agg:
            overall_bins[label] = agg

    return {
        "generated": _iso(now),
        "window_days": window_days,
        "n_pairs": len(windowed),
        "overall": {"all": _aggregate(windowed), "bins": overall_bins},
        "stations": stations_out,
    }


def cmd_score(args):
    stations = load_station_list(args.pairs)
    snap_path = os.path.join(args.data_dir, "snapshots.jsonl")
    pairs_path = os.path.join(args.data_dir, "pairs.jsonl")
    stats_path = os.path.join(args.data_dir, "stats.json")

    snapshots = _read_jsonl(snap_path)
    pairs = _read_jsonl(pairs_path)
    existing_keys = {f"{p['station']}|{p['issued']}|{p['valid']}" for p in pairs}

    station_ids = sorted({s["station"] for s in snapshots})
    print(f"Fetching observations for {len(station_ids)} stations")
    obs_by_station = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for sid, series in zip(station_ids,
                               pool.map(fetch_ndbc_series, station_ids)):
            if series:
                obs_by_station[sid] = series

    new_pairs = build_new_pairs(snapshots, obs_by_station, existing_keys)
    print(f"Scored {len(new_pairs)} new pairs")

    cutoff = _utcnow() - timedelta(days=RETENTION_DAYS)
    pairs = [p for p in pairs + new_pairs if _parse_iso(p["valid"]) >= cutoff]
    _write_jsonl(pairs_path, pairs)

    stats = compute_stats(pairs, stations)
    tmp = stats_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(stats, f, indent=1)
        f.write("\n")
    os.replace(tmp, stats_path)
    print(f"pairs.jsonl: {len(pairs)} rows; stats.json: "
          f"{stats['n_pairs']} pairs in {STATS_WINDOW_DAYS}d window")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["snapshot", "score", "run"])
    parser.add_argument("--data-dir", required=True,
                        help="checkout of the verification-data branch")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS_FILE,
                        help="path to buoy_pairs.json")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    if args.command == "snapshot":
        return cmd_snapshot(args)
    if args.command == "score":
        return cmd_score(args)
    # run: snapshot first so this cycle's forecast is banked, then score
    # whatever past snapshots now have observations. A snapshot failure
    # (site down) shouldn't stop scoring.
    snap_rc = cmd_snapshot(args)
    score_rc = cmd_score(args)
    return snap_rc or score_rc


if __name__ == "__main__":
    sys.exit(main())
