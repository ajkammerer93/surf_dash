from flask import Flask, render_template, jsonify, request, Response
from urllib.parse import quote as url_quote
import requests
import re
import numpy as np
import math
import subprocess
import time
import traceback
import os
import json
from datetime import datetime, timedelta
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _retry_request(request_fn, max_retries=2):
    """
    Retry wrapper for HTTP requests.
    Calls request_fn() up to (1 + max_retries) times.
    Retries on 5xx responses, connection errors, and timeouts.
    Does NOT retry on 4xx (client) errors.
    Returns the response on success; re-raises the last exception on failure.
    """
    last_exc = None
    last_resp = None
    for attempt in range(1 + max_retries):
        try:
            resp = request_fn()
            if resp.status_code < 500:
                return resp
            # 5xx — retry
            last_resp = resp
            if attempt < max_retries:
                print(f"  Retry {attempt+1}/{max_retries}: HTTP {resp.status_code}")
                time.sleep(1)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            last_resp = None
            if attempt < max_retries:
                print(f"  Retry {attempt+1}/{max_retries}: {type(exc).__name__}")
                time.sleep(1)
    if last_exc is not None:
        raise last_exc
    return last_resp


# Initialize Flask app
app = Flask(__name__)

def get_version():
    """Read version from VERSION file, falling back to git tag."""
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    try:
        with open(version_file) as f:
            return f.read().strip()
    except FileNotFoundError:
        pass
    try:
        return subprocess.check_output(
            ['git', 'describe', '--tags', '--abbrev=0'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'v0.1.0'

APP_VERSION = get_version()

def _load_cameras():
    """Load surf cameras from surf_cameras.json."""
    cam_file = os.path.join(os.path.dirname(__file__), 'surf_cameras.json')
    try:
        with open(cam_file) as f:
            cams = json.load(f)
        print(f"Loaded {len(cams)} surf cameras")
        return cams
    except Exception as e:
        print(f"Warning: could not load surf_cameras.json: {e}")
        return []

SURFCHEX_CAMERAS = _load_cameras()

def _load_ndbc_stations():
    """Load NDBC buoy stations from ndbc_stations.json."""
    path = os.path.join(os.path.dirname(__file__), 'ndbc_stations.json')
    try:
        with open(path) as f:
            stations = json.load(f)
        print(f"Loaded {len(stations)} NDBC stations")
        return stations
    except Exception as e:
        print(f"Warning: could not load ndbc_stations.json: {e}")
        return []

def _load_cdip_stations():
    """Load CDIP buoy stations from cdip_stations.json."""
    path = os.path.join(os.path.dirname(__file__), 'cdip_stations.json')
    try:
        with open(path) as f:
            stations = json.load(f)
        print(f"Loaded {len(stations)} CDIP stations")
        return stations
    except Exception as e:
        print(f"Warning: could not load cdip_stations.json: {e}")
        return []

NDBC_STATIONS = _load_ndbc_stations()
CDIP_STATIONS = _load_cdip_stations()


def _load_coastline_data():
    """Load preprocessed coastline data from coastline_data.npz."""
    path = os.path.join(os.path.dirname(__file__), 'coastline_data.npz')
    try:
        data = np.load(path)
        result = {
            'seg_start': data['seg_start'],     # (N, 2) [lat, lon]
            'seg_end': data['seg_end'],          # (N, 2) [lat, lon]
            'seg_mid': data['seg_mid'],          # (N, 2) [lat, lon]
            'land_vertices': data['land_vertices'],       # (M, 2)
            'land_parts_offsets': data['land_parts_offsets'],  # (P+1,)
            'land_bboxes': data['land_bboxes'],           # (P, 4) [lat_min, lat_max, lon_min, lon_max]
        }
        print(f"Loaded coastline data: {len(result['seg_start'])} segments, "
              f"{len(result['land_bboxes'])} land polygons")
        return result
    except Exception as e:
        print(f"Warning: could not load coastline_data.npz: {e}")
        return None

COASTLINE_DATA = _load_coastline_data()


def _vectorized_haversine(lat1, lon1, lat2_arr, lon2_arr):
    """Haversine distance from (lat1, lon1) to arrays of (lat2, lon2). Returns km."""
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2_arr)
    lon2_r = np.radians(lon2_arr)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing(lat1, lon1, lat2, lon2):
    """Initial bearing from (lat1,lon1) to (lat2,lon2) in degrees [0, 360)."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlon = lon2_r - lon1_r
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _offset_point(lat, lon, bearing_deg, dist_km):
    """Approximate point offset from (lat, lon) along bearing by dist_km."""
    # Simple flat-earth approximation good enough for ~5-10 km offsets
    bearing_rad = math.radians(bearing_deg)
    dlat = (dist_km / 111.32) * math.cos(bearing_rad)
    dlon = (dist_km / (111.32 * math.cos(math.radians(lat)))) * math.sin(bearing_rad)
    return lat + dlat, lon + dlon


def _point_to_segment_distance(plat, plon, slat1, slon1, slat2, slon2):
    """Approximate distance (km) from point to line segment using flat-earth projection."""
    cos_lat = math.cos(math.radians(plat))
    # Project to km-scale flat coordinates
    px = (plon - slon1) * 111.32 * cos_lat
    py = (plat - slat1) * 111.32
    sx = (slon2 - slon1) * 111.32 * cos_lat
    sy = (slat2 - slat1) * 111.32
    seg_len_sq = sx * sx + sy * sy
    if seg_len_sq < 1e-12:
        return math.sqrt(px * px + py * py)
    t = max(0.0, min(1.0, (px * sx + py * sy) / seg_len_sq))
    dx = px - t * sx
    dy = py - t * sy
    return math.sqrt(dx * dx + dy * dy)


def _point_in_polygon(plat, plon, verts_lat, verts_lon):
    """Ray-casting point-in-polygon test."""
    n = len(verts_lat)
    inside = False
    j = n - 1
    for i in range(n):
        yi, yj = verts_lat[i], verts_lat[j]
        xi, xj = verts_lon[i], verts_lon[j]
        if ((yi > plat) != (yj > plat)) and \
           (plon < (xj - xi) * (plat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _point_in_land(plat, plon, coastline_data):
    """Check if a point is inside any land polygon, using bounding box pre-filtering."""
    bboxes = coastline_data['land_bboxes']
    offsets = coastline_data['land_parts_offsets']
    verts = coastline_data['land_vertices']

    # Vectorized bbox check
    mask = ((bboxes[:, 0] <= plat) & (plat <= bboxes[:, 1]) &
            (bboxes[:, 2] <= plon) & (plon <= bboxes[:, 3]))
    candidates = np.where(mask)[0]

    for idx in candidates:
        start = offsets[idx]
        end = offsets[idx + 1]
        ring = verts[start:end]
        if _point_in_polygon(plat, plon, ring[:, 0], ring[:, 1]):
            return True
    return False


def compute_beach_facing_direction(lat, lon):
    """Compute the compass direction a beach faces at (lat, lon).

    Returns dict with beach_facing_direction, coastline_bearing,
    distance_to_coast_km, confidence — or None if inland / data unavailable.
    """
    if COASTLINE_DATA is None:
        return None

    seg_mid = COASTLINE_DATA['seg_mid']
    seg_start = COASTLINE_DATA['seg_start']
    seg_end = COASTLINE_DATA['seg_end']

    # 1. Coarse filter: bounding box on midpoints within ~2 degrees
    BOX_DEG = 2.0
    mask = ((np.abs(seg_mid[:, 0] - lat) < BOX_DEG) &
            (np.abs(seg_mid[:, 1] - lon) < BOX_DEG))
    candidates = np.where(mask)[0]

    if len(candidates) == 0:
        # Widen search to 5 degrees
        BOX_DEG = 5.0
        mask = ((np.abs(seg_mid[:, 0] - lat) < BOX_DEG) &
                (np.abs(seg_mid[:, 1] - lon) < BOX_DEG))
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            return None  # Truly inland or far from any coast

    # 2. Vectorized haversine to candidate midpoints
    dists = _vectorized_haversine(lat, lon,
                                  seg_mid[candidates, 0],
                                  seg_mid[candidates, 1])

    # Reject if nearest coast is > 200 km
    min_dist = float(np.min(dists))
    if min_dist > 200:
        return None

    # 3. Fine rank: point-to-segment distance for top ~20
    top_k = min(20, len(candidates))
    top_indices = np.argpartition(dists, top_k)[:top_k]

    best_dist = float('inf')
    best_seg_idx = None
    for idx in top_indices:
        ci = candidates[idx]
        d = _point_to_segment_distance(
            lat, lon,
            float(seg_start[ci, 0]), float(seg_start[ci, 1]),
            float(seg_end[ci, 0]), float(seg_end[ci, 1])
        )
        if d < best_dist:
            best_dist = d
            best_seg_idx = ci

    # 4. Tangent bearing along nearest segment
    s0_lat, s0_lon = float(seg_start[best_seg_idx, 0]), float(seg_start[best_seg_idx, 1])
    s1_lat, s1_lon = float(seg_end[best_seg_idx, 0]), float(seg_end[best_seg_idx, 1])
    coastline_bearing = _bearing(s0_lat, s0_lon, s1_lat, s1_lon)

    # 5. Two perpendicular candidates (±90° from coastline)
    perp_a = (coastline_bearing + 90) % 360
    perp_b = (coastline_bearing - 90) % 360

    # 6. Ocean/land disambiguation: sample test points along each perpendicular
    for offset_km in [5, 10, 20]:
        pt_a_lat, pt_a_lon = _offset_point(lat, lon, perp_a, offset_km)
        pt_b_lat, pt_b_lon = _offset_point(lat, lon, perp_b, offset_km)

        a_land = _point_in_land(pt_a_lat, pt_a_lon, COASTLINE_DATA)
        b_land = _point_in_land(pt_b_lat, pt_b_lon, COASTLINE_DATA)

        if a_land and not b_land:
            beach_facing = perp_b
            confidence = 'high' if offset_km <= 10 else 'medium'
            break
        elif b_land and not a_land:
            beach_facing = perp_a
            confidence = 'high' if offset_km <= 10 else 'medium'
            break
        elif not a_land and not b_land:
            # Both ocean (island?) — pick direction away from nearest land bbox center
            # Fallback: use the perpendicular closest to "away from continent center"
            continue
    else:
        # All attempts inconclusive — use heuristic: pick perpendicular pointing
        # more toward equator / open ocean. This handles small islands.
        # If query point is in northern hemisphere, prefer the southward perpendicular
        if lat > 0:
            # Prefer the perpendicular with more southward component
            a_south = math.cos(math.radians(perp_a))  # cos of bearing: 1=N, -1=S
            b_south = math.cos(math.radians(perp_b))
            beach_facing = perp_a if a_south < b_south else perp_b
        else:
            a_north = math.cos(math.radians(perp_a))
            b_north = math.cos(math.radians(perp_b))
            beach_facing = perp_a if a_north > b_north else perp_b
        confidence = 'low'

    return {
        'beach_facing_direction': round(beach_facing, 1),
        'coastline_bearing': round(coastline_bearing, 1),
        'distance_to_coast_km': round(best_dist, 2),
        'confidence': confidence,
    }


# Simple time-based response cache
_cache = {}
CACHE_TTL = 900  # 15 minutes
BASIN_CACHE_TTL = 1800  # 30 minutes (WW3 model updates ~every 6 hours)

def cached(key, fn, ttl=CACHE_TTL):
    """Return cached result if fresh, otherwise call fn() and cache it."""
    now = time.time()
    if key in _cache and now - _cache[key]['time'] < ttl:
        return _cache[key]['data']
    result = fn()
    if result is not None:
        _cache[key] = {'data': result, 'time': now}
    return result

DEFAULT_LAT = 34.42711
DEFAULT_LON = -77.54608

def validate_lat_lon():
    """Parse and validate lat/lon from request args. Returns (lat, lon) or a 400 Response."""
    lat = request.args.get('lat', DEFAULT_LAT, type=float)
    lon = request.args.get('lon', DEFAULT_LON, type=float)
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return jsonify({"error": "lat must be between -90 and 90, lon must be between -180 and 180."}), 400
    return lat, lon

def get_point_weather_data(latitude, longitude):
    """
    Fetches wave and wind forecast for a single point.
    Uses NOMADS GFS-Wave Atlantic 0.16deg if in coverage, falls back to Open-Meteo Marine.
    Enriches result with air and water temperature from Open-Meteo.
    """
    if _in_gfswave_atlantic_coverage(latitude, longitude):
        try:
            result = _get_point_from_nomads(latitude, longitude)
            if result:
                _enrich_with_temperatures(result, latitude, longitude)
                _enrich_with_wind(result, latitude, longitude)
                return result
            print("NOMADS point forecast returned no data, falling back to Open-Meteo Marine")
        except Exception as e:
            print(f"NOMADS point forecast failed, falling back to Open-Meteo Marine: {e}")
            traceback.print_exc()
    result = _get_point_from_open_meteo(latitude, longitude)
    if result:
        _enrich_with_temperatures(result, latitude, longitude)
        _enrich_with_wind(result, latitude, longitude)
    return result


def _enrich_with_temperatures(forecast, latitude, longitude):
    """
    Fetch current air temperature (Open-Meteo Weather) and sea surface
    temperature (Open-Meteo Marine) and add them to the first forecast entry.
    Non-critical — silently skips on failure.
    """
    try:
        # Air temperature
        air_temp = None
        resp = _retry_request(lambda: requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": latitude, "longitude": longitude,
                    "current": "temperature_2m", "timezone": "auto"},
            timeout=10
        ))
        if resp.ok:
            air_temp = resp.json().get("current", {}).get("temperature_2m")

        # Sea surface temperature
        water_temp = None
        resp = _retry_request(lambda: requests.get(
            "https://marine-api.open-meteo.com/v1/marine",
            params={"latitude": latitude, "longitude": longitude,
                    "current": "sea_surface_temperature"},
            timeout=10
        ))
        if resp.ok:
            water_temp = resp.json().get("current", {}).get("sea_surface_temperature")

        if forecast:
            forecast[0]["air_temperature"] = air_temp
            forecast[0]["water_temperature"] = water_temp
            print(f"  Temperatures: air={air_temp}°C, water={water_temp}°C")
    except Exception as e:
        print(f"  Temperature fetch failed (non-critical): {e}")


def _enrich_with_wind(forecast, latitude, longitude):
    """
    Replace wind_speed / wind_direction in the forecast list with Open-Meteo
    hourly data.  Keeps existing (NOMADS / Open-Meteo) values for any hours where
    Open-Meteo data is missing or the request fails entirely.
    Non-critical — silently skips on failure.
    """
    try:
        resp = _retry_request(lambda: requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude, "longitude": longitude,
                "hourly": "wind_speed_10m,wind_direction_10m",
                "wind_speed_unit": "kmh",
                "timezone": "UTC",
                "forecast_days": 7,
            },
            timeout=10,
        ))
        if not resp.ok:
            print(f"  Open-Meteo wind request failed (HTTP {resp.status_code}), keeping original wind")
            return
        om = resp.json().get("hourly", {})
        om_times = om.get("time", [])
        om_ws = om.get("wind_speed_10m", [])
        om_wd = om.get("wind_direction_10m", [])
        # Build time-keyed lookup (Open-Meteo returns "2025-03-01T00:00")
        om_lookup = {}
        for idx, t_str in enumerate(om_times):
            om_lookup[t_str] = idx
        matched = 0
        for entry in forecast:
            # forecast times are "2025-03-01 00:00"; Open-Meteo uses "T" separator
            key = entry["time"].replace(" ", "T")
            if key in om_lookup:
                oi = om_lookup[key]
                if oi < len(om_ws) and om_ws[oi] is not None:
                    entry["wind_speed"] = round(float(om_ws[oi]), 1)
                if oi < len(om_wd) and om_wd[oi] is not None:
                    entry["wind_direction"] = round(float(om_wd[oi]), 1)
                matched += 1
        print(f"  Open-Meteo wind: matched {matched}/{len(forecast)} hours")
    except Exception as e:
        print(f"  Open-Meteo wind fetch failed (non-critical): {e}")


def _get_point_from_open_meteo(latitude, longitude):
    """
    Fetches wave forecast for a single point using Open-Meteo Marine API.
    Returns forecast list with wave data, or None on failure.
    Wind speed/direction are left as None (filled later by _enrich_with_wind).
    """
    try:
        print(f"Open-Meteo Marine point forecast for ({latitude}, {longitude})...")
        resp = _retry_request(lambda: requests.get(
            "https://marine-api.open-meteo.com/v1/marine",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "wave_height,wave_period,wave_direction,wind_wave_height,wind_wave_period,wind_wave_direction",
                "forecast_days": 7,
                "timezone": "UTC",
            },
            timeout=30,
        ))
        if not resp.ok:
            print(f"  Open-Meteo Marine HTTP {resp.status_code}")
            return None

        data = resp.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            print("  Open-Meteo Marine returned no hourly data")
            return None

        wh = hourly.get("wave_height", [])
        wp = hourly.get("wave_period", [])
        wd = hourly.get("wave_direction", [])
        wwh = hourly.get("wind_wave_height", [])
        wwp = hourly.get("wind_wave_period", [])
        wwd = hourly.get("wind_wave_direction", [])

        # Compute sunrise/sunset for each unique date
        sunrise_map = {}
        sunset_map = {}
        for t_str in times:
            date_key = t_str[:10]  # "YYYY-MM-DD"
            if date_key not in sunrise_map:
                dt = datetime.strptime(date_key, "%Y-%m-%d")
                sr, ss = _sunrise_sunset(latitude, longitude, dt.date())
                sunrise_map[date_key] = sr
                sunset_map[date_key] = ss

        def _val(arr, idx):
            if idx < len(arr) and arr[idx] is not None:
                return round(float(arr[idx]), 2)
            return None

        forecast = []
        for i, t_str in enumerate(times):
            date_key = t_str[:10]
            # Open-Meteo times are "YYYY-MM-DDTHH:MM"; convert to space-separated
            time_str = t_str.replace("T", " ")
            forecast.append({
                "time": time_str,
                "wave_height": _val(wh, i),
                "wave_period": _val(wp, i),
                "wave_direction": _val(wd, i),
                "wind_wave_height": _val(wwh, i),
                "wind_wave_period": _val(wwp, i),
                "wind_wave_direction": _val(wwd, i),
                "wind_speed": None,
                "wind_direction": None,
                "sunrise": sunrise_map.get(date_key),
                "sunset": sunset_map.get(date_key),
            })

        print(f"  Open-Meteo Marine: {len(forecast)} hourly entries")
        return forecast

    except Exception as e:
        print(f"Error fetching point forecast from Open-Meteo Marine: {e}")
        traceback.print_exc()
        return None


def _get_point_from_erddap(latitude, longitude):
    """
    Fetches wave and wind forecast for a single point using ERDDAP.
    WW3 for waves, GFS for wind, local computation for sunrise/sunset.

    NOTE: No longer called from the main forecast path (replaced by
    _get_point_from_open_meteo). Kept for potential future use as an
    alternative data source.
    """
    try:
        # Convert longitude to 0-360 for ERDDAP
        lon_360 = longitude % 360

        # ±1° bounding box to find nearest ocean point (WW3 has land masking)
        lat_range = f"({latitude - 1}):({latitude + 1})"
        lon_range = f"({lon_360 - 1}):({lon_360 + 1})"

        # Retry with progressively older start times to handle ERDDAP model update gaps
        wave_json = None
        for hours_back in [6, 12, 24]:
            try:
                now = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours_back)
                t_start = now.isoformat() + "Z"
                time_range = f"({t_start}):(last)"

                # Fetch WW3 wave data
                print(f"Point forecast: fetching WW3 waves for ({latitude}, {longitude}), start={t_start}...")
                wave_json = _fetch_erddap_grid(
                    server="pae-paha.pacioos.hawaii.edu",
                    dataset="ww3_global",
                    variables=["Thgt", "Tper", "Tdir", "whgt", "wper"],
                    time_range=time_range,
                    lat_range=lat_range,
                    lon_range=lon_range,
                    depth=0
                )
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404 and hours_back < 24:
                    print(f"  ERDDAP 404 with start {t_start}, retrying with older start...")
                    continue
                raise
        wave = _parse_erddap_to_grids(wave_json, ["Thgt", "Tper", "Tdir", "whgt", "wper"])

        # Find nearest non-NaN ocean grid point
        wave_lats = np.array(wave['lats'])
        wave_lons = np.array(wave['lons'])
        best_dist = float('inf')
        best_lat_i, best_lon_i = 0, 0
        for li, la in enumerate(wave_lats):
            for loi, lo in enumerate(wave_lons):
                if not np.isnan(wave['grids']['Thgt'][0, li, loi]):
                    lo_180 = lo - 360 if lo > 180 else lo
                    d = haversine_distance(latitude, longitude, la, lo_180)
                    if d < best_dist:
                        best_dist = d
                        best_lat_i, best_lon_i = li, loi
        print(f"  Nearest ocean point: ({wave_lats[best_lat_i]}, {wave_lons[best_lon_i]}), distance: {best_dist:.1f} km")

        # Extract time series for the nearest ocean point
        wave_times = wave['times']
        wave_dts = [_parse_erddap_time(t) for t in wave_times]
        thgt = wave['grids']['Thgt'][:, best_lat_i, best_lon_i]
        tper = wave['grids']['Tper'][:, best_lat_i, best_lon_i]
        tdir = wave['grids']['Tdir'][:, best_lat_i, best_lon_i]
        whgt = wave['grids']['whgt'][:, best_lat_i, best_lon_i]
        wper = wave['grids']['wper'][:, best_lat_i, best_lon_i]

        # Fetch GFS wind data (also retry on 404)
        wind_json = None
        for hours_back in [6, 12, 24]:
            try:
                now = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours_back)
                t_start = now.isoformat() + "Z"
                time_range = f"({t_start}):(last)"

                print(f"Point forecast: fetching GFS wind, start={t_start}...")
                wind_json = _fetch_erddap_grid(
                    server="coastwatch.pfeg.noaa.gov",
                    dataset="NCEP_Global_Best",
                    variables=["ugrd10m", "vgrd10m"],
                    time_range=time_range,
                    lat_range=lat_range,
                    lon_range=lon_range,
                    depth=None
                )
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404 and hours_back < 24:
                    print(f"  ERDDAP 404 for wind with start {t_start}, retrying with older start...")
                    continue
                raise
        wind = _parse_erddap_to_grids(wind_json, ["ugrd10m", "vgrd10m"])

        # Find nearest wind grid point
        wind_lats = np.array(wind['lats'])
        wind_lons = np.array(wind['lons'])
        wlat_i = int(np.argmin(np.abs(wind_lats - latitude)))
        wlon_i = int(np.argmin(np.abs(wind_lons - lon_360)))

        # Interpolate 3-hourly wind to hourly wave time steps
        wind_dts = [_parse_erddap_time(t) for t in wind['times']]
        wind_u = wind['grids']['ugrd10m'][:, wlat_i, wlon_i]
        wind_v = wind['grids']['vgrd10m'][:, wlat_i, wlon_i]

        wind_secs = np.array([(wdt - wind_dts[0]).total_seconds() for wdt in wind_dts])
        wind_speed_hourly = []
        wind_dir_hourly = []
        for wdt in wave_dts:
            t_sec = (wdt - wind_dts[0]).total_seconds()
            if t_sec <= wind_secs[0]:
                u, v = float(wind_u[0]), float(wind_v[0])
            elif t_sec >= wind_secs[-1]:
                u, v = float(wind_u[-1]), float(wind_v[-1])
            else:
                idx = max(0, min(int(np.searchsorted(wind_secs, t_sec)) - 1, len(wind_secs) - 2))
                dt = wind_secs[idx + 1] - wind_secs[idx]
                w = (t_sec - wind_secs[idx]) / dt if dt > 0 else 0.0
                u = float(wind_u[idx] * (1 - w) + wind_u[idx + 1] * w)
                v = float(wind_v[idx] * (1 - w) + wind_v[idx + 1] * w)
            speed = math.sqrt(u**2 + v**2) * 3.6  # m/s to km/h
            direction = (270 - math.degrees(math.atan2(v, u))) % 360
            wind_speed_hourly.append(round(speed, 1))
            wind_dir_hourly.append(round(direction, 1))

        # Compute sunrise/sunset for each day in forecast
        sunrise_map = {}
        sunset_map = {}
        for wdt in wave_dts:
            date_key = wdt.strftime('%Y-%m-%d')
            if date_key not in sunrise_map:
                sr, ss = _sunrise_sunset(latitude, longitude, wdt.date())
                sunrise_map[date_key] = sr
                sunset_map[date_key] = ss

        # Build output in same format frontend expects
        forecast = []
        for i, wdt in enumerate(wave_dts):
            date_key = wdt.strftime('%Y-%m-%d')
            forecast.append({
                "time": wdt.strftime('%Y-%m-%d %H:%M'),
                "wave_height": float(thgt[i]) if not np.isnan(thgt[i]) else None,
                "wave_period": float(tper[i]) if not np.isnan(tper[i]) else None,
                "wave_direction": float(tdir[i]) if not np.isnan(tdir[i]) else None,
                "wind_wave_height": float(whgt[i]) if not np.isnan(whgt[i]) else None,
                "wind_wave_period": float(wper[i]) if not np.isnan(wper[i]) else None,
                "wind_speed": wind_speed_hourly[i],
                "wind_direction": wind_dir_hourly[i],
                "sunrise": sunrise_map.get(date_key),
                "sunset": sunset_map.get(date_key),
            })

        return forecast

    except Exception as e:
        print(f"Error fetching point forecast from ERDDAP: {e}")
        traceback.print_exc()
        return None

def _fetch_erddap_grid(server, dataset, variables, time_range, lat_range, lon_range, depth=None):
    """
    Fetch gridded data from an ERDDAP griddap server in JSON format.
    Returns the parsed JSON response containing a table of rows.
    """
    var_parts = []
    for var in variables:
        dims = f"[{time_range}]"
        if depth is not None:
            dims += f"[({depth})]"
        dims += f"[{lat_range}]"
        dims += f"[{lon_range}]"
        var_parts.append(f"{var}{dims}")

    query = ",".join(var_parts)
    url = f"https://{server}/erddap/griddap/{dataset}.json?{query}"
    print(f"  ERDDAP request: {url[:150]}...")

    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return response.json()

def _parse_erddap_to_grids(erddap_json, variable_names):
    """
    Parse ERDDAP JSON table response into structured numpy grid arrays.
    Returns dict with 'times', 'lats', 'lons', and 'grids' (variable name → 3D numpy array).
    """
    table = erddap_json['table']
    col_names = table['columnNames']
    rows = table['rows']

    time_col = col_names.index('time')
    lat_col = col_names.index('latitude')
    lon_col = col_names.index('longitude')
    var_cols = {var: col_names.index(var) for var in variable_names}

    # Extract unique sorted dimension values
    times = sorted(set(row[time_col] for row in rows))
    lats = sorted(set(row[lat_col] for row in rows))
    lons = sorted(set(row[lon_col] for row in rows))

    time_idx = {t: i for i, t in enumerate(times)}
    lat_idx = {la: i for i, la in enumerate(lats)}
    lon_idx = {lo: i for i, lo in enumerate(lons)}

    grids = {}
    for var in variable_names:
        grids[var] = np.full((len(times), len(lats), len(lons)), np.nan)

    for row in rows:
        ti = time_idx[row[time_col]]
        lai = lat_idx[row[lat_col]]
        loi = lon_idx[row[lon_col]]
        for var in variable_names:
            val = row[var_cols[var]]
            if val is not None:
                grids[var][ti, lai, loi] = val

    return {'times': times, 'lats': lats, 'lons': lons, 'grids': grids}

def _parse_erddap_time(t):
    """Parse ERDDAP timestamp (may or may not have trailing Z, fractional seconds, etc.)."""
    for fmt in ('%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(t.replace('Z', '+00:00').replace('+00:00', ''))

def _sunrise_sunset(lat, lon, date):
    """
    Compute sunrise and sunset times using NOAA solar equations.
    Returns (sunrise_iso, sunset_iso) strings in 'YYYY-MM-DDTHH:MM' format (UTC).
    Returns (None, None) for polar day/night.
    """
    n = date.timetuple().tm_yday
    gamma = 2 * math.pi / 365 * (n - 1)

    # Equation of time (minutes)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma)
             - 0.032077 * math.sin(gamma)
             - 0.014615 * math.cos(2 * gamma)
             - 0.040849 * math.sin(2 * gamma))

    # Solar declination (radians)
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))

    lat_rad = math.radians(lat)

    # Hour angle
    cos_ha = (math.cos(math.radians(90.833)) / (math.cos(lat_rad) * math.cos(decl))
              - math.tan(lat_rad) * math.tan(decl))

    if cos_ha > 1 or cos_ha < -1:
        return None, None  # Polar night or day

    ha = math.degrees(math.acos(cos_ha))

    # Sunrise and sunset in minutes from midnight UTC
    sunrise_min = 720 - 4 * (lon + ha) - eqtime
    sunset_min = 720 - 4 * (lon - ha) - eqtime

    def min_to_iso(minutes):
        minutes = minutes % 1440
        h = int(minutes // 60)
        m = int(minutes % 60)
        return f"{date.isoformat()}T{h:02d}:{m:02d}"

    return min_to_iso(sunrise_min), min_to_iso(sunset_min)

def _interpolate_wind_to_hourly(wind_parsed, target_times_dt, target_lats, target_lons_360):
    """
    Interpolate 3-hourly GFS U/V wind data to hourly time steps.
    Returns (wind_speed_grid, wind_dir_grid) as numpy arrays.
    Wind speed in km/h, direction in meteorological degrees.
    """
    num_times = len(target_times_dt)
    num_lats = len(target_lats)
    num_lons = len(target_lons_360)

    wind_speed_out = np.zeros((num_times, num_lats, num_lons))
    wind_dir_out = np.zeros((num_times, num_lats, num_lons))

    if not wind_parsed or not wind_parsed['times']:
        return wind_speed_out, wind_dir_out

    wind_dts = [_parse_erddap_time(t) for t in wind_parsed['times']]
    wind_u = wind_parsed['grids']['ugrd10m']
    wind_v = wind_parsed['grids']['vgrd10m']

    # Spatially align wind grid to target grid if needed
    if len(wind_parsed['lats']) == num_lats and len(wind_parsed['lons']) == num_lons:
        u_aligned = wind_u
        v_aligned = wind_v
    else:
        wind_lats_arr = np.array(wind_parsed['lats'])
        wind_lons_arr = np.array(wind_parsed['lons'])
        nn_lat = np.array([np.argmin(np.abs(wind_lats_arr - la)) for la in target_lats])
        nn_lon = np.array([np.argmin(np.abs(wind_lons_arr - lo)) for lo in target_lons_360])
        lat_mesh, lon_mesh = np.meshgrid(nn_lat, nn_lon, indexing='ij')
        u_aligned = np.array([wind_u[t][lat_mesh, lon_mesh] for t in range(len(wind_dts))])
        v_aligned = np.array([wind_v[t][lat_mesh, lon_mesh] for t in range(len(wind_dts))])

    # Linearly interpolate 3-hourly U/V to each hourly time step
    wind_secs = np.array([(wdt - wind_dts[0]).total_seconds() for wdt in wind_dts])
    for ti, tdt in enumerate(target_times_dt):
        t_sec = (tdt - wind_dts[0]).total_seconds()
        if t_sec <= wind_secs[0]:
            u_interp, v_interp = u_aligned[0], v_aligned[0]
        elif t_sec >= wind_secs[-1]:
            u_interp, v_interp = u_aligned[-1], v_aligned[-1]
        else:
            idx = max(0, min(int(np.searchsorted(wind_secs, t_sec)) - 1, len(wind_secs) - 2))
            dt = wind_secs[idx + 1] - wind_secs[idx]
            w = (t_sec - wind_secs[idx]) / dt if dt > 0 else 0.0
            u_interp = u_aligned[idx] * (1 - w) + u_aligned[idx + 1] * w
            v_interp = v_aligned[idx] * (1 - w) + v_aligned[idx + 1] * w

        wind_speed_out[ti] = np.sqrt(u_interp**2 + v_interp**2) * 3.6  # m/s → km/h
        wind_dir_out[ti] = (270 - np.degrees(np.arctan2(v_interp, u_interp))) % 360

    return wind_speed_out, wind_dir_out

# ---------------------------------------------------------------------------
# GFS-Wave Atlantic 0.16° via NOMADS OPeNDAP
# ---------------------------------------------------------------------------
NOMADS_BASE = "https://nomads.ncep.noaa.gov/dods/wave/gfswave"
GFSWAVE_ATL_LAT_MIN = 0.0
GFSWAVE_ATL_LAT_MAX = 55.0
GFSWAVE_ATL_LON_MIN = 260.0   # -100°W in 0-360
GFSWAVE_ATL_LON_MAX = 310.0   # -50°W in 0-360
GFSWAVE_ATL_RES = 1.0 / 6.0   # 0.16667°
NOMADS_CYCLE_CACHE_TTL = 1800  # 30 min

def _in_gfswave_atlantic_coverage(lat, lon):
    """Return True if lat/lon is inside the GFS-Wave Atlantic 0.16deg domain."""
    lon_360 = lon % 360
    return (GFSWAVE_ATL_LAT_MIN <= lat <= GFSWAVE_ATL_LAT_MAX and
            GFSWAVE_ATL_LON_MIN <= lon_360 <= GFSWAVE_ATL_LON_MAX)

def _nomads_lat_index(lat):
    """Convert latitude to 0-based grid index in GFS-Wave Atlantic."""
    return round((lat - GFSWAVE_ATL_LAT_MIN) / GFSWAVE_ATL_RES)

def _nomads_lon_index(lon_360):
    """Convert longitude (0-360) to 0-based grid index in GFS-Wave Atlantic."""
    return round((lon_360 - GFSWAVE_ATL_LON_MIN) / GFSWAVE_ATL_RES)

def _find_latest_nomads_cycle():
    """
    Find the latest available NOMADS GFS-Wave Atlantic cycle URL.
    Tries today's cycles (newest first), then yesterday's.
    Result is cached for 30 minutes.
    """
    cache_key = "nomads_cycle"
    now_ts = time.time()
    if cache_key in _cache and now_ts - _cache[cache_key]['time'] < NOMADS_CYCLE_CACHE_TTL:
        return _cache[cache_key]['data']

    now = datetime.utcnow()
    yesterday = now - timedelta(days=1)

    for day in [now, yesterday]:
        date_str = day.strftime('%Y%m%d')
        for cycle in ['18', '12', '06', '00']:
            # Skip obviously future cycles
            if day.date() == now.date() and int(cycle) > now.hour:
                continue
            url = f"{NOMADS_BASE}/{date_str}/gfswave.atlocn.0p16_{cycle}z"
            try:
                test_url = f"{url}.ascii?time[0:0]"
                resp = requests.get(test_url, timeout=10)
                if resp.status_code == 200 and resp.headers.get('content-type', '').startswith('text/plain'):
                    print(f"NOMADS cycle found: {date_str}/{cycle}z")
                    _cache[cache_key] = {'data': url, 'time': now_ts}
                    return url
            except Exception:
                continue

    print("No NOMADS cycle available")
    # Cache the failure for 5 minutes to avoid hammering NOMADS when it's down
    _cache[cache_key] = {'data': None, 'time': now_ts - (NOMADS_CYCLE_CACHE_TTL - 300)}
    return None

def _fetch_nomads_opendap(base_url, variables, time_slice, lat_slice, lon_slice):
    """
    Fetch data from NOMADS OPeNDAP ASCII interface.
    Slices use OPeNDAP syntax: "[start:stride:end]" (0-based, inclusive).
    Returns parsed dict with 'times', 'lats', 'lons', 'grids'.
    """
    parts = [f"{var}{time_slice}{lat_slice}{lon_slice}" for var in variables]
    constraint = ",".join(parts)
    url = f"{base_url}.ascii?{constraint}"
    print(f"  NOMADS request: {url[:180]}...")

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return _parse_opendap_ascii(resp.text, variables)

def _parse_opendap_ascii(text, variable_names):
    """
    Parse NOMADS OPeNDAP ASCII response into structured numpy grids.
    Returns dict with 'times' (list of datetime), 'lats', 'lons',
    and 'grids' (variable name -> 3D numpy array [time, lat, lon]).

    NOMADS ASCII format (no separator, dimensions repeat per variable):
        htsgwsfc, [2][3][3]
        [0][0], 9.999E20, 0.88, 1.05
        ...
        time, [2]
        739668.5, 739668.625
        lat, [3]
        34.333, 34.500, 34.667
        lon, [3]
        282.333, 282.500, 282.667
        perpwsfc, [2][3][3]
        ...
    """
    lines = text.strip().split('\n')

    raw_grids = {}
    dims = {}
    current_type = None   # 'var' or 'dim'
    current_name = None
    current_data = []
    var_set = set(variable_names)
    dim_names = {'time', 'lat', 'lon', 'latitude', 'longitude'}

    def _save():
        """Save accumulated data for the current section."""
        if current_name is None or not current_data:
            return
        if current_type == 'var':
            raw_grids[current_name] = current_data
        elif current_type == 'dim':
            dim_key = current_name.replace('latitude', 'lat').replace('longitude', 'lon')
            dims[dim_key] = current_data

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        comma_pos = stripped.find(',')
        if comma_pos > 0:
            name_part = stripped[:comma_pos].strip()
            rest = stripped[comma_pos + 1:].strip()

            # Variable header: "htsgwsfc, [2][3][3]" — multiple bracket groups
            if name_part in var_set and rest.startswith('[') and '][' in rest:
                _save()
                current_type = 'var'
                current_name = name_part
                current_data = []
                continue

            # Dimension header: "time, [2]" — single bracket group like [N]
            if name_part in dim_names and rest.startswith('[') and rest.endswith(']') and rest[1:-1].isdigit():
                _save()
                current_type = 'dim'
                current_name = name_part
                current_data = []
                continue

        # Data lines
        if current_type == 'var':
            # "[t][lat], val, val, val, ..."
            bracket_end = stripped.rfind(']')
            if bracket_end >= 0:
                vals_str = stripped[bracket_end + 1:].lstrip(',').strip()
            else:
                vals_str = stripped
            if vals_str:
                for v in vals_str.split(','):
                    v = v.strip()
                    if v:
                        current_data.append(float(v))
        elif current_type == 'dim':
            for v in stripped.rstrip(',').split(','):
                v = v.strip()
                if v:
                    current_data.append(float(v))

    _save()  # Save final section

    times_raw = dims.get('time', [])
    lats = dims.get('lat', [])
    lons = dims.get('lon', [])
    nt, nla, nlo = len(times_raw), len(lats), len(lons)
    expected = nt * nla * nlo

    # Convert NOMADS time (days since 0001-01-01 proleptic Gregorian) to datetime
    time_dts = []
    for t in times_raw:
        ordinal = int(t)
        frac = t - ordinal
        time_dts.append(datetime.fromordinal(ordinal) + timedelta(seconds=round(frac * 86400)))

    # Reshape variable data into 3D grids
    result_grids = {}
    for var in variable_names:
        if var in raw_grids and len(raw_grids[var]) == expected:
            arr = np.array(raw_grids[var])
            arr[arr > 9.99e20] = np.nan  # Replace fill values
            result_grids[var] = arr.reshape((nt, nla, nlo))
        else:
            if var in raw_grids:
                print(f"  Warning: {var} has {len(raw_grids[var])} values, expected {expected}")
            result_grids[var] = np.full((nt, nla, nlo), np.nan)

    return {'times': time_dts, 'lats': lats, 'lons': lons, 'grids': result_grids}

def _get_point_from_nomads(lat, lon):
    """
    Fetch point forecast from NOMADS GFS-Wave Atlantic 0.16deg.
    Returns forecast list in the same format as _get_point_from_open_meteo.
    """
    base_url = _find_latest_nomads_cycle()
    if not base_url:
        return None

    lon_360 = lon % 360

    # +/-1 degree bounding box in grid indices
    lat_lo = max(0, _nomads_lat_index(lat - 1))
    lat_hi = _nomads_lat_index(min(lat + 1, GFSWAVE_ATL_LAT_MAX))
    lon_lo = max(0, _nomads_lon_index(max(lon_360 - 1, GFSWAVE_ATL_LON_MIN)))
    lon_hi = _nomads_lon_index(min(lon_360 + 1, GFSWAVE_ATL_LON_MAX))

    # First 56 time steps = 7 days at 3-hourly
    time_slice = "[0:1:55]"
    lat_slice = f"[{lat_lo}:1:{lat_hi}]"
    lon_slice = f"[{lon_lo}:1:{lon_hi}]"

    variables = ["htsgwsfc", "perpwsfc", "dirpwsfc", "wvhgtsfc", "wvpersfc"]

    print(f"Point forecast: fetching NOMADS GFS-Wave for ({lat}, {lon})...")
    data = _fetch_nomads_opendap(base_url, variables, time_slice, lat_slice, lon_slice)

    # Find nearest non-NaN ocean grid point
    lats_arr = np.array(data['lats'])
    lons_arr = np.array(data['lons'])
    best_dist = float('inf')
    best_li, best_loi = 0, 0
    for li, la in enumerate(lats_arr):
        for loi, lo in enumerate(lons_arr):
            if not np.isnan(data['grids']['htsgwsfc'][0, li, loi]):
                lo_180 = lo - 360 if lo > 180 else lo
                d = haversine_distance(lat, lon, la, lo_180)
                if d < best_dist:
                    best_dist = d
                    best_li, best_loi = li, loi

    if best_dist == float('inf'):
        print("  No ocean grid points found in NOMADS data")
        return None

    print(f"  Nearest ocean point: ({lats_arr[best_li]:.2f}, {lons_arr[best_loi]:.2f}), distance: {best_dist:.1f} km")

    # Extract time series at nearest point
    time_dts = data['times']
    htsgw = data['grids']['htsgwsfc'][:, best_li, best_loi]
    perpw = data['grids']['perpwsfc'][:, best_li, best_loi]
    dirpw = data['grids']['dirpwsfc'][:, best_li, best_loi]
    wvhgt = data['grids']['wvhgtsfc'][:, best_li, best_loi]
    wvper = data['grids']['wvpersfc'][:, best_li, best_loi]

    # Interpolate 3-hourly to hourly
    src_secs = np.array([(t - time_dts[0]).total_seconds() for t in time_dts])
    hourly_dts = []
    dt = time_dts[0]
    while dt <= time_dts[-1]:
        hourly_dts.append(dt)
        dt += timedelta(hours=1)
    tgt_secs = np.array([(t - time_dts[0]).total_seconds() for t in hourly_dts])

    htsgw_h = np.interp(tgt_secs, src_secs, htsgw)
    perpw_h = np.interp(tgt_secs, src_secs, perpw)
    dirpw_h = np.interp(tgt_secs, src_secs, dirpw)
    wvhgt_h = np.interp(tgt_secs, src_secs, wvhgt)
    wvper_h = np.interp(tgt_secs, src_secs, wvper)

    # Compute sunrise/sunset for each day
    sunrise_map = {}
    sunset_map = {}
    for wdt in hourly_dts:
        date_key = wdt.strftime('%Y-%m-%d')
        if date_key not in sunrise_map:
            sr, ss = _sunrise_sunset(lat, lon, wdt.date())
            sunrise_map[date_key] = sr
            sunset_map[date_key] = ss

    # Build output
    forecast = []
    for i, wdt in enumerate(hourly_dts):
        date_key = wdt.strftime('%Y-%m-%d')
        forecast.append({
            "time": wdt.strftime('%Y-%m-%d %H:%M'),
            "wave_height": round(float(htsgw_h[i]), 2) if not np.isnan(htsgw_h[i]) else None,
            "wave_period": round(float(perpw_h[i]), 1) if not np.isnan(perpw_h[i]) else None,
            "wave_direction": round(float(dirpw_h[i]), 1) if not np.isnan(dirpw_h[i]) else None,
            "wind_wave_height": round(float(wvhgt_h[i]), 2) if not np.isnan(wvhgt_h[i]) else None,
            "wind_wave_period": round(float(wvper_h[i]), 1) if not np.isnan(wvper_h[i]) else None,
            "wind_speed": None,
            "wind_direction": None,
            "sunrise": sunrise_map.get(date_key),
            "sunset": sunset_map.get(date_key),
        })

    print(f"  NOMADS point forecast: {len(forecast)} hourly steps")
    return forecast

def _fill_nan_nearest(grid_2d, max_iterations=None):
    """Fill NaN values in a 2D grid with nearest non-NaN value via iterative neighbor expansion."""
    filled = grid_2d.copy()
    if max_iterations is None:
        max_iterations = max(filled.shape)
    for _ in range(max_iterations):
        if not np.any(np.isnan(filled)):
            break
        padded = np.pad(filled, 1, constant_values=np.nan)
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            shifted = padded[1+dy:1+dy+filled.shape[0], 1+dx:1+dx+filled.shape[1]]
            mask = np.isnan(filled) & ~np.isnan(shifted)
            filled[mask] = shifted[mask]
    return filled

def _get_grid_from_nomads(lat_min, lat_max, lon_min, lon_max):
    """
    Fetch gridded data from NOMADS GFS-Wave Atlantic 0.16deg.
    Returns dict in the same format as _get_grid_from_erddap.
    """
    base_url = _find_latest_nomads_cycle()
    if not base_url:
        return None

    lon_min_360 = lon_min % 360
    lon_max_360 = lon_max % 360

    # Grid indices
    lat_lo = max(0, _nomads_lat_index(lat_min))
    lat_hi = _nomads_lat_index(min(lat_max, GFSWAVE_ATL_LAT_MAX))
    lon_lo = max(0, _nomads_lon_index(max(lon_min_360, GFSWAVE_ATL_LON_MIN)))
    lon_hi = _nomads_lon_index(min(lon_max_360, GFSWAVE_ATL_LON_MAX))

    # First 48 time steps = 6 days at 3-hourly
    time_slice = "[0:1:47]"
    lat_slice = f"[{lat_lo}:1:{lat_hi}]"
    lon_slice = f"[{lon_lo}:1:{lon_hi}]"

    # Fetch waves + wind from NOMADS (wind U/V at same 0.16° resolution)
    variables = ["htsgwsfc", "perpwsfc", "dirpwsfc", "ugrdsfc", "vgrdsfc"]

    print(f"Grid forecast: fetching NOMADS GFS-Wave for ({lat_min},{lon_min}) to ({lat_max},{lon_max})...")
    data = _fetch_nomads_opendap(base_url, variables, time_slice, lat_slice, lon_slice)
    print(f"  NOMADS grid: {len(data['times'])} times, {len(data['lats'])}x{len(data['lons'])} grid")

    # Convert lons from 0-360 to -180..180
    lons = [lo - 360 if lo > 180 else lo for lo in data['lons']]
    lons_360 = data['lons']
    wave_dts = data['times']  # already datetime objects

    # Fill coastal wave gaps (limited iterations to smooth model coastline)
    wave_height = data['grids']['htsgwsfc'].copy()
    wave_period = data['grids']['perpwsfc'].copy()
    wave_dir = data['grids']['dirpwsfc'].copy()
    for t in range(wave_height.shape[0]):
        wave_height[t] = _fill_nan_nearest(wave_height[t], max_iterations=3)
        wave_period[t] = _fill_nan_nearest(wave_period[t], max_iterations=3)
        wave_dir[t] = _fill_nan_nearest(wave_dir[t], max_iterations=3)

    # Compute wind from NOMADS U/V (same 0.16° grid as waves)
    # Wind blows over land too — fill all NaN so arrows appear inland
    ugrd = data['grids']['ugrdsfc'].copy()
    vgrd = data['grids']['vgrdsfc'].copy()
    for t in range(ugrd.shape[0]):
        ugrd[t] = _fill_nan_nearest(ugrd[t])
        vgrd[t] = _fill_nan_nearest(vgrd[t])
    wind_speed = np.sqrt(ugrd**2 + vgrd**2) * 3.6  # m/s -> km/h
    wind_dir = (270 - np.degrees(np.arctan2(vgrd, ugrd))) % 360

    # Detect stale wind: NOMADS repeats the last valid wind grid for later timesteps.
    # Require 2+ consecutive identical grids before declaring staleness.
    consecutive_same = 0
    stale_start = None
    for t in range(1, wind_speed.shape[0]):
        if np.allclose(wind_speed[t], wind_speed[t - 1], atol=0.01) and \
           np.allclose(wind_dir[t], wind_dir[t - 1], atol=0.01):
            consecutive_same += 1
            if consecutive_same >= 2 and stale_start is None:
                stale_start = t - 1  # first stale timestep
        else:
            consecutive_same = 0
    if stale_start is not None:
        print(f"  Wind staleness detected at timestep {stale_start}/{wind_speed.shape[0]}, zeroing stale wind data")
        wind_speed[stale_start:] = 0.0
        wind_dir[stale_start:] = 0.0

    # Replace remaining NaN with 0 for JSON serialization
    wave_height = np.nan_to_num(wave_height, nan=0.0)
    wave_period = np.nan_to_num(wave_period, nan=0.0)
    wave_dir = np.nan_to_num(wave_dir, nan=0.0)
    wind_speed = np.nan_to_num(wind_speed, nan=0.0)
    wind_dir = np.nan_to_num(wind_dir, nan=0.0)

    return {
        "lats": [float(la) for la in data['lats']],
        "lons": [float(lo) for lo in lons],
        "times": [dt.strftime('%Y-%m-%d %H:%M') for dt in wave_dts],
        "wave_height": wave_height.tolist(),
        "wave_period": wave_period.tolist(),
        "wave_direction": wave_dir.tolist(),
        "wind_speed": wind_speed.tolist(),
        "wind_direction": wind_dir.tolist(),
    }

def get_grid_weather_data(lat_min, lat_max, lon_min, lon_max):
    """
    Fetches gridded wave and wind data for local map display.
    Uses NOMADS GFS-Wave Atlantic 0.16deg if in coverage, falls back to ERDDAP.
    """
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    if _in_gfswave_atlantic_coverage(center_lat, center_lon):
        try:
            result = _get_grid_from_nomads(lat_min, lat_max, lon_min, lon_max)
            if result:
                return result
            print("NOMADS grid forecast returned no data, falling back to ERDDAP")
        except Exception as e:
            print(f"NOMADS grid forecast failed, falling back to ERDDAP: {e}")
            traceback.print_exc()
    return _get_grid_from_erddap(lat_min, lat_max, lon_min, lon_max)

def _get_grid_from_erddap(lat_min, lat_max, lon_min, lon_max):
    """
    Fetches gridded wave and wind data from ERDDAP for local map display.
    WW3 at native 0.5deg resolution, GFS wind interpolated to hourly.
    """
    try:
        # Convert lon bounds to 0-360 for ERDDAP
        lon_min_360 = lon_min % 360
        lon_max_360 = lon_max % 360
        lat_range = f"({lat_min}):({lat_max})"
        lon_range = f"({lon_min_360}):({lon_max_360})"

        # Retry with progressively older start times to handle ERDDAP model update gaps
        wave_json = None
        for hours_back in [6, 12, 24]:
            try:
                now = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours_back)
                t_start = now.isoformat() + "Z"
                time_range = f"({t_start}):(last)"

                # Fetch WW3 wave data (native 0.5° resolution)
                print(f"Grid forecast: fetching WW3 waves for ({lat_min},{lon_min}) to ({lat_max},{lon_max}), start={t_start}...")
                wave_json = _fetch_erddap_grid(
                    server="pae-paha.pacioos.hawaii.edu",
                    dataset="ww3_global",
                    variables=["Thgt", "Tper", "Tdir"],
                    time_range=time_range,
                    lat_range=lat_range,
                    lon_range=lon_range,
                    depth=0
                )
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404 and hours_back < 24:
                    print(f"  ERDDAP 404 with start {t_start}, retrying with older start...")
                    continue
                raise
        wave = _parse_erddap_to_grids(wave_json, ["Thgt", "Tper", "Tdir"])
        wave_dts = [_parse_erddap_time(t) for t in wave['times']]
        print(f"  Wave data: {len(wave['times'])} times, {len(wave['lats'])}x{len(wave['lons'])} grid")

        # Fetch GFS wind data (also retry on 404)
        wind_json = None
        for hours_back in [6, 12, 24]:
            try:
                now = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours_back)
                t_start = now.isoformat() + "Z"
                time_range = f"({t_start}):(last)"

                print(f"Grid forecast: fetching GFS wind, start={t_start}...")
                wind_json = _fetch_erddap_grid(
                    server="coastwatch.pfeg.noaa.gov",
                    dataset="NCEP_Global_Best",
                    variables=["ugrd10m", "vgrd10m"],
                    time_range=time_range,
                    lat_range=lat_range,
                    lon_range=lon_range,
                    depth=None
                )
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404 and hours_back < 24:
                    print(f"  ERDDAP 404 for wind with start {t_start}, retrying with older start...")
                    continue
                raise
        wind = _parse_erddap_to_grids(wind_json, ["ugrd10m", "vgrd10m"])
        print(f"  Wind data: {len(wind['times'])} times, {len(wind['lats'])}x{len(wind['lons'])} grid")

        lats = wave['lats']
        lons_360 = wave['lons']
        lons = [lo - 360 if lo > 180 else lo for lo in lons_360]

        # Interpolate wind to hourly using shared helper
        wind_speed_grid, wind_dir_grid = _interpolate_wind_to_hourly(
            wind, wave_dts, wave['lats'], lons_360
        )

        # Wave grids
        wave_height_grid = wave['grids']['Thgt'].copy()
        wave_period_grid = wave['grids']['Tper'].copy()
        wave_dir_grid = wave['grids']['Tdir'].copy()

        # Replace NaN with 0 for JSON serialization
        wave_height_grid = np.nan_to_num(wave_height_grid, nan=0.0)
        wave_period_grid = np.nan_to_num(wave_period_grid, nan=0.0)
        wave_dir_grid = np.nan_to_num(wave_dir_grid, nan=0.0)
        wind_speed_grid = np.nan_to_num(wind_speed_grid, nan=0.0)
        wind_dir_grid = np.nan_to_num(wind_dir_grid, nan=0.0)

        return {
            "lats": [float(la) for la in lats],
            "lons": [float(lo) for lo in lons],
            "times": [dt.strftime('%Y-%m-%d %H:%M') for dt in wave_dts],
            "wave_height": wave_height_grid.tolist(),
            "wave_period": wave_period_grid.tolist(),
            "wave_direction": wave_dir_grid.tolist(),
            "wind_speed": wind_speed_grid.tolist(),
            "wind_direction": wind_dir_grid.tolist(),
        }

    except Exception as e:
        print(f"Error fetching grid data from ERDDAP: {e}")
        traceback.print_exc()
        return None

# Route for the main dashboard
@app.route('/')
def index():
    """
    Renders the main dashboard page with SEO meta tags.
    """
    lat = request.args.get('lat', DEFAULT_LAT, type=float)
    lon = request.args.get('lon', DEFAULT_LON, type=float)
    name = request.args.get('name', 'Surf City, North Carolina')

    page_title = f"Surf Forecast for {name} | Free Surf Forecast"
    meta_description = (
        f"Free 7-day surf forecast for {name}. "
        "Wave height, wave period, wind speed and direction, tide predictions, "
        "and live surf cameras. No ads, no sign-up."
    )

    # Build canonical URL with location params
    canonical_params = f"?lat={lat}&lon={lon}&name={name}" if (lat != DEFAULT_LAT or lon != DEFAULT_LON) else ""
    canonical_url = f"https://freesurfforecast.com/{canonical_params}"

    return render_template(
        'index.html',
        version=APP_VERSION,
        page_title=page_title,
        meta_description=meta_description,
        canonical_url=canonical_url,
        og_title=page_title,
        og_description=meta_description,
        location_name=name,
        location_lat=lat,
        location_lon=lon,
    )

@app.route('/robots.txt')
def robots_txt():
    """Serve robots.txt for search engine crawlers."""
    content = (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /api/\n"
        "\n"
        "Sitemap: https://freesurfforecast.com/sitemap.xml\n"
    )
    return Response(content, mimetype='text/plain')


@app.route('/sitemap.xml')
def sitemap_xml():
    """Generate sitemap.xml from surf camera locations."""
    cameras_path = os.path.join(os.path.dirname(__file__), 'surf_cameras.json')
    try:
        with open(cameras_path, 'r') as f:
            cameras = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cameras = []

    # Deduplicate by lat/lon (some cameras share coordinates)
    seen = set()
    unique_locations = []
    for cam in cameras:
        key = (round(cam['lat'], 4), round(cam['lon'], 4))
        if key not in seen:
            seen.add(key)
            unique_locations.append(cam)

    urls = ['<?xml version="1.0" encoding="UTF-8"?>']
    urls.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')

    # Root URL
    urls.append('  <url>')
    urls.append('    <loc>https://freesurfforecast.com/</loc>')
    urls.append('    <priority>1.0</priority>')
    urls.append('  </url>')

    # Location URLs from camera list
    for loc in unique_locations:
        loc_url = f"https://freesurfforecast.com/?lat={loc['lat']}&amp;lon={loc['lon']}&amp;name={url_quote(loc['name'])}"
        urls.append('  <url>')
        urls.append(f'    <loc>{loc_url}</loc>')
        urls.append('    <priority>0.7</priority>')
        urls.append('  </url>')

    urls.append('</urlset>')

    return Response('\n'.join(urls), mimetype='application/xml')


# Route for the API to get point forecast data
@app.route('/api/forecast')
def forecast():
    """
    Provides weather forecast data for a single point as JSON.
    Accepts optional lat/lon query parameters.
    """
    result = validate_lat_lon()
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], float):
        return result
    lat, lon = result

    cache_key = f"forecast:{lat:.4f},{lon:.4f}"
    data = cached(cache_key, lambda: get_point_weather_data(lat, lon))

    if data:
        return jsonify(data)
    else:
        return jsonify({"error": "Could not retrieve weather data."}), 500

# Route for the API to get gridded forecast data
@app.route('/api/map-forecast')
def map_forecast():
    """
    Provides gridded weather forecast data as JSON for a specified bounding box.
    Accepts optional lat/lon query parameters to center the bounding box.
    """
    result = validate_lat_lon()
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], float):
        return result
    center_lat, center_lon = result

    # Round to reduce URL length
    center_lat = round(center_lat, 2)
    center_lon = round(center_lon, 2)

    # Create bounding box around center point (±1.5° lat, ±2.0° lon)
    # Gives ~19×25 grid at GFS-Wave 0.16° (NOMADS) or ~7×9 at WW3 0.5° (ERDDAP fallback)
    lat_min, lat_max = center_lat - 1.5, center_lat + 1.5
    lon_min, lon_max = center_lon - 2.0, center_lon + 2.0

    cache_key = f"map:{center_lat},{center_lon}"
    data = cached(cache_key, lambda: get_grid_weather_data(lat_min, lat_max, lon_min, lon_max))

    if data:
        return jsonify(data)
    else:
        return jsonify({"error": "Could not retrieve gridded weather data."}), 500

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on earth (in km).
    """
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

SURFCHEX_MAX_DISTANCE_KM = 160  # ~100 miles

def find_nearest_cameras(lat, lon, count=2):
    """Find the nearest surf cameras to the given coordinates."""
    # Search surf camera catalog
    nearby = []
    for cam in SURFCHEX_CAMERAS:
        if cam.get('disabled'):
            continue
        dist = haversine_distance(lat, lon, cam['lat'], cam['lon'])
        if dist <= SURFCHEX_MAX_DISTANCE_KM:
            cam_type = cam.get('type', 'hls')
            entry = {
                'name': cam['name'],
                'type': cam_type,
                'distance_km': round(dist, 1),
                'page_url': cam.get('page_url', '')
            }
            if cam_type == 'link':
                entry['url'] = cam['page_url']
                entry['thumbnail_url'] = cam.get('thumbnail_url', '')
            else:
                entry['url'] = cam['stream_url']
            nearby.append(entry)
    nearby.sort(key=lambda c: c['distance_km'])
    results = nearby[:count]

    # Windy Webcams API fallback if we don't have enough Surfchex cams
    if len(results) < count:
        windy_key = os.environ.get('WINDY_API_KEY')
        if not windy_key:
            print("Windy fallback skipped: WINDY_API_KEY not set")
        else:
            try:
                needed = count - len(results)
                radius_km = SURFCHEX_MAX_DISTANCE_KM
                resp = requests.get(
                    'https://api.windy.com/webcams/api/v3/webcams',
                    params={
                        'nearby': f'{lat},{lon},{radius_km}',
                        'categories': 'beach',
                        'include': 'images,location,player',
                        'limit': needed
                    },
                    headers={'x-windy-api-key': windy_key},
                    timeout=10
                )
                if not resp.ok:
                    print(f"Windy API returned {resp.status_code}: {resp.text[:200]}")
                else:
                    webcams = resp.json().get('webcams', [])
                    print(f"Windy returned {len(webcams)} webcams")
                    for wc in webcams:
                        loc = wc.get('location', {})
                        wc_lat = loc.get('latitude', 0)
                        wc_lon = loc.get('longitude', 0)
                        dist = haversine_distance(lat, lon, wc_lat, wc_lon)
                        player = wc.get('player', {})
                        embed_url = player.get('day') or player.get('lifetime')
                        if embed_url:
                            wc_id = wc.get('webcamId', '')
                            results.append({
                                'name': wc.get('title', 'Webcam'),
                                'type': 'iframe',
                                'url': embed_url,
                                'distance_km': round(dist, 1),
                                'page_url': f'https://www.windy.com/webcams/{wc_id}' if wc_id else ''
                            })
            except Exception as e:
                print(f"Windy webcam fallback failed (non-critical): {e}")

    return results

def get_ocean_basin_data():
    """
    Fetches global wave and wind data using NOAA ERDDAP.
    Uses WW3 global wave model (PacIOOS) and GFS wind model (CoastWatch).
    """
    # WW3 global at 3° effective resolution (stride=6 on native 0.5°)
    # 3-hourly time steps to keep response size under ~30 MB for 512 MB Render tier
    lat_range = "(-77.5):6:(77.5)"
    lon_range = "(0.0):6:(359.5)"
    time_stride = 3  # every 3rd hourly step = 3-hourly

    # --- Fetch WW3 wave data (required) ---
    # Retry with progressively older start times to handle ERDDAP model update gaps
    try:
        wave_json = None
        for hours_back in [6, 12, 24]:
            try:
                now = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours_back)
                t_start = now.isoformat() + "Z"

                print(f"Fetching global wave data from ERDDAP (WW3), start={t_start}...")
                wave_json = _fetch_erddap_grid(
                    server="pae-paha.pacioos.hawaii.edu",
                    dataset="ww3_global",
                    variables=["Thgt", "Tper", "Tdir"],
                    time_range=f"({t_start}):{time_stride}:(last)",
                    lat_range=lat_range,
                    lon_range=lon_range,
                    depth=0
                )
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404 and hours_back < 24:
                    print(f"  ERDDAP 404 with start {t_start}, retrying with older start...")
                    continue
                raise
        wave = _parse_erddap_to_grids(wave_json, ["Thgt", "Tper", "Tdir"])
        print(f"  Wave data: {len(wave['times'])} times, {len(wave['lats'])}x{len(wave['lons'])} grid")
    except Exception as e:
        print(f"Error fetching global wave data from ERDDAP: {e}")
        traceback.print_exc()
        return None

    # --- Fetch GFS wind data (optional — degrade gracefully) ---
    wind = None
    try:
        wind_json = None
        for hours_back in [6, 12, 24]:
            try:
                now = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours_back)
                t_start = now.isoformat() + "Z"

                print(f"Fetching global wind data from ERDDAP (GFS), start={t_start}...")
                wind_json = _fetch_erddap_grid(
                    server="coastwatch.pfeg.noaa.gov",
                    dataset="NCEP_Global_Best",
                    variables=["ugrd10m", "vgrd10m"],
                    time_range=f"({t_start}):{time_stride}:(last)",
                    lat_range=lat_range,
                    lon_range=lon_range,
                    depth=None
                )
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404 and hours_back < 24:
                    print(f"  ERDDAP 404 for wind with start {t_start}, retrying with older start...")
                    continue
                raise
        wind = _parse_erddap_to_grids(wind_json, ["ugrd10m", "vgrd10m"])
        print(f"  Wind data: {len(wind['times'])} times, {len(wind['lats'])}x{len(wind['lons'])} grid")
    except Exception as e:
        print(f"Warning: Could not fetch wind data from ERDDAP (continuing without wind): {e}")
        traceback.print_exc()

    # --- Assemble output grids ---
    try:
        wave_dts = [_parse_erddap_time(t) for t in wave['times']]

        # Convert ERDDAP 0–360° lons back to -180..180 for the frontend
        lats = wave['lats']
        lons = [lon - 360 if lon > 180 else lon for lon in wave['lons']]
        lons_360 = wave['lons']

        # Wave grids
        wave_height_out = wave['grids']['Thgt'].copy()
        wave_period_out = wave['grids']['Tper'].copy()
        wave_dir_out = wave['grids']['Tdir'].copy()

        # Interpolate wind using shared helper
        wind_speed_out, wind_dir_out = _interpolate_wind_to_hourly(
            wind, wave_dts, wave['lats'], lons_360
        )

        # Replace NaN with 0 for JSON serialization (land points)
        wave_height_out = np.nan_to_num(wave_height_out, nan=0.0)
        wave_period_out = np.nan_to_num(wave_period_out, nan=0.0)
        wave_dir_out = np.nan_to_num(wave_dir_out, nan=0.0)
        wind_speed_out = np.nan_to_num(wind_speed_out, nan=0.0)
        wind_dir_out = np.nan_to_num(wind_dir_out, nan=0.0)

        formatted_times = [dt.strftime('%Y-%m-%d %H:%M') for dt in wave_dts]

        return {
            "lats": [float(la) for la in lats],
            "lons": [float(lo) for lo in lons],
            "times": formatted_times,
            "wave_height": wave_height_out.tolist(),
            "wave_period": wave_period_out.tolist(),
            "wave_direction": wave_dir_out.tolist(),
            "wind_speed": wind_speed_out.tolist(),
            "wind_direction": wind_dir_out.tolist(),
        }

    except Exception as e:
        print(f"Error processing ocean basin data: {e}")
        traceback.print_exc()
        return None

# Route for ocean basin data
@app.route('/api/ocean-basin')
def ocean_basin():
    """
    Provides wave data for the ocean basin around the forecast location.
    Accepts optional lat/lon query parameters.
    """
    result = validate_lat_lon()
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], float):
        return result
    center_lat, center_lon = result

    # Global data is the same for all locations — single cache entry
    cache_key = "basin:global"
    data = cached(cache_key, get_ocean_basin_data, ttl=BASIN_CACHE_TTL)

    if data:
        result = dict(data)  # shallow copy so we don't mutate the cached version
        result["center"] = {"lat": round(center_lat, 2), "lon": round(center_lon, 2)}
        return jsonify(result)
    else:
        return jsonify({"error": "Could not retrieve ocean basin data."}), 500

def find_nearest_tide_station(target_lat, target_lon):
    """
    Finds the nearest NOAA tide prediction station to the given coordinates.
    Only considers Reference stations (type='R') which have direct harmonic predictions.
    """
    try:
        # Get list of all tide prediction stations
        url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
        params = {"type": "tidepredictions"}

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        stations = data.get("stations", [])
        if not stations:
            print("No tide stations found")
            return None

        # Find the nearest Reference station (type='R')
        # Reference stations have direct harmonic predictions
        # Subordinate stations (type='S') use offsets and may not work with the predictions API
        nearest_station = None
        min_distance = float('inf')

        for station in stations:
            # Only consider Reference stations
            if station.get("type") != "R":
                continue

            station_lat = station.get("lat")
            station_lon = station.get("lng")

            if station_lat is None or station_lon is None:
                continue

            distance = haversine_distance(target_lat, target_lon, station_lat, station_lon)

            if distance < min_distance:
                min_distance = distance
                nearest_station = {
                    "id": station.get("id"),
                    "name": station.get("name"),
                    "lat": station_lat,
                    "lon": station_lon,
                    "distance_km": round(distance, 1)
                }

        return nearest_station

    except Exception as e:
        print(f"Error finding nearest tide station: {e}")
        traceback.print_exc()
        return None

def get_tide_data(station_id):
    """
    Fetches tide prediction data from NOAA CO-OPS API.
    """
    try:
        # Get tide predictions for the next 7 days
        today = datetime.now()
        end_date = today + timedelta(days=7)

        url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        params = {
            "begin_date": today.strftime("%Y%m%d"),
            "end_date": end_date.strftime("%Y%m%d"),
            "station": station_id,
            "product": "predictions",
            "datum": "MLLW",  # Mean Lower Low Water
            "time_zone": "lst_ldt",  # Local time with daylight savings
            "units": "metric",
            "format": "json",
            "interval": "h"  # Hourly predictions
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "predictions" not in data:
            print(f"No predictions in tide data: {data}")
            return None

        # Also get high/low tide times
        hilo_params = params.copy()
        hilo_params["interval"] = "hilo"
        hilo_response = requests.get(url, params=hilo_params)
        hilo_data = hilo_response.json()

        tide_forecast = {
            "hourly": [
                {
                    "time": pred["t"],
                    "height": float(pred["v"])
                }
                for pred in data["predictions"]
            ],
            "high_low": [
                {
                    "time": pred["t"],
                    "height": float(pred["v"]),
                    "type": pred["type"]  # "H" for high, "L" for low
                }
                for pred in hilo_data.get("predictions", [])
            ]
        }

        return tide_forecast

    except requests.exceptions.RequestException as e:
        print(f"Error fetching tide data: {e}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Error processing tide data: {e}")
        traceback.print_exc()
        return None

# Route for the API to get tide data
@app.route('/api/tides')
def tides():
    """
    Provides tide prediction data as JSON for the nearest station to the forecast location.
    Accepts optional lat/lon query parameters.
    """
    result = validate_lat_lon()
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], float):
        return result
    target_lat, target_lon = result

    cache_key = f"tides:{target_lat:.4f},{target_lon:.4f}"

    def fetch_tides():
        station = find_nearest_tide_station(target_lat, target_lon)
        if not station:
            return None
        data = get_tide_data(station["id"])
        if data:
            data["station"] = station
        return data

    data = cached(cache_key, fetch_tides)

    if data:
        return jsonify(data)
    else:
        return jsonify({"error": "Could not retrieve tide data."}), 500

@app.route('/api/webcams')
def webcams():
    result = validate_lat_lon()
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], float):
        return result
    lat, lon = result
    cache_key = f"webcams:{lat:.2f},{lon:.2f}"
    cameras = cached(cache_key, lambda: find_nearest_cameras(lat, lon, count=50))
    return jsonify({"cameras": cameras or []})

BUOY_MAX_DISTANCE_KM = 320  # ~200 miles

def find_nearest_buoys(lat, lon, max_km=BUOY_MAX_DISTANCE_KM, count=3):
    """Find the nearest NDBC and CDIP buoys to the given coordinates."""
    candidates = []
    for st in NDBC_STATIONS:
        dist = haversine_distance(lat, lon, st['lat'], st['lon'])
        if dist <= max_km:
            candidates.append({
                'id': st['id'], 'source': 'ndbc',
                'name': st['name'], 'lat': st['lat'], 'lon': st['lon'],
                'distance_km': round(dist, 1)
            })
    for st in CDIP_STATIONS:
        dist = haversine_distance(lat, lon, st['lat'], st['lon'])
        if dist <= max_km:
            candidates.append({
                'id': st['id'], 'source': 'cdip',
                'name': st['name'], 'lat': st['lat'], 'lon': st['lon'],
                'distance_km': round(dist, 1)
            })
    candidates.sort(key=lambda b: b['distance_km'])
    return candidates[:count]


def _fetch_ndbc_observation(station_id):
    """Fetch latest observation from NDBC realtime2 text files."""
    try:
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
        resp = requests.get(url, timeout=15)
        if not resp.ok:
            print(f"  NDBC obs {station_id}: HTTP {resp.status_code}")
            return None
        lines = resp.text.strip().split('\n')
        if len(lines) < 3:
            return None
        # Header row 0 = column names, row 1 = units, row 2+ = data (newest first)
        headers = lines[0].replace('#', '').split()
        data = lines[2].split()
        if len(data) < len(headers):
            return None

        def val(name):
            try:
                idx = headers.index(name)
                v = data[idx]
                if v == 'MM' or v == 'MM.M':
                    return None
                return float(v)
            except (ValueError, IndexError):
                return None

        yr, mo, dy, hr, mn = val('YY'), val('MM'), val('DD'), val('hh'), val('mm')
        time_str = None
        if yr and mo and dy:
            time_str = f"{int(yr)}-{int(mo):02d}-{int(dy):02d} {int(hr or 0):02d}:{int(mn or 0):02d} UTC"

        return {
            'time': time_str,
            'wave_height': val('WVHT'),
            'dominant_period': val('DPD'),
            'avg_period': val('APD'),
            'wave_direction': val('MWD'),
            'wind_speed': val('WSPD'),
            'wind_direction': val('WDIR'),
            'wind_gust': val('GST'),
            'air_temp': val('ATMP'),
            'water_temp': val('WTMP'),
            'pressure': val('PRES'),
        }
    except Exception as e:
        print(f"  NDBC obs {station_id} error: {e}")
        return None


def _fetch_ndbc_spectrum(station_id):
    """Fetch latest 1D energy spectrum from NDBC realtime2 data_spec file."""
    try:
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.data_spec"
        resp = requests.get(url, timeout=15)
        if not resp.ok:
            return None
        lines = resp.text.strip().split('\n')
        if len(lines) < 2:
            return None
        # First data line (after header) has format:
        # YYYY MM DD hh mm  sep_freq  spec_1(freq_1) spec_2(freq_2) ...
        import re
        data_line = lines[1]
        # Extract energy(freq) pairs
        pairs = re.findall(r'([\d.]+)\s*\(([\d.]+)\)', data_line)
        if not pairs:
            return None
        energy = [float(p[0]) for p in pairs]
        frequencies = [float(p[1]) for p in pairs]
        return {'frequencies': frequencies, 'energy': energy}
    except Exception as e:
        print(f"  NDBC spectrum {station_id} error: {e}")
        return None


def _fetch_ndbc_directional(station_id):
    """Fetch mean direction and spread per frequency from NDBC .swdir and .swr1 files."""
    try:
        dir_url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.swdir"
        r1_url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.swr1"
        import re

        dir_resp = requests.get(dir_url, timeout=15)
        r1_resp = requests.get(r1_url, timeout=15)
        if not dir_resp.ok or not r1_resp.ok:
            return None

        def parse_freq_values(text):
            lines = text.strip().split('\n')
            if len(lines) < 2:
                return None, None
            data_line = lines[1]
            pairs = re.findall(r'([\d.]+)\s*\(([\d.]+)\)', data_line)
            if not pairs:
                return None, None
            values = [float(p[0]) for p in pairs]
            freqs = [float(p[1]) for p in pairs]
            return freqs, values

        freqs_d, directions = parse_freq_values(dir_resp.text)
        freqs_r, r1_values = parse_freq_values(r1_resp.text)
        if not freqs_d or not directions:
            return None

        return {
            'frequencies': freqs_d,
            'directions': directions,
            'r1': r1_values or []
        }
    except Exception as e:
        print(f"  NDBC directional {station_id} error: {e}")
        return None


_cdip_dds_cache = {}

def _cdip_last_index(station_id, dim_name='waveTime'):
    """Get the last valid index for a CDIP OPeNDAP dimension."""
    import re
    if station_id not in _cdip_dds_cache or time.time() - _cdip_dds_cache[station_id]['t'] > 900:
        base = f"https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{station_id}p1_rt.nc"
        resp = requests.get(base + ".dds", timeout=10)
        if not resp.ok:
            return None
        _cdip_dds_cache[station_id] = {'text': resp.text, 't': time.time()}
    text = _cdip_dds_cache[station_id]['text']
    m = re.search(dim_name + r'\s*=\s*(\d+)', text)
    return int(m.group(1)) - 1 if m else None


def _fetch_cdip_observation(station_id):
    """Fetch latest observation from CDIP via OPeNDAP."""
    try:
        last_w = _cdip_last_index(station_id, 'waveTime')
        if last_w is None:
            return None
        last_s = _cdip_last_index(station_id, 'sstTime')
        base = f"https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{station_id}p1_rt.nc"
        query = f".ascii?waveHs[{last_w}],waveTp[{last_w}],waveDp[{last_w}],waveTa[{last_w}],waveTime[{last_w}]"
        if last_s is not None:
            query += f",sstSeaSurfaceTemperature[{last_s}]"
        resp = requests.get(base + query, timeout=15)
        if not resp.ok:
            return None
        text = resp.text
        import re

        def extract_val(varname):
            # OPeNDAP format: "varname[1]\nvalue\n" or "varname, value"
            m = re.search(varname + r'(?:\[\d+\])?\s*[\n,]\s*([\d.eE+\-]+)', text)
            if m:
                v = float(m.group(1))
                if v > 900 or v < -900:
                    return None
                return v
            return None

        time_str = None
        m = re.search(r'waveTime\[\d+\]\s*\n\s*([\d.eE+\-]+)', text)
        if m:
            epoch = float(m.group(1))
            dt = datetime.utcfromtimestamp(epoch)
            time_str = dt.strftime('%Y-%m-%d %H:%M UTC')

        return {
            'time': time_str,
            'wave_height': extract_val('waveHs'),
            'dominant_period': extract_val('waveTp'),
            'wave_direction': extract_val('waveDp'),
            'avg_period': extract_val('waveTa'),
            'water_temp': extract_val('sstSeaSurfaceTemperature'),
            'wind_speed': None,
            'wind_direction': None,
            'wind_gust': None,
            'air_temp': None,
            'pressure': None,
        }
    except Exception as e:
        print(f"  CDIP obs {station_id} error: {e}")
        return None


def _fetch_cdip_spectrum(station_id):
    """Fetch latest 1D energy spectrum from CDIP via OPeNDAP."""
    try:
        last_w = _cdip_last_index(station_id, 'waveTime')
        if last_w is None:
            return None
        # waveFrequency dimension size varies by station; get from DDS
        last_f = _cdip_last_index(station_id, 'waveFrequency')
        if last_f is None:
            return None
        base = f"https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{station_id}p1_rt.nc"
        url = base + f".ascii?waveEnergyDensity[{last_w}][0:1:{last_f}],waveFrequency[0:1:{last_f}]"
        resp = requests.get(url, timeout=15)
        if not resp.ok:
            return None
        text = resp.text
        import re

        def parse_cdip_array(text, varname):
            """Extract float array from CDIP OPeNDAP ASCII response."""
            # Find the section starting with varname[N] or varname.varname[N][M]
            # Data is on lines after the header, comma-separated
            # Format: "varname[100]\n0.025, 0.03, ..." or "[0], 0.025, 0.03, ..."
            pattern = varname + r'(?:\.' + varname + r')?\[\d+\](?:\[\d+\])?\s*\n'
            m = re.search(pattern, text)
            if not m:
                return []
            start = m.end()
            # Collect until next blank line or next variable
            remaining = text[start:]
            # Take lines until we hit a blank line or a new variable name
            data_lines = []
            for line in remaining.split('\n'):
                stripped = line.strip()
                if not stripped or (stripped[0].isalpha() and not stripped[0] == 'E'):
                    break
                data_lines.append(stripped)
            data_str = ' '.join(data_lines)
            # Remove array index prefix like "[0],"
            data_str = re.sub(r'\[\d+\],?\s*', '', data_str)
            vals = []
            for s in data_str.split(','):
                s = s.strip()
                if s:
                    try:
                        vals.append(float(s))
                    except ValueError:
                        pass
            return vals

        freq_vals = parse_cdip_array(text, 'waveFrequency')
        energy_vals = parse_cdip_array(text, 'waveEnergyDensity')

        if not freq_vals or not energy_vals:
            return None
        energy_vals = [e if e < 9000 else 0 for e in energy_vals]
        n = min(len(freq_vals), len(energy_vals))
        return {'frequencies': freq_vals[:n], 'energy': energy_vals[:n]}
    except Exception as e:
        print(f"  CDIP spectrum {station_id} error: {e}")
        return None


def _fetch_cdip_directional(station_id):
    """Fetch directional Fourier coefficients from CDIP via OPeNDAP."""
    try:
        last_w = _cdip_last_index(station_id, 'waveTime')
        last_f = _cdip_last_index(station_id, 'waveFrequency')
        if last_w is None or last_f is None:
            return None
        base = f"https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{station_id}p1_rt.nc"
        url = (base + f".ascii?waveMeanDirection[{last_w}][0:1:{last_f}],"
               f"waveA1Value[{last_w}][0:1:{last_f}],waveB1Value[{last_w}][0:1:{last_f}],"
               f"waveA2Value[{last_w}][0:1:{last_f}],waveB2Value[{last_w}][0:1:{last_f}],"
               f"waveFrequency[0:1:{last_f}]")
        resp = requests.get(url, timeout=15)
        if not resp.ok:
            return None
        text = resp.text
        import re

        def parse_cdip_dir_array(text, varname):
            """Extract float array from CDIP OPeNDAP ASCII response."""
            pattern = varname + r'(?:\.' + varname + r')?\[\d+\](?:\[\d+\])?\s*\n'
            m = re.search(pattern, text)
            if not m:
                return []
            start = m.end()
            remaining = text[start:]
            data_lines = []
            for line in remaining.split('\n'):
                stripped = line.strip()
                if not stripped or (stripped[0].isalpha() and stripped[0] != 'E' and stripped[0] != 'e'):
                    break
                data_lines.append(stripped)
            data_str = ' '.join(data_lines)
            data_str = re.sub(r'\[\d+\],?\s*', '', data_str)
            vals = []
            for s in data_str.split(','):
                s = s.strip()
                if s:
                    try:
                        vals.append(float(s))
                    except ValueError:
                        pass
            return vals

        frequencies = parse_cdip_dir_array(text, 'waveFrequency')
        directions = parse_cdip_dir_array(text, 'waveMeanDirection')
        a1 = parse_cdip_dir_array(text, 'waveA1Value')
        b1 = parse_cdip_dir_array(text, 'waveB1Value')
        a2 = parse_cdip_dir_array(text, 'waveA2Value')
        b2 = parse_cdip_dir_array(text, 'waveB2Value')

        if not frequencies or not a1:
            return None

        directions = [d if -360 <= d <= 360 else 0 for d in directions]
        n = min(len(frequencies), len(directions), len(a1), len(b1), len(a2), len(b2))
        if n == 0:
            return None

        return {
            'frequencies': frequencies[:n],
            'directions': directions[:n],
            'a1': a1[:n], 'b1': b1[:n],
            'a2': a2[:n], 'b2': b2[:n]
        }
    except Exception as e:
        print(f"  CDIP directional {station_id} error: {e}")
        return None


def get_buoy_data(lat, lon):
    """Fetch live buoy observations and spectral data for nearest buoys."""
    # Fetch extra candidates since some NDBC stations lack realtime data
    candidates = find_nearest_buoys(lat, lon, count=10)
    if not candidates:
        return {'buoys': [], 'spectrum': None, 'directional': None}

    print(f"Buoy search: {len(candidates)} candidates near ({lat}, {lon})")

    buoys = []
    for buoy in candidates:
        if len(buoys) >= 3:
            break
        if buoy['source'] == 'ndbc':
            obs = _fetch_ndbc_observation(buoy['id'])
        else:
            obs = _fetch_cdip_observation(buoy['id'])
        if obs and obs.get('wave_height') is not None:
            buoy.update(obs)
            buoys.append(buoy)
        else:
            print(f"  Skipping {buoy['source'].upper()} {buoy['id']} (no realtime data)")

    # Fetch spectrum and directional data for each buoy
    for buoy in buoys:
        bid = buoy['id']
        if buoy['source'] == 'ndbc':
            spec = _fetch_ndbc_spectrum(bid)
            dire = _fetch_ndbc_directional(bid)
        else:
            spec = _fetch_cdip_spectrum(bid)
            dire = _fetch_cdip_directional(bid)
        buoy['spectrum'] = spec
        buoy['directional'] = dire

    return {'buoys': buoys}


@app.route('/api/buoys')
def buoys_endpoint():
    result = validate_lat_lon()
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], float):
        return result
    lat, lon = result
    cache_key = f"buoys:{lat:.4f},{lon:.4f}"
    data = cached(cache_key, lambda: get_buoy_data(lat, lon))
    if data:
        return jsonify(data)
    else:
        return jsonify({"error": "Could not retrieve buoy data."}), 500


ORIENTATION_CACHE_TTL = 86400  # 24 hours (coastline doesn't change)

@app.route('/api/beach-orientation')
def beach_orientation():
    result = validate_lat_lon()
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], float):
        return result
    lat, lon = result
    cache_key = f"orientation:{lat:.4f},{lon:.4f}"
    data = cached(cache_key, lambda: compute_beach_facing_direction(lat, lon),
                  ttl=ORIENTATION_CACHE_TTL)
    if data:
        return jsonify(data)
    else:
        return jsonify({"error": "Could not determine beach orientation. Location may be too far inland."}), 404


@app.route('/api/swell-narrative')
def swell_narrative():
    """Generate a plain-English narrative about swell origin, type, and trend."""
    result = validate_lat_lon()
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], float):
        return result
    lat, lon = result

    cache_key = f"swell-narrative:{lat:.4f},{lon:.4f}"

    def _compute():
        # Reuse cached forecast data
        forecast_key = f"forecast:{lat:.4f},{lon:.4f}"
        data = cached(forecast_key, lambda: get_point_weather_data(lat, lon))
        if not data or len(data) < 2:
            return None

        # Extract swell parameters from first entry
        wave_dir = data[0].get('wave_direction')
        wave_period = data[0].get('wave_period', 0)
        wave_height = data[0].get('wave_height', 0)

        # Compass direction label
        compass_dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                        'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        source_label = 'unknown'
        if wave_dir is not None:
            source_label = compass_dirs[round(wave_dir / 22.5) % 16]

        # Swell type classification
        is_ground_swell = wave_period >= 10
        swell_type = 'Ground swell' if is_ground_swell else 'Wind swell'

        # Estimated source distance using group velocity
        # Group velocity (m/s) = 1.56 * period (deep water approximation)
        estimated_source_km = None
        if wave_period > 0:
            group_speed_ms = 1.56 * wave_period
            group_speed_kmh = group_speed_ms * 3.6
            # Assume 1-3 day travel: use 2 days as midpoint
            estimated_source_km = round(group_speed_kmh * 48)

        # Trend: compare avg wave height of first 24h vs next 24h
        first_24 = [d['wave_height'] for d in data[:24] if d.get('wave_height') is not None]
        next_24 = [d['wave_height'] for d in data[24:48] if d.get('wave_height') is not None]

        trend = 'steady'
        if first_24 and next_24:
            avg_first = sum(first_24) / len(first_24)
            avg_next = sum(next_24) / len(next_24)
            diff_pct = (avg_next - avg_first) / max(avg_first, 0.01) * 100
            if diff_pct > 15:
                trend = 'building'
            elif diff_pct < -15:
                trend = 'fading'

        # Build narrative sentence
        m_to_ft = 3.28084
        height_ft = round(wave_height * m_to_ft, 1)
        parts = []
        parts.append(f"{swell_type} from the {source_label}")
        parts.append(f"at {wave_period:.0f}s period" if wave_period else "")
        if estimated_source_km and is_ground_swell:
            parts.append(f"originating ~{estimated_source_km:,} km away")
        trend_labels = {'building': 'Building over the next 24h',
                        'fading': 'Fading over the next 24h',
                        'steady': 'Holding steady'}
        parts.append(trend_labels[trend])
        narrative = '. '.join(p for p in parts if p) + '.'

        return {
            'narrative': narrative,
            'swell_direction': wave_dir,
            'swell_type': swell_type.lower(),
            'is_building': trend == 'building',
            'trend': trend,
            'estimated_source_km': estimated_source_km
        }

    data = cached(cache_key, _compute)
    if data:
        return jsonify(data)
    else:
        return jsonify({"error": "Could not generate swell narrative."}), 500


if __name__ == '__main__':
    app.run(debug=True, threaded=True)