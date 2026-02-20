from flask import Flask, render_template, jsonify, request
import requests
import numpy as np
import math
import subprocess
import time
import traceback
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)

def get_version():
    """Read version from git tags via git describe."""
    try:
        return subprocess.check_output(
            ['git', 'describe', '--tags', '--always'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'v0.1.0'

APP_VERSION = get_version()

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

def get_point_weather_data(latitude, longitude):
    """
    Fetches wave and wind forecast for a single point using ERDDAP.
    WW3 for waves, GFS for wind, local computation for sunrise/sunset.
    """
    try:
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        t_start = now.isoformat() + "Z"
        time_range = f"({t_start}):(last)"

        # Convert longitude to 0-360 for ERDDAP
        lon_360 = longitude % 360

        # ±1° bounding box to find nearest ocean point (WW3 has land masking)
        lat_range = f"({latitude - 1}):({latitude + 1})"
        lon_range = f"({lon_360 - 1}):({lon_360 + 1})"

        # Fetch WW3 wave data
        print(f"Point forecast: fetching WW3 waves for ({latitude}, {longitude})...")
        wave_json = _fetch_erddap_grid(
            server="pae-paha.pacioos.hawaii.edu",
            dataset="ww3_global",
            variables=["Thgt", "Tper", "Tdir", "whgt", "wper"],
            time_range=time_range,
            lat_range=lat_range,
            lon_range=lon_range,
            depth=0
        )
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

        # Fetch GFS wind data
        print(f"Point forecast: fetching GFS wind...")
        wind_json = _fetch_erddap_grid(
            server="coastwatch.pfeg.noaa.gov",
            dataset="NCEP_Global_Best",
            variables=["ugrd10m", "vgrd10m"],
            time_range=time_range,
            lat_range=lat_range,
            lon_range=lon_range,
            depth=None
        )
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

def get_grid_weather_data(lat_min, lat_max, lon_min, lon_max):
    """
    Fetches gridded wave and wind data from ERDDAP for local map display.
    WW3 at native 0.5° resolution, GFS wind interpolated to hourly.
    """
    try:
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        t_start = now.isoformat() + "Z"
        time_range = f"({t_start}):(last)"

        # Convert lon bounds to 0-360 for ERDDAP
        lon_min_360 = lon_min % 360
        lon_max_360 = lon_max % 360
        lat_range = f"({lat_min}):({lat_max})"
        lon_range = f"({lon_min_360}):({lon_max_360})"

        # Fetch WW3 wave data (native 0.5° resolution)
        print(f"Grid forecast: fetching WW3 waves for ({lat_min},{lon_min}) to ({lat_max},{lon_max})...")
        wave_json = _fetch_erddap_grid(
            server="pae-paha.pacioos.hawaii.edu",
            dataset="ww3_global",
            variables=["Thgt", "Tper", "Tdir"],
            time_range=time_range,
            lat_range=lat_range,
            lon_range=lon_range,
            depth=0
        )
        wave = _parse_erddap_to_grids(wave_json, ["Thgt", "Tper", "Tdir"])
        wave_dts = [_parse_erddap_time(t) for t in wave['times']]
        print(f"  Wave data: {len(wave['times'])} times, {len(wave['lats'])}x{len(wave['lons'])} grid")

        # Fetch GFS wind data
        print(f"Grid forecast: fetching GFS wind...")
        wind_json = _fetch_erddap_grid(
            server="coastwatch.pfeg.noaa.gov",
            dataset="NCEP_Global_Best",
            variables=["ugrd10m", "vgrd10m"],
            time_range=time_range,
            lat_range=lat_range,
            lon_range=lon_range,
            depth=None
        )
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
    Renders the main dashboard page.
    """
    return render_template('index.html', version=APP_VERSION)

# Route for the API to get point forecast data
@app.route('/api/forecast')
def forecast():
    """
    Provides weather forecast data for a single point as JSON.
    Accepts optional lat/lon query parameters.
    """
    # Get lat/lon from query params, default to Surf City, North Carolina
    lat = request.args.get('lat', 34.42711, type=float)
    lon = request.args.get('lon', -77.54608, type=float)

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
    # Get center lat/lon from query params, default to Surf City, North Carolina
    center_lat = request.args.get('lat', 34.42711, type=float)
    center_lon = request.args.get('lon', -77.54608, type=float)

    # Round to reduce URL length
    center_lat = round(center_lat, 2)
    center_lon = round(center_lon, 2)

    # Create bounding box around center point (±1.5° lat, ±2.0° lon)
    # Wider box gives ~7×9 grid at WW3 native 0.5° and better coastal coverage
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

def get_ocean_basin_data():
    """
    Fetches global wave and wind data using NOAA ERDDAP.
    Uses WW3 global wave model (PacIOOS) and GFS wind model (CoastWatch).
    """
    # Use explicit ISO start time with (last) end to avoid 404 when forecast end varies
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    t_start = now.isoformat() + "Z"

    # Full WW3 global range at 3° effective resolution (stride=6 on native 0.5°)
    # stride=3 on native 0.5° = 1.5° effective resolution
    lat_range = "(-77.5):3:(77.5)"
    lon_range = "(0.0):3:(359.5)"

    # --- Fetch WW3 wave data (required) ---
    try:
        print("Fetching global wave data from ERDDAP (WW3)...")
        wave_json = _fetch_erddap_grid(
            server="pae-paha.pacioos.hawaii.edu",
            dataset="ww3_global",
            variables=["Thgt", "Tper", "Tdir"],
            time_range=f"({t_start}):(last)",
            lat_range=lat_range,
            lon_range=lon_range,
            depth=0
        )
        wave = _parse_erddap_to_grids(wave_json, ["Thgt", "Tper", "Tdir"])
        print(f"  Wave data: {len(wave['times'])} times, {len(wave['lats'])}x{len(wave['lons'])} grid")
    except Exception as e:
        print(f"Error fetching global wave data from ERDDAP: {e}")
        traceback.print_exc()
        return None

    # --- Fetch GFS wind data (optional — degrade gracefully) ---
    wind = None
    try:
        print("Fetching global wind data from ERDDAP (GFS)...")
        wind_json = _fetch_erddap_grid(
            server="coastwatch.pfeg.noaa.gov",
            dataset="NCEP_Global_Best",
            variables=["ugrd10m", "vgrd10m"],
            time_range=f"({t_start}):(last)",
            lat_range=lat_range,
            lon_range=lon_range,
            depth=None
        )
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
    # Get center lat/lon from query params (used only for map centering, not data bounds)
    center_lat = request.args.get('lat', 34.42711, type=float)
    center_lon = request.args.get('lon', -77.54608, type=float)

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
    # Get lat/lon from query params, default to Surf City, North Carolina
    target_lat = request.args.get('lat', 34.42711, type=float)
    target_lon = request.args.get('lon', -77.54608, type=float)

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

if __name__ == '__main__':
    app.run(debug=True)