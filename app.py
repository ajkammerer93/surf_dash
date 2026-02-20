from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
import numpy as np
import subprocess
import os
import time
import traceback
from datetime import datetime, timedelta
from itertools import product

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

# Rate limit cooldown — stop hitting the API for a while after 429s
_rate_limit_until = 0  # timestamp when cooldown expires
RATE_LIMIT_COOLDOWN = 300  # 5 minutes

def is_rate_limited():
    """Check if we're in a rate limit cooldown period."""
    return time.time() < _rate_limit_until

def set_rate_limited():
    """Enter rate limit cooldown mode."""
    global _rate_limit_until
    _rate_limit_until = time.time() + RATE_LIMIT_COOLDOWN
    print(f"Rate limit cooldown active for {RATE_LIMIT_COOLDOWN}s — skipping API calls until {time.strftime('%H:%M:%S', time.localtime(_rate_limit_until))}")

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
    Fetches wave data from Marine API and wind data from Weather API for a single point.
    """
    try:
        # Fetch marine data
        marine_url = "https://marine-api.open-meteo.com/v1/marine"
        marine_params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "wave_height,wave_period,wave_direction,wind_wave_height,wind_wave_period",
            "timezone": "auto"
        }
        marine_url, marine_params = _apply_api_key(marine_url, marine_params)
        marine_response = _request_with_retry(marine_url, marine_params, label='point marine')
        marine_data = marine_response.json()

        # Fetch weather data (for wind and sunrise/sunset)
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "wind_speed_10m,wind_direction_10m",
            "daily": "sunrise,sunset",
            "timezone": "auto"
        }
        weather_url, weather_params = _apply_api_key(weather_url, weather_params)
        weather_response = _request_with_retry(weather_url, weather_params, label='point weather')
        weather_data = weather_response.json()

        # Parse sunrise/sunset times into a dict by date
        daily_data = weather_data.get('daily', {})
        sunrise_times = {}
        sunset_times = {}
        if 'sunrise' in daily_data and 'sunset' in daily_data:
            for i, date in enumerate(daily_data.get('time', [])):
                sunrise_times[date] = daily_data['sunrise'][i]
                sunset_times[date] = daily_data['sunset'][i]

        # Process the data
        marine_hourly = marine_data['hourly']
        weather_hourly = weather_data['hourly']
        time_points = pd.to_datetime(marine_hourly['time'])

        # Create a structured forecast
        forecast = []
        for i, time in enumerate(time_points):
            date_str = time.strftime('%Y-%m-%d')
            sunrise = sunrise_times.get(date_str)
            sunset = sunset_times.get(date_str)

            forecast.append({
                "time": time.strftime('%Y-%m-%d %H:%M'),
                "wave_height": marine_hourly['wave_height'][i],
                "wave_period": marine_hourly['wave_period'][i],
                "wave_direction": marine_hourly['wave_direction'][i],
                "wind_wave_height": marine_hourly['wind_wave_height'][i],
                "wind_wave_period": marine_hourly['wind_wave_period'][i],
                "wind_speed": weather_hourly['wind_speed_10m'][i] if i < len(weather_hourly['wind_speed_10m']) else None,
                "wind_direction": weather_hourly['wind_direction_10m'][i] if i < len(weather_hourly['wind_direction_10m']) else None,
                "sunrise": sunrise,
                "sunset": sunset,
            })

        return forecast

    except requests.exceptions.RequestException as e:
        print(f"Error fetching point data: {e}")
        return None
    except (KeyError, TypeError) as e:
        print(f"Error processing point data: {e}")
        return None

def _apply_api_key(url, params):
    """If OPEN_METEO_API_KEY is set, switch to the commercial endpoint."""
    api_key = os.environ.get('OPEN_METEO_API_KEY')
    if api_key:
        url = url.replace('open-meteo.com', 'customer-open-meteo.com')
        params['apikey'] = api_key
    return url, params

def _short_sleep(seconds):
    """Sleep in 1-second intervals so Gunicorn sync workers can still heartbeat."""
    end = time.time() + seconds
    while time.time() < end:
        time.sleep(min(1, end - time.time()))

def _request_with_retry(url, params, label='', max_retries=4):
    """Make a GET request with 429 retry logic using short sleeps."""
    if is_rate_limited():
        raise requests.exceptions.ConnectionError(f"Skipping {label} — rate limit cooldown active")
    for attempt in range(max_retries):
        response = requests.get(url, params=params)
        if response.status_code == 429:
            if attempt < max_retries - 1:
                wait = [5, 15, 30, 45][attempt]
                print(f"Rate limited ({label}), retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                _short_sleep(wait)
                continue
            else:
                # All retries exhausted — enter cooldown to stop hammering the API
                set_rate_limited()
                response.raise_for_status()
        response.raise_for_status()
        return response
    return response

def fetch_batched(url, base_params, all_lats, all_lons, batch_size=250):
    """
    Fetches data from Open-Meteo API in batches to avoid URL length limits.
    Returns a flat list of per-point JSON results.
    """
    all_results = []
    for i, start in enumerate(range(0, len(all_lats), batch_size)):
        if i > 0:
            _short_sleep(3)  # Pause between batches to respect rate limits
        end = start + batch_size
        batch_lats = all_lats[start:end]
        batch_lons = all_lons[start:end]
        params = dict(base_params)
        params["latitude"] = batch_lats
        params["longitude"] = batch_lons
        req_url, params = _apply_api_key(url, params)
        response = _request_with_retry(req_url, params, label=f'batch {i}')
        data = response.json()
        # Single-point responses are a dict, multi-point are a list
        if isinstance(data, list):
            all_results.extend(data)
        else:
            all_results.append(data)
    return all_results

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

def get_grid_weather_data(lat_min, lat_max, lon_min, lon_max, resolution):
    """
    Fetches gridded wave data from Marine API and wind data from Weather API.
    """
    try:
        # Round to avoid floating point precision issues in URL
        lats = np.round(np.arange(lat_min, lat_max + 0.001, resolution), 2)
        lons = np.round(np.arange(lon_min, lon_max + 0.001, resolution), 2)

        all_coords = list(product(lats, lons))
        all_lats = [float(coord[0]) for coord in all_coords]
        all_lons = [float(coord[1]) for coord in all_coords]

        # Fetch marine data in batches (wave height, wave period)
        marine_url = "https://marine-api.open-meteo.com/v1/marine"
        marine_base_params = {
            "hourly": "wave_height,wave_period,wave_direction",
            "timezone": "auto"
        }

        # Fetch weather data in batches (wind speed, wind direction)
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_base_params = {
            "hourly": "wind_speed_10m,wind_direction_10m",
            "timezone": "auto"
        }

        marine_data = fetch_batched(marine_url, marine_base_params, all_lats, all_lons)
        weather_data = fetch_batched(weather_url, weather_base_params, all_lats, all_lons)

        num_lats = len(lats)
        num_lons = len(lons)

        # Use marine data for time reference
        time_points = pd.to_datetime(marine_data[0]['hourly']['time'])
        num_times = len(time_points)

        # Initialize grids with NaN to handle missing data
        wave_height_grid = np.full((num_times, num_lats, num_lons), np.nan)
        wave_period_grid = np.full((num_times, num_lats, num_lons), np.nan)
        wind_speed_grid = np.full((num_times, num_lats, num_lons), np.nan)
        wind_direction_grid = np.full((num_times, num_lats, num_lons), np.nan)

        for i, (marine_point, weather_point) in enumerate(zip(marine_data, weather_data)):
            lat_index = i // num_lons
            lon_index = i % num_lons

            # Handle potential null values from marine API (land points return nulls)
            wave_heights = marine_point['hourly'].get('wave_height', [None] * num_times)
            wave_periods = marine_point['hourly'].get('wave_period', [None] * num_times)

            for t in range(min(num_times, len(wave_heights))):
                if wave_heights[t] is not None:
                    wave_height_grid[t, lat_index, lon_index] = wave_heights[t]
                if wave_periods[t] is not None:
                    wave_period_grid[t, lat_index, lon_index] = wave_periods[t]

            # Wind data from weather API
            wind_speeds = weather_point['hourly'].get('wind_speed_10m', [None] * num_times)
            wind_dirs = weather_point['hourly'].get('wind_direction_10m', [None] * num_times)

            for t in range(min(num_times, len(wind_speeds))):
                if wind_speeds[t] is not None:
                    wind_speed_grid[t, lat_index, lon_index] = wind_speeds[t]
                if wind_dirs[t] is not None:
                    wind_direction_grid[t, lat_index, lon_index] = wind_dirs[t]

        # Replace NaN with 0 for JSON serialization
        wave_height_grid = np.nan_to_num(wave_height_grid, nan=0.0)
        wave_period_grid = np.nan_to_num(wave_period_grid, nan=0.0)
        wind_speed_grid = np.nan_to_num(wind_speed_grid, nan=0.0)
        wind_direction_grid = np.nan_to_num(wind_direction_grid, nan=0.0)

        grid_forecast = {
            "lats": lats.tolist(),
            "lons": lons.tolist(),
            "times": [time.strftime('%Y-%m-%d %H:%M') for time in time_points],
            "wave_height": wave_height_grid.tolist(),
            "wave_period": wave_period_grid.tolist(),
            "wind_speed": wind_speed_grid.tolist(),
            "wind_direction": wind_direction_grid.tolist(),
        }

        return grid_forecast

    except requests.exceptions.RequestException as e:
        print(f"Error fetching grid data: {e}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Error processing grid data: {e}")
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

    # Create bounding box around center point (±0.5° lat, ±0.75° lon)
    lat_min, lat_max = center_lat - 0.5, center_lat + 0.5
    lon_min, lon_max = center_lon - 0.75, center_lon + 0.75
    resolution = 0.1  # degrees (~11 km cells) - gives 11x16=176 points (1 batch, no sleep delays)

    cache_key = f"map:{center_lat},{center_lon}"
    data = cached(cache_key, lambda: get_grid_weather_data(lat_min, lat_max, lon_min, lon_max, resolution))

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
    Two HTTP requests replace ~150+ Open-Meteo calls.
    """
    # Full WW3 global range at 3° effective resolution (stride=6 on native 0.5°)
    lat_range = "(-77.5):6:(77.5)"
    lon_range = "(0.0):6:(359.5)"

    # --- Fetch WW3 wave data (required) ---
    try:
        print("Fetching global wave data from ERDDAP (WW3)...")
        wave_json = _fetch_erddap_grid(
            server="pae-paha.pacioos.hawaii.edu",
            dataset="ww3_global",
            variables=["Thgt", "Tper", "Tdir"],
            time_range="(last-71):(last)",
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
            time_range="(last-23):(last)",
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
        def _parse_erddap_time(t):
            """Parse ERDDAP timestamp (may or may not have trailing Z, fractional seconds, etc.)."""
            for fmt in ('%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%S'):
                try:
                    return datetime.strptime(t, fmt)
                except ValueError:
                    continue
            return datetime.fromisoformat(t.replace('Z', '+00:00').replace('+00:00', ''))

        # Use wave times as the output time axis (72 hourly steps)
        output_times = wave['times']
        wave_dts = [_parse_erddap_time(t) for t in output_times]

        # Convert ERDDAP 0–360° lons back to -180..180 for the frontend
        lats = wave['lats']
        lons = [lon - 360 if lon > 180 else lon for lon in wave['lons']]

        num_times = len(output_times)
        num_lats = len(lats)
        num_lons = len(lons)

        # Wave data maps directly (same time axis)
        wave_height_out = wave['grids']['Thgt'].copy()
        wave_period_out = wave['grids']['Tper'].copy()
        wave_dir_out = wave['grids']['Tdir'].copy()
        wind_speed_out = np.zeros((num_times, num_lats, num_lons))
        wind_dir_out = np.zeros((num_times, num_lats, num_lons))

        # Convert wind U/V → speed (km/h) + direction, interpolated to hourly
        if wind and wind['times']:
            wind_dts = [_parse_erddap_time(t) for t in wind['times']]
            wind_u = wind['grids']['ugrd10m']
            wind_v = wind['grids']['vgrd10m']

            # Step 1: Spatially align wind grid to wave grid if needed
            if len(wind['lats']) == num_lats and len(wind['lons']) == num_lons:
                print(f"  Wind/wave grids match ({num_lats}x{num_lons})")
                u_aligned = wind_u
                v_aligned = wind_v
            else:
                print(f"  Wind grid ({len(wind['lats'])}x{len(wind['lons'])}) differs from wave grid ({num_lats}x{num_lons}), using nearest-neighbor interpolation")
                wind_lats_arr = np.array(wind['lats'])
                wind_lons_arr = np.array(wind['lons'])
                nn_lat = np.array([np.argmin(np.abs(wind_lats_arr - la)) for la in wave['lats']])
                nn_lon = np.array([np.argmin(np.abs(wind_lons_arr - lo)) for lo in wave['lons']])
                lat_mesh, lon_mesh = np.meshgrid(nn_lat, nn_lon, indexing='ij')
                u_aligned = np.array([wind_u[t][lat_mesh, lon_mesh] for t in range(len(wind_dts))])
                v_aligned = np.array([wind_v[t][lat_mesh, lon_mesh] for t in range(len(wind_dts))])

            # Step 2: Linearly interpolate 3-hourly U/V to each hourly wave time step
            wind_secs = np.array([(wdt - wind_dts[0]).total_seconds() for wdt in wind_dts])
            for ti, wdt in enumerate(wave_dts):
                t_sec = (wdt - wind_dts[0]).total_seconds()
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

            print(f"  Wind data interpolated from {len(wind_dts)} to {num_times} hourly time steps")

        # Replace NaN with 0 for JSON serialization (land points)
        wave_height_out = np.nan_to_num(wave_height_out, nan=0.0)
        wave_period_out = np.nan_to_num(wave_period_out, nan=0.0)
        wave_dir_out = np.nan_to_num(wave_dir_out, nan=0.0)
        wind_speed_out = np.nan_to_num(wind_speed_out, nan=0.0)
        wind_dir_out = np.nan_to_num(wind_dir_out, nan=0.0)

        # Format times to match frontend expectation
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