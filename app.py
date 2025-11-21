from flask import Flask, render_template, jsonify
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)

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
        marine_response = requests.get(marine_url, params=marine_params)
        marine_response.raise_for_status()
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
        weather_response = requests.get(weather_url, params=weather_params)
        weather_response.raise_for_status()
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

import traceback
from itertools import product

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

        # Fetch marine data (wave height, wave period)
        marine_url = "https://marine-api.open-meteo.com/v1/marine"
        marine_params = {
            "latitude": all_lats,
            "longitude": all_lons,
            "hourly": "wave_height,wave_period,wave_direction",
            "timezone": "auto"
        }

        # Fetch weather data (wind speed, wind direction)
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": all_lats,
            "longitude": all_lons,
            "hourly": "wind_speed_10m,wind_direction_10m",
            "timezone": "auto"
        }

        marine_response = requests.get(marine_url, params=marine_params)
        marine_response.raise_for_status()
        marine_data = marine_response.json()

        weather_response = requests.get(weather_url, params=weather_params)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

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
    return render_template('index.html')

# Route for the API to get point forecast data
@app.route('/api/forecast')
def forecast():
    """
    Provides weather forecast data for a single point as JSON.
    """
    # Default to Surf City, North Carolina
    lat = 34.42711
    lon = -77.54608
    
    data = get_point_weather_data(lat, lon)
    
    if data:
        return jsonify(data)
    else:
        return jsonify({"error": "Could not retrieve weather data."}), 500

# Route for the API to get gridded forecast data
@app.route('/api/map-forecast')
def map_forecast():
    """
    Provides gridded weather forecast data as JSON for a specified bounding box.
    """
    # Tighter bounding box around Surf City, North Carolina
    lat_min, lat_max = 34.0, 35.0
    lon_min, lon_max = -78.0, -76.5
    resolution = 0.25 # degrees (~28 km cells) - gives 5x7=35 points

    data = get_grid_weather_data(lat_min, lat_max, lon_min, lon_max, resolution)

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

def get_ocean_basin_data(center_lat, center_lon):
    """
    Fetches wave data for a large ocean basin area (coarse resolution).
    """
    try:
        # Define a large bounding box for the Atlantic Ocean
        # Extended bounds for zoom-out capability
        lat_min = 15
        lat_max = 50
        lon_min = -82
        lon_max = -40

        # Use 3 degree resolution for better coverage (~330 km cells, ~195 points)
        resolution = 3.0
        lats = np.arange(lat_min, lat_max + resolution, resolution)
        lons = np.arange(lon_min, lon_max + resolution, resolution)

        all_coords = list(product(lats, lons))
        all_lats = [coord[0] for coord in all_coords]
        all_lons = [coord[1] for coord in all_coords]

        # Fetch marine data
        marine_url = "https://marine-api.open-meteo.com/v1/marine"
        marine_params = {
            "latitude": all_lats,
            "longitude": all_lons,
            "hourly": "wave_height,wave_period,wave_direction",
            "timezone": "auto"
        }

        response = requests.get(marine_url, params=marine_params)
        response.raise_for_status()
        data = response.json()

        num_lats = len(lats)
        num_lons = len(lons)

        # Just get first time point for the basin overview
        time_point = pd.to_datetime(data[0]['hourly']['time'][0])

        wave_height_grid = np.full((num_lats, num_lons), np.nan)
        wave_period_grid = np.full((num_lats, num_lons), np.nan)
        wave_direction_grid = np.full((num_lats, num_lons), np.nan)

        for i, point_data in enumerate(data):
            lat_index = i // num_lons
            lon_index = i % num_lons

            wh = point_data['hourly'].get('wave_height', [None])[0]
            wp = point_data['hourly'].get('wave_period', [None])[0]
            wd = point_data['hourly'].get('wave_direction', [None])[0]

            if wh is not None:
                wave_height_grid[lat_index, lon_index] = wh
            if wp is not None:
                wave_period_grid[lat_index, lon_index] = wp
            if wd is not None:
                wave_direction_grid[lat_index, lon_index] = wd

        # Replace NaN with 0 for JSON
        wave_height_grid = np.nan_to_num(wave_height_grid, nan=0.0)
        wave_period_grid = np.nan_to_num(wave_period_grid, nan=0.0)
        wave_direction_grid = np.nan_to_num(wave_direction_grid, nan=0.0)

        return {
            "lats": lats.tolist(),
            "lons": lons.tolist(),
            "time": time_point.strftime('%Y-%m-%d %H:%M'),
            "wave_height": wave_height_grid.tolist(),
            "wave_period": wave_period_grid.tolist(),
            "wave_direction": wave_direction_grid.tolist(),
            "center": {"lat": center_lat, "lon": center_lon}
        }

    except Exception as e:
        print(f"Error fetching ocean basin data: {e}")
        traceback.print_exc()
        return None

# Route for ocean basin data
@app.route('/api/ocean-basin')
def ocean_basin():
    """
    Provides wave data for the ocean basin around the forecast location.
    """
    # Center on Surf City, NC
    center_lat = 34.42711
    center_lon = -77.54608

    data = get_ocean_basin_data(center_lat, center_lon)

    if data:
        return jsonify(data)
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
    """
    # Surf City, NC coordinates (same as forecast location)
    target_lat = 34.42711
    target_lon = -77.54608

    # Find nearest tide station
    station = find_nearest_tide_station(target_lat, target_lon)

    if not station:
        return jsonify({"error": "Could not find a nearby tide station."}), 500

    # Get tide data for that station
    data = get_tide_data(station["id"])

    if data:
        # Include station info in response
        data["station"] = station
        return jsonify(data)
    else:
        return jsonify({"error": f"Could not retrieve tide data for station {station['name']}."}), 500

if __name__ == '__main__':
    app.run(debug=True)