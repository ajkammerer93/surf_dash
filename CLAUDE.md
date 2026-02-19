# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Surf Forecast Dashboard — a Flask web app that aggregates surf forecasting data (waves, wind, tides) from multiple free APIs and presents it as an interactive single-page dashboard.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (starts Flask dev server on http://localhost:5000 with debug/auto-reload)
python app.py

# Production (used by Render deployment)
gunicorn app:app --bind 0.0.0.0:$PORT
```

There is no test suite, linter, or build step configured.

## Architecture

The entire app is two files:

- **`app.py`** — Flask backend with 4 API endpoints that proxy/aggregate external data
- **`templates/index.html`** — Self-contained SPA frontend (HTML + embedded JS/CSS) using Bootstrap 5, Chart.js, and Leaflet maps via CDN

### Backend API Endpoints

| Route | Purpose | External API |
|---|---|---|
| `GET /api/forecast?lat=&lon=` | Point forecast (wave height/period/direction, wind, sunrise/sunset) | Open-Meteo Marine + Weather |
| `GET /api/map-forecast?lat=&lon=` | Gridded local area data (0.1° resolution) for 3 map overlays | Open-Meteo Marine + Weather |
| `GET /api/ocean-basin?lat=&lon=` | Coarse ocean basin data (3° resolution, ±15° lat/±20° lon) | Open-Meteo Marine |
| `GET /api/tides?lat=&lon=` | 7-day tide predictions from nearest NOAA reference station | NOAA CO-OPS |

All endpoints are stateless with no caching. Default location is Surf City, NC (34.43, -77.55).

### Frontend Data Flow

1. User picks location via Leaflet map modal → stores lat/lon in global state
2. JS fetches all 4 endpoints in parallel
3. Ocean basin data renders on a large-scale Leaflet map (wave height + period)
4. Local grid data renders on 3 maps (wave period, wave height, wind speed+direction) with a time slider for animation
5. Point forecast data populates Chart.js time-series charts (wave, wind, tides)
6. Surfing-specific logic: offshore wind detection (NW ±45° or <5 km/h), daylight filtering (5am–9pm)

### Key Implementation Details

- `find_nearest_tide_station()` uses haversine distance to search a hardcoded list of NOAA reference stations
- Grid visualization uses colored Leaflet rectangles with a blue→cyan→lime→yellow→red gradient
- Wind arrows are drawn every 5th grid point to reduce clutter, rotated by meteorological direction
- Reverse geocoding uses Nominatim (OpenStreetMap) with a 1 req/sec rate limit

## Git Commits

Do not include "Co-Authored-By" lines, "authored by Claude", or any AI attribution in commit messages.

## Deployment

Deployed on Render.com (see `render.yaml`). Python 3.11, Gunicorn WSGI server.
