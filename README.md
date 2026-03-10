# Free Surf Forecast

A real-time surf forecast dashboard that aggregates wave, wind, and tide data from multiple free APIs into an interactive single-page application.

**Live:** [freesurfforecast.com](https://freesurfforecast.com/)

## Features

- **Dark/Light Theme** — Toggle between dark and light modes; respects OS preference, persists to localStorage
- **Ocean Basin Maps** — Large-scale wave height and period visualization with swell narrative overlay
- **Local Detail Maps** — High-resolution wave and wind overlays (0.1° resolution) with ocean-clipped rendering
- **Unified Time Slider** — Animate all map panels through hourly forecast data with adjustable playback speed
- **Current Conditions** — At-a-glance 3×2 panel with wave height/period/direction, wind speed+direction, air/water temperature, and upcoming tides with hover popups for coastline-relative direction diagrams and a 24-hour tide sparkline
- **Wave & Wind Charts** — Time-series forecast with wave height, wind speed, direction arrows, surf condition classification, tide overlay in tooltip with high/low event markers, and period/height trend arrows
- **Location Comparison** — Side-by-side comparison of 2–3 locations with scored conditions
- **Session Planner** — Scored 3-hour surf windows (0–100) highlighting the best times to paddle out
- **Forecast Export** — Export session windows to ICS calendar files or download forecast data as CSV
- **Skill Level Selector** — Beginner/intermediate/advanced modes with tailored descriptions and scoring
- **Forecast Literacy** — Plain-English descriptions for wave height, period, and wind conditions
- **Swell Narrative** — Ocean basin overlay showing swell type, origin distance, arrival ETA, and trend
- **Model Confidence** — Forecast confidence badge (High/Moderate/Low) based on data quality and lead time
- **Condition Alerts** — Opt-in browser notifications when conditions improve (via Notifications API)
- **Surf Cameras** — Nearest live webcam feeds via Windy
- **Buoy Data** — Real-time NDBC/CDIP buoy observations with wave spectrum interpretation
- **Beach Orientation** — Auto-detected coastline facing direction for accurate offshore/onshore classification
- **Dynamic Location Selection** — Click anywhere on a global map; reverse geocoded via OpenStreetMap
- **PWA Support** — Installable as a progressive web app with offline caching via service worker

## Data Sources

| Data | Source | Cost |
|---|---|---|
| Wave height, period, direction | [Open-Meteo Marine API](https://open-meteo.com/) / NOAA WW3 (ERDDAP) / NOMADS | Free |
| Wind speed, direction | [Open-Meteo Weather API](https://open-meteo.com/) / GFS (ERDDAP) | Free |
| Tide predictions | [NOAA CO-OPS API](https://tidesandcurrents.noaa.gov/api/) | Free |
| Buoy observations | [NDBC](https://www.ndbc.noaa.gov/) / [CDIP](https://cdip.ucsd.edu/) | Free |
| Surf cameras | [Windy Webcams API](https://api.windy.com/) | Free tier |
| Coastline data | [OpenStreetMap Overpass API](https://overpass-api.de/) | Free |
| Reverse geocoding | [Nominatim (OpenStreetMap)](https://nominatim.openstreetmap.org/) | Free |

## Getting Started

### Prerequisites

- Python 3.11+

### Installation

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
python app.py
```

The dashboard will be available at `http://localhost:5000`.

## Architecture

The app is three files:

- **`app.py`** — Flask backend with API endpoints, SEO routes, and SSR content
- **`templates/index.html`** — Self-contained SPA frontend (HTML + embedded JS/CSS) using Chart.js and Leaflet via CDN
- **`templates/locations.html`** — Location index page listing all forecast locations by state

### Page Routes

| Route | Purpose |
|---|---|
| `/` | Dashboard (default location: Surf City, NC) |
| `/forecast/<slug>` | Dashboard for a specific location (e.g., `/forecast/virginia-beach`) |
| `/locations` | Index of all 126 forecast locations grouped by state |
| `/about` | About page |

Old query-param URLs (`/?lat=X&lon=Y&name=Z`) are 301-redirected to `/forecast/<slug>` for known locations.

### API Endpoints

| Route | Purpose |
|---|---|
| `GET /api/forecast?lat=&lon=` | Point forecast (wave, wind, sunrise/sunset) |
| `GET /api/map-forecast?lat=&lon=` | Gridded local area data for map overlays |
| `GET /api/ocean-basin?lat=&lon=` | Coarse ocean basin data for basin maps |
| `GET /api/tides?lat=&lon=` | 7-day tide predictions from nearest NOAA station |
| `GET /api/webcams?lat=&lon=` | Nearest surf camera feeds |
| `GET /api/buoys?lat=&lon=` | Real-time buoy observations and spectra |
| `GET /api/beach-orientation?lat=&lon=` | Coastline facing direction |
| `GET /api/swell-narrative?lat=&lon=` | Swell type, origin, and trend narrative |

All endpoints are stateless. Default location is Surf City, NC.

## Deployment

Deployed on [Render.com](https://render.com/) — see `render.yaml` for configuration.

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

## License

This project is provided as-is for personal and educational use.
