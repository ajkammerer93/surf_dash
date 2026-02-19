# Free Surf Forecast

A real-time surf forecast dashboard that aggregates wave, wind, and tide data from multiple free APIs into an interactive single-page application.

**Live:** [freesurfforecast.com](https://freesurfforecast.com/)

## Features

- **Ocean Basin Maps** — Large-scale wave height and period visualization (3° resolution, ~330 km cells)
- **Local Detail Maps** — High-resolution wave and wind overlays (0.1° resolution, ~11 km cells)
- **Unified Time Slider** — Animate all 5 map panels through hourly forecast data
- **Wave & Wind Charts** — Time-series forecast with wave height, wind speed, and wind direction arrows
- **Tidal Predictions** — 7-day tide chart from the nearest NOAA reference station with high/low markers
- **Daylight Forecast Table** — Filtered to surfable hours (5am–9pm) with offshore wind detection
- **Dynamic Location Selection** — Click anywhere on a global map to get a forecast; reverse geocoded location names via OpenStreetMap

## Data Sources

| Data | Source | Cost |
|---|---|---|
| Wave height, period, direction | [Open-Meteo Marine API](https://open-meteo.com/) | Free |
| Wind speed, direction | [Open-Meteo Weather API](https://open-meteo.com/) | Free |
| Tide predictions | [NOAA CO-OPS API](https://tidesandcurrents.noaa.gov/api/) | Free |
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

The entire app is two files:

- **`app.py`** — Flask backend with 4 API endpoints that proxy and aggregate external data
- **`templates/index.html`** — Self-contained SPA frontend (HTML + embedded JS/CSS) using Chart.js and Leaflet via CDN

### API Endpoints

| Route | Purpose |
|---|---|
| `GET /api/forecast?lat=&lon=` | Point forecast (wave, wind, sunrise/sunset) |
| `GET /api/map-forecast?lat=&lon=` | Gridded local area data for 3 map overlays |
| `GET /api/ocean-basin?lat=&lon=` | Coarse ocean basin data for 2 basin maps |
| `GET /api/tides?lat=&lon=` | 7-day tide predictions from nearest NOAA station |

All endpoints are stateless with no caching. Default location is Surf City, NC.

## Deployment

Deployed on [Render.com](https://render.com/) — see `render.yaml` for configuration.

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

## License

This project is provided as-is for personal and educational use.
