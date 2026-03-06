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

```bash
# Run SEO tests
pytest tests/test_seo.py -v

# Generate SEO health report (local)
python scripts/seo_report.py

# Generate SEO health report (live site)
python scripts/seo_report.py --base-url https://freesurfforecast.com
```

There is no linter or build step configured.

## Architecture

The app is three files:

- **`app.py`** — Flask backend with 8 API endpoints + SEO page routes that proxy/aggregate external data
- **`templates/index.html`** — Self-contained SPA frontend (HTML + embedded JS/CSS) using Bootstrap 5, Chart.js, and Leaflet maps via CDN
- **`templates/locations.html`** — Location index page listing all forecast locations grouped by state

### Page Routes

| Route | Purpose |
|---|---|
| `/` | Dashboard (default location, respects localStorage) |
| `/forecast/<slug>` | Dashboard for a known location (126 locations from `surf_cameras.json`) |
| `/locations` | Index of all locations grouped by state |
| `/about` | About page |

Old `/?lat=X&lon=Y&name=Z` URLs 301-redirect to `/forecast/<slug>` for known locations via `@app.before_request`.

### Backend API Endpoints

| Route | Purpose | External API |
|---|---|---|
| `GET /api/forecast?lat=&lon=` | Point forecast (wave height/period/direction, wind, sunrise/sunset) | Open-Meteo Marine + Weather (fallback from NOMADS/WW3) |
| `GET /api/map-forecast?lat=&lon=` | Gridded local area data (0.1° resolution) for 3 map overlays | WW3 (ERDDAP) + GFS (ERDDAP) |
| `GET /api/ocean-basin?lat=&lon=` | Coarse ocean basin data (3° resolution, ±15° lat/±20° lon) | WW3 (ERDDAP) + GFS (ERDDAP) |
| `GET /api/tides?lat=&lon=` | 7-day tide predictions from nearest NOAA reference station | NOAA CO-OPS |
| `GET /api/webcams?lat=&lon=` | Nearest surf camera feeds | Windy Webcams API |
| `GET /api/buoys?lat=&lon=` | Real-time buoy observations + wave spectra | NDBC + CDIP |
| `GET /api/beach-orientation?lat=&lon=` | Coastline facing direction for offshore/onshore classification | OpenStreetMap Overpass (cached) |
| `GET /api/swell-narrative?lat=&lon=` | Swell type, origin distance, trend | Derived from forecast data |

All endpoints are stateless with TTL caching. Default location is Surf City, NC (34.43, -77.55).

### Frontend Data Flow

1. User picks location via Leaflet map modal → stores lat/lon in global state
2. JS fetches all endpoints in parallel (forecast, tides, basin, maps, buoys, webcams, orientation, narrative)
3. Ocean basin data renders on a large-scale Leaflet map with swell narrative overlay
4. Local grid data renders on 3 maps (wave period, wave height, wind speed+direction) with a time slider for animation
5. Point forecast data populates Chart.js time-series charts (wave, wind, tides)
6. Session planner scores 3-hour windows and highlights best surf times
7. Surf condition classification: `classifySurfConditionGlobal()` uses `BEACH_FACING_DIRECTION` for offshore/onshore detection

### Key Implementation Details

- `find_nearest_tide_station()` uses haversine distance to search a hardcoded list of NOAA reference stations
- Grid visualization uses canvas-based ocean clipping (land polygons from world-atlas TopoJSON)
- Wind arrows are drawn every 5th grid point to reduce clutter, rotated by meteorological direction
- Reverse geocoding uses Nominatim (OpenStreetMap) with a 1 req/sec rate limit
- Wave direction interpolation uses sin/cos decomposition to handle 0°/360° wrap
- Group velocity formula: `0.78 * T` (deep-water approximation, g*T / 4π)
- Swell classification: ≥12s ground, ≥9s medium-period, <9s wind swell
- Scoring engine: `scoreHour()` weights wave height, period, wind condition (speed-differentiated), and tide movement
- `rebuildForecastTable()` is the single source of truth for the daylight forecast table (called on initial load and skill level change)
- `CONDITION_LABELS` is the global label map for surf conditions; `CONDITION_COLORS` was removed
- Basin narrative overlay must be detached before Leaflet init and re-appended after (Leaflet clears its container)
- Current Conditions uses `#current-conditions-stats` scoped CSS grid (3×2). Base `.stat-list` stays flex-column for the buoy panel. `.stat-split` provides left/right justified pairs within a cell.
- `window._tideHighLow` stores raw high/low tide array. Used by `updateConditionsTide()`. Cleared on location change alongside `_tideData`.
- `drawConditionsDiagram()` is a standalone version of the wave chart tooltip diagram, used for hover popups on Wave Direction and Wind cells. The original `drawTooltipDiagram()` is scoped inside `loadPointForecast`.
- Hover popups (direction diagrams, tide sparkline) use `position: fixed` with JS-calculated placement to avoid clipping by panel overflow.
- `LOCATION_BY_SLUG` and `SLUG_BY_COORDS` dicts built at startup from `surf_cameras.json` (deduped by lat/lon). `slugify()` lowercases, strips non-alphanumeric, hyphenates.
- `_get_ssr_summary()` is cache-only — renders forecast into `<section id="ssr-summary">` for crawlers without blocking page render. Removed in `loadPointForecast` success handler (NOT at `DOMContentLoaded` — must stay visible if APIs fail in Google's sandbox).
- `window._serverLocation` is only set on `/forecast/` routes (checked via `window.location.pathname`). On `/`, it's null so localStorage takes priority.
- `window._slugByCoords` JS lookup map injected via Jinja for client-side slug resolution on location change.
- Locations page groups by state using explicit `name_to_state` dict for locations without state suffix (OBX beaches, OC MD cams, etc.).

## Git Commits

Do not include "Co-Authored-By" lines, "authored by Claude", or any AI attribution in commit messages.

Only commit and push as GitHub user **ajkammerer93**. Do not change the git user config.

## Deployment

Deployed on Render.com (see `render.yaml`). Python 3.11, Gunicorn WSGI server.
