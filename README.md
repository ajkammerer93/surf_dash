# Free Surf Forecast

An open-source, data-forward surf forecast dashboard that aggregates wave, wind, and tide data from NOAA, NDBC, CDIP, and Open-Meteo into an interactive single-page app — plus a topic cluster of evergreen forecasting guides.

**Live:** [freesurfforecast.com](https://freesurfforecast.com/)

**Current version:** v0.10.0

## Dashboard

- **Next 6 Hours Hero** — Always-visible decision-relevant window above the panel grid: hour-by-hour wave height, period, wind, and condition color bar.
- **Current Conditions** — Wave height/period/direction, wind speed and direction, air and water temperature, and tide stage with hover popups for coastline-relative direction diagrams and a 24-hour tide sparkline. Includes a **wetsuit recommendation** from water temperature.
- **Wave & Wind Chart** — 7-day time-series with surf condition color strip at top, day/night band background, tide overlay in the tooltip with high/low markers, and trend arrows. Forecast confidence badge with a **per-hour mini sparkline** showing skill decay from lead 0 (~85% NOMADS, 65% Open-Meteo) out to +168h.
- **Ocean Basin + Local Detail Maps** — Wave height, period, and wind overlays at 0.1° resolution with land-clipped ocean rendering. Unified time slider animates all panels.
- **Buoy Observations** — NDBC and CDIP real-time data with 1D and 2D spectra, prominent station distance, and a "deep-water buoy" warning when the station is more than 100 km offshore.
- **Surf Cameras** — Nearest live webcams via Windy, with safe DOM-rendered attribution.
- **Session Planner** — Scored 3-hour windows (0–100) factoring swell direction × beach orientation, period, wind, and tide stage.
- **Skill-Level Selector** — Beginner / intermediate / advanced labels and scoring.
- **Swell Narrative** — Ocean basin overlay describing swell type, origin distance, trend, and peak-hour offset; flags period-vs-steepness disagreements.
- **Forecast Export** — Best-session ICS calendar files and full forecast CSV download.
- **Condition Alerts** — Opt-in browser notifications when scoring crosses thresholds.
- **Location Comparison** — Side-by-side comparison of 2–3 spots.
- **Nearby Spots** — Server-rendered "Within 150 km" sidebar with crawlable internal links on every forecast page.
- **Dark / Light Theme** — System preference by default, manual toggle, persisted across sessions.
- **PWA** — Installable, offline caching via service worker (network-first for HTML).
- **Mobile** — Responsive layout, 44 px tap targets on all dropdown menus.

## Content & SEO

- **`/forecast/<slug>`** — 126 location-specific forecast pages with unique meta descriptions, server-rendered SSR blocks, and per-location OG images (1200×630 PNG with live wave height, period, wind, condition).
- **`/learn`** — 8 long-form articles (~800 words each): forecast reading, period, swell windows, tide, beach orientation, buoys, storms, surf safety. Each with `Article` + `BreadcrumbList` JSON-LD.
- **`/glossary`** — 44 surf and oceanography terms across 7 sections with `DefinedTermSet` JSON-LD.
- **`/regions/<slug>`** — 10 regional landing pages (OBX, NC, VA, NJ, NY, FL Space Coast, SoCal, NorCal, Oregon, Hawaii) with `CollectionPage` JSON-LD and a spot roster.
- **Per-location `TouristAttraction` JSON-LD** with `geo` + `addressRegion`.
- **Sitemap** — All forecast pages at `hourly` `changefreq`; tier priorities (top-traffic spots at 0.9, others 0.6); `/learn`, `/regions`, `/glossary` included.

## Data Sources

| Data | Source | Cost |
|---|---|---|
| Wave height, peak period, direction | [Open-Meteo Marine API](https://open-meteo.com/) / NOAA WW3 (ERDDAP) / NOMADS GFS-Wave | Free |
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

Available at `http://localhost:5000`.

### Run Tests

```bash
pytest tests/                                          # full suite (104 tests)
pytest tests/test_seo.py -v                            # 57 SEO/integration tests
pytest tests/test_units.py -v                          # 47 pure-function unit tests
python scripts/seo_report.py                           # 15-check SEO health report (local)
python scripts/seo_report.py --base-url https://freesurfforecast.com   # live site
```

### Logging

Logs go to stdout via the standard `logging` module. Override level/format with env vars:

```bash
LOG_LEVEL=DEBUG python app.py
```

## Architecture

| File | Purpose |
|---|---|
| `app.py` | Flask backend — API endpoints, SEO routes, OG image rendering, SSR helpers |
| `templates/index.html` | Dashboard SPA (HTML + embedded JS + CSS, Bootstrap 5 / Chart.js / Leaflet via CDN) |
| `templates/about.html`, `locations.html`, `glossary.html` | Stand-alone pages |
| `templates/learn/{index,article}.html` | `/learn` topic cluster |
| `templates/regions/{index,page}.html` | `/regions` landing pages |
| `learn_articles.py` | Article content as Python dicts (no markdown library) |
| `region_pages.py` | Region metadata + spot rosters |
| `surf_cameras.json` | 157 webcam entries → 126 deduped locations |
| `tests/test_seo.py`, `tests/test_units.py` | Test suite |
| `static/sw.js` | Service worker (network-first for HTML, cache-first for CDN libs) |
| `.github/FUNDING.yml` | GitHub Sponsor button → Buy Me a Coffee |
| `render.yaml` | Render deploy config (Gunicorn `--workers 1 --threads 4 --preload`) |

### Page Routes

| Route | Purpose |
|---|---|
| `/` | Dashboard (default: Surf City, NC) |
| `/forecast/<slug>` | Dashboard for a specific location (126 slugs) |
| `/locations` | Index of all forecast locations grouped by state |
| `/learn`, `/learn/<slug>` | Learn topic cluster (8 articles) |
| `/glossary` | Surf & oceanography glossary (44 terms) |
| `/regions`, `/regions/<slug>` | Regional landing pages (10 regions) |
| `/about` | About page |
| `/og/<slug>.png` | Per-location OpenGraph image (1200×630 PNG) |
| `/sitemap.xml`, `/robots.txt` | SEO surfaces |

Old `/?lat=X&lon=Y&name=Z` URLs 301-redirect to `/forecast/<slug>`.

### API Endpoints

| Route | Purpose |
|---|---|
| `GET /api/forecast?lat=&lon=` | Point forecast with confidence per-hour array |
| `GET /api/map-forecast?lat=&lon=` | Local gridded data for map overlays |
| `GET /api/ocean-basin?lat=&lon=` | Coarse ocean basin data |
| `GET /api/tides?lat=&lon=` | 7-day tide predictions |
| `GET /api/webcams?lat=&lon=` | Nearest webcams |
| `GET /api/buoys?lat=&lon=` | Buoy observations + spectra |
| `GET /api/beach-orientation?lat=&lon=` | Coastline facing direction |
| `GET /api/swell-narrative?lat=&lon=` | Swell type + steepness + narrative |

All endpoints are stateless; in-process TTL caching across requests.

## Deployment

Deployed on [Render.com](https://render.com/) behind Cloudflare (DNS, CDN, DDoS, free analytics). See `render.yaml`.

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 4 --preload
```

The frontend has a security baseline of: Content-Security-Policy, X-Frame-Options DENY, Referrer-Policy strict-origin-when-cross-origin, Permissions-Policy, and SRI hashes on all CDN script and style tags.

## Support

If this site has helped you score a session, [buy me a coffee ☕](https://buymeacoffee.com/freesurfforecast) to keep the servers running. The site is and will stay free, no ads, no signup.

## License

MIT.
