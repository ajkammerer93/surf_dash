# Free Surf Forecast

An open-source, data-forward surf forecast dashboard that aggregates wave, wind, and tide data from NOAA, NDBC, CDIP, and Open-Meteo into an interactive single-page app — plus a topic cluster of evergreen forecasting guides.

**Live:** [freesurfforecast.com](https://freesurfforecast.com/)

**Current version:** v0.11.51

## Dashboard

- **Next 6 Hours Hero** — Always-visible decision window with a **go/no-go verdict** (Go surf / Worth a paddle / Maybe / Skip it) for the current session plus the best upcoming window in the next 48 hours, above hour-by-hour wave height, period, wind, and condition color bars.
- **Swell Map** — One zoom-adaptive map replaces separate basin/local panes. The global swell field renders as a smooth raster (perceptual color ramp, shoreline-masked, single image overlay per timestep — no per-frame vector work); the high-resolution local grid blends over it in the same scale. Zoomed out: swell-direction arrows, the spot's swell-window cone, groundswell arrival rings (1–4 days), and an estimated storm-origin marker. Zoomed in: nearby-spot labels with live heights (click to switch spots), camera markers, and buoy markers — all with hover tooltips. Animated wind particles at every zoom (high-res local wind when zoomed in), paused off-screen and on hidden tabs. Embedded time slider with play/speed controls and a cursor probe that reads out conditions under the pointer.
- **Current Conditions** — Wave height/period/direction, wind, air and water temperature, tide stage, hover direction diagrams, tide sparkline, and a wetsuit recommendation.
- **Wave & Wind Chart** — 7-day time-series built for at-a-glance reading: hourly stacked bars where height = surf size, the faded cap = wind chop, and the bar color = surf condition (colorblind-validated scale, green offshore → rose blown-out); a wind lane of direction arrows with speeds above the plot; a tide lane with high/low times below it; a swell period + direction row along the bottom; day/night bands; and a rich hover tooltip with a wind/swell direction diagram and per-hour forecast confidence.
- **Buoy Observations** — NDBC and CDIP real-time data with 1D and 2D spectra and a deep-water-buoy caveat beyond 100 km.
- **Surf Cameras** — Nearest webcams with safe DOM-rendered attribution, preferring embeddable live streams over link-outs. YouTube live surf cams (owner-enabled embedding only, official privacy-enhanced player) are curated via a scanner + review pipeline; providers that opt out of embedding (e.g. SurfChex) render as designed link-out cards.
- **Session Planner** — Scored 3-hour windows (0–100) factoring swell direction × beach orientation, period, wind, and tide. The "best window" highlight only appears when a window clears the Good (55+) bar.
- **Any coastline, not just a spot list** — the 145 listed spots are a starting point, not the limit. Drop a pin anywhere on the map and you get the same 7-day wave, wind and swell forecast, plus nearby cams; tides and buoy observations are US-only, since they come from NOAA CO-OPS and NDBC/CDIP.
- **Geolocation** — "Use my location" snaps to the nearest of 145 spots; first-time visitors are offered it on the welcome card, never auto-prompted.
- **Units** — Imperial/metric toggle (ft/mph ↔ m/km·h, °F ↔ °C) across every chart, map, table, and export; scoring is unit-independent.
- **Condition Alerts** — Browser notifications gated on a user-set score threshold (Fair 35+ / Good 55+ / Prime 75+). No email required.
- **Offline** — The service worker serves the last-loaded forecast for up to 24 hours offline, with the staleness badge showing the data's true age.
- **Skill-Level Selector**, **Location Comparison**, **Forecast Export** (ICS + CSV), **Dark/Light theme**, installable **PWA**, responsive mobile layout.

## Content, SEO & Distribution

- **`/forecast/<slug>`** — 145 location pages (East and West Coasts, Gulf, **Great Lakes**, Hawaii, **Puerto Rico**) with unique meta, SSR summaries kept warm by a background cache warmer, and per-location OG images.
- **`/learn`** — 8 long-form forecasting guides; **`/glossary`** — 44 terms with JSON-LD.
- **`/regions/<slug>`** — 16 regional landing pages including Great Lakes, Puerto Rico, Delmarva, New England, South Carolina & Georgia, and the Gulf Coast.
- **`/compare/surfline`**, **`/compare/magicseaweed`** — honest comparison pages for surfers looking for a free alternative.
- **`/embed/<slug>`** — a free iframe-able live forecast card for surf shops and beach-town sites (the only route that permits framing).
- **`/social/daily/<region>.{png,jpg}`** + **`/social/story/<region>.jpg`** + **`/api/social-card/<region>`** — Instagram-ready regional report cards (4:5 feed and 9:16 story) with captions, each carrying the verified 30-day forecast error. `scripts/instagram_publish.py` publishes a feed post and a matching story daily on two schedules, one per coast, so each region lands near its own local dawn.
- **`/accuracy`** — public forecast-verification report: a scheduled pipeline snapshots the site's own forecasts at 21 NDBC buoy locations every 6 hours, scores them against observed wave heights, and publishes rolling 30-day bias/MAE/RMSE stats (raw JSON at `/api/accuracy`). Where a buoy shows a persistent, well-sampled bias, `/api/forecast` automatically applies a capped correction for nearby spots — and always verifies against raw model output so the correction can't mask true model error.
  The claim is auditable rather than asserted: `/accuracy/stations.csv` for per-station stats, and every scored pair is published on the `verification-data` branch alongside the scoring script.
- **`/faq`** — surf-forecasting FAQ with `FAQPage` JSON-LD targeting question queries.
- **Sitemap** with tiered priorities; per-location `TouristAttraction` JSON-LD.

## Data Sources

| Data | Source | Cost |
|---|---|---|
| Wave height, peak period, direction | [Open-Meteo Marine API](https://open-meteo.com/) / NOAA WW3 via ERDDAP (CoastWatch + PacIOOS mirrors) | Free |
| Wind speed, direction, air temp | [Open-Meteo Weather API](https://open-meteo.com/) with NOAA CoastWatch ERDDAP (GFS) fallback | Free |
| Tide predictions | [NOAA CO-OPS API](https://tidesandcurrents.noaa.gov/api/) (non-tidal waters flagged) | Free |
| Buoy observations | [NDBC](https://www.ndbc.noaa.gov/) / [CDIP](https://cdip.ucsd.edu/) | Free |
| Surf cameras | Curated YouTube live embeds + [Windy Webcams API](https://api.windy.com/) + provider link-outs | Free tier |
| Coastline data | [OpenStreetMap Overpass API](https://overpass-api.de/), world-atlas land polygons | Free |
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
pytest tests/                                          # full suite (476 tests)
pytest tests/test_seo.py -v                            # SEO/integration tests
pytest tests/test_units.py -v                          # pure-function unit tests
pytest tests/test_failures.py -v                       # mocked upstream-failure tests
python scripts/seo_report.py                           # SEO health report (local)
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
| `app.py` | Flask backend — API endpoints, SEO routes, OG + social image rendering, SSR cache warmer |
| `templates/index.html` | Dashboard SPA (HTML + embedded JS + CSS, Bootstrap 5 / Chart.js / Leaflet via CDN) |
| `templates/about.html`, `locations.html`, `glossary.html`, `compare*.html`, `embed.html` | Stand-alone pages |
| `templates/learn/{index,article}.html` | `/learn` topic cluster |
| `templates/regions/{index,page}.html` | `/regions` landing pages |
| `learn_articles.py` | Article content as Python dicts |
| `region_pages.py` | Region metadata + spot rosters (16 regions) |
| `surf_cameras.json` | Camera + forecast-only spot catalog → 145 deduped locations |
| `youtube_cams.json` | Approved YouTube live surf cams (embed-enabled streams only) |
| `scripts/youtube_cam_scan.py` | YouTube cam discovery/verify/approve pipeline; a weekly Action keeps one rolling review issue up to date, ranked by how many scans each stream has survived |
| `scripts/forecast_verification.py` | Snapshots the site's own forecasts at NDBC buoys and scores them; publishes to the `verification-data` branch |
| `scripts/seo_audit.py` | Collects CI, site-health, Search Console and Cloudflare metrics into a daily snapshot on the `seo-data` branch |
| `scripts/instagram_publish.py` | Social pipeline publisher (Graph API or dry-run) |
| `tests/` | 182 tests across SEO, pure functions, and mocked upstream failures |
| `static/sw.js` | Service worker (network-first HTML, 24h offline API fallback) |
| `render.yaml` | Render deploy config (Gunicorn `--workers 1 --threads 8 --preload`) |

### Page Routes

| Route | Purpose |
|---|---|
| `/` | Dashboard (default: Surf City, NC; respects saved location) |
| `/forecast/<slug>` | Dashboard for a specific location (145 slugs) |
| `/locations` | All forecast locations grouped by state |
| `/learn`, `/learn/<slug>` | Learn topic cluster |
| `/glossary` | Surf & oceanography glossary |
| `/regions`, `/regions/<slug>` | Regional landing pages (16 regions) |
| `/compare/surfline`, `/compare/magicseaweed` | Comparison landing pages |
| `/embed/<slug>` | Iframe-able mini forecast card |
| `/social/daily/<region>.png`, `/api/social-card/<region>` | Social report cards |
| `/about` | About page (includes embed-widget docs) |
| `/og/<slug>.png` | Per-location OpenGraph image |
| `/sitemap.xml`, `/robots.txt` | SEO surfaces |

Old `/?lat=X&lon=Y&name=Z` URLs 301-redirect to `/forecast/<slug>`.

### API Endpoints

| Route | Purpose |
|---|---|
| `GET /api/forecast?lat=&lon=` | Point forecast with per-hour confidence |
| `GET /api/map-forecast?lat=&lon=` | Local gridded data for the Swell Map |
| `GET /api/ocean-basin?lat=&lon=` | Global basin data for the Swell Map |
| `GET /api/tides?lat=&lon=` | 7-day tide predictions (`non_tidal` flag for the Great Lakes) |
| `GET /api/webcams?lat=&lon=` | Nearest cams (with coordinates for map markers) |
| `GET /api/buoys?lat=&lon=` | Buoy observations + spectra |
| `GET /api/beach-orientation?lat=&lon=` | Coastline facing direction |
| `GET /api/swell-narrative?lat=&lon=` | Swell type + origin estimate + narrative |

| `GET /api/health-upstreams` | Diagnostic: probes each weather source from the server's vantage |

All endpoints are stateless with thread-safe, size-bounded TTL caching and stampede protection. Upstream failures degrade gracefully: the wave chain falls from Open-Meteo Marine to WW3 across two independent ERDDAP servers, wind and air-temp have ERDDAP backfills, tides degrade to hourly-only, and if every wave source fails at once the API serves the last good forecast (up to 24h, flagged `stale`) rather than going dark.

### Scheduled jobs

Master is protected, so none of the scheduled jobs commit to it. Each writes to
its own unprotected data branch, which the app or the next run reads back.

| Workflow | Cadence | Writes to |
|---|---|---|
| `forecast-verification.yml` | every 6h | `verification-data` |
| `social-post.yml` | daily, two schedules by coast | posts to Instagram |
| `seo-audit.yml` | daily | `seo-data` |
| `youtube-cam-scan.yml` | weekly | `cam-data` + the rolling review issue |
| `ux-audit.yml` | daily | `ux-data` |
| `instagram-token-refresh.yml` | monthly | rotates the 60-day token |
| `social-accuracy-post.yml` | manual only | posts the weekly accuracy card |
| `social-highlight-post.yml` | daily 22:07 UTC | posts the daily highlight card |
| `seo-tests.yml` | weekly + on push | — |

`social-accuracy-post.yml` has its schedule commented out on purpose and its
`dry_run` input defaults to true, so it publishes nothing until I uncomment the
cron. It posts a card built from the live verification stats, and refuses rather
than posting when those stats are stale, thin or malformed.

`social-highlight-post.yml` posts the day's most notable buoy reading - biggest
seas, longest swell period or strongest wind, whichever clears its bar - with a
map of where it was measured. Every number on it is an observation rather than a
forecast. It has an editorial floor as well as the usual data guards: a day where
nothing is exceptional is skipped rather than posted. Expect it to skip fairly
often through the summer, when the whole network can sit below the floors for
days at a time.

`ux-audit.yml` runs `scripts/ux_audit.py`, which measures the live site with
Playwright at two viewports across six pages and records CLS, LCP, console and
request failures, horizontal overflow, tap-target and font sizes, WCAG contrast,
heading order and image alt gaps, plus a screenshot of each. It records and never
judges; reading the snapshot happens in the review log, issue #28. It runs at
midday rather than overnight on purpose: measured after dark, every screenshot
showed the AFTER DARK badge and black camera panels instead of the site a real
visitor sees.

## Deployment

Deployed on [Render.com](https://render.com/) behind Cloudflare (DNS, CDN, DDoS, free analytics). See `render.yaml`.

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 180 --workers 1 --threads 8 --preload
```

Security baseline: Content-Security-Policy, X-Frame-Options DENY (except `/embed/*`, which allows framing via `frame-ancestors *`), Referrer-Policy, Permissions-Policy, and SRI hashes on CDN script/style tags.

## Support

If this site has helped you score a session, you can [buy me a coffee](https://buymeacoffee.com/freesurfforecast) to keep the servers running. The site is and will stay free, no ads, no signup.

## License

MIT.
