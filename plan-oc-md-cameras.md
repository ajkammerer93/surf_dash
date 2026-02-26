# Plan: Add OC MD Cameras + Attribution + Footer

## 1. Add Ocean City MD Beach Webcams

### Research findings
- Found 39 webcams on oceancity.com via sitemap
- Most streams are dynamically loaded (JavaScript-based player) — can't extract static stream URLs
- 2 cams found with YouTube embeds via SkylineWebcams
- Remaining 14 cams added as `"type": "link"` pointing to oceancity.com pages

### Embeddable cams (type: iframe)

| # | Name | YouTube ID | Source |
|---|------|-----------|--------|
| 1 | OC MD - Beach Cam | `4eQtjcG4Ddc` | SkylineWebcams |
| 2 | OC MD - Econo Lodge Beachfront | `g5nMJtlSsa8` | SkylineWebcams |

### Link-only cams (type: link)

| # | Name | Location | Page URL |
|---|------|----------|----------|
| 1 | OC MD - Inlet Cam | South inlet tip | ocean-city-inlet-cam-from-the-wedge |
| 2 | OC MD - Assateague Cam | South end, facing Assateague Island | assateague-cam-from-the-wedge-bar |
| 3 | OC MD - Ocean 1 Hotel Beach (1st St) | 1st St, panoramic beach | beach-and-boardwalk-cam-ocean-1-hotel-suites |
| 4 | OC MD - Kite Loft Beach (5th St) | 5th-6th St, beach + Atlantic | kite-loft-beach-and-boardwalk-cam |
| 5 | OC MD - HoJo 12th St Beach | 12th St, ocean views | beach-cam-at-the-howard-johnson-plaza-on-12th-street |
| 6 | OC MD - HoJo Oceanfront Beach | Oceanfront, beach view | howard-johnsons-oceanfront-boardwalk-beach-cam |
| 7 | OC MD - Park Place Beach | Boardwalk/beach | park-place-hotel-boardwalk-beach-cam |
| 8 | OC MD - Plim Plaza Oceanfront | Oceanfront boardwalk | plim-plaza-oceanfront-boardwalk-hotel |
| 9 | OC MD - Grand Hotel Ocean View | Ocean view | the-grand-hotel-ocean-view |
| 10 | OC MD - Princess Royale Beach (91st St) | 91st St, south-facing ocean | princess-royale-beach-cam |
| 11 | OC MD - Princess Royale Oceanfront | 91st St, oceanfront | princess-royale-oceanfront-hotel |
| 12 | OC MD - Carousel Beach (117th St) | 117th St, surfers/waves | beach-view-webcam-from-the-carousel-oceanfront-resort |
| 13 | OC MD - Carousel Sky Cam (117th St) | 117th St, sky/ocean | north-ocean-city-sky-cam-from-the-carousel-resort |
| 14 | OC MD - Hyatt Place Oceanfront | Oceanfront ocean view | hyatt-place-ocean-city-oceanfront-ocean-air-show-cam |

### Excluded cams (not beach)
- 2x mini golf, 3x Route 50/traffic, 1x parking lot, 3x boardwalk-only
- 2x bayside/sunset, 2x amusement ride, 2x restaurant, 1x convention center parking, 1x street park

### Coordinates
Ocean City MD runs north-south from the inlet (~38.326N) to the DE border (~38.450N). Approximate lat/lon assigned based on each camera's street address along the island.

---

## 2. Camera Panel Attribution

### Goal
Display source attribution on each camera panel: source name, logo, and link to source page.

### Data model change — `surf_cameras.json`
Add a `source` object to every camera entry:
```json
{
  "name": "Surf City Pier North",
  "lat": 34.4271,
  "lon": -77.5461,
  "stream_url": "https://streams.surfchex.com:8443/live/sc1.stream/playlist.m3u8",
  "page_url": "https://www.surfchex.com/cams/surf-city-pier-north/",
  "source": {
    "name": "SurfChex",
    "url": "https://www.surfchex.com"
  }
}
```

### Source catalog
All distinct sources across the library:

| Source Name | Domain | Camera Count |
|-------------|--------|-------------|
| SurfChex | surfchex.com | ~20 (HLS streams) |
| HDBeachCams | hdbeachcams.com | ~30 (link type) |
| TheSurfersView | thesurfersview.com | ~12 (link type) |
| NJBeachCams | njbeachcams.com | ~5 (link type) |
| OceanCity.com | oceancity.com | 14 (new, link type) |
| SkylineWebcams | skylinewebcams.com | 2 (new, iframe/YouTube) |
| Twiddy | twiddy.com | 1 (Kitty Hawk, HLS) |

### Logo approach
Use Google Favicon API for consistent, reliable icons without hosting:
```
https://www.google.com/s2/favicons?domain=surfchex.com&sz=16
```

### Frontend change — `setCamPanel()` in `index.html`
Add an attribution bar at the bottom of each cam panel content area:
```html
<div class="cam-attribution">
  <img src="https://www.google.com/s2/favicons?domain=surfchex.com&sz=16" alt="">
  <a href="https://www.surfchex.com" target="_blank" rel="noopener">SurfChex</a>
</div>
```

### CSS
```css
.cam-attribution {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 3px 8px;
  background: rgba(0, 0, 0, 0.75);
  font-size: 9px;
  z-index: 2;
}
.cam-attribution img {
  width: 12px;
  height: 12px;
}
.cam-attribution a {
  color: var(--text-muted);
  text-decoration: none;
}
.cam-attribution a:hover {
  color: var(--text);
  text-decoration: underline;
}
```

---

## 3. Site Footer

### Goal
Add a persistent footer to the page with copyright/all-rights-reserved message.

### HTML
```html
<footer class="site-footer">
  <span>&copy; 2026 Surf Dash. All rights reserved.</span>
</footer>
```

### CSS
```css
.site-footer {
  text-align: center;
  padding: 8px 16px;
  font-size: 10px;
  color: var(--text-muted);
  border-top: 1px solid var(--border);
  background: var(--surface);
}
```

Place after the closing `</div>` of the main grid-stack container.

### Data & model attributions in footer
Include links to all major data/model/camera sources:

| Source | Used For |
|--------|----------|
| Open-Meteo | Weather + marine forecast models |
| NOAA | Tides, GFS-Wave, ERDDAP/CoastWatch |
| NDBC | Buoy observations |
| CDIP | Buoy spectral data |
| OpenStreetMap | Geocoding + map data |
| CARTO | Dark basemap tiles |
| SurfChex | Surf camera streams |
| Windy | Webcam API fallback |

Format as a compact row of links, e.g.:
`Data: Open-Meteo | NOAA | NDBC | CDIP  Maps: OpenStreetMap | CARTO  Cameras: SurfChex | Windy`

---

## Implementation Steps

1. Create new branch `add-oc-md-cameras`
2. **surf_cameras.json**:
   - Add `source` object to all existing camera entries
   - Add 2 new iframe-type OC MD cams (SkylineWebcams YouTube embeds)
   - Add 14 new link-type OC MD cams (oceancity.com pages)
   - Keep existing "Ocean City MD" hdbeachcams.com entry as-is
3. **templates/index.html**:
   - Update `setCamPanel()` to render attribution bar using `source` data
   - Add `.cam-attribution` CSS
   - Add site footer HTML + CSS
4. Commit, PR, merge, tag

### Note on YouTube IDs
YouTube live stream video IDs can change if the channel restarts. The SkylineWebcams IDs (`4eQtjcG4Ddc`, `g5nMJtlSsa8`) may not be permanent — same caveat applies to existing Atlantic City and Longport entries.
