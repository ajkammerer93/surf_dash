"""
Custom pin-drop coordinate tests for freesurfforecast.com

The app forecasts any coastline on earth, not just the curated spots: the
frontend writes ?lat=&lon=&name= into the URL when a surfer drops a pin
somewhere unlisted, and those links get shared. These tests cover the "/"
route honouring those params (title, meta, OG, SSR summary), the sanitising
of the attacker-controlled name, the lat/lon validation fallback, the
preserved 301 to /forecast/<slug> for known coordinates, and the tide route
refusing to name a station on the far side of an ocean.

Run: pytest tests/test_custom_spots.py -v
"""
import json
import re
import pytest

import app as surf_app
from app import app, LOCATION_BY_SLUG, _sanitize_place_name


# Uluwatu, Bali — global waves and cams, no NOAA tides or buoys.
BALI = {'lat': -8.81, 'lon': 115.09}
XSS_PAYLOAD = '"><script>alert(1)</script>'


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta(html, pattern):
    m = re.search(pattern, html)
    return m.group(1) if m else None


def _title(html):
    return _meta(html, r'<title>(.*?)</title>')


def _description(html):
    return _meta(html, r'<meta name="description" content="(.*?)">')


def _canonical(html):
    return _meta(html, r'<link rel="canonical" href="(.*?)">')


def _og(html, prop):
    return _meta(html, r'<meta property="og:%s" content="(.*?)">' % prop)


def _json_ld_blocks(html):
    return [
        json.loads(b) for b in re.findall(
            r'<script type="application/ld\+json">\s*(\{.*?\})\s*</script>',
            html, re.DOTALL,
        )
    ]


def _custom_page(client, name=None, lat=BALI['lat'], lon=BALI['lon']):
    params = {'lat': lat, 'lon': lon}
    if name is not None:
        params['name'] = name
    r = client.get('/', query_string=params)
    assert r.status_code == 200
    return r.get_data(as_text=True)


# ===================================================================
# 1. CUSTOM COORDINATES DESCRIBE THEMSELVES
# ===================================================================

class TestCustomCoordinateMeta:
    """A shared pin-drop link must not preview as the default location."""

    def test_title_uses_custom_name(self, client):
        html = _custom_page(client, name='Uluwatu')
        assert _title(html) == 'Surf Forecast for Uluwatu | Free Surf Forecast'
        assert 'Surf City' not in _title(html)

    def test_description_uses_custom_name(self, client):
        html = _custom_page(client, name='Uluwatu')
        assert 'Uluwatu' in _description(html)
        assert 'Surf City' not in _description(html)

    def test_og_tags_use_custom_name(self, client):
        html = _custom_page(client, name='Uluwatu')
        assert 'Uluwatu' in _og(html, 'title')
        assert 'Uluwatu' in _og(html, 'description')
        assert 'Uluwatu' in _meta(
            html, r'<meta name="twitter:title" content="(.*?)">')

    def test_ssr_summary_uses_custom_location(self, client):
        html = _custom_page(client, name='Uluwatu')
        assert 'Live Surf Forecast for Uluwatu' in html
        # Coordinates are rendered into the crawlable summary block
        assert '115.09' in html

    def test_json_ld_names_the_custom_location(self, client):
        html = _custom_page(client, name='Uluwatu')
        types = {b['@type']: b for b in _json_ld_blocks(html)}
        assert types['WebApplication']['contentLocation']['name'] == 'Uluwatu'
        assert types['WebApplication']['contentLocation']['geo']['latitude'] == BALI['lat']
        assert types['TouristAttraction']['name'].startswith('Uluwatu')

    def test_missing_name_falls_back_to_coordinates(self, client):
        html = _custom_page(client, name=None)
        assert _title(html) == 'Surf Forecast for Location (-8.81, 115.09) | Free Surf Forecast'

    def test_no_tide_claim_outside_noaa_coverage(self, client):
        """Tides and buoys are NOAA-only; do not promise them in Bali."""
        desc = _description(_custom_page(client, name='Uluwatu'))
        assert 'tides' not in desc.lower()
        assert 'surf cameras' in desc

    def test_tide_claim_kept_for_us_pin_drops(self, client):
        """An unlisted US beach still gets NOAA tides, so the copy stands."""
        # Ocean Isle Beach NC — not a curated spot, well inside NDBC range.
        desc = _description(_custom_page(client, name='Ocean Isle', lat=33.89, lon=-78.43))
        assert 'tides' in desc.lower()

    def test_no_distant_nearby_spots_on_foreign_pin(self, client):
        """The min_count padding would list Carolina beaches 9,000 mi away."""
        html = _custom_page(client, name='Uluwatu')
        assert 'nearby-distance">' not in html

    def test_custom_page_uses_generic_og_image(self, client):
        """Per-slug OG images only exist for curated spots."""
        html = _custom_page(client, name='Uluwatu')
        assert _og(html, 'image') == 'https://freesurfforecast.com/static/og-image.png'


# ===================================================================
# 2. CANONICAL — NO NEW INDEXABLE URL SPACE
# ===================================================================

class TestCustomCoordinateCanonical:
    """Pin coordinates are unbounded; every variant must canonicalise to "/"."""

    def test_canonical_is_root(self, client):
        html = _custom_page(client, name='Uluwatu')
        assert _canonical(html) == 'https://freesurfforecast.com/'

    def test_og_url_is_root(self, client):
        html = _custom_page(client, name='Uluwatu')
        assert _og(html, 'url') == 'https://freesurfforecast.com/'

    def test_no_noindex_header(self, client):
        """A noindex would fight the canonical — canonical alone is correct."""
        r = client.get('/', query_string={'lat': BALI['lat'], 'lon': BALI['lon']})
        assert 'noindex' not in r.headers.get('X-Robots-Tag', '')


# ===================================================================
# 3. NAME SANITISING (reflected XSS)
# ===================================================================

class TestNameSanitising:
    """?name= is attacker-controlled and lands in markup and JSON-LD."""

    def test_script_payload_cannot_escape(self, client):
        html = _custom_page(client, name=XSS_PAYLOAD)
        assert '<script>alert' not in html
        assert '</script>alert' not in html
        # Nothing from the payload may reach the title as markup
        title = _title(html)
        for ch in '<>"':
            assert ch not in title

    def test_script_payload_does_not_break_json_ld(self, client):
        """Autoescaping does not apply inside a JSON string — parse to prove."""
        html = _custom_page(client, name=XSS_PAYLOAD)
        blocks = _json_ld_blocks(html)
        assert len(blocks) >= 3  # WebApplication, BreadcrumbList, TouristAttraction
        for block in blocks:
            assert '<' not in json.dumps(block)

    def test_attribute_breakout_payload_is_stripped(self, client):
        html = _custom_page(client, name='x" onload="alert(1)')
        assert 'onload=' not in html
        assert 'onload=' not in _description(html)

    def test_angle_brackets_and_quotes_removed(self):
        assert '<' not in _sanitize_place_name(XSS_PAYLOAD)
        assert '"' not in _sanitize_place_name(XSS_PAYLOAD)
        assert "\\" not in _sanitize_place_name('back\\slash')
        assert '&' not in _sanitize_place_name('Cape & Bay')

    def test_length_is_capped(self, client):
        html = _custom_page(client, name='A' * 500)
        assert 'A' * (surf_app.CUSTOM_NAME_MAX_LEN + 1) not in html
        assert 'A' * surf_app.CUSTOM_NAME_MAX_LEN in html

    def test_control_characters_stripped(self):
        # Newlines and tabs fold to a space rather than vanishing, so a name
        # smuggled across lines cannot re-join into a different word.
        assert _sanitize_place_name('Pipe\nline\tBeach\x00') == 'Pipe line Beach'
        assert _sanitize_place_name('Bells\r\n\r\nBeach') == 'Bells Beach'

    def test_all_markup_name_falls_back_to_coordinates(self, client):
        """Nothing survives sanitising, so use the coordinate label."""
        html = _custom_page(client, name='<<<>>>')
        assert _title(html) == 'Surf Forecast for Location (-8.81, 115.09) | Free Surf Forecast'

    def test_real_place_names_survive_intact(self):
        for name in ["Praia do Guincho", "Bells Beach, VIC", "Coco's Point",
                     "Praia do Maca", "Sao Conrado (Rio)", "Half Moon Bay"]:
            assert _sanitize_place_name(name) == name

    def test_accented_characters_preserved(self):
        assert _sanitize_place_name('Praia do Maçã') == 'Praia do Maçã'


# ===================================================================
# 4. LAT/LON VALIDATION AND FALLBACK
# ===================================================================

class TestCoordinateValidation:
    """Anything unusable must fall back to the default location exactly."""

    def _is_default(self, html):
        return _title(html).startswith('Surf Forecast for Surf City')

    @pytest.mark.parametrize('params', [
        {'lat': 999, 'lon': 115.09},
        {'lat': -8.81, 'lon': 500},
        {'lat': -91, 'lon': 0},
        {'lat': 0, 'lon': -181},
        {'lat': 'abc', 'lon': 'def'},
        {'lat': 'nan', 'lon': 'nan'},
        {'lat': 'inf', 'lon': 'inf'},
        {'lat': '', 'lon': ''},
        {'lat': -8.81},          # lon missing
        {'lon': 115.09},         # lat missing
        {'name': 'Uluwatu'},     # name only
    ])
    def test_invalid_params_fall_back_to_default(self, client, params):
        r = client.get('/', query_string=params)
        assert r.status_code == 200
        html = r.get_data(as_text=True)
        assert self._is_default(html)
        assert _canonical(html) == 'https://freesurfforecast.com/'

    def test_bare_root_still_renders_default(self, client):
        r = client.get('/')
        assert r.status_code == 200
        html = r.get_data(as_text=True)
        assert self._is_default(html)
        # Default keeps its slug-derived OG image and nearby-spot links
        assert _og(html, 'image').startswith('https://freesurfforecast.com/og/')
        assert 'nearby-distance">' in html

    @pytest.mark.parametrize('coords', [
        (90, 180), (-90, -180), (0, 0),
    ])
    def test_boundary_coordinates_accepted(self, client, coords):
        lat, lon = coords
        r = client.get('/', query_string={'lat': lat, 'lon': lon, 'name': 'Edge'})
        assert r.status_code == 200
        assert _title(r.get_data(as_text=True)) == 'Surf Forecast for Edge | Free Surf Forecast'


# ===================================================================
# 5. KNOWN COORDINATES STILL WIN
# ===================================================================

class TestKnownCoordinatesPreserved:
    """Curated spots keep their indexable /forecast/<slug> URL."""

    def test_known_coords_redirect_to_slug(self, client):
        loc = LOCATION_BY_SLUG['virginia-beach']
        r = client.get('/', query_string={
            'lat': loc['lat'], 'lon': loc['lon'], 'name': 'Whatever'})
        assert r.status_code == 301
        assert r.headers['Location'].endswith('/forecast/virginia-beach')

    def test_redirect_ignores_the_supplied_name(self, client):
        """A hostile name must not survive into the redirect target."""
        loc = LOCATION_BY_SLUG['virginia-beach']
        r = client.get('/', query_string={
            'lat': loc['lat'], 'lon': loc['lon'], 'name': XSS_PAYLOAD})
        assert r.status_code == 301
        assert 'script' not in r.headers['Location']

    def test_slug_page_meta_unchanged(self, client):
        """The is_custom copy switch must not touch curated pages."""
        html = client.get('/forecast/virginia-beach').get_data(as_text=True)
        assert _canonical(html) == 'https://freesurfforecast.com/forecast/virginia-beach'
        assert 'tides' in _description(html).lower()
        assert 'nearby-distance">' in html


# ===================================================================
# 6. TIDE STATION DISTANCE GUARD
# ===================================================================

class TestDistantTideStation:
    """Do not name a station on the far side of an ocean."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        surf_app._cache.clear()
        surf_app._cache_key_locks.clear()
        yield
        surf_app._cache.clear()
        surf_app._cache_key_locks.clear()

    def test_far_station_not_reported_as_the_station(self, client, monkeypatch):
        monkeypatch.setattr(surf_app, 'find_nearest_tide_stations',
                            lambda lat, lon, limit=None: [{
            'id': '1234567', 'name': 'Djakarta, Java',
            'lat': 6.1, 'lon': 106.8, 'distance_km': 1169.0,
        }])
        data = client.get('/api/tides', query_string=BALI).get_json()
        assert data['non_tidal'] is True
        assert data['station'] is None
        assert 'Djakarta' not in json.dumps(data)

    def test_distance_kept_as_diagnostic(self, client, monkeypatch):
        monkeypatch.setattr(surf_app, 'find_nearest_tide_stations',
                            lambda lat, lon, limit=None: [{
            'id': '1234567', 'name': 'Djakarta, Java',
            'lat': 6.1, 'lon': 106.8, 'distance_km': 1169.0,
        }])
        data = client.get('/api/tides', query_string=BALI).get_json()
        assert data['nearest_station_km'] == 1169.0

    def test_non_tidal_payload_shape_unchanged_for_frontend(self, client, monkeypatch):
        """loadTideData() reads non_tidal/hourly/high_low and never station."""
        monkeypatch.setattr(surf_app, 'find_nearest_tide_stations',
                            lambda lat, lon, limit=None: [{
            'id': '1234567', 'name': 'Djakarta, Java',
            'lat': 6.1, 'lon': 106.8, 'distance_km': 1169.0,
        }])
        data = client.get('/api/tides', query_string=BALI).get_json()
        assert data['hourly'] == []
        assert data['high_low'] == []

    # These patch find_nearest_tide_stations (plural), which is what the route
    # calls. They previously patched the singular helper; when the route moved
    # to the ranked version the patch became a no-op and the tests silently
    # went to the live NOAA API instead. Two of them still passed, which is the
    # part worth remembering -- a test that passes for the wrong reason is
    # indistinguishable from one that works.
    def test_nearby_station_still_reported(self, client, monkeypatch):
        station = {'id': '8658163', 'name': 'Wrightsville Beach',
                   'lat': 34.21, 'lon': -77.79, 'distance_km': 12.4}
        monkeypatch.setattr(surf_app, 'find_nearest_tide_stations',
                            lambda lat, lon, limit=None: [station])
        monkeypatch.setattr(surf_app, 'get_tide_data', lambda sid: {
            'hourly': [{'time': '2026-01-01T00:00:00', 'height': 1.0}],
            'high_low': [],
        })
        data = client.get('/api/tides', query_string={'lat': 34.43, 'lon': -77.55}).get_json()
        assert data.get('non_tidal') is None
        assert data['station']['name'] == 'Wrightsville Beach'

    def test_threshold_boundary_is_inclusive(self, client, monkeypatch):
        """Exactly at the threshold is still treated as a usable station."""
        station = {'id': '8658163', 'name': 'Edge Station',
                   'lat': 34.21, 'lon': -77.79,
                   'distance_km': float(surf_app.NON_TIDAL_STATION_KM)}
        monkeypatch.setattr(surf_app, 'find_nearest_tide_stations',
                            lambda lat, lon, limit=None: [station])
        monkeypatch.setattr(surf_app, 'get_tide_data', lambda sid: {
            'hourly': [], 'high_low': [],
        })
        data = client.get('/api/tides', query_string={'lat': 34.43, 'lon': -77.55}).get_json()
        assert data['station']['name'] == 'Edge Station'


class TestSeoAuditCollectors:
    """The audit runs unattended, so a missing credential must degrade to a
    recorded reason rather than an exception that kills the whole snapshot."""

    def _module(self):
        import importlib.util, os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            'seo_audit', os.path.join(root, 'scripts', 'seo_audit.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_search_console_without_credentials(self, monkeypatch):
        monkeypatch.delenv('GSC_SA_JSON', raising=False)
        out = self._module().collect_search_console({})
        assert 'unavailable' in out and 'GSC_SA_JSON' in out['unavailable']

    def test_cloudflare_without_credentials(self, monkeypatch):
        for var in ('CF_API_TOKEN', 'CF_ACCOUNT_ID', 'CF_SITE_TAG'):
            monkeypatch.delenv(var, raising=False)
        out = self._module().collect_cloudflare()
        assert 'unavailable' in out
        assert 'CF_API_TOKEN' in out['unavailable']

    def test_watched_workflows_all_exist(self):
        """A renamed workflow would silently stop being audited."""
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for wf in self._module().WATCHED_WORKFLOWS:
            path = os.path.join(root, '.github', 'workflows', wf)
            assert os.path.exists(path), f'{wf} is watched but does not exist'


class TestCamCandidatePersistence:
    """The weekly scan ranks candidates by how many scans they survive, so the
    parsing and ranking that produce that signal need pinning."""

    def _module(self):
        import importlib.util, os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            'youtube_cam_scan', os.path.join(root, 'scripts', 'youtube_cam_scan.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_parses_a_review_issue_line(self):
        m = self._module().ISSUE_LINE.match(
            '- [ ] [Dania Beach Pier Cam](https://www.youtube.com/watch?v=o1eTLR7Degs)'
            ' — City of Dania Beach Pier Cam — suggested spot: Dania FL — `o1eTLR7Degs`')
        assert m and m.group('vid') == 'o1eTLR7Degs'
        assert m.group('done') == ' '
        assert m.group('title') == 'Dania Beach Pier Cam'

    def test_checked_lines_are_distinguishable(self):
        m = self._module().ISSUE_LINE.match(
            '- [x] [Handled Cam](https://www.youtube.com/watch?v=abcdefghijk)'
            ' — Chan — suggested spot: _no spot match_ — `abcdefghijk`')
        assert m and m.group('done') == 'x'

    def test_id_comes_from_the_backticked_field_not_the_title(self):
        """Titles are attacker-controlled; a title imitating the id field must
        not be able to smuggle in a different video."""
        m = self._module().ISSUE_LINE.match(
            '- [ ] [Evil `aaaaaaaaaaa` Cam](https://www.youtube.com/watch?v=zzzzzzzzzzz)'
            ' — Chan — suggested spot: X — `zzzzzzzzzzz`')
        assert m and m.group('vid') == 'zzzzzzzzzzz'

    def test_ranking_puts_persistent_candidates_first(self):
        mod = self._module()
        ranked = mod._rank([
            {'video_id': 'a', 'times_seen': 1, 'concurrent_viewers': 900, 'title': 'a'},
            {'video_id': 'b', 'times_seen': 6, 'concurrent_viewers': 2, 'title': 'b'},
            {'video_id': 'c', 'times_seen': 6, 'concurrent_viewers': 50, 'title': 'c'},
        ])
        assert [c['video_id'] for c in ranked] == ['c', 'b', 'a']

    def test_ranking_tolerates_missing_fields(self):
        mod = self._module()
        ranked = mod._rank([{'video_id': 'x', 'title': 'x'},
                            {'video_id': 'y', 'times_seen': 3, 'title': 'y'}])
        assert [c['video_id'] for c in ranked] == ['y', 'x']


class TestCamCheckmarkDismissal:
    """Ticking a box in the review issue means "not this one, don't ask again".
    The scan rewrites that issue body, so the parsing has to survive both the
    old em-dash format and the current hyphen one -- a parser that knew only
    one of them silently matched nothing and looked like it worked."""

    def _module(self):
        import importlib.util, os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            'youtube_cam_scan', os.path.join(root, 'scripts', 'youtube_cam_scan.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_parses_the_current_hyphen_format(self):
        m = self._module().ISSUE_LINE.match(
            '- [x] [Some Cam](https://www.youtube.com/watch?v=abcdefghijk) - Chan - '
            'suggested spot: Venice Beach CA - seen 7x - 811 watching - `abcdefghijk`')
        assert m and m.group('done') == 'x' and m.group('vid') == 'abcdefghijk'

    def test_parses_the_legacy_em_dash_format(self):
        m = self._module().ISSUE_LINE.match(
            '- [ ] [Old Cam](https://www.youtube.com/watch?v=bbbbbbbbbbb) — Chan — '
            'suggested spot: _no spot match_ — `bbbbbbbbbbb`')
        assert m and m.group('done') == ' ' and m.group('vid') == 'bbbbbbbbbbb'

    def test_dismissed_ids_are_excluded_from_a_render(self, tmp_path):
        mod = self._module()
        store = {
            'candidates': [
                {'video_id': 'keepkeepkee', 'title': 'Keep', 'url': 'u',
                 'channel': 'c', 'times_seen': 5},
                {'video_id': 'dropdropdro', 'title': 'Drop', 'url': 'u',
                 'channel': 'c', 'times_seen': 9},
            ],
            'dismissed': {'dropdropdro': {'title': 'Drop', 'dismissed_on': '2026-08-21'}},
        }
        path = tmp_path / 'youtube_cam_candidates.json'
        path.write_text(__import__('json').dumps(store))
        out = tmp_path / 'issue.md'

        class Args:
            data_dir = str(tmp_path)
        args = Args()
        args.out = str(out)
        mod.cmd_render_issue(args)
        body = out.read_text()
        # Dismissed wins even though it has the higher sighting count.
        assert 'keepkeepkee' in body
        assert 'dropdropdro' not in body

    def test_the_issue_explains_that_ticking_dismisses(self, tmp_path):
        """If the issue does not say so, nobody knows the tick means anything."""
        mod = self._module()
        out = tmp_path / 'issue.md'
        mod._write_issue_markdown(
            [{'video_id': 'aaaaaaaaaaa', 'title': 'A', 'url': 'u',
              'channel': 'c', 'times_seen': 1}], str(out))
        assert 'Tick the box' in out.read_text()
