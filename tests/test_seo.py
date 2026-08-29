"""
SEO test suite for freesurfforecast.com

Validates all SEO infrastructure: clean URLs, redirects, meta tags,
structured data, sitemap, robots.txt, resource hints, cache headers,
internal linking, and SSR content.

Run: pytest tests/test_seo.py -v
"""
import json
import re
import xml.etree.ElementTree as ET
import pytest

from app import app, LOCATION_BY_SLUG, SLUG_BY_COORDS


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_ld(html, schema_type):
    """Extract JSON-LD blocks of a given @type from HTML."""
    blocks = re.findall(
        r'<script type="application/ld\+json">\s*(\{.*?\})\s*</script>',
        html, re.DOTALL,
    )
    results = []
    for block in blocks:
        try:
            data = json.loads(block)
            if data.get('@type') == schema_type:
                results.append(data)
        except json.JSONDecodeError:
            continue
    return results


# ===================================================================
# 1. CLEAN URL ROUTES
# ===================================================================

class TestCleanURLRoutes:
    """Every known location should have a working /forecast/<slug> page."""

    def test_slug_count(self):
        assert len(LOCATION_BY_SLUG) > 100, "Should have 100+ location slugs"

    def test_forecast_route_returns_200(self, client):
        r = client.get('/forecast/virginia-beach')
        assert r.status_code == 200

    def test_unknown_slug_returns_404(self, client):
        r = client.get('/forecast/nonexistent-place-xyz')
        assert r.status_code == 404

    def test_home_page_returns_200(self, client):
        r = client.get('/')
        assert r.status_code == 200

    def test_locations_page_returns_200(self, client):
        r = client.get('/locations')
        assert r.status_code == 200

    @pytest.mark.parametrize("slug", list(LOCATION_BY_SLUG.keys())[:10])
    def test_sample_forecast_pages_load(self, client, slug):
        r = client.get(f'/forecast/{slug}')
        assert r.status_code == 200


# ===================================================================
# 2. 301 REDIRECTS (old query-param URLs -> clean slugs)
# ===================================================================

class TestRedirects:
    """Old /?lat=X&lon=Y URLs should 301 to /forecast/<slug>."""

    def test_known_location_redirects(self, client):
        loc = LOCATION_BY_SLUG['virginia-beach']
        r = client.get(f'/?lat={loc["lat"]}&lon={loc["lon"]}&name=Virginia+Beach')
        assert r.status_code == 301
        assert '/forecast/virginia-beach' in r.headers['Location']

    def test_unknown_coords_stay_on_home(self, client):
        r = client.get('/?lat=99.0&lon=-99.0')
        assert r.status_code == 200

    def test_no_redirect_without_params(self, client):
        r = client.get('/')
        assert r.status_code == 200


# ===================================================================
# 3. META TAGS
# ===================================================================

class TestMetaTags:
    """Every page must have title, description, canonical, OG, and Twitter tags."""

    def _check_meta(self, html, location_name):
        assert f'<title>' in html, "Missing <title>"
        assert location_name in html, "Location name not in page"

        # Canonical
        canonical = re.search(r'<link rel="canonical" href="([^"]+)"', html)
        assert canonical, "Missing canonical link"
        assert canonical.group(1).startswith('https://freesurfforecast.com')

        # Meta description
        desc = re.search(r'<meta name="description" content="([^"]+)"', html)
        assert desc, "Missing meta description"
        assert len(desc.group(1)) >= 50, "Meta description too short"
        assert len(desc.group(1)) <= 300, "Meta description too long"

        # Open Graph
        assert 'og:title' in html, "Missing og:title"
        assert 'og:description' in html, "Missing og:description"
        assert 'og:url' in html, "Missing og:url"
        assert 'og:image' in html, "Missing og:image"

        # Twitter Card
        assert 'twitter:card' in html, "Missing twitter:card"
        assert 'twitter:title' in html, "Missing twitter:title"

    def test_home_meta_tags(self, client):
        r = client.get('/')
        self._check_meta(r.data.decode(), 'Surf City')

    def test_forecast_meta_tags(self, client):
        r = client.get('/forecast/virginia-beach')
        html = r.data.decode()
        self._check_meta(html, 'Virginia Beach')
        assert 'freesurfforecast.com/forecast/virginia-beach' in html

    def test_locations_meta_tags(self, client):
        r = client.get('/locations')
        html = r.data.decode()
        assert '<title>' in html
        assert 'meta name="description"' in html

    def test_canonical_is_unique_per_location(self, client):
        r1 = client.get('/forecast/virginia-beach')
        r2 = client.get('/forecast/wrightsville-beach')
        c1 = re.search(r'<link rel="canonical" href="([^"]+)"', r1.data.decode())
        c2 = re.search(r'<link rel="canonical" href="([^"]+)"', r2.data.decode())
        assert c1.group(1) != c2.group(1), "Canonical URLs should differ between locations"


# ===================================================================
# 4. STRUCTURED DATA (JSON-LD)
# ===================================================================

class TestStructuredData:

    def test_breadcrumb_on_forecast_page(self, client):
        r = client.get('/forecast/virginia-beach')
        html = r.data.decode()
        breadcrumbs = _parse_json_ld(html, 'BreadcrumbList')
        assert len(breadcrumbs) == 1, "Should have exactly 1 BreadcrumbList"

        items = breadcrumbs[0]['itemListElement']
        assert len(items) == 3, "Breadcrumb should have 3 levels: Home > Locations > Location"
        assert items[0]['name'] == 'Home'
        assert items[1]['name'] == 'Locations'
        assert 'Virginia Beach' in items[2]['name']

    def test_breadcrumb_on_locations_page(self, client):
        r = client.get('/locations')
        html = r.data.decode()
        breadcrumbs = _parse_json_ld(html, 'BreadcrumbList')
        assert len(breadcrumbs) == 1
        items = breadcrumbs[0]['itemListElement']
        assert items[-1]['name'] == 'Locations'

    def test_web_application_schema(self, client):
        r = client.get('/forecast/virginia-beach')
        html = r.data.decode()
        apps = _parse_json_ld(html, 'WebApplication')
        assert len(apps) == 1
        assert apps[0]['offers']['price'] == '0'

    def test_faq_schema_on_about_only(self, client):
        about = client.get('/about')
        about_html = about.data.decode()
        about_faqs = _parse_json_ld(about_html, 'FAQPage')
        assert len(about_faqs) == 1, "About page should have FAQPage schema"

        home = client.get('/')
        home_html = home.data.decode()
        home_faqs = _parse_json_ld(home_html, 'FAQPage')
        assert len(home_faqs) == 0, "FAQPage schema should not be duplicated on forecast pages"


# ===================================================================
# 5. SITEMAP
# ===================================================================

class TestSitemap:

    def test_sitemap_valid_xml(self, client):
        r = client.get('/sitemap.xml')
        assert r.status_code == 200
        assert 'application/xml' in r.content_type
        ET.fromstring(r.data)  # Raises if invalid XML

    def test_sitemap_contains_all_locations(self, client):
        r = client.get('/sitemap.xml')
        xml = r.data.decode()
        for slug in LOCATION_BY_SLUG:
            assert f'/forecast/{slug}' in xml, f"Missing {slug} from sitemap"

    def test_sitemap_contains_key_pages(self, client):
        r = client.get('/sitemap.xml')
        xml = r.data.decode()
        assert 'freesurfforecast.com/' in xml
        assert '/locations' in xml
        assert '/about' in xml

    def test_sitemap_no_query_param_urls(self, client):
        r = client.get('/sitemap.xml')
        xml = r.data.decode()
        assert '?lat=' not in xml, "Sitemap should not contain old query-param URLs"

    def test_sitemap_has_lastmod(self, client):
        r = client.get('/sitemap.xml')
        xml = r.data.decode()
        assert '<lastmod>' in xml


# ===================================================================
# 6. ROBOTS.TXT
# ===================================================================

class TestRobotsTxt:

    def test_robots_txt_exists(self, client):
        r = client.get('/robots.txt')
        assert r.status_code == 200
        assert 'text/plain' in r.content_type

    def test_robots_allows_root(self, client):
        r = client.get('/robots.txt')
        text = r.data.decode()
        assert 'Allow: /' in text

    def test_robots_disallows_api(self, client):
        r = client.get('/robots.txt')
        text = r.data.decode()
        assert 'Disallow: /api/' in text

    def test_robots_references_sitemap(self, client):
        r = client.get('/robots.txt')
        text = r.data.decode()
        assert 'sitemap.xml' in text.lower()


# ===================================================================
# 7. RESOURCE HINTS
# ===================================================================

class TestResourceHints:

    def test_preconnect_tags(self, client):
        r = client.get('/')
        html = r.data.decode()
        assert 'rel="preconnect" href="https://cdn.jsdelivr.net"' in html
        assert 'rel="preconnect" href="https://unpkg.com"' in html

    def test_dns_prefetch_tags(self, client):
        r = client.get('/')
        html = r.data.decode()
        assert 'rel="dns-prefetch" href="https://cdn.jsdelivr.net"' in html
        assert 'rel="dns-prefetch" href="https://unpkg.com"' in html


# ===================================================================
# 8. CACHE HEADERS
# ===================================================================

class TestCacheHeaders:

    def test_html_cache_control(self, client):
        r = client.get('/')
        cc = r.headers.get('Cache-Control', '')
        assert 'public' in cc
        assert 'max-age=300' in cc

    def test_forecast_page_cache_control(self, client):
        r = client.get('/forecast/virginia-beach')
        cc = r.headers.get('Cache-Control', '')
        assert 'max-age=300' in cc


# ===================================================================
# 9. INTERNAL LINKING
# ===================================================================

class TestInternalLinking:

    def test_footer_has_locations_link(self, client):
        r = client.get('/')
        html = r.data.decode()
        assert 'href="/locations"' in html

    def test_footer_has_popular_location_links(self, client):
        r = client.get('/')
        html = r.data.decode()
        assert 'href="/forecast/' in html

    def test_locations_page_links_to_all_forecasts(self, client):
        r = client.get('/locations')
        html = r.data.decode()
        for slug in LOCATION_BY_SLUG:
            assert f'/forecast/{slug}' in html, f"Locations page missing link to {slug}"

    def test_locations_page_no_other_group(self, client):
        r = client.get('/locations')
        html = r.data.decode()
        assert '>Other' not in html, "All locations should be categorized by state"


# ===================================================================
# 10. SSR CONTENT
# ===================================================================

class TestSSRContent:
    """SSR summary is always present with location-specific static content,
    plus an optional live-conditions block when the forecast cache is warm."""

    def test_ssr_always_present(self, client):
        """Even on cold cache the SSR block must render with static location info."""
        r = client.get('/forecast/virginia-beach')
        html = r.data.decode()
        assert r.status_code == 200
        assert 'id="ssr-summary"' in html, "SSR summary section must always render"

    def test_ssr_section_has_correct_structure(self, client):
        r = client.get('/forecast/virginia-beach')
        html = r.data.decode()
        assert 'id="ssr-summary"' in html
        assert 'Live Surf Forecast for Virginia Beach' in html
        # State should be present in the heading or body
        assert 'VA' in html
        # Either coordinates or facing direction should appear in the static block
        assert 'Coordinates' in html or 'facing beach' in html

    def test_ssr_includes_state_in_meta_description(self, client):
        r = client.get('/forecast/virginia-beach')
        html = r.data.decode()
        desc = re.search(r'<meta name="description" content="([^"]+)"', html)
        assert desc, "Missing meta description"
        assert 'VA' in desc.group(1), "Meta description should include state code"

    def test_ssr_content_differs_by_location(self, client):
        """Each /forecast/<slug> page should have meaningfully different SSR
        content so Google doesn't see 126 near-duplicate pages."""
        r1 = client.get('/forecast/virginia-beach').data.decode()
        r2 = client.get('/forecast/wrightsville-beach').data.decode()
        # Extract just the SSR block from each
        m1 = re.search(r'id="ssr-summary".*?</section>', r1, re.DOTALL)
        m2 = re.search(r'id="ssr-summary".*?</section>', r2, re.DOTALL)
        assert m1 and m2, "SSR block missing on one of the test pages"
        assert m1.group(0) != m2.group(0), "SSR content must differ between locations"

    def test_noscript_fallback_present(self, client):
        r = client.get('/forecast/virginia-beach')
        html = r.data.decode()
        assert '<noscript>' in html
        assert 'Virginia Beach' in html


class TestNearbySpots:
    """Server-rendered "Nearby Spots" block — crawlable internal links to
    seed PageRank into the long-tail forecast pages."""

    def test_nearby_spots_block_present(self, client):
        r = client.get('/forecast/virginia-beach')
        html = r.data.decode()
        assert 'class="nearby-spots"' in html, "Forecast page should render nearby-spots block"

    def test_nearby_spots_links_are_internal_forecast_urls(self, client):
        r = client.get('/forecast/virginia-beach')
        html = r.data.decode()
        m = re.search(r'<aside class="nearby-spots".*?</aside>', html, re.DOTALL)
        assert m, "nearby-spots block missing"
        links = re.findall(r'href="(/forecast/[^"]+)"', m.group(0))
        assert len(links) >= 2, "Should link to at least 2 nearby spots"
        # Should not link to itself
        assert all('/forecast/virginia-beach' not in l for l in links)


class TestTouristAttractionSchema:
    """Each forecast page should expose a TouristAttraction JSON-LD with geo +
    address region so it can rank for geo-intent queries."""

    def test_tourist_attraction_present(self, client):
        r = client.get('/forecast/virginia-beach')
        attractions = _parse_json_ld(r.data.decode(), 'TouristAttraction')
        assert len(attractions) == 1, "Should have exactly 1 TouristAttraction per forecast page"
        a = attractions[0]
        assert a.get('name', '').startswith('Virginia Beach')
        assert 'geo' in a and a['geo'].get('latitude') is not None
        assert a.get('address', {}).get('addressRegion') == 'VA'

    def test_tourist_attraction_unique_per_location(self, client):
        """The TouristAttraction schema's name + geo must differ across pages."""
        a1 = _parse_json_ld(client.get('/forecast/virginia-beach').data.decode(), 'TouristAttraction')
        a2 = _parse_json_ld(client.get('/forecast/wrightsville-beach').data.decode(), 'TouristAttraction')
        assert a1 and a2
        assert a1[0]['name'] != a2[0]['name']
        assert a1[0]['geo']['latitude'] != a2[0]['geo']['latitude']


# ===================================================================
# 11. JS SLUG INFRASTRUCTURE
# ===================================================================

class TestJSSlugInfra:

    def test_slug_lookup_map_present(self, client):
        r = client.get('/forecast/virginia-beach')
        html = r.data.decode()
        assert '_slugByCoords' in html
        assert 'findSlugForCoords' in html

    def test_server_location_set_on_forecast_route(self, client):
        r = client.get('/forecast/virginia-beach')
        html = r.data.decode()
        assert '"virginia-beach"' in html
        assert '_serverLocation' in html

    def test_history_replace_state_in_location_handler(self, client):
        r = client.get('/')
        html = r.data.decode()
        assert 'history.replaceState' in html


# ===================================================================
# 12. SLUG CONSISTENCY
# ===================================================================

class TestSlugConsistency:
    """Every location should round-trip: slug -> coords -> slug."""

    def test_all_slugs_have_coord_reverse_lookup(self):
        for slug, loc in LOCATION_BY_SLUG.items():
            coord_key = (round(loc['lat'], 4), round(loc['lon'], 4))
            reverse_slug = SLUG_BY_COORDS.get(coord_key)
            assert reverse_slug == slug, (
                f"Slug {slug} at {coord_key} reverse-maps to {reverse_slug}"
            )

    def test_slugs_are_url_safe(self):
        for slug in LOCATION_BY_SLUG:
            assert re.match(r'^[a-z0-9-]+$', slug), f"Slug '{slug}' contains invalid chars"
            assert not slug.startswith('-'), f"Slug '{slug}' starts with hyphen"
            assert not slug.endswith('-'), f"Slug '{slug}' ends with hyphen"
            assert '--' not in slug, f"Slug '{slug}' has double hyphens"


# ===================================================================
# 13. COMPARISON PAGE + REGION EXPANSION
# ===================================================================

class TestComparePage:
    """/compare/surfline landing page targeting Surfline-alternative queries."""

    def test_compare_page_returns_200(self, client):
        resp = client.get('/compare/surfline')
        assert resp.status_code == 200

    def test_compare_meta_tags(self, client):
        html = client.get('/compare/surfline').data.decode()
        assert '<title>' in html and 'Surfline Alternative' in html
        assert 'name="description"' in html
        assert 'rel="canonical" href="https://freesurfforecast.com/compare/surfline"' in html

    def test_compare_has_faq_schema(self, client):
        html = client.get('/compare/surfline').data.decode()
        faqs = _parse_json_ld(html, 'FAQPage')
        assert len(faqs) == 1

    def test_compare_has_trademark_disclaimer(self, client):
        html = client.get('/compare/surfline').data.decode()
        assert 'not affiliated' in html

    def test_compare_in_sitemap(self, client):
        xml = client.get('/sitemap.xml').data.decode()
        assert 'https://freesurfforecast.com/compare/surfline' in xml

    def test_about_links_to_compare(self, client):
        html = client.get('/about').data.decode()
        assert 'href="/compare/surfline"' in html


class TestRegionExpansion:
    """New regional hubs: Delmarva, SC/GA, New England, Gulf Coast."""

    NEW_REGIONS = ['delmarva', 'south-carolina-georgia', 'new-england', 'gulf-coast']

    @pytest.mark.parametrize('slug', NEW_REGIONS)
    def test_region_page_returns_200(self, client, slug):
        resp = client.get(f'/regions/{slug}')
        assert resp.status_code == 200

    @pytest.mark.parametrize('slug', NEW_REGIONS)
    def test_region_in_sitemap(self, client, slug):
        xml = client.get('/sitemap.xml').data.decode()
        assert f'https://freesurfforecast.com/regions/{slug}' in xml

    def test_all_region_slug_lists_resolve(self):
        from region_pages import REGIONS
        for region in REGIONS:
            for slug in (region.get('slug_list') or []):
                assert slug in LOCATION_BY_SLUG, (
                    f"Region {region['slug']} references unknown spot {slug}"
                )

    def test_region_pages_list_their_spots(self, client):
        html = client.get('/regions/gulf-coast').data.decode()
        assert 'galveston-tx' in html
        assert 'gulf-shores-al' in html


class TestMagicSeaweedPage:
    """/compare/magicseaweed landing page for MSW-replacement queries."""

    def test_returns_200(self, client):
        assert client.get('/compare/magicseaweed').status_code == 200

    def test_meta_and_canonical(self, client):
        html = client.get('/compare/magicseaweed').data.decode()
        assert 'MagicSeaweed Replacement' in html
        assert 'rel="canonical" href="https://freesurfforecast.com/compare/magicseaweed"' in html

    def test_has_faq_schema(self, client):
        html = client.get('/compare/magicseaweed').data.decode()
        assert len(_parse_json_ld(html, 'FAQPage')) == 1

    def test_in_sitemap(self, client):
        xml = client.get('/sitemap.xml').data.decode()
        assert 'https://freesurfforecast.com/compare/magicseaweed' in xml

    def test_cross_linked_from_surfline_page(self, client):
        html = client.get('/compare/surfline').data.decode()
        assert 'href="/compare/magicseaweed"' in html


class TestEmbedWidget:
    """Iframe-embeddable forecast card at /embed/<slug>."""

    def test_embed_returns_200(self, client):
        slug = next(iter(LOCATION_BY_SLUG))
        assert client.get(f'/embed/{slug}').status_code == 200

    def test_embed_unknown_slug_404(self, client):
        assert client.get('/embed/not-a-real-spot').status_code == 404

    def test_embed_is_frameable(self, client):
        slug = next(iter(LOCATION_BY_SLUG))
        resp = client.get(f'/embed/{slug}')
        assert 'X-Frame-Options' not in resp.headers
        assert 'frame-ancestors *' in resp.headers.get('Content-Security-Policy', '')

    def test_other_pages_still_deny_framing(self, client):
        resp = client.get('/')
        assert resp.headers.get('X-Frame-Options') == 'DENY'
        assert 'frame-ancestors *' not in resp.headers.get('Content-Security-Policy', '')

    def test_embed_noindex(self, client):
        slug = next(iter(LOCATION_BY_SLUG))
        html = client.get(f'/embed/{slug}').data.decode()
        assert 'noindex' in html

    def test_about_documents_embed(self, client):
        html = client.get('/about').data.decode()
        assert '/embed/' in html


class TestSocialCards:
    """Social content pipeline: /social/daily/<region>.{png,jpg} + caption API."""

    FAKE_FORECAST = {
        'forecast': [{
            'time': '2020-01-01T00:00Z',
            'wave_height': 1.0, 'wave_period': 10.0,
            'wind_speed': 15.0, 'wind_direction': 270.0,
        }],
        'location_timezone': 'America/New_York', 'source': 'Open-Meteo',
    }

    # 0.152 m is the real 30-day MAE at the time of writing -> 0.5 ft.
    # A publishable stats shape: the accuracy line on this card answers to
    # the same guards as the weekly accuracy card (recent, enough pairs,
    # enough stations), so a fixture that would be refused in production must
    # not be the one the caption test passes with.
    @staticmethod
    def _fake_verification():
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        return {
            'generated': (_dt.now(_tz.utc) - _td(hours=2)).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'window_days': 30,
            'n_pairs': 50000,
            'overall': {'all': {'mae_m': 0.152, 'n': 50000}},
            'stations': {f'440{i:02d}': {'all': {'mae_m': 0.16, 'n': 2000}}
                         for i in range(8)},
        }

    def _with_fake_upstream(self, client, path, verification=True):
        from unittest.mock import patch
        import app as a
        a._cache.pop('social:virginia-coast', None)
        stats = self._fake_verification() if verification else None
        # Patched, not merely faked: the real lookup would reach out to
        # raw.githubusercontent on a 10s timeout during the test run.
        with patch('app.get_point_weather_data', return_value=self.FAKE_FORECAST), \
                patch('app._get_verification_stats', return_value=stats):
            return client.get(path)

    def test_png_route(self, client):
        resp = self._with_fake_upstream(client, '/social/daily/virginia-coast.png')
        assert resp.status_code == 200
        assert resp.content_type == 'image/png'
        assert resp.data[:8] == b'\x89PNG\r\n\x1a\n'

    def test_jpg_route(self, client):
        """Instagram's publishing API accepts JPEG only, so the .jpg variant
        must exist and really be a JPEG."""
        resp = self._with_fake_upstream(client, '/social/daily/virginia-coast.jpg')
        assert resp.status_code == 200
        assert resp.content_type == 'image/jpeg'
        assert resp.data[:3] == b'\xff\xd8\xff'

    def test_caption_api(self, client):
        resp = self._with_fake_upstream(client, '/api/social-card/virginia-coast')
        assert resp.status_code == 200
        d = resp.get_json()
        assert 'freesurfforecast.com' in d['caption']
        assert '#surf' in d['caption']
        # The publisher posts this URL straight to the Graph API, so it has to
        # be the JPEG -- a PNG here would fail every automated post.
        assert d['image_url'].endswith('/social/daily/virginia-coast.jpg')
        assert d['image_url_png'].endswith('/social/daily/virginia-coast.png')
        assert d['spots'][0]['height_ft'] == 3.3  # 1.0 m

    def test_unknown_region_404(self, client):
        assert client.get('/social/daily/atlantis.png').status_code == 404
        assert client.get('/social/daily/atlantis.jpg').status_code == 404
        assert client.get('/api/social-card/atlantis').status_code == 404

    def test_caption_carries_verified_accuracy(self, client):
        resp = self._with_fake_upstream(client, '/api/social-card/virginia-coast')
        d = resp.get_json()
        assert d['accuracy_mae_ft'] == 0.5
        assert '0.5ft average error' in d['caption']
        assert 'NOAA buoys' in d['caption']

    def test_caption_drops_the_accuracy_line_when_the_stats_are_stale(self, client):
        """The daily card states the same measured error as the weekly accuracy
        card, off the same file, so it answers to the same publication guards.
        Stale stats reach here looking healthy -- the app serves the last good
        stats.json for up to a week -- and this card posts every day, so it is
        the surface that would state an old number first."""
        from unittest.mock import patch
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        import app as a
        a._cache.pop('social:virginia-coast', None)
        stale = self._fake_verification()
        stale['generated'] = (_dt.now(_tz.utc) - _td(days=6)).strftime('%Y-%m-%dT%H:%M:%SZ')
        with patch('app.get_point_weather_data', return_value=self.FAKE_FORECAST), \
                patch('app._get_verification_stats', return_value=stale):
            d = client.get('/api/social-card/virginia-coast').get_json()
        assert d['accuracy_mae_ft'] is None
        assert 'average error' not in d['caption']

    def test_caption_drops_the_accuracy_line_when_too_few_stations(self, client):
        from unittest.mock import patch
        import app as a
        a._cache.pop('social:virginia-coast', None)
        thin = self._fake_verification()
        thin['stations'] = {'44001': {'all': {'mae_m': 0.16, 'n': 2000}}}
        with patch('app.get_point_weather_data', return_value=self.FAKE_FORECAST), \
                patch('app._get_verification_stats', return_value=thin):
            d = client.get('/api/social-card/virginia-coast').get_json()
        assert d['accuracy_mae_ft'] is None

    def test_card_renders_without_verification_stats(self, client):
        """The accuracy line is a bonus, never a hard dependency -- if the
        stats fetch is down the card must still render."""
        resp = self._with_fake_upstream(
            client, '/api/social-card/virginia-coast', verification=False)
        assert resp.status_code == 200
        d = resp.get_json()
        assert d['accuracy_mae_ft'] is None
        assert 'average error' not in d['caption']

        png = self._with_fake_upstream(
            client, '/social/daily/virginia-coast.png', verification=False)
        assert png.status_code == 200
        assert png.data[:8] == b'\x89PNG\r\n\x1a\n'

    def test_story_route(self, client):
        """Stories need a 9:16 canvas, not the 4:5 feed card letterboxed."""
        from PIL import Image
        import io as _io
        resp = self._with_fake_upstream(client, '/social/story/virginia-coast.jpg')
        assert resp.status_code == 200
        assert resp.content_type == 'image/jpeg'
        assert Image.open(_io.BytesIO(resp.data)).size == (1080, 1920)

    def test_feed_and_story_sizes_differ(self, client):
        from PIL import Image
        import io as _io
        feed = self._with_fake_upstream(client, '/social/daily/virginia-coast.jpg')
        assert Image.open(_io.BytesIO(feed.data)).size == (1080, 1350)

    def test_caption_api_exposes_story_url(self, client):
        resp = self._with_fake_upstream(client, '/api/social-card/virginia-coast')
        d = resp.get_json()
        assert d['story_image_url'].endswith('/social/story/virginia-coast.jpg')

    def _publisher(self):
        import importlib.util, os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            'instagram_publish', os.path.join(root, 'scripts', 'instagram_publish.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_every_rotation_region_has_a_posting_group(self):
        """Each region must map to a schedule group, or its day silently
        posts nothing once the cron is split by coast.

        The rotation used to be exactly 7 long because it was indexed by
        weekday. It is indexed by day of year now so the length is free --
        which is what lets a high-demand region appear more than once per
        cycle -- so length is no longer the invariant. Group coverage is.
        """
        mod = self._publisher()
        assert mod.REGION_ROTATION, 'rotation is empty'
        for region in mod.REGION_ROTATION:
            assert region in mod.REGION_GROUPS, f'{region} has no posting group'
            assert mod.REGION_GROUPS[region] in ('east', 'west')
        groups = {mod.REGION_GROUPS[r] for r in mod.REGION_ROTATION}
        assert groups == {'east', 'west'}, (
            'both cron groups must appear in the rotation, or one coast never '
            f'posts; got {groups}')

    def test_rotation_covers_the_most_read_spots(self):
        """The feed should advertise the coastlines people actually read.
        These four regions carry the top pages on the site; a rotation without
        them posts about places the audience does not visit."""
        mod = self._publisher()
        for region in ('north-carolina-coast', 'jersey-shore',
                       'virginia-coast', 'northern-california'):
            assert region in mod.REGION_ROTATION, (
                f'{region} carries a top-read spot but never posts')

    def test_group_lookup_never_returns_the_wrong_coast(self):
        """rotation_region(day, group) is what each cron uses; returning a
        region from the other group would post Hawaii on the east schedule."""
        import datetime
        mod = self._publisher()
        for offset in range(0, 366):
            day = datetime.date(2026, 1, 1) + datetime.timedelta(days=offset)
            for group in ('east', 'west'):
                got = mod.rotation_region(day, group)
                assert got is not None, f'no {group} region for {day}'
                assert mod.REGION_GROUPS[got] == group, (
                    f'{day} {group} -> {got} which is '
                    f'{mod.REGION_GROUPS[got]}')

    def test_scheduled_slot_survives_a_delivery_past_midnight(self):
        """A late run must post the slot it was FIRED for, not the region for
        whatever day it happens to start on.

        Real case, 2026-08-28: the 16:07 UTC west slot was delivered at 00:32
        the next day. Reading the wall clock gave Saturday's region and posted
        it a day early; Saturday's own east cron then found that region was
        not east and exited "nothing to do", so a slot posted nothing.
        """
        import datetime
        mod = self._publisher()
        ran_at = datetime.datetime(2026, 8, 29, 0, 32,
                                   tzinfo=datetime.timezone.utc)
        slot = mod.scheduled_fire('16:07', ran_at)
        assert slot == datetime.datetime(2026, 8, 28, 16, 7,
                                         tzinfo=datetime.timezone.utc)
        by_slot = mod.rotation_region(slot.date())
        by_wall = mod.rotation_region(ran_at.date())
        assert by_slot != by_wall, (
            'this fixture only tests something if the two days differ')

    def test_scheduled_fire_never_returns_a_future_slot(self):
        import datetime
        mod = self._publisher()
        for hour in range(24):
            now = datetime.datetime(2026, 8, 29, hour, 30,
                                    tzinfo=datetime.timezone.utc)
            for hhmm in ('11:07', '16:07', '22:07', '15:07'):
                fire = mod.scheduled_fire(hhmm, now)
                assert fire <= now, (hhmm, hour, fire)
                assert (now - fire) < datetime.timedelta(days=1)

    def test_dead_hours_are_measured_in_the_region_s_own_time(self):
        """23:00-05:00 local. The whole point is that a Hawaii card and a New
        Jersey card are asleep at different UTC hours."""
        import datetime
        mod = self._publisher()
        # 12:00 UTC: 08:00 in NJ (fine), 02:00 in Hawaii (dead)
        t = datetime.datetime(2026, 8, 29, 12, 0, tzinfo=datetime.timezone.utc)
        assert not mod.in_dead_hours('America/New_York', t)
        assert mod.in_dead_hours('Pacific/Honolulu', t)
        # 06:00 UTC: 02:00 in NJ (dead), 20:00 in Hawaii (fine)
        t = datetime.datetime(2026, 8, 29, 6, 0, tzinfo=datetime.timezone.utc)
        assert mod.in_dead_hours('America/New_York', t)
        assert not mod.in_dead_hours('Pacific/Honolulu', t)

    def test_dead_window_boundaries(self):
        import datetime
        mod = self._publisher()
        def ny(hour):
            # 2026-08-29 is EDT, UTC-4
            return datetime.datetime(2026, 8, 29, (hour + 4) % 24,
                                     tzinfo=datetime.timezone.utc)
        assert not mod.in_dead_hours('America/New_York', ny(22))   # 22:00 ok
        assert mod.in_dead_hours('America/New_York', ny(23))       # 23:00 dead
        assert mod.in_dead_hours('America/New_York', ny(4))        # 04:00 dead
        assert not mod.in_dead_hours('America/New_York', ny(5))    # 05:00 ok

    def test_every_rotation_region_has_a_timezone(self):
        """A region with no entry falls back to Eastern, which would quietly
        judge a Pacific card against the wrong clock."""
        mod = self._publisher()
        for region in mod.REGION_ROTATION:
            assert region in mod.REGION_TZ, f'{region} has no timezone'

    def test_flat_day_swap_stays_within_the_group(self):
        """The swap exists to avoid posting a flat card, NOT to rank coasts.
        Picking the biggest surf anywhere would send an east-coast audience a
        Pacific card nearly every day, starving the regions the rotation was
        weighted to serve."""
        mod = self._publisher()
        cards = {r: {'spots': [{'score': 5.0}]} for r in mod.REGION_GROUPS}
        cards['jersey-shore'] = {'spots': [{'score': 4.0}]}      # flat, scheduled
        cards['virginia-coast'] = {'spots': [{'score': 40.0}]}   # best east
        cards['hawaii-north-shore'] = {'spots': [{'score': 99.0}]}  # best overall
        mod.fetch_card = lambda base, slug, retries=3, **kw: cards[slug]
        got = mod.pick_region('http://x', 'jersey-shore', 'east')
        assert got == 'virginia-coast', got
        assert mod.REGION_GROUPS[got] == 'east'

    def test_a_firing_region_is_never_swapped_away(self):
        mod = self._publisher()
        cards = {r: {'spots': [{'score': 90.0}]} for r in mod.REGION_GROUPS}
        cards['jersey-shore'] = {'spots': [{'score': 45.0}]}
        mod.fetch_card = lambda base, slug, retries=3, **kw: cards[slug]
        assert mod.pick_region('http://x', 'jersey-shore', 'east') == 'jersey-shore'

    def test_flat_day_check_failure_keeps_the_scheduled_region(self):
        """A fetch problem during the check must not cost the day's post."""
        mod = self._publisher()
        def boom(base, slug, retries=3, **kw):
            raise RuntimeError('upstream down')
        mod.fetch_card = boom
        assert mod.pick_region('http://x', 'jersey-shore', 'east') == 'jersey-shore'

    def test_card_title_fits_canvas_for_every_region(self):
        """Long region names used to run off the 1080px canvas at a fixed
        58px title. Every region must fit within the padded width."""
        from PIL import Image, ImageDraw
        import app as a
        from region_pages import REGIONS_BY_SLUG

        draw = ImageDraw.Draw(Image.new('RGB', (1080, 1350)))
        max_width = 1080 - 2 * 64
        for region in REGIONS_BY_SLUG.values():
            title = f"{region['short_title']} Surf Check"
            font, _size, lines = a._fit_card_title(draw, title, max_width)
            widest = max(draw.textlength(line, font=font) for line in lines)
            assert widest <= max_width, f"{title} overflows at {widest}px"
            assert len(lines) <= 2

    def test_simple_score_ranking(self):
        import app as a
        big_clean = {'wave_height': 2.0, 'wave_period': 14, 'wind_speed': 5}
        small_blown = {'wave_height': 0.5, 'wave_period': 5, 'wind_speed': 35}
        assert a._simple_surf_score(big_clean) > a._simple_surf_score(small_blown)


class TestInternalLinkMesh:
    """Search Console showed 130 of 187 URLs discovered but never crawled.
    Nearby-spot links are the main crawl path into those pages, so the mesh
    must not leave any spot unreachable."""

    def _map(self):
        import app as a
        a._nearby_for_slug('wrightsville-beach')  # force the lazy build
        return a._NEARBY_BY_SLUG

    def test_no_spot_is_orphaned(self):
        import app as a
        from collections import Counter
        inbound = Counter()
        for spots in self._map().values():
            for spot in spots:
                inbound[spot['slug']] += 1
        orphans = [s for s in a.LOCATION_BY_SLUG if inbound[s] == 0]
        assert orphans == [], f'unreachable from any nearby list: {orphans}'

    def test_every_page_emits_a_minimum_of_links(self):
        import app as a
        for slug, spots in self._map().items():
            assert len(spots) >= a.NEARBY_MIN_COUNT, f'{slug} emits {len(spots)}'

    def test_isolated_spot_still_gets_links(self):
        """Westport WA's nearest neighbour is 261 miles away -- outside the
        radius, so it used to emit nothing at all."""
        import app as a
        spots = a._nearby_for_slug('westport-wa')
        assert len(spots) >= a.NEARBY_MIN_COUNT
        assert all(s['distance_mi'] > 0 for s in spots)

    def test_nearby_links_render_on_an_isolated_page(self, client):
        import re
        html = client.get('/forecast/key-west-fl').data.decode()
        assert len(re.findall(r'href="/forecast/', html)) >= 4


class TestSitemapLastmod:
    def test_lastmod_is_stable_not_todays_date(self):
        """Every URL used to claim it changed today, every day. Google
        discounts lastmod once it proves unreliable."""
        import app as a
        first = a._content_lastmod()
        assert first == a._content_lastmod()
        import datetime as dt
        dt.date.fromisoformat(first)  # raises if malformed

    def test_sitemap_uses_one_content_date(self, client):
        import re
        import app as a
        sm = client.get('/sitemap.xml').data.decode()
        values = set(re.findall(r'<lastmod>([^<]+)</lastmod>', sm))
        assert values == {a._content_lastmod()}


class TestInstagramLanding:
    """/ig is the Instagram bio link. Cloudflare Web Analytics drops query
    strings and reports no referrer for in-app browsers, so the path itself
    is the only attribution signal -- it must render the dashboard while
    staying out of the index."""

    def test_ig_returns_200(self, client):
        r = client.get('/ig')
        assert r.status_code == 200

    def test_ig_sets_noindex_header(self, client):
        r = client.get('/ig')
        tag = r.headers.get('X-Robots-Tag', '')
        assert 'noindex' in tag
        assert 'follow' in tag

    def test_ig_canonical_points_at_homepage(self, client):
        html = client.get('/ig').data.decode()
        canonical = re.search(r'<link rel="canonical" href="([^"]+)"', html)
        assert canonical, "Missing canonical link"
        assert canonical.group(1) == 'https://freesurfforecast.com/'

    def test_ig_absent_from_sitemap(self, client):
        xml = client.get('/sitemap.xml').data.decode()
        assert 'freesurfforecast.com/ig' not in xml

    def test_robots_does_not_block_ig(self, client):
        """The noindex header only works if crawlers are allowed to fetch it."""
        text = client.get('/robots.txt').data.decode()
        assert 'Disallow: /ig' not in text

    def test_ig_renders_the_dashboard(self, client):
        html = client.get('/ig').data.decode()
        assert 'id="dashboard-grid"' in html
        assert 'Surf City' in html

    def test_ig_html_is_cacheable_like_other_pages(self, client):
        r = client.get('/ig')
        assert 'max-age=300' in r.headers.get('Cache-Control', '')


class TestLayoutStability:
    """Guards the layout-shift work: the SSR block sits below the dashboard so
    its removal only reflows below-the-fold content, and the hero card reserves
    the height it settles at instead of inflating the page when data lands."""

    def test_ssr_summary_sits_below_the_dashboard(self, client):
        html = client.get('/forecast/virginia-beach').data.decode()
        grid = html.find('id="dashboard-grid"')
        ssr = html.find('id="ssr-summary"')
        end_main = html.find('</main>')
        assert grid > 0 and ssr > 0 and end_main > 0
        assert grid < ssr < end_main, "SSR block must render after the grid, inside <main>"

    def test_hero_ships_hidden_with_a_reveal_script(self, client):
        """Hidden in markup so a no-JS client is not shown an empty card, then
        revealed inline before first paint."""
        html = client.get('/forecast/virginia-beach').data.decode()
        assert '<section id="forecast-hero" class="forecast-hero" style="display:none"' in html
        assert "getElementById('forecast-hero')" in html

    def test_hero_height_is_reserved(self, client):
        html = client.get('/forecast/virginia-beach').data.decode()
        assert re.search(r'#forecast-hero\s*\{\s*min-height:\s*184px', html)
        assert re.search(r'#forecast-hero\s*\{\s*min-height:\s*294px', html)
        assert re.search(r'#forecast-hero-rows\s*\{\s*min-height:\s*183px', html)
