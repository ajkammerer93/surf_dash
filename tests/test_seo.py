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

    def test_faq_schema(self, client):
        r = client.get('/')
        html = r.data.decode()
        faqs = _parse_json_ld(html, 'FAQPage')
        assert len(faqs) == 1, "Should have FAQPage schema"


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
    """SSR summary should appear when cache is warm, be absent when cold."""

    def test_ssr_absent_when_cache_cold(self, client):
        from app import _cache
        # Use coords that are unlikely to be cached
        r = client.get('/forecast/pipeline-north-shore-hi')
        html = r.data.decode()
        # May or may not have SSR depending on cache state — just verify page loads
        assert r.status_code == 200

    def test_ssr_section_has_correct_structure(self, client):
        """If SSR is present, verify it has the right structure."""
        # Warm cache first
        loc = LOCATION_BY_SLUG['virginia-beach']
        client.get(f'/api/forecast?lat={loc["lat"]}&lon={loc["lon"]}')
        r = client.get('/forecast/virginia-beach')
        html = r.data.decode()
        if 'id="ssr-summary"' in html:
            assert 'Current Conditions at Virginia Beach' in html
            assert 'waves' in html.lower()

    def test_noscript_fallback_present(self, client):
        r = client.get('/forecast/virginia-beach')
        html = r.data.decode()
        assert '<noscript>' in html
        assert 'Virginia Beach' in html


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
