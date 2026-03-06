#!/usr/bin/env python3
"""
SEO metrics collector and report generator.

Collects SEO health metrics from the app (locally or against a live URL),
appends to a historical JSON file, and generates a markdown report.

Usage:
    # Against local Flask app (default):
    python scripts/seo_report.py

    # Against live site:
    python scripts/seo_report.py --base-url https://freesurfforecast.com

    # Output report to file:
    python scripts/seo_report.py --output reports/seo-report.md
"""
import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import urlparse

# Add project root to path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'seo_metrics_history.json',
)

# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------

def collect_metrics_local():
    """Collect metrics using Flask test client (no network needed)."""
    from app import app, LOCATION_BY_SLUG, SLUG_BY_COORDS

    metrics = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'source': 'local',
        'locations': {},
        'sitemap': {},
        'robots': {},
        'pages': {},
        'internal_links': {},
        'structured_data': {},
        'technical': {},
    }

    with app.test_client() as c:
        # --- Location infrastructure ---
        metrics['locations'] = {
            'total_slugs': len(LOCATION_BY_SLUG),
            'total_coord_lookups': len(SLUG_BY_COORDS),
            'slug_coord_match': len(LOCATION_BY_SLUG) == len(SLUG_BY_COORDS),
        }

        # --- Sitemap ---
        r = c.get('/sitemap.xml')
        xml_text = r.data.decode()
        metrics['sitemap']['status'] = r.status_code
        metrics['sitemap']['valid_xml'] = True
        try:
            root = ET.fromstring(xml_text)
            ns = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            urls = root.findall('.//s:loc', ns)
            url_texts = [u.text for u in urls]
            metrics['sitemap']['total_urls'] = len(url_texts)
            metrics['sitemap']['forecast_urls'] = sum(
                1 for u in url_texts if '/forecast/' in u
            )
            metrics['sitemap']['has_locations_page'] = any(
                '/locations' in u for u in url_texts
            )
            metrics['sitemap']['has_about'] = any(
                '/about' in u for u in url_texts
            )
            metrics['sitemap']['has_query_param_urls'] = any(
                '?lat=' in u for u in url_texts
            )
            metrics['sitemap']['all_have_lastmod'] = all(
                loc.find('s:lastmod', ns) is not None
                for loc in root.findall('.//s:url', ns)
            )
        except ET.ParseError:
            metrics['sitemap']['valid_xml'] = False

        # --- Robots.txt ---
        r = c.get('/robots.txt')
        robots = r.data.decode()
        metrics['robots'] = {
            'status': r.status_code,
            'allows_root': 'Allow: /' in robots,
            'disallows_api': 'Disallow: /api/' in robots,
            'has_sitemap_ref': 'sitemap' in robots.lower(),
        }

        # --- Sample pages (home, forecast, locations) ---
        pages_to_check = {
            'home': '/',
            'forecast_virginia_beach': '/forecast/virginia-beach',
            'forecast_wrightsville': '/forecast/wrightsville-beach',
            'locations': '/locations',
            'about': '/about',
        }
        page_metrics = {}
        for name, path in pages_to_check.items():
            t0 = time.time()
            r = c.get(path)
            elapsed = round((time.time() - t0) * 1000)
            html = r.data.decode()

            pm = {
                'status': r.status_code,
                'response_time_ms': elapsed,
                'size_bytes': len(r.data),
                'has_title': '<title>' in html,
                'has_meta_description': 'name="description"' in html,
                'has_canonical': 'rel="canonical"' in html,
                'has_og_tags': 'og:title' in html,
                'has_twitter_tags': 'twitter:card' in html,
                'cache_control': r.headers.get('Cache-Control', ''),
            }

            # Meta description length
            desc_match = re.search(
                r'<meta name="description" content="([^"]+)"', html
            )
            if desc_match:
                pm['meta_description_length'] = len(desc_match.group(1))

            # Canonical URL
            can_match = re.search(r'<link rel="canonical" href="([^"]+)"', html)
            if can_match:
                pm['canonical_url'] = can_match.group(1)

            # H1 count
            pm['h1_count'] = len(re.findall(r'<h1[ >]', html))

            page_metrics[name] = pm

        metrics['pages'] = page_metrics

        # --- Internal links ---
        r = c.get('/')
        html = r.data.decode()
        footer_forecast_links = len(re.findall(r'href="/forecast/', html))
        has_locations_link = 'href="/locations"' in html

        r2 = c.get('/locations')
        loc_html = r2.data.decode()
        locations_forecast_links = len(re.findall(r'href="/forecast/', loc_html))

        metrics['internal_links'] = {
            'footer_forecast_links': footer_forecast_links,
            'footer_has_locations_link': has_locations_link,
            'locations_page_forecast_links': locations_forecast_links,
            'locations_covers_all_slugs': locations_forecast_links >= len(LOCATION_BY_SLUG),
        }

        # --- Structured data ---
        r = c.get('/forecast/virginia-beach')
        html = r.data.decode()
        json_ld_blocks = re.findall(
            r'<script type="application/ld\+json">\s*(\{.*?\})\s*</script>',
            html, re.DOTALL,
        )
        schema_types = []
        for block in json_ld_blocks:
            try:
                data = json.loads(block)
                schema_types.append(data.get('@type', 'unknown'))
            except json.JSONDecodeError:
                schema_types.append('invalid_json')

        metrics['structured_data'] = {
            'json_ld_block_count': len(json_ld_blocks),
            'schema_types': schema_types,
            'has_breadcrumb': 'BreadcrumbList' in schema_types,
            'has_web_application': 'WebApplication' in schema_types,
            'has_faq': 'FAQPage' in schema_types,
        }

        # --- Technical ---
        metrics['technical'] = {
            'has_preconnect': 'rel="preconnect"' in html,
            'has_dns_prefetch': 'rel="dns-prefetch"' in html,
            'has_manifest': 'rel="manifest"' in html,
            'has_theme_color': 'theme-color' in html,
            'has_noscript_fallback': '<noscript>' in html,
            'has_ssr_js_removal': "getElementById('ssr-summary')" in html,
            'has_slug_lookup_js': '_slugByCoords' in html,
            'has_history_replace': 'history.replaceState' in html,
        }

        # --- Redirect check ---
        from app import LOCATION_BY_SLUG as lbs
        sample_loc = lbs.get('virginia-beach')
        if sample_loc:
            r = c.get(
                f'/?lat={sample_loc["lat"]}&lon={sample_loc["lon"]}'
                f'&name=Virginia+Beach'
            )
            metrics['technical']['redirect_301_works'] = r.status_code == 301

    return metrics


def collect_metrics_remote(base_url):
    """Collect metrics against a live URL."""
    import requests as req

    metrics = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'source': base_url,
        'sitemap': {},
        'robots': {},
        'pages': {},
        'technical': {},
    }

    # --- Sitemap ---
    t0 = time.time()
    r = req.get(f'{base_url}/sitemap.xml', timeout=15)
    metrics['sitemap']['status'] = r.status_code
    metrics['sitemap']['response_time_ms'] = round((time.time() - t0) * 1000)
    try:
        root = ET.fromstring(r.text)
        ns = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [u.text for u in root.findall('.//s:loc', ns)]
        metrics['sitemap']['valid_xml'] = True
        metrics['sitemap']['total_urls'] = len(urls)
        metrics['sitemap']['forecast_urls'] = sum(
            1 for u in urls if '/forecast/' in u
        )
        metrics['sitemap']['has_query_param_urls'] = any(
            '?lat=' in u for u in urls
        )
    except ET.ParseError:
        metrics['sitemap']['valid_xml'] = False

    # --- Robots.txt ---
    r = req.get(f'{base_url}/robots.txt', timeout=10)
    metrics['robots'] = {
        'status': r.status_code,
        'allows_root': 'Allow: /' in r.text,
        'disallows_api': 'Disallow: /api/' in r.text,
    }

    # --- Sample pages ---
    pages = {
        'home': '/',
        'forecast': '/forecast/virginia-beach',
        'locations': '/locations',
    }
    page_metrics = {}
    for name, path in pages.items():
        t0 = time.time()
        r = req.get(f'{base_url}{path}', timeout=15)
        elapsed = round((time.time() - t0) * 1000)
        page_metrics[name] = {
            'status': r.status_code,
            'response_time_ms': elapsed,
            'size_bytes': len(r.content),
            'has_title': '<title>' in r.text,
            'has_meta_description': 'name="description"' in r.text,
            'has_canonical': 'rel="canonical"' in r.text,
            'has_og_tags': 'og:title' in r.text,
            'cache_control': r.headers.get('Cache-Control', ''),
        }
    metrics['pages'] = page_metrics

    # --- Structured data (from forecast page) ---
    forecast_html = req.get(
        f'{base_url}/forecast/virginia-beach', timeout=15
    ).text
    json_ld_blocks = re.findall(
        r'<script type="application/ld\+json">\s*(\{.*?\})\s*</script>',
        forecast_html, re.DOTALL,
    )
    schema_types = []
    for block in json_ld_blocks:
        try:
            data = json.loads(block)
            schema_types.append(data.get('@type', 'unknown'))
        except json.JSONDecodeError:
            schema_types.append('invalid_json')
    metrics['structured_data'] = {
        'json_ld_block_count': len(json_ld_blocks),
        'schema_types': schema_types,
        'has_breadcrumb': 'BreadcrumbList' in schema_types,
        'has_web_application': 'WebApplication' in schema_types,
        'has_faq': 'FAQPage' in schema_types,
    }

    # --- Internal links ---
    home_html = req.get(f'{base_url}/', timeout=15).text
    loc_html = req.get(f'{base_url}/locations', timeout=15).text
    metrics['internal_links'] = {
        'footer_forecast_links': len(re.findall(r'href="/forecast/', home_html)),
        'footer_has_locations_link': 'href="/locations"' in home_html,
        'locations_page_forecast_links': len(
            re.findall(r'href="/forecast/', loc_html)
        ),
        'locations_covers_all_slugs': len(
            re.findall(r'href="/forecast/', loc_html)
        ) >= metrics['sitemap'].get('forecast_urls', 0),
    }

    # --- Technical ---
    metrics['technical'] = {
        'has_preconnect': 'rel="preconnect"' in forecast_html,
        'has_dns_prefetch': 'rel="dns-prefetch"' in forecast_html,
        'has_noscript_fallback': '<noscript>' in forecast_html,
        'has_ssr_js_removal': "getElementById('ssr-summary')" in forecast_html,
    }

    # --- Redirect check ---
    r = req.get(
        f'{base_url}/?lat=36.8529&lon=-75.978&name=Virginia+Beach',
        allow_redirects=False, timeout=10,
    )
    metrics['technical']['redirect_301_works'] = r.status_code == 301

    return metrics


# ---------------------------------------------------------------------------
# History tracking
# ---------------------------------------------------------------------------

def load_history():
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(metrics, history):
    """Generate a markdown report from current metrics and history."""
    ts = metrics['timestamp'][:19].replace('T', ' ') + ' UTC'
    lines = [
        f'# SEO Health Report',
        f'',
        f'Generated: {ts}  ',
        f'Source: `{metrics["source"]}`',
        f'',
    ]

    # --- Score card ---
    checks = []
    def check(name, passed):
        checks.append((name, passed))
        return passed

    sm = metrics.get('sitemap', {})
    check('Sitemap returns 200', sm.get('status') == 200)
    check('Sitemap is valid XML', sm.get('valid_xml', False))
    check('No query-param URLs in sitemap', not sm.get('has_query_param_urls', True))

    rb = metrics.get('robots', {})
    check('Robots.txt returns 200', rb.get('status') == 200)
    check('Robots allows /', rb.get('allows_root', False))
    check('Robots disallows /api/', rb.get('disallows_api', False))

    sd = metrics.get('structured_data', {})
    check('Has BreadcrumbList schema', sd.get('has_breadcrumb', False))
    check('Has WebApplication schema', sd.get('has_web_application', False))
    check('Has FAQPage schema', sd.get('has_faq', False))

    il = metrics.get('internal_links', {})
    check('Footer links to /locations', il.get('footer_has_locations_link', False))
    check('Locations page covers all slugs', il.get('locations_covers_all_slugs', False))

    tech = metrics.get('technical', {})
    check('301 redirect works', tech.get('redirect_301_works', False))
    check('Has preconnect hints', tech.get('has_preconnect', False))
    check('Has noscript fallback', tech.get('has_noscript_fallback', False))
    check('Has SSR JS removal', tech.get('has_ssr_js_removal', False))

    passed = sum(1 for _, p in checks if p)
    total = len(checks)
    pct = round(100 * passed / total) if total else 0

    lines.append(f'## Score: {passed}/{total} ({pct}%)')
    lines.append('')
    lines.append('| Check | Status |')
    lines.append('|---|---|')
    for name, p in checks:
        icon = 'PASS' if p else 'FAIL'
        lines.append(f'| {name} | {icon} |')
    lines.append('')

    # --- Key metrics ---
    lines.append('## Key Metrics')
    lines.append('')
    lines.append('| Metric | Value |')
    lines.append('|---|---|')
    lm = metrics.get('locations', {})
    if lm:
        lines.append(f'| Location slugs | {lm.get("total_slugs", "?")} |')
    lines.append(f'| Sitemap URLs | {sm.get("total_urls", "?")} |')
    lines.append(f'| Forecast URLs in sitemap | {sm.get("forecast_urls", "?")} |')
    if sd:
        lines.append(f'| JSON-LD blocks | {sd.get("json_ld_block_count", "?")} |')
        lines.append(
            f'| Schema types | {", ".join(sd.get("schema_types", []))} |'
        )
    if il:
        lines.append(
            f'| Footer forecast links | {il.get("footer_forecast_links", "?")} |'
        )
        lines.append(
            f'| Locations page links | {il.get("locations_page_forecast_links", "?")} |'
        )
    lines.append('')

    # --- Page performance ---
    pages = metrics.get('pages', {})
    if pages:
        lines.append('## Page Performance')
        lines.append('')
        lines.append('| Page | Status | Time (ms) | Size (KB) | Title | Desc | Canonical | OG |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for name, pm in pages.items():
            size_kb = round(pm.get('size_bytes', 0) / 1024, 1)
            yn = lambda v: 'Y' if v else 'N'
            lines.append(
                f'| {name} | {pm.get("status", "?")} '
                f'| {pm.get("response_time_ms", "?")} '
                f'| {size_kb} '
                f'| {yn(pm.get("has_title"))} '
                f'| {yn(pm.get("has_meta_description"))} '
                f'| {yn(pm.get("has_canonical"))} '
                f'| {yn(pm.get("has_og_tags"))} |'
            )
        lines.append('')

    # --- Trend (if history available) ---
    if len(history) >= 2:
        lines.append('## Trend')
        lines.append('')
        lines.append('| Date | Slugs | Sitemap URLs | Score |')
        lines.append('|---|---|---|---|')
        for entry in history[-10:]:
            date = entry['timestamp'][:10]
            slugs = entry.get('locations', {}).get('total_slugs', '?')
            sitemap_urls = entry.get('sitemap', {}).get('total_urls', '?')

            # Recompute score for historical entries
            h_checks = 0
            h_total = 0
            for section in ['sitemap', 'robots', 'structured_data',
                            'internal_links', 'technical']:
                for k, v in entry.get(section, {}).items():
                    if isinstance(v, bool):
                        h_total += 1
                        if v:
                            h_checks += 1
            h_pct = round(100 * h_checks / h_total) if h_total else '?'

            lines.append(f'| {date} | {slugs} | {sitemap_urls} | {h_pct}% |')
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SEO metrics report')
    parser.add_argument(
        '--base-url',
        help='Base URL to check (default: local Flask test client)',
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file for markdown report (default: stdout)',
    )
    parser.add_argument(
        '--no-history',
        action='store_true',
        help='Skip saving to history file',
    )
    args = parser.parse_args()

    if args.base_url:
        metrics = collect_metrics_remote(args.base_url)
    else:
        metrics = collect_metrics_local()

    # Save to history
    if not args.no_history:
        history = load_history()
        history.append(metrics)
        save_history(history)
    else:
        history = load_history()

    # Generate report
    report = generate_report(metrics, history)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(report)
        print(f'Report written to {args.output}')
    else:
        print(report)


if __name__ == '__main__':
    main()
