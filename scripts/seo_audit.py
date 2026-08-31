#!/usr/bin/env python3
"""Collect CI, site-health and search metrics into a single audit snapshot.

Deliberately dumb: this script gathers numbers and writes JSON. All judgement
about what the numbers mean happens elsewhere, so the collection is cheap,
deterministic and testable, and a bad reading is a data problem rather than a
reasoning one.

    python scripts/seo_audit.py --data-dir seodata

Every collector degrades on its own. A missing credential or a dead upstream
records an "unavailable" section with the reason and leaves the rest intact --
an audit that refuses to run because one API is down tells you nothing about
the four that are fine.

Credentials, all optional, all read from the environment:
    GSC_SA_JSON      Google service-account key JSON (the whole blob), for a
                     service account added as a user on the Search Console
                     property.
    GSC_PROPERTY     Search Console property, e.g. sc-domain:freesurfforecast.com
    CF_API_TOKEN     Cloudflare token with Account Analytics: Read
    CF_ACCOUNT_ID    Cloudflare account id
    CF_SITE_TAG      Cloudflare Web Analytics SITE TAG. This is NOT the
                     beacon token embedded in the page's JS snippet -- they are
                     two different values for the same site, and using the
                     beacon token yields a valid account with zero events.
    GITHUB_TOKEN     Raises the GitHub API rate limit; the repo is public, so
                     unauthenticated works too.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

SITE = 'https://freesurfforecast.com'
REPO = 'ajkammerer93/surf_dash'
GITHUB_API = 'https://api.github.com'

# The scheduled workflows whose health actually matters day to day.
WATCHED_WORKFLOWS = [
    'forecast-verification.yml',
    'social-post.yml',
    'social-highlight-post.yml',
    'social-accuracy-post.yml',
    'seo-tests.yml',
    'youtube-cam-scan.yml',
    'instagram-token-refresh.yml',
]

# Pages worth checking on every run: one of each kind, not all 179.
KEY_PAGES = ['/', '/locations', '/accuracy', '/faq', '/compare/surfline',
             '/forecast/wrightsville-beach', '/regions/great-lakes', '/ig']

# URL Inspection is quota-limited (2000/day), and the point is a trend rather
# than a census, so each run inspects a slice and rotates through the sitemap.
INSPECT_PER_RUN = 60


def _utcnow():
    return datetime.now(timezone.utc)


def _get(url, **kw):
    kw.setdefault('timeout', 30)
    return requests.get(url, **kw)


def collect_ci():
    """Recent run outcome for each watched workflow."""
    headers = {'Accept': 'application/vnd.github+json'}
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'

    out = {'workflows': {}, 'failing': []}
    for wf in WATCHED_WORKFLOWS:
        try:
            r = _get(f'{GITHUB_API}/repos/{REPO}/actions/workflows/{wf}/runs',
                     headers=headers, params={'per_page': 10})
            if not r.ok:
                out['workflows'][wf] = {'error': f'HTTP {r.status_code}'}
                continue
            runs = r.json().get('workflow_runs', [])
            if not runs:
                out['workflows'][wf] = {'runs': 0}
                continue
            latest = runs[0]
            recent = [x for x in runs if x.get('conclusion')][:5]
            failures = [x for x in recent if x['conclusion'] != 'success']
            entry = {
                'latest_status': latest.get('status'),
                'latest_conclusion': latest.get('conclusion'),
                'latest_at': latest.get('run_started_at'),
                'recent_failures': len(failures),
                'recent_checked': len(recent),
                'url': latest.get('html_url'),
            }
            out['workflows'][wf] = entry
            if latest.get('conclusion') not in (None, 'success'):
                out['failing'].append(wf)
        except requests.RequestException as e:
            out['workflows'][wf] = {'error': str(e)}
    return out


def collect_site():
    """Response health for a representative page of each kind, plus sitemap
    integrity and the freshness of the verification stats the accuracy page
    depends on."""
    out = {'pages': {}, 'problems': []}
    for path in KEY_PAGES:
        try:
            t0 = time.time()
            r = _get(SITE + path, allow_redirects=True)
            out['pages'][path] = {
                'status': r.status_code,
                'ms': round((time.time() - t0) * 1000),
                'bytes': len(r.content),
            }
            if r.status_code != 200:
                out['problems'].append(f'{path} returned {r.status_code}')
        except requests.RequestException as e:
            out['pages'][path] = {'error': str(e)}
            out['problems'].append(f'{path} unreachable: {e}')

    try:
        r = _get(f'{SITE}/sitemap.xml')
        body = r.text
        out['sitemap'] = {
            'status': r.status_code,
            'urls': body.count('<loc>'),
            'has_query_params': '?lat=' in body or '?utm' in body,
        }
        if out['sitemap']['has_query_params']:
            out['problems'].append('sitemap contains query-parameter URLs')
    except requests.RequestException as e:
        out['sitemap'] = {'error': str(e)}

    try:
        r = _get(f'{SITE}/api/accuracy')
        stats = r.json()
        generated = stats.get('generated')
        age_h = None
        if generated:
            gen = datetime.fromisoformat(generated.replace('Z', '+00:00'))
            age_h = round((_utcnow() - gen).total_seconds() / 3600, 1)
        out['verification'] = {
            'n_pairs': stats.get('n_pairs'),
            'mae_m': ((stats.get('overall') or {}).get('all') or {}).get('mae_m'),
            'stations': len(stats.get('stations') or {}),
            'age_hours': age_h,
        }
        # The pipeline runs every 6 hours; a full day of silence is a failure
        # nobody would otherwise notice, because the page keeps serving.
        if age_h is not None and age_h > 24:
            out['problems'].append(f'verification stats are {age_h}h stale')
    except (requests.RequestException, ValueError) as e:
        out['verification'] = {'error': str(e)}

    try:
        r = _get(f'{SITE}/api/health-upstreams', timeout=90)
        out['upstreams'] = r.json() if r.ok else {'error': f'HTTP {r.status_code}'}
    except (requests.RequestException, ValueError) as e:
        out['upstreams'] = {'error': str(e)}
    return out


def _gsc_service():
    """Authorised Search Console client, or None with a reason."""
    raw = os.environ.get('GSC_SA_JSON')
    if not raw:
        return None, 'GSC_SA_JSON not set'
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        return None, 'google-api-python-client not installed'
    try:
        info = json.loads(raw)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=['https://www.googleapis.com/auth/webmasters.readonly'])
        return build('searchconsole', 'v1', credentials=creds,
                     cache_discovery=False), None
    except Exception as e:
        return None, f'auth failed: {e}'


def collect_search_console(state):
    """Search Analytics plus a rotating slice of URL Inspection.

    Search Console has no API for the aggregate Index Coverage report, which is
    where the number that matters most lives -- how many of our URLs Google has
    actually crawled. URL Inspection does expose it per-URL, so we walk the
    sitemap a slice at a time and keep the verdicts; a few days of runs give the
    same picture the Coverage report would, and it updates itself from then on.
    """
    service, reason = _gsc_service()
    if not service:
        return {'unavailable': reason}

    prop = os.environ.get('GSC_PROPERTY', 'sc-domain:freesurfforecast.com')
    out = {'property': prop}
    end = (_utcnow() - timedelta(days=2)).date()      # GSC lags ~2 days
    start = end - timedelta(days=28)

    def query(dimensions, limit=100):
        return service.searchanalytics().query(siteUrl=prop, body={
            'startDate': start.isoformat(), 'endDate': end.isoformat(),
            'dimensions': dimensions, 'rowLimit': limit,
        }).execute().get('rows', [])

    try:
        totals = query([], limit=1)
        if totals:
            t = totals[0]
            out['totals_28d'] = {
                'clicks': t.get('clicks'), 'impressions': t.get('impressions'),
                'ctr': round(t.get('ctr', 0), 4),
                'position': round(t.get('position', 0), 2),
            }
        out['top_queries'] = [{
            'query': r['keys'][0], 'clicks': r.get('clicks'),
            'impressions': r.get('impressions'),
            'position': round(r.get('position', 0), 1),
        } for r in query(['query'], 100)]
        out['top_pages'] = [{
            'page': r['keys'][0].replace(SITE, ''), 'clicks': r.get('clicks'),
            'impressions': r.get('impressions'),
            'position': round(r.get('position', 0), 1),
        } for r in query(['page'], 100)]
        # Queries close enough to page one that a small change pays off.
        out['striking_distance'] = [
            q for q in out['top_queries'] if 8 <= (q['position'] or 99) <= 25
        ]
    except Exception as e:
        out['search_analytics_error'] = str(e)

    try:
        sitemap_urls = _sitemap_urls()
        cursor = int(state.get('inspect_cursor', 0)) % max(len(sitemap_urls), 1)
        slice_ = sitemap_urls[cursor:cursor + INSPECT_PER_RUN]
        if len(slice_) < INSPECT_PER_RUN:
            slice_ += sitemap_urls[:INSPECT_PER_RUN - len(slice_)]
        verdicts = {}
        for url in slice_:
            try:
                res = service.urlInspection().index().inspect(body={
                    'inspectionUrl': url, 'siteUrl': prop,
                }).execute()
                idx = (res.get('inspectionResult') or {}).get('indexStatusResult') or {}
                verdicts[url.replace(SITE, '')] = {
                    'verdict': idx.get('verdict'),
                    'coverage': idx.get('coverageState'),
                    'last_crawl': idx.get('lastCrawlTime'),
                }
            except Exception as e:
                verdicts[url.replace(SITE, '')] = {'error': str(e)[:120]}
        out['inspection'] = verdicts
        out['inspection_cursor'] = (cursor + INSPECT_PER_RUN) % max(len(sitemap_urls), 1)
        indexed = sum(1 for v in verdicts.values() if v.get('verdict') == 'PASS')
        out['inspection_summary'] = {
            'checked': len(verdicts), 'indexed': indexed,
            'not_indexed': len(verdicts) - indexed,
        }
    except Exception as e:
        out['inspection_error'] = str(e)
    return out


def _sitemap_urls():
    import re
    r = _get(f'{SITE}/sitemap.xml')
    return re.findall(r'<loc>([^<]+)</loc>', r.text)


def _cf_site_tags(token, account):
    """Web Analytics sites on the account, as (site_tag, host) pairs."""
    try:
        r = requests.get(
            f'https://api.cloudflare.com/client/v4/accounts/{account}/rum/site_info/list',
            headers={'Authorization': f'Bearer {token}'}, timeout=30)
        if not r.ok:
            return {'error': f'HTTP {r.status_code} listing RUM sites'}
        return [{'site_tag': s.get('site_tag'),
                 'host': (s.get('ruleset') or {}).get('zone_name') or s.get('site_token'),
                 'auto_install': s.get('auto_install')}
                for s in (r.json().get('result') or [])]
    except (requests.RequestException, ValueError) as e:
        return {'error': str(e)[:200]}


def _cf_graphql(token, query, variables=None, timeout=60):
    """POST one GraphQL query. Returns (data, error_string); exactly one is None.

    A GraphQL endpoint answers HTTP 200 with an "errors" array, so checking the
    status code alone reports success for a query that returned nothing. Both
    failure shapes are collapsed here so no caller can accidentally check only
    the easy one.
    """
    try:
        r = requests.post(
            'https://api.cloudflare.com/client/v4/graphql',
            headers={'Authorization': f'Bearer {token}',
                     'Content-Type': 'application/json'},
            json={'query': query, 'variables': variables or {}},
            timeout=timeout)
        body = r.json()
        if body.get('errors'):
            return None, str(body['errors'])[:400]
        if not r.ok:
            return None, f'HTTP {r.status_code}'
        return body.get('data'), None
    except (requests.RequestException, ValueError) as e:
        return None, f'{type(e).__name__}: {str(e)[:200]}'


# Read off the live schema by introspection on 2026-08-24, not from docs --
# the dataset is Beta and its fields are not published anywhere checkable. The
# short names you would guess (clsP75, lcpP75) do not exist; the metric names
# are spelled out in full. The schema hint below is what produced this list and
# stays wired up, because a Beta dataset can rename a field again.
CF_VITALS_DATASET = 'rumWebVitalsEventsAdaptiveGroups'
CF_VITALS_QUANTILE_FIELDS = [
    'cumulativeLayoutShiftP75',
    'largestContentfulPaintP75',
    'interactionToNextPaintP75',
    'firstContentfulPaintP75',
    'timeToFirstByteP75',
]


# Timing metrics come back in MICROSECONDS. Cloudflare's own dashboard renders
# the same figure in milliseconds -- its p75 LCP of 2,492ms is this API's
# 2492000 -- so a raw value read next to the UI is off by a thousand and looks
# like a page that took twenty minutes to paint. Converted on the way in, with
# the unit in the key so nobody has to remember this again.
CF_TIMING_METRIC_HINTS = ('Paint', 'Delay', 'Byte', 'Time', 'Load')


def _scale_quantiles(quantiles):
    """Microseconds to milliseconds for timings. CLS is unitless and untouched."""
    out = {}
    for key, value in (quantiles or {}).items():
        if value is None:
            out[key] = None
        elif any(hint in key for hint in CF_TIMING_METRIC_HINTS):
            # A suffix, not an infix: replacing the first "P" would turn
            # largestContentfulPaintP75 into largestContentfulMsPaintP75,
            # which is not a name anyone could look up.
            out[f'{key}_ms'] = round(value / 1000.0, 1)
        else:
            out[key] = value
    return out


def _vitals_group(row):
    """One result row as a flat dict: sample count plus its scaled quantiles."""
    out = {'samples': row.get('count')}
    out.update(_scale_quantiles(row.get('quantiles')))
    return out


def _cf_vitals_schema_hint(token):
    """Ask the live schema what this dataset actually offers.

    Only runs when the real query failed, so it costs nothing on a healthy day.
    The Account probe is the one that matters: it finds the dataset even if the
    name above is wrong, which is the failure the rest of this cannot detect.
    """
    q = """
    query {
      account: __type(name: "Account") { fields { name } }
      group: __type(name: "AccountRumWebVitalsEventsAdaptiveGroups") { fields { name } }
      quantiles: __type(name: "AccountRumWebVitalsEventsAdaptiveGroupsQuantiles") { fields { name } }
      dimensions: __type(name: "AccountRumWebVitalsEventsAdaptiveGroupsDimensions") { fields { name } }
    }"""
    data, err = _cf_graphql(token, q, timeout=45)
    if err:
        return {'introspection_error': err}

    def names(key):
        node = (data or {}).get(key)
        return [f['name'] for f in (node or {}).get('fields', [])] if node else None

    # This came back empty on the 2026-08-24 run even though the dataset name
    # was right, so its silence means "probe did not resolve", not "no RUM
    # datasets". Reported as None rather than [] so the two cannot be confused
    # -- an empty list here would read as a missing dataset and send the next
    # reader hunting for the wrong thing.
    account_fields = names('account')
    rum = (sorted(f for f in account_fields if 'rum' in f.lower())
           if account_fields else None)
    return {
        'rum_datasets_on_account': rum,
        'group_fields': names('group'),
        'quantile_fields': names('quantiles'),
        'dimension_fields': names('dimensions'),
    }


# How many spots to check per run. The whole list rotates through in about a
# week, which is the same trick the URL-inspection collector uses -- a bounded
# number of upstream calls per run, full coverage over time.
TIDE_SPOTS_PER_RUN = 15

# Words that mean a station is NOT on the open coast. This is a REVIEW TRIGGER,
# never an automatic substitution: the same heuristic was tried for choosing
# stations and rejected, because it reads "Smith Creek, Flagler Beach" as
# ocean-side on the word "Beach". A false positive here costs a human one
# glance; a false positive in the selector costs every user of that spot wrong
# tides. Different tolerance, so a heuristic that is unfit for one job is fine
# for the other.
ENCLOSED_MARKERS = (
    'sound', 'creek', 'ditch', 'slough', 'bay', 'icww', 'intracoastal',
    '(inside)', 'landing', 'bight', 'bayou', 'canal', 'marina', 'harbor',
    'harbour', 'basin', 'lagoon', 'river', 'lake', 'bridge', 'ferry',
)
# Above this, a station is far enough from the spot to be worth a look even if
# nothing else about it is suspicious.
TIDE_DISTANCE_REVIEW_KM = 25.0


def _spot_slugs():
    """Curated spots, from the repo rather than the network."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'surf_cameras.json')) as f:
        cams = json.load(f)
    rows = cams if isinstance(cams, list) else cams.get('cameras', [])
    seen, out = set(), []
    for c in rows:
        lat, lon = c.get('lat'), c.get('lon')
        if lat is None or lon is None:
            continue
        key = (round(lat, 4), round(lon, 4))
        if key in seen:
            continue
        seen.add(key)
        out.append({'name': c.get('name') or '', 'lat': lat, 'lon': lon})
    out.sort(key=lambda s: (s['lat'], s['lon']))
    return out


def collect_tide_stations(state):
    """Which spots are reading a tide station that deserves a second look.

    This exists because the failure is silent by construction. A spot assigned
    a station in the wrong body of water still returns a complete, plausible
    tide curve -- it is simply the wrong water, off by hours and sometimes by
    most of the range. Nothing errors, so nothing else in this audit would
    ever notice. Measured on 2026-08-30: 66 of 140 tidal spots sit where the
    two nearest stations disagree by more than 30 minutes, and several ocean
    breaks were reading a sound, a lagoon or a ditch.

    Only flags for review. Fixing means adding a zone to TIDE_STATION_ZONES in
    app.py, which is a judgement call backed by measured phase and range.
    """
    out = {}
    try:
        spots = _spot_slugs()
        if not spots:
            out['unavailable'] = 'no spots found'
            return out
        cursor = int(state.get('tide_cursor', 0)) % len(spots)
        slice_ = spots[cursor:cursor + TIDE_SPOTS_PER_RUN]
        if len(slice_) < TIDE_SPOTS_PER_RUN:
            slice_ += spots[:TIDE_SPOTS_PER_RUN - len(slice_)]

        checked, review = 0, []
        for spot in slice_:
            try:
                r = requests.get(f'{SITE}/api/tides',
                                 params={'lat': spot['lat'], 'lon': spot['lon']},
                                 timeout=45)
                if not r.ok:
                    continue
                d = r.json()
            except Exception:
                continue
            if d.get('non_tidal'):
                continue
            st = d.get('station') or {}
            name = st.get('name') or ''
            if not name:
                continue
            checked += 1
            hl = d.get('high_low') or []
            highs = [e['height'] for e in hl if e.get('type') == 'H']
            lows = [e['height'] for e in hl if e.get('type') == 'L']
            rng = (round(sum(highs) / len(highs) - sum(lows) / len(lows), 3)
                   if highs and lows else None)
            low = name.lower()
            reasons = []
            hit = next((m for m in ENCLOSED_MARKERS if m in low), None)
            if hit:
                reasons.append(f'station name suggests enclosed water ({hit!r})')
            if (st.get('distance_km') or 0) > TIDE_DISTANCE_REVIEW_KM:
                reasons.append(f"{st['distance_km']} km away")
            if reasons:
                review.append({
                    'spot': spot['name'], 'lat': spot['lat'], 'lon': spot['lon'],
                    'station': st.get('id'), 'station_name': name,
                    'distance_km': st.get('distance_km'),
                    'range_m': rng, 'reasons': reasons,
                })
        # Warn only on findings that have not been seen before. The first
        # sweep surfaces a backlog of dozens; warning about all of them every
        # run would turn the whole audit into wallpaper and this check would
        # be the thing that trained people to skim past warnings. The full
        # list still goes into the snapshot every run -- it is the alerting
        # that is deduplicated, not the record.
        seen = set(state.get('tide_seen') or [])
        fresh = [r for r in review if f"{r['spot']}|{r['station']}" not in seen]
        out['checked'] = checked
        out['review'] = review
        out['new'] = fresh
        out['seen_total'] = len(seen | {f"{r['spot']}|{r['station']}" for r in review})
        out['cursor'] = (cursor + TIDE_SPOTS_PER_RUN) % len(spots)
        out['spots_total'] = len(spots)
    except Exception as e:
        out['error'] = str(e)[:200]
    return out


def collect_web_vitals():
    """Core Web Vitals at P75 from Cloudflare RUM, 7-day and 1-day windows.

    Two windows on purpose. Cloudflare reports a percentile over whatever range
    is asked for, and this site takes roughly 40 pageviews a day, so a 7-day
    P75 is still about 90% pre-change samples the day after a fix ships -- long
    enough that a real improvement looks like no improvement and gets undone.
    The 1-day number moves first and is noisy; the 7-day number is stable and
    late. Recording both is the only way the pair tells you which you are
    looking at.

    Reported per page as well as sitewide, because a single bad route drags a
    sitewide P75 that is fine everywhere else, and the fix is route-specific.
    """
    token = os.environ.get('CF_API_TOKEN')
    account = os.environ.get('CF_ACCOUNT_ID')
    site_tag = os.environ.get('CF_SITE_TAG')
    missing = [n for n, v in (('CF_API_TOKEN', token), ('CF_ACCOUNT_ID', account),
                              ('CF_SITE_TAG', site_tag)) if not v]
    if missing:
        return {'unavailable': f'not set: {", ".join(missing)}'}

    quantile_selection = ' '.join(CF_VITALS_QUANTILE_FIELDS)
    # The whole CLS distribution, sitewide. The first real run returned p75 = 1
    # for four separate elements and -1 for others, and a CLS score cannot be
    # negative -- so whatever this field is, it is not simply the score, and a
    # single percentile cannot tell the difference. A distribution can: values
    # varying smoothly across the percentiles mean it is a score, values pinned
    # to -1/0/1 with interpolated fractions between them mean it is a per-event
    # rating being read as if it were one. Nothing should act on a CLS figure
    # from this API until that is settled.
    cls_spread = ' '.join(f'cumulativeLayoutShiftP{p}'
                          for p in (25, 50, 75, 90, 95, 99, 999))
    gql = """
    query($account: String!, $tag: String!, $d1: Time!, $d7: Time!, $end: Time!) {
      viewer {
        accounts(filter: {accountTag: $account}) {
          sitewide_7d: %(ds)s(
            limit: 1,
            filter: {siteTag: $tag, datetime_geq: $d7, datetime_leq: $end}
          ) { count quantiles { %(q)s } }
          sitewide_1d: %(ds)s(
            limit: 1,
            filter: {siteTag: $tag, datetime_geq: $d1, datetime_leq: $end}
          ) { count quantiles { %(q)s } }
          by_path_7d: %(ds)s(
            limit: 20, orderBy: [count_DESC],
            filter: {siteTag: $tag, datetime_geq: $d7, datetime_leq: $end}
          ) { count quantiles { %(q)s } dimensions { requestPath } }
          cls_spread_7d: %(ds)s(
            limit: 1,
            filter: {siteTag: $tag, datetime_geq: $d7, datetime_leq: $end}
          ) { count quantiles { %(spread)s } }
          by_device_7d: %(ds)s(
            limit: 5, orderBy: [count_DESC],
            filter: {siteTag: $tag, datetime_geq: $d7, datetime_leq: $end}
          ) { count quantiles { %(q)s } dimensions { deviceType } }
          cls_elements_7d: %(ds)s(
            limit: 15, orderBy: [count_DESC],
            filter: {siteTag: $tag, datetime_geq: $d7, datetime_leq: $end}
          ) { count quantiles { cumulativeLayoutShiftP75 }
              dimensions { cumulativeLayoutShiftElement cumulativeLayoutShiftPath } }
        }
      }
    }""" % {'ds': CF_VITALS_DATASET, 'q': quantile_selection,
             'spread': cls_spread}

    end = _utcnow()
    data, err = _cf_graphql(token, gql, {
        'account': account, 'tag': site_tag,
        'd1': (end - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'd7': (end - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'end': end.strftime('%Y-%m-%dT%H:%M:%SZ')})

    if err:
        # The whole point of the hint: a Beta schema that renamed a field must
        # not degrade into a missing section that reads as "no problems".
        return {'error': err, 'schema_hint': _cf_vitals_schema_hint(token)}

    accounts = ((data or {}).get('viewer') or {}).get('accounts') or []
    if not accounts:
        return {'error': 'no account matched CF_ACCOUNT_ID for web vitals'}
    acc = accounts[0]

    def one(key):
        rows = acc.get(key) or []
        return _vitals_group(rows[0]) if rows else None

    def grouped(key, *dims):
        out = []
        for row in (acc.get(key) or []):
            d = row.get('dimensions') or {}
            entry = {dim: d.get(dim) for dim in dims}
            entry.update(_vitals_group(row))
            out.append(entry)
        return out

    result = {
        'percentile': 'p75',
        'sitewide_7d': one('sitewide_7d'),
        'sitewide_1d': one('sitewide_1d'),
        'cls_spread_7d': one('cls_spread_7d'),
        'by_path_7d': grouped('by_path_7d', 'requestPath'),
        'by_device_7d': grouped('by_device_7d', 'deviceType'),
        # The reason this dataset is worth querying at all rather than reading
        # the dashboard: Cloudflare records WHICH element shifted, for real
        # visitors on real devices. Every CLS number in this project until now
        # came from driving one machine on a fast connection and inferring the
        # cause; this is the page telling us directly. Keep it even when the
        # score is healthy -- it is what makes the next regression a lookup
        # instead of an investigation.
        'cls_elements_7d': grouped('cls_elements_7d',
                                   'cumulativeLayoutShiftElement',
                                   'cumulativeLayoutShiftPath'),
    }

    # A sample count is not decoration here. P75 over a handful of pageloads is
    # one visitor's phone, and acting on it is how a fine page gets "fixed".
    samples_7d = (result['sitewide_7d'] or {}).get('samples') or 0
    if samples_7d < 100:
        result['low_sample_warning'] = (
            f'{samples_7d} pageloads in 7d -- p75 is not meaningful yet')
    return result


def collect_cloudflare():
    """Web Analytics pageviews and top paths.

    Cloudflare deliberately does not log query strings, so campaign tagging is
    invisible and paths are the only attribution we get -- which is why /ig
    exists as its own route rather than a utm parameter.
    """
    token = os.environ.get('CF_API_TOKEN')
    account = os.environ.get('CF_ACCOUNT_ID')
    site_tag = os.environ.get('CF_SITE_TAG')
    missing = [n for n, v in (('CF_API_TOKEN', token), ('CF_ACCOUNT_ID', account),
                              ('CF_SITE_TAG', site_tag)) if not v]
    if missing:
        return {'unavailable': f'not set: {", ".join(missing)}'}

    end = _utcnow()
    start = end - timedelta(days=7)
    gql = """
    query($account: String!, $tag: String!, $start: Time!, $end: Time!) {
      viewer {
        accounts(filter: {accountTag: $account}) {
          total: rumPageloadEventsAdaptiveGroups(
            limit: 1,
            filter: {siteTag: $tag, datetime_geq: $start, datetime_leq: $end}
          ) { count }
          pages: rumPageloadEventsAdaptiveGroups(
            limit: 25, orderBy: [count_DESC],
            filter: {siteTag: $tag, datetime_geq: $start, datetime_leq: $end}
          ) { count dimensions { requestPath } }
          referrers: rumPageloadEventsAdaptiveGroups(
            limit: 25, orderBy: [count_DESC],
            filter: {siteTag: $tag, datetime_geq: $start, datetime_leq: $end}
          ) { count dimensions { refererHost } }
        }
      }
    }"""
    try:
        r = requests.post(
            'https://api.cloudflare.com/client/v4/graphql',
            headers={'Authorization': f'Bearer {token}',
                     'Content-Type': 'application/json'},
            json={'query': gql, 'variables': {
                'account': account, 'tag': site_tag,
                'start': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'end': end.strftime('%Y-%m-%dT%H:%M:%SZ')}},
            timeout=60)
        body = r.json()
        if body.get('errors'):
            return {'error': str(body['errors'])[:300]}
        accounts = body['data']['viewer']['accounts'] or []
        if not accounts:
            # An empty accounts list is not "no traffic" -- it means the filter
            # matched nothing, i.e. the account id is wrong or the token cannot
            # see that account. Silently returning zeros here would look like a
            # dead site rather than a misconfiguration.
            return {'error': 'no account matched CF_ACCOUNT_ID (check the id, '
                             'and that the token has Account Analytics: Read '
                             'on that account)'}
        acc = accounts[0]
        if not acc.get('total'):
            # Zero events with a valid account is almost always the wrong site
            # tag, so enumerate the account's actual Web Analytics sites rather
            # than leaving someone to guess. These tags are embedded in the
            # public page's beacon, so listing them reveals nothing private.
            available = _cf_site_tags(token, account)
            return {'window_days': 7, 'pageviews': 0,
                    'note': 'account matched but no RUM events in the window',
                    'configured_site_tag_suffix': (site_tag or '')[-6:],
                    'available_sites': available,
                    'top_paths': [], 'top_referrers': []}
        return {
            'window_days': 7,
            'pageviews': (acc.get('total') or [{}])[0].get('count'),
            'top_paths': [{'path': p['dimensions']['requestPath'],
                           'views': p['count']} for p in acc.get('pages', [])],
            'top_referrers': [{'host': p['dimensions']['refererHost'] or '(none)',
                               'views': p['count']} for p in acc.get('referrers', [])],
        }
    except (requests.RequestException, ValueError, KeyError, IndexError) as e:
        return {'error': str(e)[:300]}


# Endpoints the edge is expected to cache, one dynamic and one static. Both are
# cheap and both are in app.py's API_EDGE_TTL / static handling, so a DYNAMIC on
# either means the CDN is passing everything through.
EDGE_CACHE_PROBES = (
    ('/api/forecast?lat=34.43&lon=-77.55', 'dynamic JSON'),
    ('/static/og-image.png', 'static asset'),
)


def collect_edge_cache():
    """Check the CDN is still caching, because nothing else can.

    Edge caching is a dashboard-only setting -- the Blueprint spec has no field
    for it, so unlike the gunicorn flags it cannot be pinned in render.yaml or
    covered by a test. On 2026-08-25, shortly after a Blueprint sync, the
    dashboard read "None" while the edge was demonstrably still creating cache
    entries, so the stored value and the observed behaviour can disagree and
    neither is self-announcing.

    What it costs to miss: before edge caching, one gunicorn worker served the
    19.6 MB ocean-basin payload to every visitor individually, about 2.5s of
    thread time each, capping the site near three visitors a second. A silent
    reset puts it straight back there, and the symptom is "the site feels slow"
    weeks later rather than anything that looks like a configuration change.

    Two fetches of the same URL. HIT or MISS on the second is fine -- the entry
    either was already warm or has just been created. DYNAMIC is the failure: it
    means the edge considered the response ineligible and went to the origin.
    """
    results = {}
    for path, label in EDGE_CACHE_PROBES:
        statuses = []
        try:
            for _ in range(2):
                r = _get(f'{SITE}{path}')
                statuses.append(r.headers.get('cf-cache-status', '(absent)'))
        except Exception as e:
            results[path] = {'label': label, 'error': f'{type(e).__name__}: {str(e)[:120]}'}
            continue
        results[path] = {
            'label': label,
            'statuses': statuses,
            # A single DYNAMIC anywhere in the pair is enough to report. Waiting
            # for both would hide a partial reset, and a check that only fires on
            # total failure is most of the way to no check at all.
            'cached': all(s != 'DYNAMIC' for s in statuses),
        }
    working = [v for v in results.values() if v.get('cached')]
    return {
        'probes': results,
        'all_cached': len(working) == len(EDGE_CACHE_PROBES),
    }


def build_audit(state):
    return {
        'generated': _utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'ci': collect_ci(),
        'site': collect_site(),
        'search_console': collect_search_console(state),
        'cloudflare': collect_cloudflare(),
        'web_vitals': collect_web_vitals(),
        'edge_cache': collect_edge_cache(),
        'tide_stations': collect_tide_stations(state),
    }


def main():
    ap = argparse.ArgumentParser(description='Collect a CI + SEO audit snapshot')
    ap.add_argument('--data-dir', default='.',
                    help='directory to write audit-latest.json and history into')
    ap.add_argument('--print', action='store_true', help='also print the JSON')
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    state_path = os.path.join(args.data_dir, 'audit-state.json')
    state = {}
    if os.path.exists(state_path):
        try:
            with open(state_path) as f:
                state = json.load(f)
        except (OSError, ValueError):
            state = {}

    audit = build_audit(state)

    cursor = audit.get('search_console', {}).get('inspection_cursor')
    if cursor is not None:
        state['inspect_cursor'] = cursor
    tide = audit.get('tide_stations') or {}
    if tide.get('cursor') is not None:
        state['tide_cursor'] = tide['cursor']
    if tide.get('review') is not None:
        # Remember what has already been reported so a standing backlog does
        # not re-alert daily. Delete tide_seen from audit-state.json to force
        # the whole queue to be re-surfaced.
        seen = set(state.get('tide_seen') or [])
        seen |= {f"{r['spot']}|{r['station']}" for r in tide['review']}
        state['tide_seen'] = sorted(seen)
    state['last_run'] = audit['generated']
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    with open(os.path.join(args.data_dir, 'audit-latest.json'), 'w') as f:
        json.dump(audit, f, indent=2)

    # A trimmed row per run, so trends stay readable without loading every
    # snapshot in full.
    sc = audit.get('search_console', {})
    row = {
        'generated': audit['generated'],
        'ci_failing': audit['ci'].get('failing', []),
        'site_problems': audit['site'].get('problems', []),
        'sitemap_urls': (audit['site'].get('sitemap') or {}).get('urls'),
        'verification': audit['site'].get('verification'),
        'gsc_totals_28d': sc.get('totals_28d'),
        'inspection_summary': sc.get('inspection_summary'),
        'cf_pageviews': (audit.get('cloudflare') or {}).get('pageviews'),
        # Both windows in the row, not just the stable one: the 7-day figure
        # lags a fix by about a week at this traffic level, and a trend line
        # built only from it says a change did nothing for six days.
        'cwv_p75_7d': (audit.get('web_vitals') or {}).get('sitewide_7d'),
        'cwv_p75_1d': (audit.get('web_vitals') or {}).get('sitewide_1d'),
        'edge_cache_ok': (audit.get('edge_cache') or {}).get('all_cached'),
    }
    with open(os.path.join(args.data_dir, 'audit-history.jsonl'), 'a') as f:
        f.write(json.dumps(row) + '\n')

    if args.print:
        print(json.dumps(audit, indent=2))

    problems = audit['ci'].get('failing', []) + audit['site'].get('problems', [])
    print(f"Audit written to {args.data_dir}. "
          f"{len(problems)} problem(s): {problems or 'none'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
