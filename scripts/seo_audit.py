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


def build_audit(state):
    return {
        'generated': _utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'ci': collect_ci(),
        'site': collect_site(),
        'search_console': collect_search_console(state),
        'cloudflare': collect_cloudflare(),
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
