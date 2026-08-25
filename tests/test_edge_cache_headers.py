"""Tests for the API Cache-Control headers that let Cloudflare cache at the edge.

These exist because the failure modes here are silent in opposite directions.
Too little: no header means cf-cache-status DYNAMIC, every visitor reaches the
single gunicorn worker, and the 19.6 MB ocean-basin payload is served
individually to each of them -- which is the state this replaced. Too much: a
header on a 500 during an upstream outage would pin that failure at the edge
for the whole TTL, turning a blip into an outage nobody can clear by retrying.

Neither shows up as a test failure anywhere else, and neither is visible in the
app's own behaviour -- only in a response header nobody reads by accident.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as A  # noqa: E402


@pytest.fixture
def client():
    return A.app.test_client()


def test_every_mapped_ttl_is_below_the_server_side_ttl():
    """The edge copy must expire before the server copy that would refresh it.

    An edge TTL longer than the origin's means a visitor can be served data the
    origin has already replaced, with nothing to trigger a refetch.
    """
    assert A.API_EDGE_TTL['/api/ocean-basin'] < A.BASIN_CACHE_TTL
    assert A.API_EDGE_TTL['/api/forecast'] < A.CACHE_TTL
    assert A.API_EDGE_TTL['/api/beach-orientation'] <= A.ORIENTATION_CACHE_TTL


def test_health_upstreams_is_never_edge_cached():
    """It is the endpoint reached for during an incident.

    A cached answer there is worse than a slow one: it reports the state of the
    world minutes ago while someone is trying to find out what is broken now.
    """
    assert '/api/health-upstreams' not in A.API_EDGE_TTL


def test_a_200_carries_its_configured_max_age(client):
    r = client.get('/api/beach-orientation?lat=34.43&lon=-77.55')
    assert r.status_code == 200
    assert r.headers['Cache-Control'] == 'public, max-age=86400'


def test_an_error_response_is_explicitly_no_store(client):
    """The regression that would hurt most: a cached upstream failure.

    This used to assert header ABSENCE, on the theory the edge would bypass
    an unheaded response. True of Cloudflare, false of Render, which
    default-caches unheaded responses -- the same silence-disagreement the
    test below narrates for 200s applies to errors, and an edge-cached 503
    "warming" answer would keep announcing an outage long after the cache
    had warmed. Errors now say no-store out loud.
    """
    r = client.get('/api/forecast?lat=999&lon=999')
    assert r.status_code == 400
    assert r.headers['Cache-Control'] == 'no-store' 


def test_an_unmapped_api_endpoint_is_explicitly_uncacheable(client):
    """Fail closed -- but the two edges disagree about what silence means.

    Cloudflare's rule bypasses cache when no Cache-Control is present. Render's
    edge does the opposite: a 200 with no directive gets a DEFAULT 120-minute
    TTL. So an omission that meant "do not cache" on one edge means "cache for
    two hours" on the other, and the endpoint it would hurt most is the incident
    diagnostic. Silence is not a safe default anywhere now; the header is
    explicit either way.
    """
    r = client.get('/api/health-upstreams')
    assert r.headers.get('Cache-Control') == 'no-store'


def test_healthz_is_cheap_and_uncacheable(client):
    """The platform health check must not depend on anyone else's uptime.

    A probe that touches an upstream hands that upstream a restart button: NOAA
    has a bad afternoon, the probe fails, and the platform recycles a process
    that was serving cached forecasts perfectly well.
    """
    r = client.get('/healthz')
    assert r.status_code == 200
    assert r.get_json()['status'] == 'ok'
    assert r.headers['Cache-Control'] == 'no-store'


def test_healthz_is_not_the_upstream_diagnostic():
    """Guards against someone later pointing healthCheckPath at the wrong one."""
    assert '/healthz' not in A.API_EDGE_TTL
    assert '/api/health-upstreams' not in A.API_EDGE_TTL


def test_html_caching_is_unchanged(client):
    r = client.get('/about')
    assert r.headers['Cache-Control'] == 'public, max-age=300'


def test_static_caching_is_unchanged(client):
    r = client.get('/static/manifest.json')
    if r.status_code == 200:
        assert r.headers['Cache-Control'] == 'public, max-age=86400'


def test_no_mapped_path_is_a_prefix_match():
    """The lookup is an exact path match, not a prefix.

    /api/social-card/<slug> and /api/social-card/accuracy are not in the map and
    must not pick up a TTL from a shorter entry that happens to prefix them.
    """
    for path in A.API_EDGE_TTL:
        assert not path.endswith('/')
        others = [p for p in A.API_EDGE_TTL if p != path]
        assert not any(o.startswith(path + '/') for o in others)
