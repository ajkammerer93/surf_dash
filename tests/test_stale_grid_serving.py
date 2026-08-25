"""Serve-stale and warming behaviour for the two grid endpoints.

Until 2026-08-25 both endpoints had exactly two states: fresh data or a 500.
When every ERDDAP mirror collapsed onto one degraded server, that meant a
blank swell map AND -- because the fetches ran on request threads -- a dead
site. The rules now:

  /api/ocean-basin NEVER fetches upstream on a request thread. The background
  warmer is the only writer of basin:global; requests serve fresh, then stale
  up to a day (flagged), then an honest fast 503 {"warming": true}.

  /api/map-forecast still fetches inline (the breaker bounds the failure to
  milliseconds) but falls back to a flagged stale entry before 500ing.
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as surf_app  # noqa: E402

BASIN_FIXTURE = {
    'lats': [0.0, 3.0], 'lons': [0.0, 3.0],
    'times': ['2026-08-25T00:00Z'],
    'wave_height': [[[1.0, 1.1], [1.2, 1.3]]],
    'wave_period': [[[8.0, 8.1], [8.2, 8.3]]],
    'wave_direction': [[[90, 91], [92, 93]]],
    'wind_speed': [[[10.0, 10.1], [10.2, 10.3]]],
    'wind_direction': [[[180, 181], [182, 183]]],
}


@pytest.fixture(autouse=True)
def _clean_cache():
    with surf_app._cache_lock:
        surf_app._cache.clear()
        surf_app._cache_bytes = 0
    # A warm wave store would let the endpoint answer 200 from RAM and mask
    # every cold-path assertion below.
    surf_app._wave_basin = None
    yield
    with surf_app._cache_lock:
        surf_app._cache.clear()
        surf_app._cache_bytes = 0
    surf_app._wave_basin = None


def _seed(key, data, age_s):
    size = surf_app._deep_size(data)
    with surf_app._cache_lock:
        surf_app._cache_store(key, data, size)
        surf_app._cache[key]['time'] = time.time() - age_s


@pytest.fixture
def client():
    return surf_app.app.test_client()


def test_basin_request_path_never_fetches_inline(client, monkeypatch):
    """The invariant the whole outage taught: no request thread pays for the
    global fetch. If this producer runs during a request, eight visitors can
    take the site down again."""
    def forbidden(*a, **kw):
        raise AssertionError("get_ocean_basin_data ran on a request thread")
    monkeypatch.setattr(surf_app, 'get_ocean_basin_data', forbidden)

    # cold cache: an honest warming answer, instantly, not a fetch
    resp = client.get('/api/ocean-basin?lat=34.43&lon=-77.55')
    assert resp.status_code == 503
    body = resp.get_json()
    assert body['warming'] is True
    assert 'error' in body, "old edge-cached JS keys off .error"
    assert resp.headers['Retry-After'] == '60'

    # fresh cache: served, producer still never called
    _seed('basin:global', dict(BASIN_FIXTURE), age_s=0)
    resp = client.get('/api/ocean-basin?lat=34.43&lon=-77.55')
    assert resp.status_code == 200
    assert 'stale' not in resp.get_json()


def test_basin_serves_stale_with_flag_and_wind_filter(client, monkeypatch):
    monkeypatch.setattr(surf_app, 'get_ocean_basin_data',
                        lambda *a, **kw: None)
    _seed('basin:global', dict(BASIN_FIXTURE),
          age_s=surf_app.BASIN_CACHE_TTL + 600)     # expired but retained

    resp = client.get('/api/ocean-basin?lat=34.43&lon=-77.55')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['stale'] is True
    # ISO UTC with a trailing Z, parseable by Date.parse on the client
    assert body['stale_at'].endswith('Z') and 'T' in body['stale_at']
    # the wind opt-in filter must apply to stale data exactly as to fresh --
    # zero-filled wind would permanently disable the client's lazy fetch
    assert 'wind_speed' not in body
    with_wind = client.get('/api/ocean-basin?lat=34.43&lon=-77.55&wind=1').get_json()
    assert 'wind_speed' in with_wind


def test_basin_stale_older_than_a_day_is_warming_not_served(client):
    _seed('basin:global', dict(BASIN_FIXTURE),
          age_s=surf_app.BASIN_STALE_MAX_AGE + 60)
    resp = client.get('/api/ocean-basin?lat=34.43&lon=-77.55')
    assert resp.status_code == 503


def test_warming_503_is_no_store(client):
    """Render's edge default-caches header-less responses. An edge-cached
    warming answer would keep announcing the outage long after the cache
    warmed."""
    resp = client.get('/api/ocean-basin?lat=34.43&lon=-77.55')
    assert resp.status_code == 503
    assert resp.headers['Cache-Control'] == 'no-store'


def test_map_forecast_serves_stale_on_fetch_failure(client, monkeypatch):
    monkeypatch.setattr(surf_app, 'get_grid_weather_data',
                        lambda *a, **kw: None)
    _seed('map:34.43,-77.55', dict(BASIN_FIXTURE),
          age_s=surf_app.CACHE_TTL + 600)
    resp = client.get('/api/map-forecast?lat=34.43&lon=-77.55')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['stale'] is True and body['stale_at'].endswith('Z')


def test_map_forecast_without_stale_still_500s(client, monkeypatch):
    monkeypatch.setattr(surf_app, 'get_grid_weather_data',
                        lambda *a, **kw: None)
    resp = client.get('/api/map-forecast?lat=34.43&lon=-77.55')
    assert resp.status_code == 500
    assert resp.headers['Cache-Control'] == 'no-store'
