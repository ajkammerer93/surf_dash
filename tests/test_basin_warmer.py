"""The background basin warmer: the only writer of basin:global.

Design constraints pinned here, each learned the hard way on 2026-08-25:
the fetch runs for minutes so no lock may be held across it; the store must
go through _cache_store with _deep_size so the byte accounting stays honest
(a byte bound whose total has drifted is worse than no bound); and the
warmer must be OFF whenever SSR_WARM is off, because the pinned CI command
only sets SSR_WARM=0 and a separately-gated warmer would launch a live
PacIOOS fetch inside the test suite.
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as surf_app  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_cache():
    with surf_app._cache_lock:
        surf_app._cache.clear()
        surf_app._cache_bytes = 0
    yield
    with surf_app._cache_lock:
        surf_app._cache.clear()
        surf_app._cache_bytes = 0


def _run_one_warmer_pass(monkeypatch, producer, sleeps):
    """Execute exactly one loop body of _basin_cache_warmer."""
    monkeypatch.setattr(surf_app, 'get_ocean_basin_data', producer)
    monkeypatch.setattr(surf_app.time, 'sleep',
                        lambda s: sleeps.append(s) or (_ for _ in ()).throw(StopIteration))
    with pytest.raises(StopIteration):
        surf_app._basin_cache_warmer()


def test_warmer_stores_through_cache_store_with_accounting(monkeypatch):
    sleeps = []
    payload = {'lats': [0.0], 'lons': [0.0], 'times': ['t'],
               'wave_height': [[[1.0]]]}
    calls = {}

    def producer(wave_timeouts=None, wind_timeouts=None):
        calls['wave'] = wave_timeouts
        calls['wind'] = wind_timeouts
        return payload

    # skip the initial boot sleep: make the first sleep a no-op, stop on 2nd
    seen = []

    def fake_sleep(s):
        seen.append(s)
        if len(seen) >= 2:
            raise StopIteration
    monkeypatch.setattr(surf_app, 'get_ocean_basin_data', producer)
    monkeypatch.setattr(surf_app.time, 'sleep', fake_sleep)
    with pytest.raises(StopIteration):
        surf_app._basin_cache_warmer()

    entry = surf_app._cache.get('basin:global')
    assert entry is not None and entry['data'] is payload
    assert entry['bytes'] == surf_app._deep_size(payload)
    assert surf_app._cache_bytes == entry['bytes'], "byte accounting drifted"
    # the warmer's defining property: budgets no request could be allowed
    assert calls['wave'] == surf_app.BASIN_WARM_WAVE_TIMEOUTS
    assert calls['wind'] == surf_app.BASIN_WARM_WIND_TIMEOUTS


def test_warmer_skips_when_fresh_and_backs_off_after_failure(monkeypatch):
    # fresh entry -> the producer must not run
    size = surf_app._deep_size({'x': 1})
    with surf_app._cache_lock:
        surf_app._cache_store('basin:global', {'x': 1}, size)

    def forbidden(**kw):
        raise AssertionError("fetched while fresh")
    seen = []

    def fake_sleep(s):
        seen.append(s)
        if len(seen) >= 2:
            raise StopIteration
    monkeypatch.setattr(surf_app, 'get_ocean_basin_data', forbidden)
    monkeypatch.setattr(surf_app.time, 'sleep', fake_sleep)
    with pytest.raises(StopIteration):
        surf_app._basin_cache_warmer()
    assert seen[-1] == surf_app.BASIN_WARM_CHECK_S

    # cold cache + failing producer -> the longer retry sleep
    with surf_app._cache_lock:
        surf_app._cache.clear()
        surf_app._cache_bytes = 0
    seen.clear()
    monkeypatch.setattr(surf_app, 'get_ocean_basin_data', lambda **kw: None)
    with pytest.raises(StopIteration):
        surf_app._basin_cache_warmer()
    assert seen[-1] == surf_app.BASIN_WARM_RETRY_S


def test_warmer_never_holds_cache_lock_during_fetch(monkeypatch):
    """The fetch takes minutes against a degraded upstream. Holding
    _cache_lock (or the basin key lock) across it would freeze every cache
    read on the site for the duration -- the warmer exists precisely to keep
    that cost OFF the serving path."""
    observed = {}

    def producer(**kw):
        observed['cache_lock_held'] = surf_app._cache_lock.locked()
        key_lock = surf_app._cache_key_locks.get('basin:global')
        observed['key_lock_held'] = key_lock.locked() if key_lock else False
        return {'ok': 1}
    seen = []

    def fake_sleep(s):
        seen.append(s)
        if len(seen) >= 2:
            raise StopIteration
    monkeypatch.setattr(surf_app, 'get_ocean_basin_data', producer)
    monkeypatch.setattr(surf_app.time, 'sleep', fake_sleep)
    with pytest.raises(StopIteration):
        surf_app._basin_cache_warmer()
    assert observed == {'cache_lock_held': False, 'key_lock_held': False}


def test_basin_warm_default_inherits_ssr_warm():
    """SSR_WARM=0 (the pinned CI command) must imply the basin warmer is off.
    Source-level pin: the default chains through os.environ.get('SSR_WARM')."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'app.py')).read()
    assert "'BASIN_WARM', os.environ.get('SSR_WARM'" in src


def test_warmer_fetch_is_breaker_exempt(monkeypatch):
    """A warmer locked out by the breaker could never close it: request-path
    probes fail against a server that needs 130s, so only the generous
    background budget can record the success that resets the host."""
    surf_app._erddap_breakers.clear()
    observed = {}

    def producer(**kw):
        observed['exempt'] = getattr(surf_app._BREAKER_EXEMPT, 'active', False)
        return {'ok': 1}
    seen = []

    def fake_sleep(s):
        seen.append(s)
        if len(seen) >= 2:
            raise StopIteration
    monkeypatch.setattr(surf_app, 'get_ocean_basin_data', producer)
    monkeypatch.setattr(surf_app.time, 'sleep', fake_sleep)
    with pytest.raises(StopIteration):
        surf_app._basin_cache_warmer()
    assert observed['exempt'] is True
