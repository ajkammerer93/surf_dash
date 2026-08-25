"""The per-host circuit breaker and the wall-clock deadline on ERDDAP fetches.

Both exist because of 2026-08-25, when the site went down without a single
component "failing": PacIOOS drip-fed responses at ~120 KB/s, requests' read
timeout (which is between-bytes, not end-to-end) never tripped, the nominal
70s budget ran 177s, and eight such requests pinned every gunicorn thread --
502 for everything, /healthz included.

The breaker protects THREAD TIME, not data availability. A host that answers
quickly with a 404 or a 500 costs nothing and must not open the breaker (the
mirror chain's walk-the-clock-back behaviour depends on fast 404s). Only the
outcomes that pin a thread count: timeouts, connection failures, and redirect
shims.
"""
import os
import sys
import threading
import time

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402

HOST = 'breaker-test.example'


@pytest.fixture(autouse=True)
def _clean_breakers():
    """Breaker state is module-level and would otherwise leak between tests
    -- the same trap the Open-Meteo cooldown fixture in test_failures.py
    guards against."""
    app._erddap_breakers.clear()
    yield
    app._erddap_breakers.clear()


def _fail(n=1, host=HOST):
    for _ in range(n):
        app._breaker_record(host, ok=False)


def test_three_consecutive_timeouts_open_the_breaker():
    _fail(2)
    app._breaker_preflight(HOST)          # still closed after two
    _fail(1)
    with pytest.raises(app.ErddapCircuitOpen):
        app._breaker_preflight(HOST)


def test_a_completed_http_response_resets_the_count():
    """404s and 500s are the host ANSWERING. Two timeouts + a 404 + two more
    timeouts must not open -- otherwise the chain's 404 clock-walk would
    charge the host for having not yet published an hour."""
    _fail(2)
    app._breaker_record(HOST, ok=True)    # a 404 records as ok
    _fail(2)
    app._breaker_preflight(HOST)          # still closed


def test_open_breaker_fails_without_a_network_call(monkeypatch):
    _fail(3)

    def no_network(*a, **kw):
        raise AssertionError("breaker open -- the socket must never open")
    monkeypatch.setattr(app.requests, 'get', no_network)
    started = time.monotonic()
    with pytest.raises(app.ErddapCircuitOpen):
        app._fetch_erddap_grid(HOST, 'DS', ['Thgt'], '(x):(y)', '(0):(1)', '(0):(1)')
    assert time.monotonic() - started < 0.05


def test_half_open_admits_exactly_one_probe():
    _fail(3)
    # age the breaker past its cooldown
    app._erddap_breakers[HOST]['opened_at'] = (
        time.monotonic() - app.ERDDAP_BREAKER_COOLDOWN_S - 1)
    assert app._breaker_preflight(HOST) is True      # this call is the probe
    with pytest.raises(app.ErddapCircuitOpen):
        app._breaker_preflight(HOST)                 # concurrent call: still fast-fail


def test_probe_success_closes_and_probe_failure_reopens():
    _fail(3)
    app._erddap_breakers[HOST]['opened_at'] = (
        time.monotonic() - app.ERDDAP_BREAKER_COOLDOWN_S - 1)
    probe = app._breaker_preflight(HOST)
    app._breaker_record(HOST, ok=False, probe=probe)
    with pytest.raises(app.ErddapCircuitOpen):       # re-opened, fresh cooldown
        app._breaker_preflight(HOST)

    app._erddap_breakers[HOST]['opened_at'] = (
        time.monotonic() - app.ERDDAP_BREAKER_COOLDOWN_S - 1)
    probe = app._breaker_preflight(HOST)
    app._breaker_record(HOST, ok=True, probe=probe)
    app._breaker_preflight(HOST)                     # closed again


def test_background_exemption_bypasses_preflight_but_still_records(monkeypatch):
    """The warmer's generous budget is the only call shape that can succeed
    against a server needing 130s+. If the exemption did not exist,
    short-budget request probes would claim the half-open slot, fail, and
    re-open the breaker forever -- the warmer must both get through and be
    the thing whose success closes the breaker for everyone."""
    _fail(3)

    class _Resp:
        status_code = 200
        headers = {}

        @staticmethod
        def iter_content(chunk_size):
            return iter([b'{"table": {"ok": true}}'])

        @staticmethod
        def close():
            return None

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {'table': {'ok': True}}

    monkeypatch.setattr(app.requests, 'get',
                        lambda *a, **kw: _Resp())
    with app._breaker_exempt():
        result = app._fetch_erddap_grid(HOST, 'DS', ['Thgt'],
                                        '(x):(y)', '(0):(1)', '(0):(1)')
    assert result == {'table': {'ok': True}}
    app._breaker_preflight(HOST)          # success recorded: breaker closed


def test_deadline_bounds_wall_clock_not_read_gaps(monkeypatch):
    """The keystone: a server that keeps the socket alive and drips bytes
    forever never trips requests' between-bytes read timeout. The deadline
    wrapper must cut it off on WALL CLOCK. Without this, every budget in the
    mirror chain is decorative -- measured 177s against a '70s budget' on
    2026-08-25."""
    class _DripResp:
        status_code = 200
        headers = {}

        @staticmethod
        def iter_content(chunk_size):
            while True:                    # drips forever, never times out
                yield b'x'

        @staticmethod
        def close():
            return None

    monkeypatch.setattr(app.requests, 'get', lambda *a, **kw: _DripResp())
    started = time.monotonic()
    with pytest.raises(requests.Timeout):
        app._get_json_with_deadline('https://drip.example/x', deadline_s=0.2)
    assert time.monotonic() - started < 2.0


def test_snapshot_names_states():
    _fail(3)
    app._breaker_record('healthy.example', ok=True)
    snap = app._breaker_snapshot()
    assert snap[HOST]['state'] == 'open'
    assert snap[HOST]['retry_in_s'] > 0
    assert snap['healthy.example']['state'] == 'closed'


class TestHealthUpstreamsHonesty:
    """The probe has been deceived twice; these pin the synthesis.

    Deception one (2026-08-24): an expensive time[(last)] query read healthy
    hosts as dead. Deception two (2026-08-25): metadata probes read a
    redirect shim as healthy while every data query it received was being
    302'd to a degraded server. The probe now asks for one cell by index on
    the DATA path, never follows redirects, and names the redirect target.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        app._erddap_breakers.clear()
        with app._cache_lock:
            app._cache.pop('health-upstreams', None)
        yield
        app._erddap_breakers.clear()
        with app._cache_lock:
            app._cache.pop('health-upstreams', None)

    def test_redirecting_probe_reports_target_not_fake_200(self, monkeypatch):
        class _R:
            status_code = 302
            headers = {'Location': 'https://pae-paha.pacioos.hawaii.edu/erddap/x'}
        monkeypatch.setattr(app.requests, 'get', lambda *a, **kw: _R())
        client = app.app.test_client()
        body = client.get('/api/health-upstreams').get_json()
        probe = body['erddap-upwell-ww3']
        assert probe['status'] == 302
        assert probe['redirects_to'] == 'pae-paha.pacioos.hawaii.edu'

    def test_probes_never_follow_redirects(self, monkeypatch):
        seen = []

        class _R:
            status_code = 200
            headers = {}

        def fake(url, *a, **kw):
            seen.append(kw.get('allow_redirects', True))
            return _R()
        monkeypatch.setattr(app.requests, 'get', fake)
        app.app.test_client().get('/api/health-upstreams')
        assert seen and all(v is False for v in seen)

    def test_breaker_state_is_live_not_probe_cached(self, monkeypatch):
        """A breaker that opened 10 seconds ago must not hide behind the 60s
        probe cache for the rest of the minute."""
        class _R:
            status_code = 200
            headers = {}
        monkeypatch.setattr(app.requests, 'get', lambda *a, **kw: _R())
        client = app.app.test_client()
        first = client.get('/api/health-upstreams').get_json()
        assert first['breakers'] == {}

        for _ in range(3):
            app._breaker_record('pae-paha.pacioos.hawaii.edu', ok=False)
        second = client.get('/api/health-upstreams').get_json()
        assert second['breakers']['pae-paha.pacioos.hawaii.edu']['state'] == 'open'
        # and the cached probe dict itself was never polluted
        entry = app._cache.get('health-upstreams')
        assert 'breakers' not in entry['data']
