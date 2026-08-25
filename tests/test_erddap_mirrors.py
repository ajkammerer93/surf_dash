"""Tests for the ERDDAP mirror chain in app.py.

These exist because the failure this code exists to survive is one that cannot
be reproduced on demand: a mirror is only degraded during an incident, and by
the time anyone looks it has usually recovered. That is exactly what happened
on 2026-08-24 -- PacIOOS took 20s to serve a static page and never finished a
global wave query, the swell map returned 500 for hours, and by the time the
fix was verified end to end the mirror was answering in a second again. So the
fallback was proved against a healthy upstream, which proves the happy path and
nothing else.

The interesting behaviour is what happens when a mirror is bad, and the two bad
cases need OPPOSITE responses: a 404 means this host is fine but has not
published the hour yet, so walk the clock back on the SAME host; a timeout
means the host is no good, so stop walking and go to the next one. Getting that
backwards spends the entire request budget proving one dead box is dead three
times over, which is what a per-host retry loop does by default.
"""
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402


def _http_error(status):
    resp = requests.Response()
    resp.status_code = status
    return requests.exceptions.HTTPError(response=resp)


@pytest.fixture
def calls(monkeypatch):
    """Record every _fetch_erddap_grid call and script its outcomes."""
    log = []
    outcomes = {}

    def fake(server, dataset, variables, time_range, lat_range, lon_range,
             depth=None, timeout=30):
        log.append({'server': server, 'dataset': dataset,
                    'time_range': time_range, 'timeout': timeout})
        outcome = outcomes.get(server, {'ok': True})
        if outcome.get('ok'):
            return {'table': {'served_by': server}}
        exc = outcome['raise']
        raise exc() if isinstance(exc, type) else exc

    monkeypatch.setattr(app, '_fetch_erddap_grid', fake)
    return log, outcomes


def _run(**kw):
    params = dict(
        mirrors=[('first.example', 'DS_ONE'), ('second.example', 'DS_TWO')],
        timeouts=(11, 22),
        variables=['Thgt'],
        lat_range='(0):(1)',
        lon_range='(0):(1)',
        depth=0,
        label='test',
    )
    params.update(kw)
    return app._fetch_erddap_grid_chain(**params)


def test_first_healthy_mirror_wins_and_second_is_never_touched(calls):
    log, _ = calls
    result = _run()
    assert result['table']['served_by'] == 'first.example'
    assert [c['server'] for c in log] == ['first.example']


def test_each_mirror_gets_its_own_timeout_budget(calls):
    log, outcomes = calls
    outcomes['first.example'] = {'ok': False, 'raise': requests.Timeout}
    _run()
    assert [c['timeout'] for c in log] == [11, 22]


def test_timeout_moves_to_the_next_mirror_without_retrying_the_dead_one(calls):
    log, outcomes = calls
    outcomes['first.example'] = {'ok': False, 'raise': requests.Timeout}
    result = _run()
    assert result['table']['served_by'] == 'second.example'
    # One attempt against the dead host, not one per hours_back option.
    assert [c['server'] for c in log] == ['first.example', 'second.example']


def test_connection_error_behaves_like_a_timeout(calls):
    log, outcomes = calls
    outcomes['first.example'] = {'ok': False, 'raise': requests.ConnectionError}
    assert _run()['table']['served_by'] == 'second.example'
    assert [c['server'] for c in log] == ['first.example', 'second.example']


def test_404_walks_the_clock_back_on_the_same_mirror(calls):
    log, outcomes = calls
    outcomes['first.example'] = {'ok': False, 'raise': _http_error(404)}
    result = _run()
    # Three start times against the host that 404s, then the next mirror.
    served = [c['server'] for c in log]
    assert served == ['first.example'] * 3 + ['second.example']
    assert len({c['time_range'] for c in log[:3]}) == 3
    assert result['table']['served_by'] == 'second.example'


def test_non_404_http_error_abandons_the_mirror_immediately(calls):
    log, outcomes = calls
    outcomes['first.example'] = {'ok': False, 'raise': _http_error(500)}
    _run()
    assert [c['server'] for c in log] == ['first.example', 'second.example']


def test_every_mirror_failing_raises_rather_than_returning_none(calls):
    _, outcomes = calls
    outcomes['first.example'] = {'ok': False, 'raise': requests.Timeout}
    outcomes['second.example'] = {'ok': False, 'raise': requests.Timeout}
    # A None here would be re-interpreted downstream as "no data for this
    # region" instead of "every upstream is down", which is how an outage gets
    # rendered as an empty ocean.
    with pytest.raises(requests.Timeout):
        _run()


def test_time_stride_is_applied_when_given(calls):
    log, _ = calls
    _run(time_stride=3)
    assert ':3:(last)' in log[0]['time_range']
    log.clear()
    _run()
    assert ':3:' not in log[0]['time_range']
    assert log[0]['time_range'].endswith(':(last)')


def test_dataset_travels_with_its_host(calls):
    log, outcomes = calls
    outcomes['first.example'] = {'ok': False, 'raise': requests.Timeout}
    _run()
    # The mirrors host the same model under different dataset ids, so pairing
    # them is not cosmetic -- a host with the other's dataset id is a 404.
    assert [(c['server'], c['dataset']) for c in log] == [
        ('first.example', 'DS_ONE'), ('second.example', 'DS_TWO')]


def test_shipped_mirror_lists_are_truthful_about_the_consolidation():
    """The lists must contain only hosts that answer data queries THEMSELVES.

    This test used to assert the opposite -- at least two mirrors per list --
    and that requirement became actively harmful on 2026-08-25, when NOAA
    consolidated the West Coast ERDDAP family: upwell and coastwatch.pfeg
    began 302-redirecting every griddap data query to pae-paha. A second
    entry that redirects to the first is worse than no second entry: it
    doubles the worst case (two budgets spent proving one sick server sick)
    and it lies to the per-host breaker, which counts the shim and its
    target as independent hosts. Redundancy that upstream has quietly
    removed must be removed from the list too, not simulated.
    """
    retired_shims = {'upwell.pfeg.noaa.gov', 'coastwatch.pfeg.noaa.gov'}
    for mirrors in (app.WW3_WAVE_MIRRORS, app.GFS_WIND_MIRRORS):
        assert len(mirrors) >= 1
        hosts = [host for host, _ in mirrors]
        assert len(set(hosts)) == len(hosts)
        assert not retired_shims.intersection(hosts), (
            "a retired redirect shim is back in a mirror list")


def test_redirect_is_raised_not_followed(monkeypatch):
    """A 302 on a data query means the host is a shim for another server.

    requests follows redirects silently by default, which is exactly how the
    consolidation hid: every query 'to upwell' was really a query to the
    degraded PacIOOS, billed against upwell's budget. The fetch layer must
    surface the redirect as a failure of THIS host, carrying the target so
    the log names the server actually being pointed at.
    """
    class _Resp:
        status_code = 302
        headers = {'Location': 'https://pae-paha.pacioos.hawaii.edu/erddap/x'}

        @staticmethod
        def close():
            return None

    followed = {'n': 0}

    def fake_get(url, timeout=None, allow_redirects=True, stream=False):
        assert allow_redirects is False, "data queries must never auto-follow"
        followed['n'] += 1
        return _Resp()

    monkeypatch.setattr(app.requests, 'get', fake_get)
    with pytest.raises(app.ErddapRedirect) as exc:
        app._fetch_erddap_grid('shim.example', 'DS', ['Thgt'],
                               '(x):(y)', '(0):(1)', '(0):(1)')
    assert 'pae-paha' in str(exc.value)
    assert followed['n'] == 1


def test_redirect_moves_the_chain_without_walking_the_clock(calls):
    """A redirect is a host-level failure, like a timeout: one attempt, no
    hours_back walk. Walking the clock against a shim would re-issue the
    same redirected request three times for nothing."""
    log, outcomes = calls
    outcomes['first.example'] = {
        'ok': False, 'raise': app.ErddapRedirect('first.example', 'https://x')}
    result = _run()
    assert result['table']['served_by'] == 'second.example'
    assert [c['server'] for c in log] == ['first.example', 'second.example']


def test_no_erddap_call_site_is_single_homed():
    """The defect this whole change fixes, pinned so it cannot come back.

    On 2026-08-24 the point forecast survived a degraded PacIOOS and the swell
    map did not, purely because only one of them looped over mirrors.
    """
    source = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'app.py')).read()
    assert 'server="pae-paha.pacioos.hawaii.edu"' not in source
    assert 'server="coastwatch.pfeg.noaa.gov"' not in source
    assert 'server="upwell.pfeg.noaa.gov"' not in source
