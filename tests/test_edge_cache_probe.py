"""Tests for the SEO audit's edge-caching probe.

This probe is the only thing watching a setting that cannot be watched any other
way. Edge caching has no field in Render's Blueprint spec, so unlike the gunicorn
flags it cannot live in render.yaml and cannot be pinned by test_deploy_config.
On 2026-08-25 the dashboard read "None" shortly after a Blueprint sync while the
edge was still demonstrably creating cache entries -- the stored value and the
behaviour disagreed, and neither announced itself.

The cost of missing a reset is not subtle: without the edge, one gunicorn worker
serves the ocean-basin payload to every visitor individually, which capped the
site near three visitors a second. But the symptom is "the site feels slow"
discovered weeks later, which is why this has to be a warning and not a number
someone might skim.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'scripts'))

import seo_audit  # noqa: E402


class FakeResponse:
    def __init__(self, status):
        self.headers = {'cf-cache-status': status} if status else {}


def _stub(monkeypatch, sequence):
    """Serve cf-cache-status values in order across successive _get calls."""
    calls = {'n': 0}

    def fake_get(url, **kw):
        i = calls['n']
        calls['n'] += 1
        return FakeResponse(sequence[i % len(sequence)])

    monkeypatch.setattr(seo_audit, '_get', fake_get)
    return calls


def test_hit_after_miss_counts_as_working(monkeypatch):
    _stub(monkeypatch, ['MISS', 'HIT'])
    out = seo_audit.collect_edge_cache()
    assert out['all_cached'] is True


def test_already_warm_counts_as_working(monkeypatch):
    """A HIT on the first fetch is a pass, not an inconclusive result.

    The probe does not cache-bust on purpose: a real visitor may have warmed the
    entry, and busting would create a fresh cache entry on every audit run for no
    extra signal.
    """
    _stub(monkeypatch, ['HIT', 'HIT'])
    assert seo_audit.collect_edge_cache()['all_cached'] is True


def test_dynamic_is_the_failure_signal(monkeypatch):
    """DYNAMIC means the edge declared the response ineligible and hit origin.

    That is exactly the state the site was in before edge caching was enabled.
    """
    _stub(monkeypatch, ['DYNAMIC', 'DYNAMIC'])
    out = seo_audit.collect_edge_cache()
    assert out['all_cached'] is False
    assert all(not p['cached'] for p in out['probes'].values())


def test_a_single_dynamic_in_the_pair_still_fails(monkeypatch):
    """Waiting for both to fail would hide a partial reset.

    A check that only fires on total failure is most of the way to no check.
    """
    _stub(monkeypatch, ['HIT', 'DYNAMIC'])
    assert seo_audit.collect_edge_cache()['all_cached'] is False


def test_both_a_dynamic_and_a_static_probe_are_checked(monkeypatch):
    """One of each, because the two are governed by different settings.

    "Common static files" would cache the PNG and pass a static-only probe while
    leaving every API response uncached -- which is the configuration that looks
    fine and fixes nothing.
    """
    labels = {label for _, label in seo_audit.EDGE_CACHE_PROBES}
    assert labels == {'dynamic JSON', 'static asset'}
    _stub(monkeypatch, ['HIT', 'HIT'])
    assert len(seo_audit.collect_edge_cache()['probes']) == 2


def test_a_network_failure_is_recorded_not_swallowed(monkeypatch):
    def boom(url, **kw):
        raise ConnectionError('nope')
    monkeypatch.setattr(seo_audit, '_get', boom)
    out = seo_audit.collect_edge_cache()
    assert out['all_cached'] is False
    assert all('error' in p for p in out['probes'].values())


def test_a_missing_header_is_not_treated_as_success(monkeypatch):
    """An absent cf-cache-status is not a HIT. It is unknown, and unknown here
    should not read as healthy -- but it is also not DYNAMIC, so it passes with
    the raw value recorded for whoever reads the snapshot."""
    _stub(monkeypatch, [None, None])
    out = seo_audit.collect_edge_cache()
    statuses = list(out['probes'].values())[0]['statuses']
    assert statuses == ['(absent)', '(absent)']


def test_expired_is_a_working_cache_not_a_failure(monkeypatch):
    """Seen in the first live run against production.

    EXPIRED means the edge held an entry, found it stale and revalidated it --
    the cache is working exactly as intended. Only DYNAMIC says the edge declined
    to cache at all. Treating EXPIRED as a failure would make this probe cry wolf
    every time a TTL rolled over, and a check that fires on healthy days gets
    ignored on the day it matters.
    """
    _stub(monkeypatch, ['EXPIRED', 'HIT'])
    assert seo_audit.collect_edge_cache()['all_cached'] is True


def test_bypass_is_also_not_a_failure(monkeypatch):
    """BYPASS is a deliberate no-store, which is what /api/health-upstreams asks
    for. The probes never target such an endpoint, but the distinction should be
    explicit rather than accidental."""
    _stub(monkeypatch, ['BYPASS', 'BYPASS'])
    assert seo_audit.collect_edge_cache()['all_cached'] is True
