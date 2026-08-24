"""Tests for the Cloudflare Web Vitals collector in scripts/seo_audit.py.

The dataset is Beta and its schema is not published anywhere that can be
checked without an account token, so the field names in the collector are an
informed guess. That is survivable only because a wrong guess has to be LOUD:
the failure this guards against is a renamed field producing a missing section,
which reads downstream as a fast site rather than as a broken collector. These
tests pin that behaviour, not the guess.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'scripts'))

import seo_audit  # noqa: E402


@pytest.fixture
def cf_env(monkeypatch):
    monkeypatch.setenv('CF_API_TOKEN', 'tok')
    monkeypatch.setenv('CF_ACCOUNT_ID', 'acct')
    monkeypatch.setenv('CF_SITE_TAG', 'tag')


def _stub_graphql(monkeypatch, *responses):
    """Serve queued (data, error) pairs in order; record the queries sent."""
    sent = []
    queue = list(responses)

    def fake(token, query, variables=None, timeout=60):
        sent.append(query)
        return queue.pop(0) if queue else (None, 'unexpected extra call')

    monkeypatch.setattr(seo_audit, '_cf_graphql', fake)
    return sent


def _ok_payload():
    q = {'cumulativeLayoutShiftP75': 0.004,
         'largestContentfulPaintP75': 1800,
         'interactionToNextPaintP75': 90}
    return {'viewer': {'accounts': [{
        'sitewide_7d': [{'count': 250, 'quantiles': q}],
        'sitewide_1d': [{'count': 40,
                         'quantiles': dict(q, cumulativeLayoutShiftP75=0.003)}],
        'by_path_7d': [
            {'count': 150, 'quantiles': q, 'dimensions': {'requestPath': '/'}},
            {'count': 100, 'quantiles': dict(q, cumulativeLayoutShiftP75=0.42),
             'dimensions': {'requestPath': '/forecast/x'}},
        ],
        'by_device_7d': [
            {'count': 200, 'quantiles': dict(q, cumulativeLayoutShiftP75=0.09),
             'dimensions': {'deviceType': 'mobile'}},
            {'count': 50, 'quantiles': q, 'dimensions': {'deviceType': 'desktop'}},
        ],
        'cls_elements_7d': [
            {'count': 80, 'quantiles': {'cumulativeLayoutShiftP75': 0.31},
             'dimensions': {'cumulativeLayoutShiftElement': 'div#hero-verdict',
                            'cumulativeLayoutShiftPath': '/'}},
        ],
    }]}}


def test_missing_credentials_name_which_ones(monkeypatch):
    for var in ('CF_API_TOKEN', 'CF_ACCOUNT_ID', 'CF_SITE_TAG'):
        monkeypatch.delenv(var, raising=False)
    out = seo_audit.collect_web_vitals()
    assert 'CF_API_TOKEN' in out['unavailable']
    assert 'CF_SITE_TAG' in out['unavailable']


def test_happy_path_reports_both_windows_and_per_path(monkeypatch, cf_env):
    _stub_graphql(monkeypatch, (_ok_payload(), None))
    out = seo_audit.collect_web_vitals()
    assert out['percentile'] == 'p75'
    assert out['sitewide_7d'] == {
        'samples': 250, 'cumulativeLayoutShiftP75': 0.004,
        'largestContentfulPaintP75_ms': 1.8, 'interactionToNextPaintP75_ms': 0.1}
    assert out['sitewide_1d']['cumulativeLayoutShiftP75'] == 0.003
    assert [r['requestPath'] for r in out['by_path_7d']] == ['/', '/forecast/x']
    assert out['by_path_7d'][1]['cumulativeLayoutShiftP75'] == 0.42


def test_both_windows_are_collected_not_just_the_stable_one(monkeypatch, cf_env):
    """The 7d p75 lags a fix by about a week at this traffic level.

    A history built only from it shows a shipped improvement as nothing for
    six days, which is long enough for someone to revert a fix that worked.
    """
    _stub_graphql(monkeypatch, (_ok_payload(), None))
    out = seo_audit.collect_web_vitals()
    assert out['sitewide_1d'] and out['sitewide_7d']
    assert out['sitewide_1d'] != out['sitewide_7d']


def test_a_failed_query_introspects_and_reports_the_real_schema(monkeypatch, cf_env):
    hint = {'viewer': None,
            'account': {'fields': [{'name': 'rumWebVitalsEventsAdaptiveGroups'},
                                   {'name': 'rumPageloadEventsAdaptiveGroups'},
                                   {'name': 'httpRequests1dGroups'}]},
            'group': {'fields': [{'name': 'count'}, {'name': 'quantiles'}]},
            'quantiles': {'fields': [{'name': 'cumulativeLayoutShiftP75'}]},
            'dimensions': {'fields': [{'name': 'requestPath'}]}}
    _stub_graphql(monkeypatch,
                  (None, 'Unknown field "clsP75"'),
                  (hint, None))
    out = seo_audit.collect_web_vitals()
    assert 'Unknown field' in out['error']
    sh = out['schema_hint']
    # The actual name we would need is handed over, not left to be guessed at.
    assert sh['quantile_fields'] == ['cumulativeLayoutShiftP75']
    # And the dataset probe filters the Account type down to the RUM ones.
    assert sh['rum_datasets_on_account'] == [
        'rumPageloadEventsAdaptiveGroups', 'rumWebVitalsEventsAdaptiveGroups']
    assert 'httpRequests1dGroups' not in sh['rum_datasets_on_account']


def test_unresolved_account_probe_reports_none_not_empty(monkeypatch, cf_env):
    """[] would read as "this account has no RUM datasets", which is a lie.

    The live run on 2026-08-24 hit exactly this: the probe did not resolve
    while the dataset name was in fact correct.
    """
    _stub_graphql(monkeypatch,
                  (None, 'unknown field'),
                  ({'account': None, 'group': None,
                    'quantiles': None, 'dimensions': None}, None))
    sh = seo_audit.collect_web_vitals()['schema_hint']
    assert sh['rum_datasets_on_account'] is None


def test_cls_elements_carry_the_element_and_the_path(monkeypatch, cf_env):
    """The whole reason for querying this dataset instead of reading the UI."""
    _stub_graphql(monkeypatch, (_ok_payload(), None))
    els = seo_audit.collect_web_vitals()['cls_elements_7d']
    assert els[0]['cumulativeLayoutShiftElement'] == 'div#hero-verdict'
    assert els[0]['cumulativeLayoutShiftPath'] == '/'
    assert els[0]['cumulativeLayoutShiftP75'] == 0.31


def test_device_breakdown_separates_mobile_from_desktop(monkeypatch, cf_env):
    # CLS on this site has always been a mobile defect; a blended figure hides it.
    _stub_graphql(monkeypatch, (_ok_payload(), None))
    by_dev = {r['deviceType']: r for r in
              seo_audit.collect_web_vitals()['by_device_7d']}
    assert by_dev['mobile']['cumulativeLayoutShiftP75'] == 0.09
    assert by_dev['desktop']['cumulativeLayoutShiftP75'] == 0.004


def test_failure_never_returns_a_clean_looking_empty_result(monkeypatch, cf_env):
    """The specific regression: a broken collector must not look like a fast site."""
    _stub_graphql(monkeypatch, (None, 'boom'), (None, 'introspection also failed'))
    out = seo_audit.collect_web_vitals()
    assert out.get('error')
    assert 'sitewide_7d' not in out
    assert out.get('schema_hint', {}).get('introspection_error')


def test_low_sample_count_is_flagged(monkeypatch, cf_env):
    payload = _ok_payload()
    payload['viewer']['accounts'][0]['sitewide_7d'][0]['count'] = 12
    _stub_graphql(monkeypatch, (payload, None))
    out = seo_audit.collect_web_vitals()
    assert '12 pageloads' in out['low_sample_warning']


def test_ample_samples_are_not_flagged(monkeypatch, cf_env):
    _stub_graphql(monkeypatch, (_ok_payload(), None))
    assert 'low_sample_warning' not in seo_audit.collect_web_vitals()


def test_no_matching_account_is_an_error_not_zero(monkeypatch, cf_env):
    _stub_graphql(monkeypatch, ({'viewer': {'accounts': []}}, None))
    out = seo_audit.collect_web_vitals()
    assert 'no account matched' in out['error']
    assert 'sitewide_7d' not in out


def test_query_asks_for_one_day_and_seven_day_windows(monkeypatch, cf_env):
    sent = _stub_graphql(monkeypatch, (_ok_payload(), None))
    seo_audit.collect_web_vitals()
    q = sent[0]
    assert 'sitewide_1d' in q and 'sitewide_7d' in q and 'by_path_7d' in q
    assert seo_audit.CF_VITALS_DATASET in q
    for field in seo_audit.CF_VITALS_QUANTILE_FIELDS:
        assert field in q


def test_graphql_helper_treats_a_200_with_errors_as_failure(monkeypatch):
    class Resp:
        ok = True
        status_code = 200

        def json(self):
            return {'errors': [{'message': 'nope'}], 'data': None}

    monkeypatch.setattr(seo_audit.requests, 'post', lambda *a, **k: Resp())
    data, err = seo_audit._cf_graphql('tok', '{}')
    # A GraphQL endpoint returns 200 with an errors array; checking the status
    # code alone would call this a success and hand back None as data.
    assert data is None and 'nope' in err


def test_timing_metrics_are_converted_from_microseconds(monkeypatch, cf_env):
    """Cloudflare's UI shows ms; this API returns us. Its own dashboard rendered
    p75 LCP as 2,492ms while the API returned 2492000 for the same window."""
    out = seo_audit._scale_quantiles({'largestContentfulPaintP75': 2492000})
    assert out == {'largestContentfulPaintP75_ms': 2492.0}


def test_the_unit_suffix_does_not_corrupt_the_metric_name(monkeypatch):
    # Replacing the first "P" gives largestContentfulMsPaintP75, which is not a
    # field anyone could look up.
    out = seo_audit._scale_quantiles({'largestContentfulPaintP75': 1000})
    assert 'largestContentfulMsPaintP75' not in out
    assert 'largestContentfulPaintP75_ms' in out


def test_cls_is_left_unscaled(monkeypatch):
    # CLS is unitless. Dividing it by 1000 would have hidden a real regression.
    out = seo_audit._scale_quantiles({'cumulativeLayoutShiftP75': 0.25})
    assert out == {'cumulativeLayoutShiftP75': 0.25}


def test_timing_metrics_are_converted_from_microseconds():
    """Cloudflare's UI shows ms; the API returns us for the same window.

    Its dashboard rendered p75 LCP as 2,492ms while this API returned 2492000.
    Read raw next to the dashboard, every timing is wrong by a factor of 1000.
    """
    assert seo_audit._scale_quantiles({'largestContentfulPaintP75': 2492000}) == {
        'largestContentfulPaintP75_ms': 2492.0}


def test_the_unit_suffix_does_not_corrupt_the_metric_name():
    out = seo_audit._scale_quantiles({'largestContentfulPaintP75': 1000})
    assert 'largestContentfulMsPaintP75' not in out
    assert 'largestContentfulPaintP75_ms' in out


def test_cls_is_never_scaled():
    # CLS is unitless. Dividing it by 1000 would turn a failing 0.25 into a
    # spotless 0.00025 and the metric would never fire again.
    assert seo_audit._scale_quantiles({'cumulativeLayoutShiftP75': 0.25}) == {
        'cumulativeLayoutShiftP75': 0.25}


def test_a_none_quantile_survives_scaling():
    assert seo_audit._scale_quantiles({'largestContentfulPaintP75': None}) == {
        'largestContentfulPaintP75': None}


def test_cls_spread_is_requested_across_percentiles(monkeypatch, cf_env):
    """One percentile cannot distinguish a score from a rating; a spread can."""
    sent = _stub_graphql(monkeypatch, (_ok_payload(), None))
    seo_audit.collect_web_vitals()
    for p in (25, 50, 90, 99, 999):
        assert f'cumulativeLayoutShiftP{p}' in sent[0]
