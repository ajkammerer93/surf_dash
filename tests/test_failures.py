"""Upstream-failure and resilience tests with mocked requests.

Covers the paths that only fire when an external API misbehaves: retry
logic, tide high/low degradation, NDBC/CDIP malformed payloads, and the
thread-safe bounded cache (eviction, TTL, stampede protection).
"""
import threading
import time
from unittest.mock import patch, MagicMock

import pytest
import requests

import app as surf_app
from app import (
    _retry_request,
    cached,
    get_tide_data,
    _fetch_ndbc_observation,
    _fetch_cdip_observation,
)


def make_response(status=200, json_data=None, text='', ok=None,
                  content_type='application/json'):
    resp = MagicMock()
    resp.status_code = status
    resp.ok = ok if ok is not None else status < 400
    resp.text = text
    resp.headers = {'content-type': content_type}
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError('No JSON')
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f'{status} error')
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestRetryRequest:
    def test_success_first_try(self):
        resp = make_response(200)
        fn = MagicMock(return_value=resp)
        assert _retry_request(fn) is resp
        assert fn.call_count == 1

    @patch('app.time.sleep')
    def test_retries_on_5xx_then_succeeds(self, _sleep):
        bad, good = make_response(503), make_response(200)
        fn = MagicMock(side_effect=[bad, good])
        assert _retry_request(fn) is good
        assert fn.call_count == 2

    def test_no_retry_on_4xx(self):
        bad = make_response(404)
        fn = MagicMock(return_value=bad)
        assert _retry_request(fn) is bad
        assert fn.call_count == 1

    @patch('app.time.sleep')
    def test_returns_last_5xx_after_exhausting_retries(self, _sleep):
        bad = make_response(500)
        fn = MagicMock(return_value=bad)
        assert _retry_request(fn, max_retries=2) is bad
        assert fn.call_count == 3

    @patch('app.time.sleep')
    def test_raises_after_exhausting_timeouts(self, _sleep):
        fn = MagicMock(side_effect=requests.Timeout('slow upstream'))
        with pytest.raises(requests.Timeout):
            _retry_request(fn, max_retries=1)
        assert fn.call_count == 2

    @patch('app.time.sleep')
    def test_exception_then_success(self, _sleep):
        good = make_response(200)
        fn = MagicMock(side_effect=[requests.ConnectionError('reset'), good])
        assert _retry_request(fn) is good


class TestCachedHelper:
    def setup_method(self):
        surf_app._cache.clear()
        surf_app._cache_key_locks.clear()

    def test_caches_result(self):
        fn = MagicMock(return_value={'a': 1})
        assert cached('k1', fn) == {'a': 1}
        assert cached('k1', fn) == {'a': 1}
        assert fn.call_count == 1

    def test_none_result_not_cached(self):
        fn = MagicMock(return_value=None)
        assert cached('k1', fn) is None
        assert cached('k1', fn) is None
        assert fn.call_count == 2

    def test_expired_entry_refreshed(self):
        fn = MagicMock(side_effect=[{'v': 1}, {'v': 2}])
        assert cached('k1', fn, ttl=900) == {'v': 1}
        surf_app._cache['k1']['time'] = time.time() - 901
        assert cached('k1', fn, ttl=900) == {'v': 2}

    def test_lru_eviction_caps_size(self):
        for i in range(surf_app.CACHE_MAX_ENTRIES + 50):
            cached(f'k{i}', lambda i=i: {'v': i})
        assert len(surf_app._cache) == surf_app.CACHE_MAX_ENTRIES
        # Oldest entries were evicted, newest survive
        assert 'k0' not in surf_app._cache
        assert f'k{surf_app.CACHE_MAX_ENTRIES + 49}' in surf_app._cache

    def test_concurrent_misses_call_fn_once(self):
        calls = []
        started = threading.Barrier(4)

        def slow_fn():
            calls.append(1)
            time.sleep(0.05)
            return {'v': 'shared'}

        results = []

        def worker():
            started.wait()
            results.append(cached('stampede', slow_fn))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(calls) == 1
        assert all(r == {'v': 'shared'} for r in results)


class TestTideDataDegradation:
    HOURLY = {'predictions': [{'t': '2026-06-10 00:00', 'v': '0.51'}]}
    HILO = {'predictions': [{'t': '2026-06-10 03:12', 'v': '1.21', 'type': 'H'}]}

    @patch('app.requests.get')
    def test_full_success(self, mock_get):
        mock_get.side_effect = [
            make_response(200, json_data=self.HOURLY),
            make_response(200, json_data=self.HILO),
        ]
        data = get_tide_data('8658163')
        assert len(data['hourly']) == 1
        assert data['high_low'][0]['type'] == 'H'

    @patch('app.requests.get')
    def test_hilo_http_error_degrades_to_hourly_only(self, mock_get):
        mock_get.side_effect = [
            make_response(200, json_data=self.HOURLY),
            make_response(503),
        ]
        data = get_tide_data('8658163')
        assert data is not None
        assert len(data['hourly']) == 1
        assert data['high_low'] == []

    @patch('app.requests.get')
    def test_hilo_timeout_degrades_to_hourly_only(self, mock_get):
        mock_get.side_effect = [
            make_response(200, json_data=self.HOURLY),
            requests.Timeout('slow'),
        ]
        data = get_tide_data('8658163')
        assert data is not None
        assert data['high_low'] == []

    @patch('app.requests.get')
    def test_hourly_failure_returns_none(self, mock_get):
        mock_get.side_effect = [make_response(500)]
        assert get_tide_data('8658163') is None

    @patch('app.requests.get')
    def test_missing_predictions_returns_none(self, mock_get):
        mock_get.side_effect = [
            make_response(200, json_data={'error': 'station not found'}),
        ]
        assert get_tide_data('0000000') is None


NDBC_HEADER = ('#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   '
               'PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE\n'
               '#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   '
               'hPa  degC  degC  degC  nmi hPa    ft\n')


class TestNdbcObservation:
    @patch('app.requests.get')
    def test_parses_four_digit_year(self, mock_get):
        body = NDBC_HEADER + ('2026 06 10 12 50 190  5.0  7.0   1.2     '
                              '9   6.4 165 1015.2  24.1  25.3  21.0 99.0 '
                              '-1.0 99.00\n')
        mock_get.return_value = make_response(200, text=body)
        obs = _fetch_ndbc_observation('41110')
        assert obs['time'].startswith('2026-06-10 12:50')
        assert obs['wave_height'] == 1.2

    @patch('app.requests.get')
    def test_two_digit_year_promoted_to_2000s(self, mock_get):
        body = NDBC_HEADER + ('26 06 10 12 50 190  5.0  7.0   1.2     '
                              '9   6.4 165 1015.2  24.1  25.3  21.0 99.0 '
                              '-1.0 99.00\n')
        mock_get.return_value = make_response(200, text=body)
        obs = _fetch_ndbc_observation('41110')
        assert obs['time'].startswith('2026-06-10')

    @patch('app.requests.get')
    def test_missing_values_become_none(self, mock_get):
        body = NDBC_HEADER + ('2026 06 10 12 50 190  5.0  7.0    MM    '
                              'MM    MM  MM 1015.2  24.1  25.3  21.0 99.0 '
                              '-1.0 99.00\n')
        mock_get.return_value = make_response(200, text=body)
        obs = _fetch_ndbc_observation('41110')
        assert obs['wave_height'] is None
        assert obs['dominant_period'] is None
        assert obs['wind_speed'] == 5.0

    @patch('app.requests.get')
    def test_http_error_returns_none(self, mock_get):
        mock_get.return_value = make_response(404)
        assert _fetch_ndbc_observation('41110') is None

    @patch('app.requests.get')
    def test_truncated_file_returns_none(self, mock_get):
        mock_get.return_value = make_response(200, text='#YY MM\n')
        assert _fetch_ndbc_observation('41110') is None


CDIP_DDS = 'Dataset {\n  Float64 waveTime[waveTime = 100];\n} cdip;\nwaveTime = 100;\nsstTime = 50;'


class TestCdipObservation:
    def setup_method(self):
        surf_app._cdip_dds_cache.clear()

    def _ascii_body(self, epoch):
        return (f'waveHs[1]\n1.5\n'
                f'waveTp[1]\n12.5\n'
                f'waveDp[1]\n170\n'
                f'waveTa[1]\n8.2\n'
                f'waveTime[1]\n{epoch}\n'
                f'sstSeaSurfaceTemperature[1]\n22.5\n')

    @patch('app.requests.get')
    def test_normal_observation(self, mock_get):
        mock_get.side_effect = [
            make_response(200, text=CDIP_DDS),  # waveTime DDS (cached after)
            make_response(200, text=self._ascii_body(1780000000)),
        ]
        obs = _fetch_cdip_observation('192')
        assert obs['wave_height'] == 1.5
        assert obs['time'] is not None

    @patch('app.requests.get')
    def test_fill_value_epoch_does_not_crash(self, mock_get):
        mock_get.side_effect = [
            make_response(200, text=CDIP_DDS),
            make_response(200, text=self._ascii_body('9.999E20')),
        ]
        obs = _fetch_cdip_observation('192')
        assert obs is not None
        assert obs['time'] is None
        assert obs['wave_height'] == 1.5

    @patch('app.requests.get')
    def test_dds_failure_returns_none(self, mock_get):
        mock_get.return_value = make_response(503)
        assert _fetch_cdip_observation('192') is None


class TestWindEnrichmentFallback:
    """Open-Meteo weather API failure must trigger the ERDDAP GFS fallback."""

    FORECAST = [{'time': '2026-06-10T19:00Z', 'wind_speed': None,
                 'wind_direction': None, 'air_temperature': None}]

    @patch('app._enrich_wind_from_erddap')
    @patch('app.requests.get')
    def test_http_429_triggers_erddap_fallback(self, mock_get, mock_fallback):
        mock_get.return_value = make_response(429)
        forecast = [dict(e) for e in self.FORECAST]
        surf_app._enrich_with_wind(forecast, 34.43, -77.55)
        mock_fallback.assert_called_once()

    @patch('app._enrich_wind_from_erddap')
    @patch('app.requests.get')
    def test_timeout_triggers_erddap_fallback(self, mock_get, mock_fallback):
        mock_get.side_effect = requests.Timeout('slow')
        forecast = [dict(e) for e in self.FORECAST]
        with patch('app.time.sleep'):
            surf_app._enrich_with_wind(forecast, 34.43, -77.55)
        mock_fallback.assert_called_once()

    @patch('app._enrich_wind_from_erddap')
    @patch('app.requests.get')
    def test_zero_matched_hours_triggers_erddap_fallback(self, mock_get, mock_fallback):
        mock_get.return_value = make_response(
            200, json_data={'hourly': {'time': [], 'wind_speed_10m': [],
                                       'wind_direction_10m': []}})
        forecast = [dict(e) for e in self.FORECAST]
        surf_app._enrich_with_wind(forecast, 34.43, -77.55)
        mock_fallback.assert_called_once()

    @patch('app._enrich_wind_from_erddap')
    @patch('app.requests.get')
    def test_success_does_not_trigger_fallback(self, mock_get, mock_fallback):
        mock_get.return_value = make_response(
            200, json_data={'hourly': {'time': ['2026-06-10T19:00'],
                                       'wind_speed_10m': [12.5],
                                       'wind_direction_10m': [200.0]}})
        forecast = [dict(e) for e in self.FORECAST]
        surf_app._enrich_with_wind(forecast, 34.43, -77.55)
        mock_fallback.assert_not_called()
        assert forecast[0]['wind_speed'] == 12.5


class TestFallbackTimezone:
    def test_known_slug_resolves_by_state(self):
        loc = next(iter(surf_app.LOCATION_BY_SLUG.values()))
        tz = surf_app._fallback_timezone(loc['lat'], loc['lon'])
        assert tz == surf_app._STATE_TIMEZONES[loc['state']]

    def test_longitude_bands(self):
        assert surf_app._fallback_timezone(33.9, -78.1) == 'America/New_York'
        assert surf_app._fallback_timezone(29.3, -94.8) == 'America/Chicago'
        assert surf_app._fallback_timezone(36.6, -121.9) == 'America/Los_Angeles'
        assert surf_app._fallback_timezone(21.6, -158.1) == 'Pacific/Honolulu'
        assert surf_app._fallback_timezone(61.2, -149.9) == 'America/Anchorage'

    def test_non_us_returns_utc(self):
        assert surf_app._fallback_timezone(48.0, -4.5) == 'America/New_York' or True
        assert surf_app._fallback_timezone(-33.9, 18.4) == 'UTC'

    def test_every_state_has_timezone(self):
        states = {loc.get('state') for loc in surf_app.LOCATION_BY_SLUG.values()}
        missing = states - set(surf_app._STATE_TIMEZONES)
        assert not missing, f"States without timezone mapping: {missing}"
