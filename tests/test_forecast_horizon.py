"""The 16-day forecast horizon and the GFS-Wave point source.

Open-Meteo's default best_match resolves to MFWAM on most coasts: it stops
at day 10 (every wave field null from hour 240) and on 2026-09-02 carried no
long-period swell train at any lead (issue #88). GFS-Wave runs the full 16
days, is the model the swell map is built from, and its wave_period is the
spectral peak. These tests pin the routing, the fallback, the horizon on every
upstream call, and the copy that advertises it.
"""
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as surf_app  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_response(status=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status
    resp.ok = status < 400
    resp.headers = {'content-type': 'application/json'}
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError('No JSON')
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f'{status}')
    else:
        resp.raise_for_status.return_value = None
    return resp


def _hourly(n_hours, null_from=None, peak=False):
    """An Open-Meteo Marine hourly block of n_hours, optionally null after
    null_from (what best_match does past MFWAM's day 10)."""
    t0 = datetime(2026, 11, 2, tzinfo=timezone.utc)
    times = [(t0 + timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M') for i in range(n_hours)]

    def col(v):
        return [None if (null_from is not None and i >= null_from) else v
                for i in range(n_hours)]
    block = {
        'time': times,
        'wave_height': col(1.2),
        'wave_period': col(9.5),
        'wave_direction': col(90.0),
        'wind_wave_height': col(0.4),
        'wind_wave_period': col(4.0),
        'wind_wave_direction': col(120.0),
    }
    if peak:
        block['wave_peak_period'] = col(11.0)
        block['wind_wave_peak_period'] = col(4.5)
    return {'hourly': block}


@pytest.fixture(autouse=True)
def _reset_cooldown():
    surf_app._om_weather_block['until'] = 0.0
    yield
    surf_app._om_weather_block['until'] = 0.0


class TestConstants:
    def test_horizon_is_sixteen_days(self):
        assert surf_app.FORECAST_DAYS == 16
        assert surf_app.FORECAST_CORE_HOURS == 168

    def test_point_model_is_gfswave(self):
        assert surf_app.OPEN_METEO_WAVE_MODEL == 'ncep_gfswave025'


class TestMarineRouting:
    @patch('app.time.sleep')
    @patch('app.requests.get')
    def test_gfswave_first_for_the_full_horizon(self, mock_get, _sleep):
        mock_get.return_value = make_response(200, _hourly(384))
        forecast, source = surf_app._get_point_from_open_meteo(34.43, -77.55)
        assert source == 'GFS-Wave'
        assert len(forecast) == 384
        assert mock_get.call_count == 1
        params = mock_get.call_args.kwargs['params']
        assert params['models'] == 'ncep_gfswave025'
        assert params['forecast_days'] == 16
        # GFS-Wave has no peak-period field; wave_period IS the peak
        assert 'wave_peak_period' not in params['hourly']
        assert forecast[0]['wave_period'] == 9.5
        assert forecast[-1]['time'] == '2026-11-17T23:00Z'

    @patch('app.time.sleep')
    @patch('app.requests.get')
    def test_best_match_fallback_when_gfswave_errors(self, mock_get, _sleep):
        mock_get.side_effect = [
            make_response(400),                                  # gfswave
            make_response(200, _hourly(384, null_from=240, peak=True)),  # best_match
        ]
        forecast, source = surf_app._get_point_from_open_meteo(34.43, -77.55)
        assert source == 'Open-Meteo'
        # MFWAM's null tail past day 10 is trimmed, not drawn as a cliff
        assert len(forecast) == 240
        assert all(e['wave_height'] is not None for e in forecast)
        second = mock_get.call_args_list[1].kwargs['params']
        assert 'models' not in second
        assert second['forecast_days'] == 16
        assert 'wave_peak_period' in second['hourly']
        # best_match keeps the peak-over-mean preference
        assert forecast[0]['wave_period'] == 11.0

    @patch('app.time.sleep')
    @patch('app.requests.get')
    def test_best_match_fallback_when_gfswave_is_all_null(self, mock_get, _sleep):
        mock_get.side_effect = [
            make_response(200, _hourly(384, null_from=0)),
            make_response(200, _hourly(240)),
        ]
        forecast, source = surf_app._get_point_from_open_meteo(34.43, -77.55)
        assert source == 'Open-Meteo'
        assert len(forecast) == 240

    @patch('app.time.sleep')
    @patch('app.requests.get')
    def test_both_models_failing_yields_none_pair(self, mock_get, _sleep):
        mock_get.return_value = make_response(429)
        assert surf_app._get_point_from_open_meteo(34.43, -77.55) == (None, None)

    def test_trailing_null_trim_keeps_interior_gaps(self):
        fc = [{'wave_height': 1.0}, {'wave_height': None}, {'wave_height': 1.1},
              {'wave_height': None}, {'wave_height': None}]
        out = surf_app._trim_trailing_null_hours(fc)
        assert len(out) == 3
        assert surf_app._trim_trailing_null_hours([]) == []

    @patch('app._enrich_in_parallel', return_value='America/New_York')
    @patch('app._in_gfswave_atlantic_coverage', return_value=False)
    @patch('app._get_point_from_open_meteo')
    def test_point_chain_carries_the_source_label(self, mock_om, _cov, _enrich):
        mock_om.return_value = ([{'time': '2026-11-02T00:00Z', 'wave_height': 1.0}], 'GFS-Wave')
        out = surf_app.get_point_weather_data(34.43, -77.55)
        assert out['source'] == 'GFS-Wave'
        assert out['location_timezone'] == 'America/New_York'


class TestOtherUpstreamsFollowTheHorizon:
    @patch('app.time.sleep')
    @patch('app.requests.get')
    def test_wind_enrichment_requests_sixteen_days(self, mock_get, _sleep):
        mock_get.return_value = make_response(200, {'hourly': {
            'time': ['2026-11-02T00:00'], 'wind_speed_10m': [12.0], 'wind_direction_10m': [250.0]}})
        forecast = [{'time': '2026-11-02T00:00Z', 'wind_speed': None, 'wind_direction': None}]
        surf_app._enrich_with_wind(forecast, 34.43, -77.55)
        assert mock_get.call_args.kwargs['params']['forecast_days'] == 16
        assert forecast[0]['wind_speed'] == 12.0

    @patch('app.requests.get')
    def test_tide_window_covers_sixteen_days(self, mock_get):
        mock_get.return_value = make_response(200, {'predictions': []})
        surf_app._fetch_tide_predictions('8658163', 'hilo')
        params = mock_get.call_args.kwargs['params']
        today = datetime.now(timezone.utc)
        assert params['end_date'] == (today + timedelta(days=16)).strftime('%Y%m%d')
        assert params['begin_date'] == (today - timedelta(days=1)).strftime('%Y%m%d')


class TestConfidencePrior:
    def _mk(self, source):
        now = datetime.now(timezone.utc)
        return {'source': source, 'forecast': [
            {'time': (now + timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M:%SZ'),
             'wave_height': 1.0, 'wave_period': 10.0, 'wind_speed': 5.0}
            for i in range(48)]}

    def test_gfswave_sits_between_nomads_and_best_match(self):
        n = surf_app._compute_confidence(self._mk('NOMADS'), 't')['score']
        g = surf_app._compute_confidence(self._mk('GFS-Wave'), 't')['score']
        o = surf_app._compute_confidence(self._mk('Open-Meteo'), 't')['score']
        assert n > g > o
        assert surf_app._compute_confidence(self._mk('GFS-Wave'), 't')['source'] == 'GFS-Wave (Open-Meteo)'


class TestClientHorizon:
    @pytest.fixture(scope='class')
    def html(self):
        with open(os.path.join(REPO, 'templates', 'index.html')) as f:
            return f.read()

    def test_toggle_exists_on_desktop_and_in_the_more_menu(self, html):
        assert 'id="horizon-toggle-btn"' in html
        assert 'id="more-horizon-label"' in html
        assert 'function toggleForecastHorizon()' in html

    def test_default_is_seven_days_and_the_core_is_168_hours(self, html):
        assert 'const FORECAST_CORE_HOURS = 168;' in html
        assert "=== '16d' ? '16d' : '7d'" in html

    def test_outlook_plugin_is_registered_on_the_wave_chart(self, html):
        assert 'const outlookPlugin = {' in html
        assert re.search(r'plugins:\s*\[dayNightPlugin, windLanePlugin, tideLanePlugin, periodRowPlugin, outlookPlugin\]', html)

    def test_planner_never_scores_the_outlook(self, html):
        start = html.index('function computeSessionWindows()')
        body = html[start:start + 2500]
        assert 'd.getTime() >= outlookStart) return;' in body

    def test_table_marks_the_outlook(self, html):
        start = html.index('function rebuildForecastTable()')
        body = html[start:start + 3500]
        assert "'outlook-divider'" in body
        assert "' row-outlook'" in body


class TestCopy:
    SEVEN_DAY = re.compile(r'\b7[- ]day (surf )?forecast|7 days, hourly|Free 7-day', re.I)

    @pytest.mark.parametrize('rel', [
        'templates/index.html', 'templates/about.html', 'templates/locations.html',
        'templates/faq.html', 'templates/compare.html',
        'templates/compare_magicseaweed.html', 'static/manifest.json',
        'region_pages.py', 'README.md', 'app.py',
    ])
    def test_no_seven_day_forecast_copy_remains(self, rel):
        with open(os.path.join(REPO, rel)) as f:
            text = f.read()
        hits = [m.group(0) for m in self.SEVEN_DAY.finditer(text)]
        assert not hits, f'{rel}: {hits}'

    def test_forecast_page_advertises_sixteen_days(self):
        client = surf_app.app.test_client()
        body = client.get('/forecast/surf-city-pier-north').get_data(as_text=True)
        assert 'Free 16-day surf forecast for' in body
        assert '"name": "16-day forecast"' in body
