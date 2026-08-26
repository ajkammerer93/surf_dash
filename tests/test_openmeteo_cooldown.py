"""Open-Meteo WEATHER API 429 cooldown.

The weather API rate-limits per IP and Render's shared egress sits over the
limit chronically -- 200 from a residential IP and 429 from the server in
the same minute, documented across two incidents. Re-paying ~10s per cold
forecast to rediscover a chronic condition is what pushed cold forecasts to
20s+; after any 429 the weather API is skipped for a cooldown and the
fallbacks run immediately. The MARINE API is a separate pool and must be
completely unaffected.
"""
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as surf_app  # noqa: E402


def make_response(status=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status
    resp.ok = 200 <= status < 300
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


@pytest.fixture(autouse=True)
def _reset_cooldown():
    surf_app._om_weather_block['until'] = 0.0
    yield
    surf_app._om_weather_block['until'] = 0.0


def test_a_429_arms_the_cooldown():
    assert surf_app._om_weather_available() is True
    surf_app._om_weather_note(make_response(429))
    assert surf_app._om_weather_available() is False
    until = surf_app._om_weather_block['until']
    assert until - time.monotonic() <= surf_app.OPENMETEO_WEATHER_COOLDOWN_S + 1


def test_non_429_responses_do_not_arm_it():
    for status in (200, 404, 500, 503):
        surf_app._om_weather_note(make_response(status))
        assert surf_app._om_weather_available() is True


@patch('app._enrich_wind_from_erddap')
@patch('app.requests.get')
def test_wind_goes_straight_to_fallback_while_blocked(mock_get, mock_fallback):
    surf_app._om_weather_block['until'] = time.monotonic() + 600
    surf_app._enrich_with_wind([{'time': '2026-06-10T19:00Z'}], 34.43, -77.55)
    mock_fallback.assert_called_once()
    mock_get.assert_not_called()          # not even one probing request


@patch('app.requests.get')
def test_current_temp_call_is_exempt_from_the_cooldown(mock_get):
    """The cooldown exists to protect the ~10s hourly-wind call. Gating the
    ~0.5s current-temp call behind it turned "air temp missing sometimes"
    into "air temp missing ALWAYS" in production (168/168 null hours,
    2026-08-25): with the rate limit intermittent, every request that could
    have landed in a 200 window was itself skipped. The temp call now always
    tries -- and still NOTES a 429 so the wind call stays protected."""
    surf_app._om_weather_block['until'] = time.monotonic() + 600
    mock_get.return_value = make_response(
        200, json_data={'current': {'temperature_2m': 27.5,
                                    'sea_surface_temperature': 22.5},
                        'timezone': 'America/New_York'})
    forecast = [{'air_temperature': None, 'water_temperature': None}]
    surf_app._enrich_with_temperatures(forecast, 34.43, -77.55)
    urls = [c.args[0] for c in mock_get.call_args_list]
    assert any('api.open-meteo.com' in u for u in urls), "temp call was gated"
    assert forecast[0]['air_temperature'] == 27.5
    assert forecast[0]['water_temperature'] == 22.5


@patch('app.requests.get')
def test_a_429_on_the_temp_call_rearms_the_cooldown(mock_get):
    surf_app._om_weather_block['until'] = 0.0
    mock_get.return_value = make_response(429)
    surf_app._enrich_with_temperatures(
        [{'air_temperature': None, 'water_temperature': None}], 34.43, -77.55)
    assert surf_app._om_weather_available() is False


@patch('app._enrich_wind_from_erddap')
@patch('app.requests.get')
def test_a_wind_429_arms_the_cooldown_for_the_next_caller(mock_get, mock_fallback):
    mock_get.return_value = make_response(429)
    surf_app._enrich_with_wind([{'time': '2026-06-10T19:00Z'}], 34.43, -77.55)
    assert surf_app._om_weather_available() is False
