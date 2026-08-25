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
def test_temperatures_skip_air_but_keep_water_while_blocked(mock_get):
    """The marine pool keeps working through a weather 429; blocking both
    would throw away a healthy upstream."""
    surf_app._om_weather_block['until'] = time.monotonic() + 600
    mock_get.return_value = make_response(
        200, json_data={'current': {'sea_surface_temperature': 22.5}})
    forecast = [{'air_temperature': None, 'water_temperature': None}]
    surf_app._enrich_with_temperatures(forecast, 34.43, -77.55)
    urls = [c.args[0] for c in mock_get.call_args_list]
    assert all('marine' in u for u in urls), f"weather API was called: {urls}"
    assert forecast[0]['water_temperature'] == 22.5


@patch('app._enrich_wind_from_erddap')
@patch('app.requests.get')
def test_a_wind_429_arms_the_cooldown_for_the_next_caller(mock_get, mock_fallback):
    mock_get.return_value = make_response(429)
    surf_app._enrich_with_wind([{'time': '2026-06-10T19:00Z'}], 34.43, -77.55)
    assert surf_app._om_weather_available() is False
