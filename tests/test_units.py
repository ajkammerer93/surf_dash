"""Pure-function unit tests for app.py helpers.

Network-dependent helpers (find_nearest_tide_station, get_buoy_data) are
intentionally out of scope here — they need mocked upstreams which is
PR 12.2 work, not PR 12.1. This file covers pure functions: slugify,
confidence math, cardinal direction conversion, seasonal note lookup,
nearby-spot resolution, sunrise/sunset, GFS-Wave coverage check, and the
NaN-fill helpers.
"""
import math
import numpy as np
import pytest

from app import (
    slugify,
    _compute_confidence,
    _deg_to_cardinal,
    _seasonal_climatology_note,
    _find_nearby_spots,
    _sunrise_sunset,
    _in_gfswave_atlantic_coverage,
    _fill_nan_nearest,
    _fill_nan_circular,
    haversine_distance,
)


class TestSlugify:
    def test_basic_lowercase(self):
        assert slugify('Surf City') == 'surf-city'

    def test_punctuation_stripped(self):
        assert slugify("Nags Head - Jennette's Pier") == 'nags-head-jennette-s-pier'

    def test_leading_trailing_dashes_stripped(self):
        assert slugify('--Wrightsville Beach--') == 'wrightsville-beach'

    def test_state_suffix_preserved(self):
        assert slugify('Manasquan NJ') == 'manasquan-nj'

    def test_empty_string(self):
        assert slugify('') == ''

    def test_unicode_dropped(self):
        # Non-alphanumeric (including unicode) collapsed to dashes
        result = slugify('Café Beach')
        # The é gets stripped; only alphanumeric remains
        assert 'café' not in result.lower()
        assert 'beach' in result

    def test_idempotent(self):
        s = slugify('Some Wild Name!')
        assert slugify(s) == s


class TestDegToCardinal:
    def test_north(self):
        assert _deg_to_cardinal(0) == 'N'
        assert _deg_to_cardinal(360) == 'N'

    def test_east(self):
        assert _deg_to_cardinal(90) == 'E'

    def test_south(self):
        assert _deg_to_cardinal(180) == 'S'

    def test_west(self):
        assert _deg_to_cardinal(270) == 'W'

    def test_intercardinals(self):
        assert _deg_to_cardinal(45) == 'NE'
        assert _deg_to_cardinal(135) == 'SE'
        assert _deg_to_cardinal(225) == 'SW'
        assert _deg_to_cardinal(315) == 'NW'

    def test_none(self):
        assert _deg_to_cardinal(None) == ''

    def test_boundary_rounding(self):
        # 22.5° is the boundary between N and NNE — rounds to NNE
        assert _deg_to_cardinal(22.5) == 'NNE'
        # 11.25° rounds to N
        assert _deg_to_cardinal(11) == 'N'


class TestSeasonalClimatologyNote:
    def test_known_state(self):
        note = _seasonal_climatology_note('NC')
        assert note
        assert 'hurricane' in note.lower() or 'nor' in note.lower()

    def test_unknown_state_returns_none(self):
        assert _seasonal_climatology_note('XX') is None

    def test_none_input(self):
        assert _seasonal_climatology_note(None) is None
        assert _seasonal_climatology_note('') is None


class TestComputeConfidence:
    def _mk(self, source='NOMADS', n_hours=168, lead_offset=0):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        forecast = []
        for i in range(n_hours):
            t = now + timedelta(hours=i + lead_offset)
            forecast.append({
                'time': t.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'wave_height': 1.0,
                'wave_period': 10.0,
                'wind_speed': 5.0,
            })
        return {'source': source, 'forecast': forecast}

    def test_returns_required_fields(self):
        result = _compute_confidence(self._mk(), 'test')
        assert 'score' in result
        assert 'source' in result
        assert 'factors' in result
        assert 'per_hour' in result

    def test_per_hour_matches_forecast_length(self):
        data = self._mk(n_hours=72)
        result = _compute_confidence(data, 'test')
        assert len(result['per_hour']) == 72

    def test_score_decays_with_lead_time(self):
        result = _compute_confidence(self._mk(n_hours=168), 'test')
        per_hour = result['per_hour']
        # Score at lead 0 should be substantially higher than at lead 167
        assert per_hour[0] > per_hour[-1]
        # Monotonic decay (allow small rounding noise at adjacent indices)
        # Compare 24-hour buckets to avoid integer-rounding flips
        assert per_hour[0] > per_hour[24]
        assert per_hour[24] > per_hour[72]
        assert per_hour[72] > per_hour[144]

    def test_nomads_outscores_open_meteo(self):
        n = _compute_confidence(self._mk(source='NOMADS'), 'test')
        o = _compute_confidence(self._mk(source='Open-Meteo'), 'test')
        assert n['score'] > o['score']

    def test_unknown_source_gets_low_score(self):
        u = _compute_confidence(self._mk(source='Unknown'), 'test')
        # Score should be in the range corresponding to the 'other' prior (0.55)
        # adjusted for lead-0 decay (no decay) and full completeness
        assert 50 <= u['score'] <= 60

    def test_empty_forecast(self):
        result = _compute_confidence({'source': 'NOMADS', 'forecast': []}, 'test')
        assert result['per_hour'] == []
        # Score should fall back to base * completeness*0 = 0
        assert result['score'] == 0

    def test_factors_completeness_drops_when_missing_fields(self):
        data = self._mk(n_hours=24)
        # Strip wind_speed from all entries
        for entry in data['forecast']:
            del entry['wind_speed']
        result = _compute_confidence(data, 'test')
        assert result['factors']['completeness'] < 100


class TestFindNearbySpots:
    def test_excludes_self_slug(self):
        # Virginia Beach center coords
        spots = _find_nearby_spots(36.8529, -75.978, exclude_slug='virginia-beach', count=5, max_km=200)
        assert all(s['slug'] != 'virginia-beach' for s in spots)

    def test_sorted_by_distance(self):
        spots = _find_nearby_spots(36.8529, -75.978, exclude_slug='virginia-beach', count=5, max_km=300)
        distances = [s['distance_km'] for s in spots]
        assert distances == sorted(distances)

    def test_count_limit(self):
        spots = _find_nearby_spots(36.8529, -75.978, count=3, max_km=300)
        assert len(spots) <= 3

    def test_max_km_filter(self):
        # 0.5 km radius from an offshore Atlantic coord — no spots there
        spots = _find_nearby_spots(35.0, -65.0, count=10, max_km=0.5)
        assert spots == []

    def test_returns_state_in_results(self):
        spots = _find_nearby_spots(36.8529, -75.978, exclude_slug='virginia-beach', count=3, max_km=200)
        for s in spots:
            assert 'state' in s


class TestSunriseSunset:
    """Regression test for the date-wraparound bug fixed in v0.8.1."""

    def test_returns_iso_with_z(self):
        from datetime import date
        sr, ss = _sunrise_sunset(34.43, -77.55, date(2026, 5, 11))
        assert sr.endswith('Z')
        assert ss.endswith('Z')

    def test_sunset_after_sunrise_for_us_east_coast(self):
        """Bug v0.8.1: US East Coast sunset was placed ~24h BEFORE sunrise
        because the formula gave sunset_min >= 1440 (next UTC day) but the
        old min_to_iso modulo-wrapped without advancing the date."""
        from datetime import date, datetime
        sr, ss = _sunrise_sunset(34.43, -77.55, date(2026, 5, 11))
        sr_dt = datetime.fromisoformat(sr.replace('Z', '+00:00'))
        ss_dt = datetime.fromisoformat(ss.replace('Z', '+00:00'))
        assert ss_dt > sr_dt, f"Sunset {ss_dt} should be AFTER sunrise {sr_dt}"

    def test_typical_daylight_duration_in_may(self):
        from datetime import date, datetime
        sr, ss = _sunrise_sunset(34.43, -77.55, date(2026, 5, 11))
        sr_dt = datetime.fromisoformat(sr.replace('Z', '+00:00'))
        ss_dt = datetime.fromisoformat(ss.replace('Z', '+00:00'))
        duration_hours = (ss_dt - sr_dt).total_seconds() / 3600
        # Mid-May, lat 34 — ~13.5–14 hours of daylight
        assert 13 < duration_hours < 15

    def test_polar_night_returns_none(self):
        from datetime import date
        # Above the Arctic Circle in winter — polar night
        sr, ss = _sunrise_sunset(80.0, 0.0, date(2026, 12, 21))
        assert sr is None and ss is None

    def test_polar_day_returns_none(self):
        from datetime import date
        # Above the Arctic Circle at summer solstice — polar day
        sr, ss = _sunrise_sunset(80.0, 0.0, date(2026, 6, 21))
        assert sr is None and ss is None


class TestGfsWaveCoverage:
    def test_us_east_coast_in_coverage(self):
        # Surf City NC
        assert _in_gfswave_atlantic_coverage(34.43, -77.55)

    def test_atlantic_florida_in_coverage(self):
        # Cocoa Beach
        assert _in_gfswave_atlantic_coverage(28.32, -80.61)

    def test_pacific_california_not_in_coverage(self):
        # Surfrider in Malibu — GFS-Wave Atlantic only
        assert not _in_gfswave_atlantic_coverage(34.03, -118.68)

    def test_hawaii_not_in_coverage(self):
        # North Shore Oahu
        assert not _in_gfswave_atlantic_coverage(21.65, -158.05)


class TestHaversineDistance:
    def test_zero_distance(self):
        d = haversine_distance(34.43, -77.55, 34.43, -77.55)
        assert d == 0

    def test_one_degree_lat_is_about_111km(self):
        d = haversine_distance(0, 0, 1, 0)
        assert 110 < d < 112

    def test_symmetric(self):
        d1 = haversine_distance(34.43, -77.55, 36.85, -75.98)
        d2 = haversine_distance(36.85, -75.98, 34.43, -77.55)
        assert abs(d1 - d2) < 0.01

    def test_known_distance_surf_city_to_va_beach(self):
        # Surf City NC to Virginia Beach: ~300 km
        d = haversine_distance(34.43, -77.55, 36.85, -75.98)
        assert 290 < d < 315


class TestFillNanNearest:
    def test_no_nan_returns_unchanged(self):
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _fill_nan_nearest(grid)
        np.testing.assert_array_equal(result, grid)

    def test_fills_isolated_nan_with_neighbor_mean(self):
        grid = np.array([[1.0, 2.0, 3.0],
                         [4.0, np.nan, 6.0],
                         [7.0, 8.0, 9.0]])
        result = _fill_nan_nearest(grid)
        # Center cell should be filled (mean of 4 neighbors = 5)
        assert not np.isnan(result[1, 1])
        assert abs(result[1, 1] - 5.0) < 0.01

    def test_min_valid_neighbors_blocks_single_neighbor_fill(self):
        # Cell has only 1 valid neighbor — should NOT fill with min=2
        grid = np.array([[np.nan, np.nan, np.nan],
                         [np.nan, np.nan, np.nan],
                         [np.nan, 5.0, np.nan]])
        result = _fill_nan_nearest(grid, max_iterations=1, min_valid_neighbors=2)
        # Top-corner cells have at most 1 valid neighbor — stay NaN
        assert np.isnan(result[0, 0])
        # Cell at (2,0): has 1 valid neighbor at (2,1) → stays NaN with min=2
        assert np.isnan(result[2, 0])


class TestFillNanCircular:
    def test_handles_360_0_wrap(self):
        # 350° and 10° should average to 0°, not 180°
        grid = np.array([[350.0, np.nan, 10.0]])
        result = _fill_nan_circular(grid, max_iterations=1)
        # Center value should be near 0°/360°, not 180°
        assert not np.isnan(result[0, 1])
        v = result[0, 1] % 360
        # Should be in [340°, 20°] band, not near 180°
        assert v < 30 or v > 330

    def test_passes_through_valid_values(self):
        grid = np.array([[45.0, 90.0]])
        result = _fill_nan_circular(grid)
        np.testing.assert_array_almost_equal(result % 360, [[45.0, 90.0]], decimal=1)
