"""Storm watch and incoming-swell analysis over the wave store.

Synthetic fields throughout: every threshold here (gale mask, persistence,
bearing window, land shadow, group-velocity ETA) is a claim about physics or
geometry that a unit can pin exactly, and the live fields exercise none of
the failure paths on a calm day.
"""
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402


def _store(frames=20, ny=45, nx=90, step_deg=4.0, calm=True):
    """A small synthetic basin store. 4-degree cells keep the arrays tiny;
    nothing in the analysis assumes the production grid size."""
    now = time.time()
    st = {
        'times': np.array([now - 3600 + i * 10800 for i in range(frames)], np.int64),
        'lats': np.arange(-88.0, 92.0, step_deg, np.float32)[:ny],
        'lons': np.arange(2.0, 362.0, step_deg, np.float32)[:nx],
        'cycle': f"synthetic-{np.random.randint(1_000_000_000)}",
        'cycle_epoch': int(now - 6 * 3600),
        'built_at_epoch': now,
    }
    base = 1.0 if calm else np.nan
    for key in app._WAVE_BASIN_KEYS:
        st[key] = np.full((frames, ny, nx), base, np.float16)
    st['wind'][:] = 10.0
    st['perpw'][:] = 9.0
    return st


def _blob(st, frame_lo, frame_hi, r0, r1, c0, c1, wind=80.0, hs=6.0, period=15.0):
    st['wind'][frame_lo:frame_hi, r0:r1, c0:c1] = wind
    st['htsgw'][frame_lo:frame_hi, r0:r1, c0:c1] = hs
    st['perpw'][frame_lo:frame_hi, r0:r1, c0:c1] = period


@pytest.fixture(autouse=True)
def _clean():
    app._wave_basin = None
    with app._storm_tracks_lock:
        app._storm_tracks_cache['cycle'] = None
        app._storm_tracks_cache['tracks'] = None
    yield
    app._wave_basin = None
    with app._storm_tracks_lock:
        app._storm_tracks_cache['cycle'] = None
        app._storm_tracks_cache['tracks'] = None


class TestClustering:
    def test_gale_blob_is_one_cluster(self):
        st = _store()
        _blob(st, 0, 8, 30, 34, 40, 45)
        tracks = app._storm_tracks(st)
        assert len(tracks) == 1
        assert len(tracks[0]) == 8
        peak = max(tracks[0], key=lambda c: c['max_wind'])
        assert 30 <= peak['lat'] <= 50            # rows 30-33 at 4 deg from -88
        assert peak['max_wind'] == pytest.approx(80.0, abs=0.5)

    def test_clusters_merge_across_the_lon_seam(self):
        """A storm straddling 0/360 must be one system, not two half-storms
        whose centroids both sit hundreds of km from the real centre."""
        st = _store()
        st['wind'][0:8, 30:34, 0:2] = 80.0
        st['htsgw'][0:8, 30:34, 0:2] = 6.0
        st['wind'][0:8, 30:34, -2:] = 80.0
        st['htsgw'][0:8, 30:34, -2:] = 6.0
        tracks = app._storm_tracks(st)
        assert len(tracks) == 1

    def test_short_lived_gale_is_not_reported(self):
        """Fetch DURATION is what makes swell; 9 hours of gale is weather."""
        st = _store()
        _blob(st, 0, 3, 30, 34, 40, 45)          # 3 frames = 9h < 12h
        assert app._storm_tracks(st) == []

    def test_tiny_blob_is_noise(self):
        st = _store()
        _blob(st, 0, 8, 30, 31, 40, 41)          # 1 cell < STORM_MIN_CELLS
        assert app._storm_tracks(st) == []

    def test_tracks_are_cached_per_cycle(self):
        st = _store()
        _blob(st, 0, 8, 30, 34, 40, 45)
        first = app._storm_tracks(st)
        st['wind'][:] = 10.0                      # mutate: cache must not care
        assert app._storm_tracks(st) is first


class TestSpotFilter:
    # A spot on a synthetic west-facing coastline at (30, 210) -- mid-Pacific
    SPOT_LAT, SPOT_LON = 30.0, -150.0

    def _armed_store(self, storm_cols, storm_rows=(30, 34)):
        st = _store()
        _blob(st, 0, 8, storm_rows[0], storm_rows[1], storm_cols[0], storm_cols[1])
        app._wave_basin = st
        return st

    def test_storm_inside_the_window_is_reported(self):
        # storm ~west of the spot: lons 2+4*c deg; col 40 -> 162E; spot 210E
        self._armed_store((38, 43))
        out = app._potential_swells(self.SPOT_LAT, self.SPOT_LON,
                                    facing_direction=270.0)
        assert len(out) == 1
        p = out[0]
        assert p['confidence'] in ('forecast', 'storm-watch')
        assert p['max_wind_kmh'] == 80
        assert p['peak_period_s'] == 15.0

    def test_bearing_window_rejects_a_storm_behind_the_beach(self):
        self._armed_store((38, 43))
        out = app._potential_swells(self.SPOT_LAT, self.SPOT_LON,
                                    facing_direction=90.0)   # east-facing beach
        assert out == []

    def test_land_shadowed_storm_is_rejected(self):
        st = self._armed_store((38, 43))
        # drop a continent between the storm (cols 38-43) and the spot (col 52)
        st['htsgw'][:, :, 46:50] = np.nan
        out = app._potential_swells(self.SPOT_LAT, self.SPOT_LON,
                                    facing_direction=270.0)
        assert out == []

    def test_unknown_facing_admits_by_distance_alone(self):
        self._armed_store((38, 43))
        out = app._potential_swells(self.SPOT_LAT, self.SPOT_LON,
                                    facing_direction=None)
        assert len(out) == 1

    def test_stale_store_returns_none_not_empty(self):
        st = self._armed_store((38, 43))
        st['built_at_epoch'] = time.time() - app.WAVE_ARTIFACT_MAX_AGE_S - 60
        assert app._potential_swells(self.SPOT_LAT, self.SPOT_LON, 270.0) is None


class TestEtaMath:
    def test_group_velocity_arrival(self):
        """3,000 km at 15 s: cg = 2.808 * 15 = 42.1 km/h -> ~71 h."""
        assert 3000 / (app.GROUP_VELOCITY_KMH_PER_S * 15) == pytest.approx(71.2, abs=0.5)

    def test_eta_window_is_ordered(self):
        st = _store()
        _blob(st, 0, 8, 30, 34, 38, 43)
        app._wave_basin = st
        out = app._potential_swells(30.0, -150.0, 270.0)
        assert out, "fixture storm should be reported"
        p = out[0]
        assert p['eta_utc'] <= p['eta_late_utc']


class TestIncomingSwells:
    def test_events_segment_and_label(self):
        st = _store()
        app._wave_basin = st
        ri, ci = app._nearest_ocean_cell(st, 30.0, -150.0)
        # partition 1: a building swell peaking ~30h out
        st['sw1h'][:, ri, ci] = 0.0
        st['sw1h'][4:16, ri, ci] = np.linspace(0.5, 2.4, 12).astype(np.float16)
        st['sw1p'][:, ri, ci] = 14.0
        st['sw1d'][:, ri, ci] = 300.0
        events = app._incoming_swells(30.0, -150.0)
        assert len(events) == 1
        e = events[0]
        assert e['partition'] == 'sw1'
        assert e['phase'] == 'building'
        assert e['period_min_s'] == 14.0
        assert e['compass'] in ('WNW', 'NW')
        assert e['peak_height_m'] == pytest.approx(2.4, abs=0.05)

    def test_short_period_chop_is_not_an_event(self):
        st = _store()
        app._wave_basin = st
        ri, ci = app._nearest_ocean_cell(st, 30.0, -150.0)
        st['sw1h'][:, ri, ci] = 1.5
        st['sw1p'][:, ri, ci] = 6.0            # under the 8s floor
        assert app._incoming_swells(30.0, -150.0) == []

    def test_fading_event_still_in_window_is_labelled_fading(self):
        st = _store()
        app._wave_basin = st
        ri, ci = app._nearest_ocean_cell(st, 30.0, -150.0)
        st['sw1h'][0:10, ri, ci] = np.linspace(2.0, 0.6, 10).astype(np.float16)
        st['sw1h'][10:, ri, ci] = 0.0
        st['sw1p'][:, ri, ci] = 12.0
        st['sw1d'][:, ri, ci] = 300.0
        events = app._incoming_swells(30.0, -150.0)
        assert len(events) == 1
        assert events[0]['phase'] in ('fading', 'peaking')


class TestApiSurface:
    @pytest.fixture(autouse=True)
    def _cache(self):
        with app._cache_lock:
            app._cache.clear()
            app._cache_bytes = 0
        yield
        with app._cache_lock:
            app._cache.clear()
            app._cache_bytes = 0

    def test_fields_swell_serves_the_triplet(self, monkeypatch):
        from tests.test_wave_store import _mk_store
        app._wave_basin = _mk_store()
        client = app.app.test_client()
        default = client.get('/api/ocean-basin?lat=30&lon=-150').get_json()
        assert 'swell_direction' not in default
        with_swell = client.get(
            '/api/ocean-basin?lat=30&lon=-150&fields=swell').get_json()
        assert 'swell_direction' in with_swell
        assert 'wind_speed' not in with_swell, "groups must not leak into each other"
        both = client.get(
            '/api/ocean-basin?lat=30&lon=-150&fields=swell&wind=1').get_json()
        assert 'swell_direction' in both and 'wind_speed' in both

    def test_narrative_omits_annotations_when_store_is_cold(self, monkeypatch):
        app._wave_basin = None
        monkeypatch.setattr(app, 'get_point_weather_data', lambda lat, lon: [
            {'time': '2026-08-25T12:00Z', 'wave_direction': 270.0,
             'wave_period': 12.0, 'wave_height': 1.5},
            {'time': '2026-08-25T13:00Z', 'wave_direction': 270.0,
             'wave_period': 12.0, 'wave_height': 1.5}])
        monkeypatch.setattr(app, 'compute_beach_facing_direction',
                            lambda lat, lon: {'beach_facing_direction': 270.0})
        client = app.app.test_client()
        body = client.get('/api/swell-narrative?lat=30&lon=-150').get_json()
        assert body.get('narrative')
        assert 'incoming_swells' not in body
        assert 'potential_swells' not in body
