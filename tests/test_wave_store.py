"""The in-process wave store: gfswave artifacts served from RAM.

The store replaces ERDDAP as the load-bearing source for both grid endpoints.
What these tests pin is the CONTRACT under a source swap -- the client checks
none of this and fails silently when it drifts: [t][lat][lon] order, evenly
spaced ascending lats, lons as -180..180 VALUES in stored 0..360 ORDER (the
client reorders columns itself; pre-sorting here would double-reorder), land
as 0.0 rather than NaN, ISO times, and wind absent unless asked for.
"""
import io
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402


def _mk_store(age_s=0, frames=129, ny=90, nx=180):
    now = time.time()
    times = np.array([now - 3600 + i * 10800 for i in range(frames)], np.int64)
    store = {
        'times': times,
        'lats': np.arange(-89.125, 90, 2.0, np.float32)[:ny],
        'lons': np.arange(0.875, 360, 2.0, np.float32)[:nx],
        'cycle': '2026-08-25T12:00Z',
        'cycle_epoch': int(now - 6 * 3600),
        'built_at_epoch': now - age_s,
    }
    for key in app._WAVE_BASIN_KEYS:
        arr = np.full((frames, ny, nx), 1.5, np.float16)
        arr[:, 0, 0] = np.nan          # a land cell
        store[key] = arr
    # the wire grid: 3-degree, 57 frames, its own axes
    store['client_lats'] = np.arange(-88.625, 90, 3.0, np.float32)[:60]
    store['client_lons'] = np.arange(1.375, 360, 3.0, np.float32)[:120]
    for key in app._WAVE_TILE_KEYS:
        arr = np.full((app.WAVE_BASIN_CLIENT_FRAMES, 60, 120), 1.5, np.float16)
        arr[:, 0, 0] = np.nan
        store['client_' + key] = arr
    return store


@pytest.fixture(autouse=True)
def _clean_store():
    app._wave_basin = None
    app._wave_tiles.clear()
    yield
    app._wave_basin = None
    app._wave_tiles.clear()


class TestLoader:
    def _npz_bytes(self, drop=None, bad_shape=None):
        arrays = {
            'times': np.arange(3, dtype=np.int64),
            'lats': np.array([1.0, 2.0], np.float32),
            'lons': np.array([10.0, 12.0, 14.0], np.float32),
        }
        for key in app._WAVE_TILE_KEYS:
            arrays[key] = np.zeros((3, 2, 3), np.float16)
        if drop:
            del arrays[drop]
        if bad_shape:
            arrays[bad_shape] = np.zeros((3, 2, 2), np.float16)
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return buf.getvalue()

    def test_valid_artifact_loads(self):
        store = app._load_wave_npz(self._npz_bytes(), app._WAVE_TILE_KEYS, frames=3)
        assert store['htsgw'].shape == (3, 2, 3)

    def test_missing_field_rejected_whole(self):
        """A half-valid artifact swapped in would move the failure from load
        time (loud) to request time (a KeyError in every response)."""
        with pytest.raises(ValueError, match='missing key'):
            app._load_wave_npz(self._npz_bytes(drop='wdir'),
                               app._WAVE_TILE_KEYS, frames=3)

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match='shape'):
            app._load_wave_npz(self._npz_bytes(bad_shape='wind'),
                               app._WAVE_TILE_KEYS, frames=3)

    def test_descending_lats_rejected(self):
        arrays_src = self._npz_bytes()
        npz = np.load(io.BytesIO(arrays_src))
        arrays = {k: npz[k] for k in npz.files}
        arrays['lats'] = arrays['lats'][::-1].copy()
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        with pytest.raises(ValueError, match='ascending'):
            app._load_wave_npz(buf.getvalue(), app._WAVE_TILE_KEYS, frames=3)


class TestBasinFromStore:
    def test_matches_the_client_contract(self):
        app._wave_basin = _mk_store()
        result = app._basin_from_wave_store()
        assert result is not None
        # frames already inside the forecast window are dropped (the store's
        # first frame here is one hour old), never padded
        assert 0 < len(result['times']) <= app.WAVE_BASIN_CLIENT_FRAMES
        assert len(result['lats']) == 60 and len(result['lons']) == 120, (
            "the wire must carry the 3-degree client grid, not the 2-degree "
            "analysis grid -- 2 degrees measured 3.0MB brotli / 28MB parsed")
        # ISO minute-precision Z times, parseable by Date.parse
        datetime.strptime(result['times'][0], '%Y-%m-%dT%H:%MZ')
        # lats ascending and evenly spaced -- the client indexes by
        # (last-first)/(n-1) and silently misplaces pixels otherwise
        lats = result['lats']
        steps = {round(b - a, 3) for a, b in zip(lats, lats[1:])}
        assert len(steps) == 1 and lats[0] < lats[-1]
        # lons are signed VALUES in stored 0..360 ORDER: starts near 0, wraps
        lons = result['lons']
        assert -180 <= min(lons) and max(lons) <= 180
        assert lons[0] < 90 and lons[0] >= 0, "stored order must be preserved"
        assert any(v < 0 for v in lons), "western hemisphere missing"
        # land is 0.0, not NaN/None
        assert result['wave_height'][0][0][0] == 0.0
        # all five contract fields present (wind filtering is the endpoint's
        # job, not the producer's)
        for key in ('wave_height', 'wave_period', 'wave_direction',
                    'wind_speed', 'wind_direction'):
            assert key in result

    def test_stale_pipeline_returns_none(self):
        """built_at is the freshness signal: the newest complete CYCLE is
        legitimately 5-11h old, but a manifest not rebuilt in 12h means the
        workflow is dead and the ERDDAP fallback should take over."""
        app._wave_basin = _mk_store(age_s=app.WAVE_ARTIFACT_MAX_AGE_S + 60)
        assert app._basin_from_wave_store() is None

    def test_get_ocean_basin_data_prefers_the_store(self, monkeypatch):
        app._wave_basin = _mk_store()

        def no_erddap(*a, **kw):
            raise AssertionError("ERDDAP touched while the store is fresh")
        monkeypatch.setattr(app, '_fetch_erddap_grid_chain', no_erddap)
        result = app.get_ocean_basin_data()
        assert result is not None and result.get('source') == 'gfswave'


class TestTiles:
    def _mk_tile(self, lat0=20, lon0=-80, frames=121):
        lats = np.arange(lat0 - 2.5, lat0 + 22.501, 0.25, np.float32)
        lons = np.arange(lon0 - 2.5, lon0 + 22.501, 0.25, np.float32)
        now = time.time()
        tile = {
            'times': np.array([now - 3600 + i * 3600 for i in range(frames)],
                              np.int64),
            'lats': lats, 'lons': lons,
        }
        for key in app._WAVE_TILE_KEYS:
            tile[key] = np.full((frames, len(lats), len(lons)), 2.0, np.float16)
        return tile

    def test_every_shipped_spot_bbox_fits_one_tile(self):
        """The 2.5-degree halo exists so the +-1.5/+-2.0-degree map-forecast
        bbox around any covered spot stays inside the tile that contains the
        spot. If a spot ever lands close enough to a tile edge that its bbox
        leaks past the halo, the local map silently falls back to ERDDAP."""
        import json as _json
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw = _json.load(open(os.path.join(root, 'surf_cameras.json')))
        entries = raw if isinstance(raw, list) else list(raw.values())[0]
        for c in entries:
            lat, lon = float(c['lat']), float(c['lon'])
            lat0, lon0 = app._tile_key_for(round(lat, 2), round(lon, 2))
            assert lat0 - 2.5 <= lat - 1.5 and lat + 1.5 <= lat0 + 22.5, c['name']
            assert lon0 - 2.5 <= lon - 2.0 and lon + 2.0 <= lon0 + 22.5, c['name']

    def test_grid_from_tile_slices_the_bbox(self, monkeypatch):
        app._wave_basin = _mk_store()
        tile = self._mk_tile()
        monkeypatch.setattr(app, '_load_wave_tile', lambda lat0, lon0: tile)
        result = app._grid_from_wave_tile(32.93, 35.93, -79.55, -75.55)
        assert result is not None
        assert result['source'] == 'gfswave'
        assert result['lats'][0] >= 32.93 and result['lats'][-1] <= 35.93
        # hourly cadence
        t0 = datetime.strptime(result['times'][0], '%Y-%m-%dT%H:%MZ')
        t1 = datetime.strptime(result['times'][1], '%Y-%m-%dT%H:%MZ')
        assert (t1 - t0).total_seconds() == 3600

    def test_bbox_leaking_past_the_halo_returns_none(self, monkeypatch):
        app._wave_basin = _mk_store()
        tile = self._mk_tile()
        monkeypatch.setattr(app, '_load_wave_tile', lambda lat0, lon0: tile)
        assert app._grid_from_wave_tile(16.0, 19.0, -79.0, -75.0) is None

    def test_tile_lru_caps_and_keys_by_cycle(self, monkeypatch):
        app._wave_basin = _mk_store()
        cycle = app._wave_basin['cycle']
        payloads = {}

        def fake_fetch(rel, timeout=60):
            buf = io.BytesIO()
            tile = self._mk_tile()
            np.savez_compressed(buf, **tile)
            payloads[rel] = payloads.get(rel, 0) + 1
            return buf.getvalue()
        monkeypatch.setattr(app, '_wave_fetch', fake_fetch)
        for i in range(6):
            app._load_wave_tile(20, -180 + i * 20)
        assert len(app._wave_tiles) == app.WAVE_TILE_LRU
        assert all(k[0] == cycle for k in app._wave_tiles)


class TestWindBackfill:
    def test_prefers_tile_over_erddap(self, monkeypatch):
        app._wave_basin = TestBasinFromStore and _mk_store()
        tile = TestTiles()._mk_tile()
        monkeypatch.setattr(app, '_load_wave_tile', lambda lat0, lon0: tile)

        def no_erddap(*a, **kw):
            raise AssertionError("ERDDAP wind fetch ran with a warm tile")
        monkeypatch.setattr(app, '_enrich_wind_from_erddap', no_erddap)
        now_iso = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')
        forecast = [{'time': now_iso, 'wind_speed': None, 'wind_direction': None}]
        assert app._enrich_wind_from_tile(forecast, 34.43, -77.55) is True
        assert forecast[0]['wind_speed'] == 2.0

    def test_hours_past_the_tile_horizon_stay_untouched(self, monkeypatch):
        app._wave_basin = _mk_store()
        tile = TestTiles()._mk_tile(frames=5)     # 5-hour horizon
        monkeypatch.setattr(app, '_load_wave_tile', lambda lat0, lon0: tile)
        far = datetime.fromtimestamp(time.time() + 6 * 86400, timezone.utc) \
            .strftime('%Y-%m-%dT%H:%MZ')
        forecast = [{'time': far, 'wind_speed': None, 'wind_direction': None}]
        app._enrich_wind_from_tile(forecast, 34.43, -77.55)
        assert forecast[0]['wind_speed'] is None, (
            "an hour past the horizon must not be filled with the last frame")


def test_wave_sync_default_inherits_ssr_warm():
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'app.py')).read()
    assert "'WAVE_SYNC', os.environ.get('SSR_WARM'" in src


def test_deep_size_counts_numpy_nbytes():
    arr = np.zeros((100, 100), np.float32)
    assert app._deep_size(arr) >= arr.nbytes
