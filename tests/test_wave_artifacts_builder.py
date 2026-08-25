"""Units for scripts/build_wave_artifacts.py that need no network or eccodes.

The builder rewrites a data branch unattended; the failure that matters is a
build that half-works and publishes wrong geometry -- the site would keep
serving, with rasters quietly misplaced, and nothing would look wrong.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _module():
    path = os.path.join(ROOT, 'scripts', 'build_wave_artifacts.py')
    spec = importlib.util.spec_from_file_location('build_wave_artifacts', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IDX_SAMPLE = """1:0:d=2026082506:WIND:surface:48 hour fcst:
2:703753:d=2026082506:WDIR:surface:48 hour fcst:
5:3126315:d=2026082506:HTSGW:surface:48 hour fcst:
9:5417242:d=2026082506:SWELL:1 in sequence:48 hour fcst:
19:11258217:d=2026082506:SWDIR:3 in sequence:48 hour fcst:
"""


class TestIdxParsing:
    def test_offsets_become_ranges(self):
        m = _module()
        entries = m.parse_idx(IDX_SAMPLE)
        ranges = m.wanted_ranges(entries, [('WIND', 'surface', 'wind'),
                                           ('HTSGW', 'surface', 'htsgw')])
        assert ranges['wind'] == (0, 703752)
        assert ranges['htsgw'] == (3126315, 5417241)

    def test_final_band_runs_to_end_of_file(self):
        m = _module()
        entries = m.parse_idx(IDX_SAMPLE)
        ranges = m.wanted_ranges(entries, [('SWDIR', '3 in sequence', 'sw3d')])
        assert ranges['sw3d'] == (11258217, None)

    def test_partition_levels_are_distinguished(self):
        """SWELL appears three times differing only by level text; matching
        on variable name alone would hand partition 1 to all three keys."""
        m = _module()
        entries = m.parse_idx(IDX_SAMPLE)
        got = m.wanted_ranges(entries, [('SWELL', '1 in sequence', 'sw1h'),
                                        ('SWELL', '2 in sequence', 'sw2h')])
        assert 'sw1h' in got and 'sw2h' not in got


class TestPooling:
    def test_block_mean_for_scalars(self):
        m = _module()
        grid = np.full((720, 1440), 2.0, np.float32)
        pooled = m.pool_basin(grid, 'htsgw').astype(np.float32)
        assert pooled.shape == (90, 180)
        assert np.allclose(pooled, 2.0, atol=0.01)

    def test_circular_mean_for_directions(self):
        """350 and 10 average to 0/360, never to 180 -- the arithmetic-mean
        bug that points every north swell south."""
        m = _module()
        grid = np.full((720, 1440), 350.0, np.float32)
        grid[:, ::2] = 10.0
        pooled = m.pool_basin(grid, 'dirpw').astype(np.float32)
        centered = np.minimum(pooled % 360.0, 360.0 - pooled % 360.0)
        assert float(np.nanmax(centered)) < 1.0

    def test_mostly_land_blocks_pool_to_nan(self):
        m = _module()
        grid = np.full((720, 1440), np.nan, np.float32)
        grid[0:8, 0] = 1.0          # 1 of 64 cells wet: below the 25% floor
        pooled = m.pool_basin(grid, 'htsgw').astype(np.float32)
        assert not np.isfinite(pooled[0, 0])

    def test_coastal_blocks_keep_their_ocean_value(self):
        m = _module()
        grid = np.full((720, 1440), np.nan, np.float32)
        grid[0:8, 0:4] = 3.0        # half the block wet
        pooled = m.pool_basin(grid, 'htsgw').astype(np.float32)
        assert pooled[0, 0] == pytest.approx(3.0, abs=0.01)


class TestTileGeometry:
    def test_axes_are_101_cells_with_halo(self):
        m = _module()
        lats, lons = m.tile_axes(20, -80)
        assert len(lats) == 101 and len(lons) == 101
        assert lats[0] == 17.5 and lats[-1] == 42.5
        assert lons[0] == -82.5 and lons[-1] == -57.5

    def test_slice_matches_full_grid_values(self):
        m = _module()
        grid = np.arange(720 * 1440, dtype=np.float32).reshape(720, 1440)
        tile = m.slice_tile(grid, 20, -80)
        # spot-check the Surf City cell: lat 34.5 row (34.5+90)/0.25=498,
        # lon -77.5 -> 282.5 -> col 1130
        lats, lons = m.tile_axes(20, -80)
        ri = int(round((34.5 - lats[0]) / 0.25))
        ci = int(round((-77.5 - lons[0]) / 0.25))
        assert tile[ri, ci] == grid[498, 1130]

    def test_every_spot_gets_a_tile(self):
        m = _module()
        tiles = set(m.spot_tiles())
        raw = json.load(open(os.path.join(ROOT, 'surf_cameras.json')))
        entries = raw if isinstance(raw, list) else list(raw.values())[0]
        for c in entries:
            lat, lon = float(c['lat']), float(c['lon'])
            lat0 = int(np.floor(lat / 20) * 20)
            lon0 = int(np.floor(lon / 20) * 20)
            assert (lat0, lon0) in tiles, c['name']


class TestFloat16Tolerance:
    def test_representative_heights_survive(self):
        m = _module()
        m.assert_float16_tolerance(
            np.array([0.01, 0.5, 2.34, 8.75, 15.5], np.float32))

    def test_a_coarse_value_fails_loudly(self):
        m = _module()
        with pytest.raises(AssertionError):
            m.assert_float16_tolerance(np.array([3000.7], np.float32))
