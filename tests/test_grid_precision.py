"""Tests for the display-precision contract on the gridded map fields.

The failure this guards against is invisible in every other way. The five grids
serialize straight to JSON, and until v0.11.69 they went out as raw float64:
17-18 characters per value to express a wind direction drawn as an arrow rotated
to the nearest degree. That made the two wind fields two-thirds of a 19.6 MB
response, and it broke no test, changed no pixel, and produced no error -- it
just quietly cost 4.9 MB of every visitor's bandwidth.

So the thing worth pinning is not the arithmetic, which is obvious, but that all
three grid producers keep going through the shared helper. A future refactor that
reaches for a bare .tolist() again would restore the bytes silently.
"""
import json
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as A  # noqa: E402

APP_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app.py')
GRID_FIELDS = ('wave_height', 'wave_period', 'wave_direction',
               'wind_speed', 'wind_direction')


def test_every_grid_field_has_a_declared_precision():
    assert set(A.GRID_ROUNDING) == set(GRID_FIELDS)


def test_directions_are_whole_numbers_not_floats():
    """`0` costs one byte; `0.0` costs three, on 318,240 values per field.

    Directions are drawn as rotated arrows, so a degree is already finer than the
    glyph can express and the fractional part is pure payload.
    """
    grid = np.array([[[357.334826224105, 0.0, 21.4363594]]])
    for name in ('wave_direction', 'wind_direction'):
        out = A._round_grid(grid, A.GRID_ROUNDING[name])
        assert all(isinstance(v, int) for v in out[0][0]), name
        assert out[0][0] == [357, 0, 21]


def test_rounding_stays_within_its_own_tolerance():
    rng = np.random.default_rng(0)
    grid = rng.uniform(0, 360, size=(3, 4, 5))
    for name, decimals in A.GRID_ROUNDING.items():
        out = np.array(A._round_grid(grid, decimals), dtype=float)
        tolerance = 0.5 if decimals == 0 else 0.5 * 10 ** -decimals
        assert np.abs(out - grid).max() <= tolerance + 1e-9, name


def test_grid_fields_returns_exactly_the_five_fields():
    grid = np.zeros((1, 1, 1))
    assert set(A._grid_fields(grid, grid, grid, grid, grid)) == set(GRID_FIELDS)


def test_nan_free_grids_survive_the_round_trip():
    """Land arrives as 0.0 (nan_to_num) and must stay a plain zero.

    The client re-hides zeros via maskZero, so a NaN or None leaking through here
    would render as a hole in the ocean rather than as land.
    """
    grid = np.array([[[0.0, 1.5, 0.0]]])
    out = A._round_grid(grid, 2)
    assert out[0][0][0] == 0.0
    assert json.dumps(out) == '[[[0.0, 1.5, 0.0]]]'


def test_no_grid_producer_bypasses_the_shared_helper():
    """The regression that would silently restore several megabytes.

    All five producers -- NOMADS local, ERDDAP local, the ERDDAP basin, and
    the two wave-store producers (basin and tile) -- must build their field
    block with _grid_fields rather than calling .tolist() on the arrays
    directly.

    Both call shapes count. The original pin grepped only for
    '**_grid_fields(' and the wave-store producers used
    '.update(_grid_fields(...))' -- same helper, same precision, invisible to
    the count. A pin that can be satisfied by accident is not pinning.
    """
    src = open(APP_SRC).read()
    offenders = re.findall(r'"(?:wave_height|wave_period|wave_direction|'
                           r'wind_speed|wind_direction)":\s*\w+\.tolist\(\)', src)
    assert offenders == [], f"raw .tolist() on a grid field: {offenders}"
    # and the helper really is used by all five producers, in either form
    uses = src.count('**_grid_fields(') + src.count('.update(_grid_fields(')
    assert uses == 5, f"expected 5 producers through _grid_fields, found {uses}"


def test_precision_actually_shrinks_a_realistic_payload():
    """Guards the size claim itself, not just the rounding.

    A change that kept the tolerances but serialized differently (say, back to
    float64 after rounding) would pass every test above and still be large.
    """
    rng = np.random.default_rng(1)
    shape = (8, 20, 30)
    speed = rng.uniform(0, 30, size=shape)
    direction = rng.uniform(0, 360, size=shape)
    raw = json.dumps({
        'wind_speed': speed.tolist(), 'wind_direction': direction.tolist(),
    }, separators=(',', ':'))
    rounded = json.dumps({
        'wind_speed': A._round_grid(speed, A.GRID_ROUNDING['wind_speed']),
        'wind_direction': A._round_grid(direction, A.GRID_ROUNDING['wind_direction']),
    }, separators=(',', ':'))
    # measured on the live payload: the wind pair goes from ~17.5 to ~4.5 B/value
    assert len(rounded) < 0.45 * len(raw), (len(rounded), len(raw))
