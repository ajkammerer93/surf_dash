"""The directional spectrum is filtered to a swell band window.

NDBC and CDIP publish energy down to about 2.5 s. On an ordinary day the largest
value in the whole directional grid lives in that short-period tail -- it is
local wind chop, it is broadband in direction, and the polar plot normalises its
colour ramp to the grid maximum. Drawing it meant the swell bands, the only
reason a surfer opens the plot, rendered in the dark end of the ramp under a
saturated blob in the middle of the disc.

Two things are pinned here, because either one alone leaves the bug half-fixed:

1. THE BANDS ARE DROPPED BEFORE `maxEnergy` IS TAKEN, not at draw time. Skipping
   the wedge but still folding the value into the maximum keeps the colour bar
   scaled to chop, which was the actual complaint.

2. THE RADIAL AXIS SPANS THE SAME WINDOW. The old mapping was `period / 25`, so
   the centre was 0 s and the inner fifth of the radius drew periods that are now
   filtered out -- an empty hole, and 20% of the plot area spent on nothing.
   Every radius must go through `spectrumRadius`, and the hover tooltip must
   invert it with `spectrumPeriodAt`; a tooltip left on the old linear-from-zero
   mapping reports a period that does not match the band it is pointing at.

Both the buoy panel and the share-card renderer build this grid, from two copies
of the same loop. A fix applied to one of them is the failure mode this file
exists to catch.

The 1D energy spectrum is deliberately NOT filtered: it is a line chart, the chop
hump costs nothing to read past, and its height is informative.
"""
import json
import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "templates", "index.html")


def _src():
    with open(INDEX) as f:
        return f.read()


def _balanced(src, marker, opener="{", closer="}"):
    """Text of the balanced block starting at the first opener at/after marker."""
    start = src.index(marker)
    i = src.index(opener, start)
    depth, j = 0, i
    while j < len(src):
        if src[j] == opener:
            depth += 1
        elif src[j] == closer:
            depth -= 1
            if depth == 0:
                return src[i:j + 1]
        j += 1
    raise AssertionError("unbalanced block after " + marker)


class TestBandWindow:
    def test_window_constants_are_five_to_twentyfive(self):
        src = _src()
        assert re.search(r"const SPECTRUM_MIN_PERIOD = 5\b", src)
        assert re.search(r"const SPECTRUM_MAX_PERIOD = 25\b", src)

    def test_both_grid_builders_skip_out_of_window_bands(self):
        """Panel plot and share card. Two copies of the loop, one window."""
        src = _src()
        guard = ("if (bandPeriod < SPECTRUM_MIN_PERIOD || "
                 "bandPeriod > SPECTRUM_MAX_PERIOD) continue;")
        assert src.count(guard) == 2, (
            "expected the band guard in both render2DSpectrum and the share-card "
            "renderer, found %d" % src.count(guard))

    def test_the_guard_precedes_the_energy_maximum(self):
        """Filtering at draw time would leave the colour bar scaled to chop."""
        body = _balanced(_src(), "function render2DSpectrum(")
        guard = body.index("bandPeriod < SPECTRUM_MIN_PERIOD")
        maxima = body.index("if (val > maxEnergy) maxEnergy = val;")
        assert guard < maxima


class TestRadialAxis:
    def test_no_period_over_twentyfive_radius_mapping_survives(self):
        """The old centre-is-zero mapping, in any of its five spellings."""
        src = _src()
        assert not re.search(r"\(\s*periodNext?\s*/\s*25\s*\)", src)
        assert not re.search(r"\(\s*p\s*/\s*25\s*\)\s*\*\s*(maxR|specR)", src)
        assert not re.search(r"\(\s*dist\s*/\s*d\.maxR\s*\)\s*\*\s*25", src)

    def test_rings_start_at_ten_not_five(self):
        """5 s is the centre point now, so a 5 s ring would be a dot."""
        src = _src()
        assert src.count("var ringPeriods = [10, 15, 20, 25];") == 2
        assert "var ringPeriods = [5, 10, 15, 20];" not in src

    def test_tooltip_inverts_the_same_mapping(self):
        assert "var period = spectrumPeriodAt(dist, d.maxR);" in _src()

    def test_tooltip_indexes_retained_bands_not_raw_frequencies(self):
        """`freqs` includes the dropped bands; indexing it against the filtered
        grid would read the wrong row, or run off the end of it."""
        src = _src()
        assert "periods: periods," in src
        assert "freqs: freqs," not in src
        assert "for (var i = 0; i < d.periods.length; i++)" in src


class TestOneDimensionalSpectrumUntouched:
    def test_1d_chart_keeps_the_full_range(self):
        body = _balanced(_src(), "function render1DSpectrum(")
        assert "SPECTRUM_MIN_PERIOD" not in body
        assert "SPECTRUM_MAX_PERIOD" not in body


class TestLabel:
    def test_the_plot_names_the_window_it_shows(self):
        assert "swell only, 5&ndash;25s" in _src()


NODE = shutil.which("node")

# Stubs the handful of DOM and theme calls render2DSpectrum makes on its way to
# building the grid. Nothing here draws: the assertions read `_polarData`, which
# is the same object the hover tooltip reads.
HARNESS = """
const ctx = new Proxy({}, {
    get: (t, k) => (k in t ? t[k] : (t[k] = () => {})),
    set: (t, k, v) => (t[k] = v, true),
});
ctx.measureText = () => ({ width: 10 });
const canvas = { width: 0, height: 0, style: {}, getContext: () => ctx,
                 parentElement: { clientWidth: 300, clientHeight: 300 } };
global.document = { getElementById: () => canvas };
global.window = { devicePixelRatio: 1 };
global.isDarkTheme = () => true;
global.themeColors = () => ({ canvasBg: '#000' });
"""

# A groundswell at 14 s under a wind-chop peak at 3.5 s that is 7x its height.
# This is an ordinary summer afternoon at an Atlantic buoy, not a corner case.
FIXTURE = """
const freqs = [], energy = [];
for (let f = 0.033; f <= 0.486; f += 0.01) {
    freqs.push(+f.toFixed(4));
    const p = 1 / f;
    energy.push(1.2 * Math.exp(-Math.pow(p - 14, 2) / 4) +
                9.0 * Math.exp(-Math.pow(p - 3.5, 2) / 0.8));
}
const directional = { directions: freqs.map(() => 45), r1: freqs.map(() => 0.6) };
render2DSpectrum(directional, { frequencies: freqs, energy: energy });
const d = document.getElementById()._polarData;
let peak = 0, bi = -1;
d.periods.forEach((p, i) => {
    const v = Math.max.apply(null, d.grid[i]);
    if (v > peak) { peak = v; bi = i; }
});
console.log(JSON.stringify({
    bands: d.periods.length,
    minPeriod: Math.min.apply(null, d.periods),
    maxPeriod: Math.max.apply(null, d.periods),
    peakPeriod: d.periods[bi],
    peakFraction: peak / d.maxEnergy,
}));
"""


def _js_function(src, marker):
    """Source of the function declaration starting at marker."""
    i = src.index(marker)
    j = src.index("{", i)
    depth, k = 0, j
    while True:
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
        k += 1


@pytest.mark.skipif(NODE is None, reason="node not available")
class TestHelperArithmetic:
    """The helpers are three lines each and pure, so run them for real."""

    def _eval(self, expr):
        src = _src()
        start = src.index("const SPECTRUM_MIN_PERIOD")
        end = src.index("// === Theme Toggle ===")
        prelude = src[start:end]
        out = subprocess.run(
            [NODE, "-e", prelude + "\nconsole.log(JSON.stringify(" + expr + "));"],
            capture_output=True, text=True, check=True)
        return json.loads(out.stdout)

    def test_centre_is_the_minimum_period(self):
        assert self._eval("spectrumRadius(SPECTRUM_MIN_PERIOD, 100)") == 0

    def test_rim_is_the_maximum_period(self):
        assert self._eval("spectrumRadius(SPECTRUM_MAX_PERIOD, 100)") == 100

    def test_a_swell_band_lands_where_the_ring_says(self):
        """15 s is halfway between 5 and 25, so it must sit on the middle ring."""
        assert self._eval("spectrumRadius(15, 100)") == 50

    def test_the_tooltip_mapping_is_the_exact_inverse(self):
        got = self._eval(
            "[8, 12, 17].map(function(p) { "
            "return spectrumPeriodAt(spectrumRadius(p, 240), 240); })")
        assert got == pytest.approx([8, 12, 17])


@pytest.mark.skipif(NODE is None, reason="node not available")
class TestColourBarScaling:
    """render2DSpectrum, run for real against a chop-dominated spectrum.

    Every pin above can hold while the plot is still wrong; this is the one that
    checks the thing the user actually sees. On master the colour-bar maximum was
    3.147 (the 3.5 s chop band) and the 14 s groundswell rendered at fraction
    0.131 -- the dark-blue bottom of the ramp.
    """

    def _run(self, tmp_path):
        src = _src()
        i = src.index("const SPECTRUM_MIN_PERIOD")
        j = src.index("// === Theme Toggle ===")
        script = tmp_path / "spec.js"
        script.write_text(
            HARNESS + src[i:j] + "\n"
            + _js_function(src, "function spectrumColor(") + "\n"
            + _js_function(src, "function render2DSpectrum(") + "\n"
            + FIXTURE)
        out = subprocess.run([NODE, str(script)], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        return json.loads(out.stdout)

    def test_the_colour_bar_tops_out_on_the_swell_not_the_chop(self, tmp_path):
        r = self._run(tmp_path)
        assert r["peakPeriod"] == pytest.approx(14, abs=1.0)
        assert r["peakFraction"] == pytest.approx(1.0)

    def test_no_band_outside_the_window_reaches_the_grid(self, tmp_path):
        r = self._run(tmp_path)
        assert r["minPeriod"] >= 5
        assert r["maxPeriod"] <= 25
        assert r["bands"] > 8, "the window must not starve the plot of resolution"
