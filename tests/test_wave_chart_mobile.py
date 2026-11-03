"""Wave chart on a phone: day-of-week labels must survive, and a tapped
tooltip must be dismissable.

Chart.js runs autoSkip on every time scale regardless of ticks.autoSkip,
keeping every Nth tick from the FIRST one. On a 360px phone that kept the
06:00/18:00 ticks and dropped every noon tick, so the axis carried no day of
the week at all (reported 2026-11-03). On touch there is no mouseout, so a
tapped tooltip sat over the chart until the next tap.
"""
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def html():
    with open(os.path.join(REPO, 'templates', 'index.html')) as f:
        return f.read()


def _fn(html, name, span=3000):
    start = html.index('function ' + name)
    return html[start:start + span]


class TestDayLabelsOnNarrowAxes:
    def test_tick_density_follows_pixels_per_day(self, html):
        body = _fn(html, 'buildLocalTimeTicks')
        assert 'pxPerDay' in body
        assert 'axis._compactDayLabels = pxPerDay < 60' in body
        # Density thresholds are sized to a rotated label (~27px per tick),
        # not the viewport: four ticks a day from 110 px/day, two from 55.
        assert 'if (pxPerDay < 55) minor = false;' in body
        assert 'else if (pxPerDay < 110) minor = (localHour === 0);' in body
        # Noon (the day label) is emitted at every density
        assert "localHour === 12" in body

    def test_compact_labels_keep_the_weekday(self, html):
        body = _fn(html, 'formatLocalTimeTick', 800)
        assert "this._compactDayLabels" in body
        assert "formatLocationTime(d, { weekday: 'short' }) + ' ' + parseInt(getLocationParts(d).day, 10)" in body

    def test_majors_survive_chartjs_autoskip_on_both_time_axes(self, html):
        # One for the wave chart, one for the tide chart
        assert len(re.findall(r"major:\s*\{\s*enabled:\s*true\s*\}", html)) == 2


class TestTooltipDismiss:
    def test_tooltip_builds_a_close_button(self, html):
        start = html.index('function waveChartExternalTooltip')
        body = html[start:html.index('// Compute time range for sync', start)]
        assert "closeBtn.className = 'tt-close'" in body
        assert "clearSyncedTooltip(chart)" in body
        assert "el.classList.add('tt-hidden')" in body
        # The canvas mouseleave path must set the same hidden state
        ml = html[html.index("getElementById('waveChart').addEventListener('mouseleave'"):]
        assert "ttEl.classList.add('tt-hidden')" in ml[:400]
        # No stopPropagation: the document's outside-click closers must still run
        assert 'ev.stopPropagation()' not in body[body.index("closeBtn.addEventListener"):body.index("el.appendChild(closeBtn)")]

    def test_close_button_is_touch_only_and_the_only_hit_target(self, html):
        assert '.wave-tooltip .tt-close {' in html
        # Hidden is visibility: hidden, so the button is neither a hit
        # target nor a focus stop while the tooltip is faded out.
        hidden = html[html.index('.wave-tooltip.tt-hidden {'):]
        assert 'visibility: hidden;' in hidden[:120]
        hover_rule = html[html.index('@media (hover: hover) and (pointer: fine)'):]
        assert '.wave-tooltip .tt-close { display: none; }' in hover_rule[:300]
