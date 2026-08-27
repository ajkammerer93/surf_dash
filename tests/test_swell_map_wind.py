"""Source pins for the swell map's wind-particle layer.

Every defect pinned here is invisible to a rendering test and re-introducible by
deleting a single line, which is why they are source assertions rather than
behavioural ones. They came out of an audit of the layer on 2026-08-26.

The four that matter:

1. THE PARTICLES MUST DRAW ABOVE THE FIELD RASTERS. leaflet-velocity defaults to
   Leaflet's overlayPane (z-index 400), which sits UNDER fieldBase (405). Wave
   Height is on by default, so out of the box the layer composited beneath a
   saturated ramp and read as a faint ghost -- the layer looked broken while
   being entirely functional.

2. THE SLIDER'S CLOCK IS THE LONGEST-HORIZON GRID, NOT THE FINEST. The basin is
   3-hourly out to 7 days; the local tile is hourly and stops at f120. Choosing
   the axis with `localGridData.times || basinGridData.times` picked the SHORTER
   horizon the moment the local grid landed, while slider.max still held the
   basin's frame count -- the reachable map collapsed from +168h to about +56h,
   and which of the two a user got depended on whether the pane happened to be
   in the viewport when the basin response arrived.

3. NO RAW SLIDER INDEX MAY REACH A LOCAL GRID ARRAY. Once the basin drives the
   clock, a slider position past the local tail has no local frame. The old
   `Math.min(t, times.length - 1)` clamp smeared the last local hour across the
   entire f120+ tail -- the same "frame zero impersonating other hours" mistake
   the basin-store backfill guards against server-side with its 90-minute rule.

4. LAZY LOADERS MUST NOT ASSERT VISIBILITY. loadBasinWind's success handler
   passed a literal `true` for the visible argument, resurrecting the particle
   layer after the user unticked Wind or the pane scrolled off, with nothing
   left to tear it down -- an always-on off-screen animation loop, which the
   whole map rework exists to prevent.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "templates", "index.html")


def _src():
    with open(INDEX) as f:
        return f.read()


def _balanced(src, marker, opener="{", closer="}"):
    """Text of the balanced block that starts at the first opener at/after
    marker. Regex cannot do this: every block here nests."""
    start = src.index(marker)
    i = src.index(opener, start)
    depth, j = 0, i
    while j < len(src):
        if src[j] == opener:
            depth += 1
        elif src[j] == closer:
            depth -= 1
            if depth == 0:
                return src[start:j + 1]
        j += 1
    raise AssertionError("unbalanced block after %r" % marker)


def _fn(src, name):
    return _balanced(src, "function %s(" % name)


def _velocity_options(src):
    return _balanced(src, "L.velocityLayer(")


def _decommented(text):
    """Drop // line comments. Pins that forbid an identifier must not trip over
    the comment that explains why it is forbidden."""
    return "\n".join(
        re.sub(r"//.*$", "", line) for line in text.splitlines())


def _pane_zindex(src, name):
    """z-index assigned to a Leaflet pane, however it is written."""
    inline = re.search(
        r"createPane\(['\"]%s['\"]\)\.style\.zIndex\s*=\s*(\d+)" % name, src)
    if inline:
        return int(inline.group(1))
    block = re.search(
        r"createPane\(['\"]%s['\"]\)[\s\S]{0,200}?zIndex\s*=\s*(\d+)" % name, src)
    assert block, "no z-index found for pane %r" % name
    return int(block.group(1))


class TestWindParticlePane:
    """The layer has to be ABOVE the rasters or it is invisible in its default
    state (Wave Height ships on)."""

    def test_velocity_layer_names_its_pane(self):
        src = _src()
        opts = _velocity_options(src)
        assert "paneName: 'fieldWind'" in opts, (
            "velocity layer must declare paneName: 'fieldWind' -- without it "
            "leaflet-velocity falls back to overlayPane (400), under fieldBase "
            "(405), and the particles vanish beneath the wave raster")

    def test_wind_pane_sits_between_detail_and_arrows(self):
        src = _src()
        wind = _pane_zindex(src, "fieldWind")
        detail = _pane_zindex(src, "fieldDetail")
        arrows = _pane_zindex(src, "fieldArrows")
        assert detail < wind < arrows, (
            "fieldWind (%d) must sit strictly between fieldDetail (%d) and "
            "fieldArrows (%d)" % (wind, detail, arrows))

    def test_wind_pane_created_on_both_map_setup_paths(self):
        """renderSwellMap and the basin-failure fallback both build panes; a
        pane created in only one leaves the other path with an undefined
        z-index, which is how the ordering silently regresses."""
        src = _src()
        assert len(re.findall(r"createWindPane\(", src)) >= 3, (
            "expected createWindPane to be defined once and called from BOTH "
            "renderSwellMap and swellMapDataError's fallback pane setup")

    def test_wind_pane_does_not_eat_cursor_events(self):
        src = _src()
        assert "function createWindPane(" in src, "createWindPane helper is missing"
        assert "pointerEvents" in _fn(src, "createWindPane"), (
            "the particle canvas must not intercept the cursor probe")


class TestSliderClock:
    """One clock, and it must be the axis that reaches furthest."""

    def test_slider_times_helper_exists(self):
        assert "function sliderTimes()" in _src(), (
            "axis selection must live in one accessor, not be re-derived at "
            "each call site")

    def test_axis_selection_is_not_local_or_basin(self):
        """The `local || basin` idiom is the bug: it prefers the SHORTER
        horizon. Every reader must go through sliderTimes()."""
        src = _src()
        bad = re.findall(
            r"\(localGridData && localGridData\.times\)\s*\|\|\s*"
            r"\(basinGridData && basinGridData\.times\)", src)
        assert not bad, (
            "%d call site(s) still pick the time axis with `local || basin`, "
            "which truncates the slider to the local grid's f120 horizon; use "
            "sliderTimes()" % len(bad))

    def test_slider_times_prefers_longer_horizon(self):
        """Pin the comparison itself, not just the helper's existence."""
        src = _src()
        body = _fn(src, "sliderTimes")
        assert "Date.parse" in body and ">" in body, (
            "sliderTimes must compare the two axes' LAST timestamps, not their "
            "lengths -- a 121-frame hourly grid is shorter in wall-clock than "
            "a 57-frame 3-hourly one")

    def test_local_grid_handler_recomputes_slider_max(self):
        """slider.max is written only by onMapDataReady; if the local-grid
        handler does not re-run it, max keeps the basin's frame count while the
        clock moves to the local axis."""
        src = _src()
        handler = re.search(
            r"localGridData = gridData;[\s\S]{0,900}?stale_at", src)
        assert handler, "local-grid success handler not found"
        assert "onMapDataReady()" in handler.group(0), (
            "the local-grid success handler must call onMapDataReady() so "
            "slider.max is re-derived once both grids exist")

    def test_onmapdataready_preserves_wall_clock_position(self):
        src = _src()
        body = _fn(src, "onMapDataReady")
        assert "frameIndexForTime" in body, (
            "re-running onMapDataReady must remap the user's current position "
            "by timestamp, not reset the slider to 0 under them")


class TestNoRawLocalIndex:
    """A slider position is not a local frame index."""

    def test_local_frame_resolver_exists(self):
        src = _src()
        assert "function localFrameForSlider(" in src
        assert "return -1" in _fn(src, "localFrameForSlider"), (
            "must signal 'no local frame covers this time' rather than clamp")

    def test_no_clamped_index_into_local_arrays(self):
        """`Math.min(idx, localGridData.times.length - 1)` is the smear."""
        src = _src()
        bad = re.findall(
            r"Math\.min\([^)]*,\s*localGridData\.times\.length\s*-\s*1\s*\)", src)
        assert not bad, (
            "%d raw-index clamp(s) into the local grid remain; resolve the "
            "frame with localFrameForSlider() instead -- a clamp makes the "
            "last local hour impersonate every hour past f120" % len(bad))


class TestLazyWindLoader:
    """The pause paths and the resume paths must agree."""

    def test_load_basin_wind_does_not_force_visible(self):
        src = _src()
        body = _fn(src, "loadBasinWind")
        # NB: the argument list nests parens -- `(parseInt(x) || 0)` -- so a
        # [^)]* character class silently never matches and the pin is vacuous.
        assert not re.search(r"updateSwellVelocity\([\s\S]*?,\s*true\s*\)", body), (
            "loadBasinWind must not assert visibility -- a fetch in flight "
            "would resurrect the particle layer after the Wind box was "
            "unticked or the pane scrolled off, with nothing left to stop it. "
            "Route through updateSwellMap, as loadBasinSwell does.")

    def test_load_basin_wind_keeps_its_own_time_axis(self):
        """The wind response is assembled independently and its start frame
        moves with wall-clock time, so it may not share the wave axis."""
        src = _src()
        assert "wind_times" in _fn(src, "loadBasinWind"), (
            "loadBasinWind must retain data.times as wind_times so wind frames "
            "resolve against their own axis")

    def test_build_velocity_data_guards_the_frame(self):
        src = _src()
        body = _fn(src, "buildVelocityData")
        assert "wind_times" in body, (
            "wind frames must resolve against wind_times when present")
        assert re.search(r"if \(!speedFrame \|\| !dirFrame\) return null;", body), (
            "a diverged wind axis can index past the end; return null (which "
            "updateSwellVelocity already treats as 'nothing to draw') rather "
            "than throwing out of updateSwellMap")

    def test_stale_source_layer_torn_down_before_early_return(self):
        """Zooming out before basin wind arrives used to return with the
        local-extent layer still attached and animating over a world view."""
        src = _src()
        body = _fn(src, "updateSwellVelocity")
        teardown = body.find("_fsfSource !== sourceName")
        early_return = body.find("loadBasinWind()")
        assert teardown != -1 and early_return != -1
        assert teardown < early_return, (
            "the _fsfSource mismatch teardown must run BEFORE the "
            "loadBasinWind() early return, or a layer built for the local "
            "extent keeps animating over the basin view")


class TestReadoutConventions:
    """The readout must not quietly disagree with the rest of the dashboard."""

    def test_angle_convention_is_meteorological(self):
        src = _src()
        assert "angleConvention: 'meteoCW'" in _velocity_options(src), (
            "leaflet-velocity defaults to a bearing (where the wind is GOING); "
            "every other wind direction on this site is meteorological (where "
            "it comes FROM). meteoCCW is mirrored against our u/v signs.")

    def test_speed_unit_follows_the_site_preference(self):
        src = _src()
        opts = _velocity_options(src)
        assert "speedUnit" in opts and "METRIC_UNITS" in opts, (
            "the readout defaulted to m/s, a unit that appears nowhere else on "
            "the dashboard (mph, or km/h in metric mode)")

    def test_readout_is_not_under_the_narrative_bar(self):
        """.basin-narrative spans the full width at the map bottom and is
        re-appended after Leaflet builds its controls, so it paints over
        anything in the bottom-left corner."""
        src = _src()
        # The control reads `position`/`emptyString`. The displayPosition /
        # displayEmptyString spellings are not options and were silently
        # ignored -- pin that they never come back, or the readout drifts to
        # the default bottomleft under the narrative bar again.
        opts = _decommented(_velocity_options(src))
        assert "displayPosition" not in opts and "displayEmptyString" not in opts, (
            "displayPosition/displayEmptyString are not leaflet-velocity "
            "options; use position/emptyString")
        assert re.search(r"position:\s*'topright'", opts), (
            "the velocity readout must not sit in bottomleft, where the basin "
            "narrative bar spans the full width and paints over it -- worst in "
            "the zoomed-out state where the particles are the main content")
        assert "emptyString:" in opts, (
            "without emptyString the control shows the library default, "
            "'Unavailable', which reads as a broken layer")
