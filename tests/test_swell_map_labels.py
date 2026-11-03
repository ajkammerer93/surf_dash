"""Swell map spot-label placement (resolveSpotLabelCollisions).

Review of the first version (PR #110) found four user-facing defects in a
resolver whose overlap arithmetic was itself correct: a label pushed into a
neighbour's invisible icon box clicked through to the neighbour's popup;
labels at the default zoom were pushed under the storm-watch narrative bar;
on a north-south coastline the push-down cascade left labels 150px from
their dots; and the zoom/pan rerun was wired only on the normal map, not
the one every visitor gets between a deploy and the basin warming. These
pin the shape of the fix. Behaviour is verified in a browser, not here.
"""
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def html():
    with open(os.path.join(REPO, 'templates', 'index.html')) as f:
        return f.read()


@pytest.fixture(scope='module')
def resolver(html):
    start = html.index('function resolveSpotLabelCollisions()')
    return html[start:html.index('function updateLocalCamMarkers(', start)]


class TestHitTesting:
    def test_only_the_dot_and_label_take_pointer_events(self, html):
        assert '.leaflet-marker-icon.spot-marker.leaflet-interactive { pointer-events: none; }' in html
        assert re.search(r'\.spot-marker \.spot-marker-dot,\s*\.spot-marker \.spot-marker-label \{ pointer-events: auto; \}', html)


class TestPlacement:
    def test_slots_are_bounded_in_label_heights(self, html):
        assert re.search(r'const SPOT_LABEL_SLOTS = \[\[0, 1\], \[0, -1\], \[1, 1\], \[1, -1\], \[2, 1\], \[-2, 1\], \[-2, -1\]\];', html)
        # one row up is never a slot: it lands on the dot itself
        assert '[-1, 1]' not in html[html.index('const SPOT_LABEL_SLOTS'):][:120]

    def test_unplaceable_labels_hide_instead_of_drifting(self, resolver):
        assert "it.label.style.visibility = 'hidden'" in resolver
        # and every run starts from a clean slate
        assert "label.style.visibility = '';" in resolver
        assert "label.style.transform = '';" in resolver

    def test_controls_narrative_and_map_edges_are_obstacles(self, resolver):
        assert "container.querySelectorAll('.leaflet-control, .basin-narrative')" in resolver
        assert 'cand.top < bounds.top || cand.bottom > bounds.bottom' in resolver
        assert 'cand.left < bounds.left || cand.right > bounds.right' in resolver

    def test_every_other_marker_is_an_obstacle(self, resolver):
        # cameras, buoys and the current-spot star, by the class Leaflet
        # gives every marker icon -- not a hand-kept list of layers
        assert "container.querySelectorAll('.leaflet-marker-icon:not(.spot-marker)')" in resolver
        assert 'SPOT_OBSTACLE_INSET' in resolver

    def test_narrative_show_and_hide_rerun_the_resolver(self, html):
        start = html.index("closeBtn.className = 'basin-narrative-close'")
        assert 'requestAnimationFrame(resolveSpotLabelCollisions)' in html[start:start + 400]
        shown = html.index("// The bar is an obstacle for the spot labels")
        assert 'requestAnimationFrame(resolveSpotLabelCollisions)' in html[shown:shown + 300]


class TestHiddenLabelSpotsStayReachable:
    def test_spot_dot_paints_above_its_camera(self, html):
        start = html.index("className: 'spot-marker',")
        assert 'zIndexOffset: 1000' in html[start - 400:start]

    def test_camera_popup_offers_the_co_located_spot(self, html):
        start = html.index('function updateLocalCamMarkers(')
        body = html[start:html.index('function updateSwellBuoyMarkers(', start)]
        assert "sw.textContent = 'Switch to this spot';" in body
        assert 'Math.abs(sp.lat - cam.lat) < 1e-3' in body

    def test_mid_zoom_animation_is_skipped(self, resolver):
        assert 'if (mapSwell._animatingZoom) return;' in resolver

    def test_reads_are_batched_after_resets(self, resolver):
        # all transform resets happen in the first eachLayer pass, before
        # any getBoundingClientRect
        first_read = resolver.index('getBoundingClientRect')
        assert resolver.index("label.style.transform = '';") < first_read


class TestRerunWiring:
    def test_rerun_is_wired_from_the_marker_builder(self, html):
        start = html.index('function updateLocalSpotMarkers(')
        body = html[start:html.index('function wireSpotLabelRerun(', start)]
        assert 'wireSpotLabelRerun(mapSwell);' in body

    def test_rerun_is_wired_once_per_map(self, html):
        start = html.index('function wireSpotLabelRerun(')
        body = html[start:start + 600]
        assert 'map._spotLabelRerunWired' in body
        assert "map.on('zoomend moveend'" in body
        # the old direct registration in renderSwellMap is gone
        assert html.count("mapSwell.on('zoomend moveend'") == 0
