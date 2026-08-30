"""Tests for NOAA tide station selection.

These exist because the site shipped visibly wrong tide times for months and
nothing caught it. find_nearest_tide_station() filtered NOAA's station list to
Reference stations (type='R') on the reasoning, written in the code, that
Subordinate stations "may not work with the predictions API". That is not true
-- subordinate stations serve the same product=predictions endpoint, verified
against 20 random ones nationally and all 77 that the curated spots would
newly select -- and the filter discarded 2,243 of NOAA's 3,499 stations.

The failure was not "the station is a bit further away". A spot's nearest
REFERENCE station is frequently inside an inlet, sound or harbour, which is a
different tidal regime. Surf City NC was reading Hampstead, 17.5 km up the
Intracoastal Waterway, and ran 28-51 minutes late with high tide 0.85 ft low
against the ocean pier station 5.6 km away at the beach itself. Pipeline, on
Oahu's NORTH shore, was being handed Ford Island in Pearl Harbor -- the south
shore, 50 minutes out.

Distance alone does not predict the error, which is why these tests pin the
chosen STATION rather than a distance threshold: San Clemente sits 32 km from
its reference station and is only 2-8 minutes off, because open-coast Pacific
stations are in phase.

No test here touches the network. The station list is a fixture whose geometry
mirrors the real NOAA data for the spots that were wrong.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402

# Captured before the autouse fixture replaces it, so the caching test can
# exercise the real implementation rather than its own stub.
_REAL_LOAD_TIDE_STATIONS = app._load_tide_stations


# Geometry copied from NOAA's live metadata for the spots this bug affected.
# 'type' is retained precisely because the old code branched on it.
FIXTURE_STATIONS = [
    # Surf City NC (34.43, -77.55): the pier is subordinate and 3x closer.
    {"id": "8657419", "name": "Ocean City Beach (fishing pier)",
     "lat": 34.4589, "lng": -77.5361, "type": "S"},
    {"id": "8657813", "name": "Hampstead",
     "lat": 34.3394, "lng": -77.7060, "type": "R"},
    # Oahu: Haleiwa is north shore, Ford Island is Pearl Harbor on the south.
    {"id": "1611400", "name": "Haleiwa, Waialua Bay",
     "lat": 21.5906, "lng": -158.1064, "type": "S"},
    {"id": "1612401", "name": "Ford Island, Pearl Harbor",
     "lat": 21.3675, "lng": -157.9600, "type": "R"},
    # A station with no coordinates: NOAA's list contains these and the
    # distance maths must skip them rather than raise.
    {"id": "9999999", "name": "Broken station", "lat": None, "lng": None,
     "type": "R"},
]

SURF_CITY = (34.43, -77.55)
PIPELINE = (21.665, -158.053)


@pytest.fixture(autouse=True)
def _stub_station_list(monkeypatch):
    """Serve the fixture list and fail loudly if anything reaches the network."""
    monkeypatch.setattr(app, "_load_tide_stations", lambda: FIXTURE_STATIONS)
    yield


def test_surf_city_uses_the_ocean_pier_not_the_intracoastal_station():
    """The reported bug. Hampstead is a Reference station and 3x further."""
    station = app.find_nearest_tide_station(*SURF_CITY)
    assert station["id"] == "8657419", (
        f"expected the ocean pier, got {station['name']} — the type='R' filter is back"
    )
    assert station["distance_km"] < 7


def test_pipeline_does_not_get_pearl_harbor():
    """North shore must not be served by a south shore harbour station."""
    station = app.find_nearest_tide_station(*PIPELINE)
    assert station["id"] == "1611400"
    assert "Pearl Harbor" not in station["name"]


def test_subordinate_stations_are_eligible():
    """The whole point: type must not be used to exclude a station."""
    chosen = {app.find_nearest_tide_station(*SURF_CITY)["id"],
              app.find_nearest_tide_station(*PIPELINE)["id"]}
    subordinate = {s["id"] for s in FIXTURE_STATIONS if s["type"] == "S"}
    assert chosen <= subordinate


def test_candidates_are_ranked_nearest_first():
    candidates = app.find_nearest_tide_stations(*SURF_CITY, limit=4)
    distances = [c["distance_km"] for c in candidates]
    assert distances == sorted(distances)
    assert candidates[0]["id"] == "8657419"


def test_stations_without_coordinates_are_skipped():
    """A None lat/lng must not raise or win by comparing as smallest."""
    ids = {c["id"] for c in app.find_nearest_tide_stations(*SURF_CITY, limit=99)}
    assert "9999999" not in ids


def test_limit_is_respected():
    assert len(app.find_nearest_tide_stations(*SURF_CITY, limit=2)) == 2


def test_single_station_helper_matches_the_ranked_head():
    ranked = app.find_nearest_tide_stations(*SURF_CITY, limit=3)
    assert app.find_nearest_tide_station(*SURF_CITY) == ranked[0]


def test_empty_station_list_returns_empty_not_none(monkeypatch):
    monkeypatch.setattr(app, "_load_tide_stations", lambda: [])
    assert app.find_nearest_tide_stations(*SURF_CITY) == []
    assert app.find_nearest_tide_station(*SURF_CITY) is None


def test_upstream_failure_is_swallowed_not_raised(monkeypatch):
    def boom():
        raise RuntimeError("NOAA metadata down")
    monkeypatch.setattr(app, "_load_tide_stations", boom)
    assert app.find_nearest_tide_stations(*SURF_CITY) == []


def test_station_list_is_cached_between_calls(monkeypatch):
    """The list sits on every tide request and ranking multiplies the fetch."""
    calls = {"n": 0}

    class _Resp:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"stations": FIXTURE_STATIONS}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        return _Resp()

    monkeypatch.setattr(app, "_load_tide_stations", _REAL_LOAD_TIDE_STATIONS)
    monkeypatch.setattr(app.requests, "get", fake_get)
    app._tide_stations_cache["stations"] = []
    app._tide_stations_cache["at"] = 0.0
    try:
        app._load_tide_stations()
        app._load_tide_stations()
        app._load_tide_stations()
        assert calls["n"] == 1, "station metadata refetched on every call"
    finally:
        app._tide_stations_cache["stations"] = []
        app._tide_stations_cache["at"] = 0.0


class TestTideEndpointFallback:
    """The endpoint must survive a station NOAA declines to serve.

    Every nearest-station candidate for the 146 curated spots was checked and
    all 77 subordinate ones served predictions, so no such failure is known.
    This is insurance: a station that quietly returns nothing would otherwise
    take the tide panel down for that spot with no way to tell it apart from a
    genuine outage.
    """

    def _clear_cache(self):
        app._cache.clear()

    def test_falls_back_to_the_next_station_when_the_closest_is_empty(self, monkeypatch):
        self._clear_cache()
        served = []

        def fake_get_tide_data(station_id):
            served.append(station_id)
            if station_id == "8657419":
                return None          # closest station declines
            return {"hourly": [], "high_low": []}

        monkeypatch.setattr(app, "get_tide_data", fake_get_tide_data)
        client = app.app.test_client()
        resp = client.get("/api/tides?lat=34.43&lon=-77.55")
        try:
            assert resp.status_code == 200
            body = resp.get_json()
            assert served[0] == "8657419", "should try the closest station first"
            assert len(served) > 1, "should have tried a second station"
            assert body["station"]["id"] == served[-1], (
                "the reported station must be the one that actually served the data"
            )
        finally:
            self._clear_cache()

    def test_reports_the_station_that_served_the_data(self, monkeypatch):
        self._clear_cache()
        monkeypatch.setattr(app, "get_tide_data",
                            lambda sid: {"hourly": [], "high_low": []})
        client = app.app.test_client()
        resp = client.get("/api/tides?lat=34.43&lon=-77.55")
        try:
            body = resp.get_json()
            assert body["station"]["id"] == "8657419"
            assert body["station"]["name"] == "Ocean City Beach (fishing pier)"
        finally:
            self._clear_cache()

    def test_all_candidates_failing_is_an_error_not_a_silent_empty(self, monkeypatch):
        self._clear_cache()
        monkeypatch.setattr(app, "get_tide_data", lambda sid: None)
        client = app.app.test_client()
        resp = client.get("/api/tides?lat=34.43&lon=-77.55")
        try:
            assert resp.status_code == 500
        finally:
            self._clear_cache()

    def test_great_lakes_still_reported_non_tidal(self, monkeypatch):
        """The 300 km guard predates this change and must survive it.

        Uluwatu was once told its tides came from Djakarta, 1169 km away.
        Widening the station pool must not quietly reintroduce that by finding
        some closer-but-still-meaningless station.
        """
        self._clear_cache()
        called = []
        monkeypatch.setattr(app, "get_tide_data",
                            lambda sid: called.append(sid) or {"hourly": []})
        client = app.app.test_client()
        resp = client.get("/api/tides?lat=43.0517&lon=-87.8760")   # Milwaukee
        try:
            body = resp.get_json()
            assert body["non_tidal"] is True
            assert body["station"] is None
            assert not called, "must not fetch predictions for a non-tidal point"
        finally:
            self._clear_cache()


class TestHourlyReconstruction:
    """Subordinate stations serve interval=hilo but NOT interval=h.

    This is the part the original code comment was actually right about, and
    testing hilo alone hides it completely -- every subordinate station I
    checked "worked" until the endpoint asked for the hourly series the app
    really uses. So the extremes come from NOAA for the chosen station and only
    the SHAPE between them is borrowed from its reference station.

    Half-cosine interpolation was measured and rejected: against true hourly
    data it reaches 32.8 cm of error at Newport Bay, because Pacific extremes
    are markedly unequal and the curve is not a half-cosine. Warping a real
    neighbouring shape holds the same case to 2.5 cm.
    """

    @staticmethod
    def _preds(pairs, types=None):
        out = []
        for i, (t, v) in enumerate(pairs):
            row = {"t": t, "v": str(v)}
            if types:
                row["type"] = types[i]
            out.append(row)
        return out

    def test_reference_station_uses_its_own_hourly_series(self, monkeypatch):
        calls = []

        def fake(sid, interval):
            calls.append((sid, interval))
            if interval == "h":
                return self._preds([("2026-08-25 00:00", 1.0),
                                    ("2026-08-25 01:00", 1.2)])
            return self._preds([("2026-08-25 00:00", 1.0)], ["L"])

        monkeypatch.setattr(app, "_fetch_tide_predictions", fake)
        data = app.get_tide_data("8657813")
        assert len(data["hourly"]) == 2
        assert data["hourly"][0]["height"] == 1.0
        # No reference-station lookup should happen for a station with hourly.
        assert all(sid == "8657813" for sid, _ in calls)

    def test_subordinate_station_rebuilds_hourly_from_its_reference(self, monkeypatch):
        target_hilo = self._preds(
            [("2026-08-25 00:00", 0.0), ("2026-08-25 06:00", 2.0)], ["L", "H"])
        ref_hilo = self._preds(
            [("2026-08-25 00:30", 0.5), ("2026-08-25 06:30", 1.5)], ["L", "H"])
        ref_hourly = self._preds([("2026-08-25 00:30", 0.5),
                                  ("2026-08-25 03:30", 1.0),
                                  ("2026-08-25 06:30", 1.5)])

        def fake(sid, interval):
            if sid == "SUB":
                if interval == "h":
                    raise ValueError("subordinate stations have no hourly series")
                return target_hilo
            return ref_hourly if interval == "h" else ref_hilo

        monkeypatch.setattr(app, "_fetch_tide_predictions", fake)
        monkeypatch.setattr(app, "_tide_reference_station", lambda sid: "REF")
        data = app.get_tide_data("SUB")

        assert data["high_low"][0]["height"] == 0.0, "extremes must be NOAA's own"
        assert data["hourly"], "hourly curve should have been rebuilt"
        # The warped curve must span the TARGET's range, not the reference's.
        heights = [h["height"] for h in data["hourly"]]
        assert min(heights) == pytest.approx(0.0, abs=1e-6)
        assert max(heights) == pytest.approx(2.0, abs=1e-6)

    def test_warp_lands_exactly_on_the_target_extremes(self):
        shape_hilo = self._preds([("2026-08-25 00:00", 1.0), ("2026-08-25 06:00", 2.0)])
        shape_hourly = self._preds([("2026-08-25 00:00", 1.0), ("2026-08-25 06:00", 2.0)])
        target_hilo = self._preds([("2026-08-25 01:00", 0.2), ("2026-08-25 07:30", 3.0)])
        out = app._warp_hourly_onto_extremes(shape_hourly, shape_hilo, target_hilo)
        assert out[0]["time"] == "2026-08-25T01:00Z"
        assert out[0]["height"] == pytest.approx(0.2)
        assert out[-1]["time"] == "2026-08-25T07:30Z"
        assert out[-1]["height"] == pytest.approx(3.0)

    def test_warp_is_empty_when_there_is_nothing_to_map(self):
        assert app._warp_hourly_onto_extremes([], [], []) == []
        one = self._preds([("2026-08-25 00:00", 1.0)])
        assert app._warp_hourly_onto_extremes(one, one, one) == []

    def test_high_low_survives_when_no_reference_station_is_known(self, monkeypatch):
        """Losing the curve must not lose the tide times."""
        hilo = self._preds([("2026-08-25 00:00", 0.3)], ["L"])

        def fake(sid, interval):
            if interval == "h":
                raise ValueError("no hourly")
            return hilo

        monkeypatch.setattr(app, "_fetch_tide_predictions", fake)
        monkeypatch.setattr(app, "_tide_reference_station", lambda sid: None)
        data = app.get_tide_data("SUB")
        assert data["hourly"] == []
        assert data["high_low"][0]["height"] == 0.3
        assert data["high_low"][0]["type"] == "L"

    def test_station_serving_nothing_returns_none(self, monkeypatch):
        def fake(sid, interval):
            raise ValueError("dead station")
        monkeypatch.setattr(app, "_fetch_tide_predictions", fake)
        assert app.get_tide_data("DEAD") is None


class TestTideLabelPrecision:
    """Tide extremes do not land on the hour, so an hour-only label is wrong.

    Intl.DateTimeFormat TRUNCATES rather than rounds, so `{hour:'numeric'}`
    rendered a high at 8:46am as "8a" -- up to 59 minutes early, and always
    early, never late. Current Conditions and the chart tooltip both show
    minutes, so the same tide event was being presented two ways that
    disagreed by up to an hour. Reported by the user for the Surf City /
    Topsail area on 2026-08-29.
    """

    @staticmethod
    def _markers_block():
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, 'templates', 'index.html')) as f:
            src = f.read()
        start = src.index('const hl = window._tideHighLow || [];')
        end = src.index('ctx.restore();', start)
        return src[start:end]

    def test_chart_tide_markers_show_minutes(self):
        block = self._markers_block()
        assert "minute: '2-digit'" in block, (
            "the high/low markers on the wave chart must render minutes; "
            "hour-only truncates a 8:46am high to '8a'")

    def test_no_hour_only_formatting_survives_in_the_markers(self):
        import re
        block = self._markers_block()
        bad = re.findall(r"formatLocationTime\([^)]*\{\s*hour:\s*'numeric'\s*\}", block)
        assert not bad, (
            f"{len(bad)} hour-only formatter(s) left in the tide markers")

    def test_collision_gate_widened_for_the_longer_label(self):
        """'8:46a' is roughly twice the width of '8a'. Leaving the gate at its
        old 32px would overlap neighbouring labels rather than drop one."""
        import re
        block = self._markers_block()
        m = re.search(r'TIDE_LABEL_MIN_PX\s*=\s*(\d+)', block)
        assert m, 'TIDE_LABEL_MIN_PX not found in the marker block'
        assert int(m.group(1)) >= 48, (
            f'gate {m.group(1)}px is too narrow for a HH:MM label')


class TestTideStationOverrides:
    """Some ocean spots have an enclosed-water station closer than the coast.

    Ranking by distance -- which is what cured the earlier, worse bug of
    considering only Reference stations -- still walks into a bay, sound,
    ditch or creek whenever one sits nearer than the open coast. On a barrier
    island the wrong side is ALWAYS nearer, so distance is actively misleading
    exactly where it matters most. Measured against NOAA on 2026-08-30:
    Rodanthe read Pamlico Sound (~3 h late), Ocean City MD read Little
    Assawoman Bay (high tide 3 h 52 m late, 0.21 m of range against the ocean
    pier's 1.20 m), Indian Rocks Beach read the station NOAA labels
    "(inside)".

    These tests stay offline, like the rest of this file: the mechanism is
    exercised against a purpose-built list, and the table itself is checked
    for internal consistency only.
    """

    # A pinned ocean station plus a nearer decoy in the wrong water body.
    OVERRIDE_FIXTURE = [
        {"id": "8654400", "name": "CAPE HATTERAS FISHING PIER",
         "lat": 35.2228, "lng": -75.6358, "type": "R"},
        {"id": "8653215", "name": "Rodanthe, Pamlico Sound",
         "lat": 35.5961, "lng": -75.4700, "type": "S"},
        {"id": "8652678", "name": "Oregon Inlet (USCG Station)",
         "lat": 35.7950, "lng": -75.5480, "type": "R"},
    ]

    def test_overrides_name_real_spots(self):
        """A renamed slug would silently drop back to the nearest station,
        which is the bug this table exists to prevent."""
        for slug in app.TIDE_STATION_OVERRIDES:
            assert slug in app.LOCATION_BY_SLUG, (
                f'{slug} is pinned but is not a known spot')

    def test_a_spot_is_pinned_at_most_once(self):
        keys = list(app.TIDE_STATION_OVERRIDES)
        assert len(keys) == len(set(keys))

    def test_pinned_ids_look_like_station_ids(self):
        for slug, sid in app.TIDE_STATION_OVERRIDES.items():
            assert isinstance(sid, str) and sid.strip(), slug

    def test_pinned_station_is_promoted_over_a_nearer_one(self, monkeypatch):
        monkeypatch.setattr(app, "_load_tide_stations",
                            lambda: self.OVERRIDE_FIXTURE)
        loc = app.LOCATION_BY_SLUG['rodanthe']
        monkeypatch.setitem(app.TIDE_STATION_OVERRIDES, 'rodanthe', '8654400')
        top = app.find_nearest_tide_stations(loc['lat'], loc['lon'])
        assert top[0]['id'] == '8654400', (
            f"expected the pinned ocean pier, got {top[0]['name']}")
        # ...even though the sound station is far nearer
        assert top[0]['distance_km'] > top[1]['distance_km']

    def test_the_displaced_station_survives_as_fallback(self, monkeypatch):
        """Promotion, not replacement: if NOAA declines the pinned station the
        caller must still have somewhere to go rather than losing tides."""
        monkeypatch.setattr(app, "_load_tide_stations",
                            lambda: self.OVERRIDE_FIXTURE)
        loc = app.LOCATION_BY_SLUG['rodanthe']
        top = app.find_nearest_tide_stations(loc['lat'], loc['lon'])
        ids = [s['id'] for s in top]
        assert ids[0] == '8654400'
        assert '8653215' in ids[1:], 'the nearest station was dropped, not demoted'

    def test_no_duplicate_entry_for_the_pinned_station(self, monkeypatch):
        monkeypatch.setattr(app, "_load_tide_stations",
                            lambda: self.OVERRIDE_FIXTURE)
        loc = app.LOCATION_BY_SLUG['rodanthe']
        ids = [s['id'] for s in app.find_nearest_tide_stations(loc['lat'], loc['lon'])]
        assert ids.count('8654400') == 1

    def test_unpinned_spot_keeps_the_nearest_station(self, monkeypatch):
        monkeypatch.setattr(app, "_load_tide_stations",
                            lambda: self.OVERRIDE_FIXTURE)
        # A coordinate with no override must be untouched by the table.
        top = app.find_nearest_tide_stations(35.7950, -75.5480)
        assert top[0]['id'] == '8652678'

    def test_unknown_coordinates_have_no_override(self):
        assert app.tide_station_override(0.0, 0.0) is None

    def test_a_pin_to_a_missing_station_falls_back_quietly(self, monkeypatch):
        """If NOAA drops a station from its list the spot must degrade to the
        nearest one, not lose tides."""
        monkeypatch.setattr(app, "_load_tide_stations",
                            lambda: self.OVERRIDE_FIXTURE)
        monkeypatch.setitem(app.TIDE_STATION_OVERRIDES, 'rodanthe', '0000000')
        loc = app.LOCATION_BY_SLUG['rodanthe']
        top = app.find_nearest_tide_stations(loc['lat'], loc['lon'])
        assert top and top[0]['id'] == '8653215'
