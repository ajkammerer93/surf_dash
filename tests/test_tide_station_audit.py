"""The daily audit's tide-station review queue.

This check exists because the failure it hunts is silent by construction. A
spot assigned a station in the wrong body of water still returns a complete,
plausible tide curve -- it is simply the wrong water, off by hours and
sometimes by most of the range. Nothing raises, nothing 4xxs, so no other
collector in the audit would ever notice. Measured 2026-08-30: 66 of 140
tidal spots sit where the two nearest stations disagree by more than 30
minutes.

Two design points are load-bearing and both are tested here:

  It only FLAGS. The name test that drives it is the same heuristic that was
  rejected for choosing stations automatically -- it reads "Smith Creek,
  Flagler Beach" as ocean-side on the word "Beach". A false positive in a
  review queue costs a human one glance; the same false positive in the
  selector costs every user of that spot wrong tides. The heuristic is unfit
  for one job and fine for the other, and the difference is who checks it.

  It alerts only on findings not seen before. The first sweep surfaces dozens.
  Warning about all of them every run would make the audit wallpaper, and this
  check would become the thing that taught people to skim past warnings.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)

import seo_audit as audit  # noqa: E402


class _Resp:
    def __init__(self, payload, ok=True):
        self._p, self.ok = payload, ok

    def json(self):
        return self._p


def _tide_payload(station_id, name, distance_km=2.0, non_tidal=False):
    if non_tidal:
        return {'non_tidal': True}
    return {
        'station': {'id': station_id, 'name': name, 'distance_km': distance_km},
        'high_low': [{'type': 'H', 'height': 1.0}, {'type': 'L', 'height': 0.0},
                     {'type': 'H', 'height': 1.2}, {'type': 'L', 'height': 0.1}],
        'hourly': [],
    }


@pytest.fixture
def spots(monkeypatch):
    rows = [{'name': f'Spot {i}', 'lat': 30.0 + i, 'lon': -80.0} for i in range(6)]
    monkeypatch.setattr(audit, '_spot_slugs', lambda: rows)
    monkeypatch.setattr(audit, 'TIDE_SPOTS_PER_RUN', 3)
    return rows


def _serve(monkeypatch, by_lat):
    def fake_get(url, params=None, timeout=None, **kw):
        return _Resp(by_lat[round(params['lat'], 4)])
    monkeypatch.setattr(audit.requests, 'get', fake_get)


class TestFlagging:
    def test_enclosed_station_names_are_flagged(self, monkeypatch, spots):
        _serve(monkeypatch, {
            30.0: _tide_payload('1', 'Rodanthe, Pamlico Sound'),
            31.0: _tide_payload('2', 'Donald Ross Bridge, ICWW'),
            32.0: _tide_payload('3', 'CAPE HATTERAS FISHING PIER'),
        })
        out = audit.collect_tide_stations({})
        flagged = {r['station_name'] for r in out['review']}
        assert 'Rodanthe, Pamlico Sound' in flagged
        assert 'Donald Ross Bridge, ICWW' in flagged
        assert 'CAPE HATTERAS FISHING PIER' not in flagged

    def test_a_distant_station_is_flagged_even_if_its_name_is_clean(
            self, monkeypatch, spots):
        _serve(monkeypatch, {
            30.0: _tide_payload('1', 'Somewhere Pier', distance_km=60.0),
            31.0: _tide_payload('2', 'Nearby Pier', distance_km=3.0),
            32.0: _tide_payload('3', 'Another Pier', distance_km=1.0),
        })
        out = audit.collect_tide_stations({})
        assert len(out['review']) == 1
        assert out['review'][0]['station'] == '1'
        assert 'km away' in ' '.join(out['review'][0]['reasons'])

    def test_non_tidal_spots_are_skipped_even_if_they_name_a_station(
            self, monkeypatch, spots):
        """The Great Lakes have no harmonic stations, so a non-tidal response
        must never enter the queue.

        Today /api/tides sends "station": None alongside non_tidal, so the
        name test would skip it anyway -- which is why this passes a non-tidal
        payload that DOES name an enclosed-sounding station. Without the
        explicit non_tidal guard, a future response shape that keeps the
        nearest station for diagnostics would flood the queue with lakes.
        """
        payload = _tide_payload('1', 'Some Lake Bay')
        payload['non_tidal'] = True
        _serve(monkeypatch, {
            30.0: payload,
            31.0: _tide_payload('2', 'Clean Pier'),
            32.0: _tide_payload('3', 'Clean Pier Two'),
        })
        out = audit.collect_tide_stations({})
        assert out['checked'] == 2, 'a non-tidal spot must not be counted'
        assert out['review'] == [], 'a non-tidal spot must not be flagged'

    def test_the_range_is_recorded_for_review(self, monkeypatch, spots):
        _serve(monkeypatch, {
            30.0: _tide_payload('1', 'Little Assawoman Bay'),
            31.0: _tide_payload('2', 'Clean Pier'),
            32.0: _tide_payload('3', 'Clean Pier Two'),
        })
        out = audit.collect_tide_stations({})
        # mean high 1.1, mean low 0.05
        assert out['review'][0]['range_m'] == pytest.approx(1.05, abs=1e-6)


class TestQueueDoesNotBecomeWallpaper:
    def test_repeat_findings_do_not_re_alert(self, monkeypatch, spots):
        _serve(monkeypatch, {
            30.0: _tide_payload('1', 'Rodanthe, Pamlico Sound'),
            31.0: _tide_payload('2', 'Clean Pier'),
            32.0: _tide_payload('3', 'Clean Pier Two'),
        })
        first = audit.collect_tide_stations({})
        assert len(first['new']) == 1
        state = {'tide_seen': [f"{r['spot']}|{r['station']}" for r in first['review']]}
        second = audit.collect_tide_stations(state)
        assert second['review'], 'the record must still list the finding'
        assert second['new'] == [], 'a known finding must not warn again'

    def test_a_station_change_on_a_known_spot_is_new_again(self, monkeypatch, spots):
        """Keyed on spot AND station, so a spot that moves to a different
        wrong station is surfaced rather than silently absorbed."""
        _serve(monkeypatch, {
            30.0: _tide_payload('1', 'Rodanthe, Pamlico Sound'),
            31.0: _tide_payload('2', 'Clean Pier'),
            32.0: _tide_payload('3', 'Clean Pier Two'),
        })
        first = audit.collect_tide_stations({})
        state = {'tide_seen': [f"{r['spot']}|{r['station']}" for r in first['review']]}
        _serve(monkeypatch, {
            30.0: _tide_payload('9', 'Some Other Creek'),
            31.0: _tide_payload('2', 'Clean Pier'),
            32.0: _tide_payload('3', 'Clean Pier Two'),
        })
        second = audit.collect_tide_stations(state)
        assert len(second['new']) == 1
        assert second['new'][0]['station'] == '9'


class TestRotation:
    def test_the_cursor_advances_and_wraps(self, monkeypatch, spots):
        _serve(monkeypatch, {round(30.0 + i, 4): _tide_payload(str(i), 'Clean Pier')
                             for i in range(6)})
        out = audit.collect_tide_stations({'tide_cursor': 0})
        assert out['cursor'] == 3
        out = audit.collect_tide_stations({'tide_cursor': out['cursor']})
        assert out['cursor'] == 0, 'must wrap so the list keeps rotating'

    def test_a_short_tail_wraps_to_the_front(self, monkeypatch, spots):
        """Otherwise the last runs of a cycle check fewer spots than the rest."""
        _serve(monkeypatch, {round(30.0 + i, 4): _tide_payload(str(i), 'Clean Pier')
                             for i in range(6)})
        out = audit.collect_tide_stations({'tide_cursor': 5})
        assert out['checked'] == 3


class TestResilience:
    def test_an_upstream_failure_does_not_fail_the_audit(self, monkeypatch, spots):
        def boom(*a, **kw):
            raise RuntimeError('tides down')
        monkeypatch.setattr(audit.requests, 'get', boom)
        out = audit.collect_tide_stations({})
        assert out.get('checked') == 0
        assert 'error' not in out or out['checked'] == 0

    def test_a_missing_spot_list_reports_rather_than_raising(self, monkeypatch):
        monkeypatch.setattr(audit, '_spot_slugs', lambda: [])
        out = audit.collect_tide_stations({})
        assert out.get('unavailable')
