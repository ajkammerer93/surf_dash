"""Forecast verification pipeline and bias-correction tests.

Pins the invariants that keep the closed loop honest:
  - NDBC parsing and forecast/observation pairing are correct
  - stats aggregation (bias/MAE/RMSE, lead bins, rolling window) is correct
  - bias correction is gated (sample size, distance) and capped, never
    mutates the shared cache entry, and preserves the raw model value
  - /faq ships valid FAQPage JSON-LD that matches the visible content
  - the new pages exist and are in the sitemap
"""

import importlib.util
import json
import os
import re
from datetime import datetime, timedelta, timezone

import pytest

import app as app_module
from app import app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_spec = importlib.util.spec_from_file_location(
    "forecast_verification",
    os.path.join(ROOT, "scripts", "forecast_verification.py"))
fv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fv)


NDBC_FIXTURE = """\
#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa    ft
2026 07 17 12 40 190  5.0  7.0   1.2   9.1   6.4 175 1015.2  27.1  27.9  24.0 99.0 +0.2 99.00
2026 07 17 11 40 200  4.0  6.0   1.1   8.9   6.2 170 1015.0  26.8  27.9  24.0 99.0 +0.1 99.00
2026 07 17 10 40 210  3.0  5.0    MM    MM   6.0 165 1014.8  26.5  27.8  24.0 99.0 -0.1 99.00
2026 07 17 09 40 220  2.0  4.0   0.9   8.0   5.9 160 1014.5  26.1  27.8  24.0 99.0 -0.2 99.00
"""


class TestNdbcParsing:
    def test_parses_rows_and_skips_missing_wvht(self):
        series = fv.parse_ndbc_stdmet(NDBC_FIXTURE)
        assert len(series) == 3  # the MM row is dropped
        dt, wvht, dpd = series[0]
        assert dt == datetime(2026, 7, 17, 12, 40, tzinfo=timezone.utc)
        assert wvht == 1.2
        assert dpd == 9.1

    def test_empty_and_garbage_input(self):
        assert fv.parse_ndbc_stdmet("") == []
        assert fv.parse_ndbc_stdmet("#YY MM\n#yr mo\nnot numbers") == []


class TestPairing:
    def _series(self):
        return fv.parse_ndbc_stdmet(NDBC_FIXTURE)

    def test_match_within_tolerance_picks_nearest(self):
        valid = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
        obs = fv.match_observation(self._series(), valid)
        assert obs == (1.1, 8.9)  # 11:40 (20 min away) beats 12:40 (40 min)

    def test_no_match_outside_tolerance(self):
        valid = datetime(2026, 7, 17, 6, 0, tzinfo=timezone.utc)
        assert fv.match_observation(self._series(), valid) is None

    def test_build_new_pairs_scores_past_and_dedupes(self):
        now = datetime(2026, 7, 17, 13, 0, tzinfo=timezone.utc)
        snap = {
            "issued": "2026-07-17T06:00:00Z",
            "station": "41110",
            "source": "Open-Meteo",
            "times": ["2026-07-17T09:00:00Z", "2026-07-17T12:00:00Z",
                      "2026-07-17T18:00:00Z"],  # 18Z is still in the future
            "wave_height": [1.0, 1.3, 1.5],
            "wave_period": [8.0, 9.0, 10.0],
        }
        obs = {"41110": self._series()}
        keys = set()
        pairs = fv.build_new_pairs([snap], obs, keys, now=now)
        assert len(pairs) == 2
        by_valid = {p["valid"]: p for p in pairs}
        assert by_valid["2026-07-17T12:00:00Z"]["fc_wh"] == 1.3
        assert by_valid["2026-07-17T12:00:00Z"]["ob_wh"] == 1.1
        assert by_valid["2026-07-17T12:00:00Z"]["lead_h"] == 6.0
        # Second run with the same keys set scores nothing new
        assert fv.build_new_pairs([snap], obs, keys, now=now) == []


class TestStats:
    def _pair(self, valid, lead_h, fc, ob, station="41110"):
        return {"station": station, "issued": "x", "valid": valid,
                "lead_h": lead_h, "fc_wh": fc, "ob_wh": ob,
                "fc_tp": None, "ob_tp": None, "source": "test"}

    def test_bias_mae_rmse(self):
        now = datetime(2026, 7, 17, tzinfo=timezone.utc)
        recent = "2026-07-16T12:00:00Z"
        pairs = [self._pair(recent, 6, 1.5, 1.0),   # err +0.5
                 self._pair(recent, 12, 0.7, 1.0)]  # err -0.3
        stats = fv.compute_stats(pairs, [{"id": "41110", "name": "Test",
                                          "lat": 34.1, "lon": -77.7,
                                          "region": "SE"}], now=now)
        agg = stats["overall"]["all"]
        assert agg["n"] == 2
        assert agg["bias_m"] == pytest.approx(0.1)
        assert agg["mae_m"] == pytest.approx(0.4)
        assert agg["rmse_m"] == pytest.approx(0.412, abs=1e-3)
        assert "0-24" in stats["stations"]["41110"]["bins"]

    def test_window_excludes_old_pairs(self):
        now = datetime(2026, 7, 17, tzinfo=timezone.utc)
        old = (now - timedelta(days=35)).strftime("%Y-%m-%dT%H:%M:%SZ")
        stats = fv.compute_stats([self._pair(old, 6, 2.0, 1.0)], [], now=now)
        assert stats["n_pairs"] == 0
        assert stats["overall"]["all"] is None

    def test_lead_bins(self):
        assert fv._bin_label(0) == "0-24"
        assert fv._bin_label(23.9) == "0-24"
        assert fv._bin_label(24) == "24-48"
        assert fv._bin_label(72) == "48-72"
        assert fv._bin_label(73) is None


def _stats_for(station_id, lat, lon, bias_m, n=50):
    agg = {"n": n, "bias_m": bias_m, "mae_m": abs(bias_m), "rmse_m": abs(bias_m)}
    return {
        "generated": "2026-07-17T00:00:00Z",
        "window_days": 30,
        "n_pairs": 3 * n,
        "overall": {"all": agg, "bins": {}},
        "stations": {station_id: {
            "name": "Test Buoy", "lat": lat, "lon": lon, "region": "Test",
            "all": agg,
            "bins": {"0-24": dict(agg), "24-48": dict(agg), "48-72": dict(agg)},
        }},
    }


def _forecast_data(wave_height=1.0, hours=6):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return {
        "source": "Open-Meteo",
        "forecast": [
            {"time": (now + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%MZ"),
             "wave_height": wave_height, "wave_period": 10.0}
            for h in range(hours)
        ],
    }


class TestBiasCorrection:
    def test_correction_applied_and_capped(self):
        # Model over-forecasts by 0.2 m at a buoy 0 km away
        stats = _stats_for("41110", 34.43, -77.55, bias_m=0.2)
        data = _forecast_data(wave_height=1.0)
        out = self._apply(data, 34.43, -77.55, stats)
        entry = out["forecast"][0]
        assert entry["wave_height"] == pytest.approx(0.8)
        assert entry["wave_height_raw"] == 1.0
        assert out["bias_correction"]["applied"] is True
        assert out["bias_correction"]["station"] == "41110"

    def _apply(self, data, lat, lon, stats):
        orig = app_module._get_verification_stats
        app_module._get_verification_stats = lambda: stats
        try:
            return app_module._apply_bias_correction(data, lat, lon)
        finally:
            app_module._get_verification_stats = orig

    def test_cap_limits_large_corrections(self):
        # Huge measured bias must be clamped to 30% of the forecast value
        stats = _stats_for("41110", 34.43, -77.55, bias_m=1.5)
        out = self._apply(_forecast_data(wave_height=1.0), 34.43, -77.55, stats)
        assert out["forecast"][0]["wave_height"] == pytest.approx(0.7)

    def test_gated_below_min_pairs(self):
        stats = _stats_for("41110", 34.43, -77.55, bias_m=0.3, n=10)
        data = _forecast_data()
        out = self._apply(data, 34.43, -77.55, stats)
        assert out is data  # unchanged, same object

    def test_gated_by_distance(self):
        stats = _stats_for("46225", 32.9, -117.4, bias_m=0.3)  # San Diego
        data = _forecast_data()
        out = self._apply(data, 34.43, -77.55, stats)  # request in NC
        assert out is data

    def test_never_mutates_cached_dict(self):
        stats = _stats_for("41110", 34.43, -77.55, bias_m=0.2)
        data = _forecast_data(wave_height=1.0)
        before = json.dumps(data, sort_keys=True)
        self._apply(data, 34.43, -77.55, stats)
        assert json.dumps(data, sort_keys=True) == before

    def test_correction_never_goes_negative(self):
        stats = _stats_for("41110", 34.43, -77.55, bias_m=0.5)
        out = self._apply(_forecast_data(wave_height=0.1), 34.43, -77.55, stats)
        assert out["forecast"][0]["wave_height"] >= 0.0


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def stats_file(tmp_path):
    """Point the app at a local stats file and clear the stats cache."""
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(_stats_for("41110", 34.142, -77.715, 0.15)))
    orig = app_module.VERIFICATION_STATS_FILE
    app_module.VERIFICATION_STATS_FILE = str(path)
    with app_module._cache_lock:
        app_module._cache.pop("verification:stats", None)
    yield path
    app_module.VERIFICATION_STATS_FILE = orig
    with app_module._cache_lock:
        app_module._cache.pop("verification:stats", None)


class TestRoutes:
    def test_api_accuracy_serves_stats(self, client, stats_file):
        resp = client.get("/api/accuracy")
        assert resp.status_code == 200
        assert resp.get_json()["stations"]["41110"]["name"] == "Test Buoy"

    def test_accuracy_page_renders_stats(self, client, stats_file):
        resp = client.get("/accuracy")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Forecast Accuracy Report" in html
        assert "Test Buoy" in html
        assert "Methodology" in html

    def test_accuracy_page_empty_state(self, client, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({
            "generated": "2026-07-17T00:00:00Z", "window_days": 30,
            "n_pairs": 0, "overall": {"all": None, "bins": {}},
            "stations": {}}))
        orig = app_module.VERIFICATION_STATS_FILE
        app_module.VERIFICATION_STATS_FILE = str(path)
        with app_module._cache_lock:
            app_module._cache.pop("verification:stats", None)
        try:
            resp = client.get("/accuracy")
            assert resp.status_code == 200
            assert "collecting data" in resp.get_data(as_text=True)
        finally:
            app_module.VERIFICATION_STATS_FILE = orig
            with app_module._cache_lock:
                app_module._cache.pop("verification:stats", None)

    def test_faq_page_renders(self, client):
        resp = client.get("/faq")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Frequently Asked Questions" in html
        assert "/accuracy" in html

    def test_faq_page_has_valid_fappage_jsonld(self, client):
        html = client.get("/faq").get_data(as_text=True)
        blocks = re.findall(
            r'<script type="application/ld\+json">(.*?)</script>',
            html, re.DOTALL)
        faq_ld = None
        for block in blocks:
            parsed = json.loads(block)
            if parsed.get("@type") == "FAQPage":
                faq_ld = parsed
        assert faq_ld is not None
        questions = faq_ld["mainEntity"]
        assert len(questions) >= 10
        for q in questions:
            assert q["@type"] == "Question"
            assert q["name"].strip()
            assert q["acceptedAnswer"]["@type"] == "Answer"
            assert q["acceptedAnswer"]["text"].strip()
            # visible content must match the markup (Google requirement)
            assert q["name"] in html

    def test_sitemap_includes_new_pages(self, client):
        xml = client.get("/sitemap.xml").get_data(as_text=True)
        assert "https://freesurfforecast.com/faq" in xml
        assert "https://freesurfforecast.com/accuracy" in xml

    def test_buoy_pairs_file_is_valid(self):
        with open(os.path.join(ROOT, "data", "verification",
                               "buoy_pairs.json")) as f:
            data = json.load(f)
        ids = [s["id"] for s in data["stations"]]
        assert len(ids) == len(set(ids))
        for s in data["stations"]:
            assert -90 < s["lat"] < 90 and -180 < s["lon"] < 180
            assert s["name"] and s["region"]
