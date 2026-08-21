"""Tests for the daily highlight card.

The card states a measured number in public, unattended, so the tests care most
about two things: that it refuses when it should, and that nothing it draws can
run off the canvas or land on top of something else.
"""
import io
import re
from datetime import datetime, timedelta, timezone

import pytest

import app as app_module
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def _obs_line(sid, lat, lon, when, wspd='MM', gust='MM',
              wvht='MM', dpd='MM', apd='MM', mwd='MM', wtmp='20.0'):
    return ' '.join([
        sid, f'{lat:.3f}', f'{lon:.3f}',
        f'{when.year}', f'{when.month:02d}', f'{when.day:02d}',
        f'{when.hour:02d}', f'{when.minute:02d}',
        '270', str(wspd), str(gust), str(wvht), str(dpd), str(apd), str(mwd),
        '1015.0', 'MM', '18.0', str(wtmp), 'MM', 'MM', 'MM',
    ])


HEADER = ('#STN       LAT      LON  YYYY MM DD hh mm WDIR WSPD   GST WVHT  DPD '
          'APD MWD   PRES  PTDY  ATMP  WTMP  DEWP  VIS   TIDE\n'
          '#text      deg      deg   yr mo day hr mn degT  m/s   m/s   m   sec '
          'sec degT   hPa   hPa  degC  degC  degC  nmi     ft\n')


def _known_ids(n):
    """Real station ids, because the parser deliberately drops anything it
    cannot put a name to."""
    return [str(s['id']) for s in app_module.NDBC_STATIONS[:n]]


def _feed(rows):
    return HEADER + '\n'.join(rows) + '\n'


def _clear_caches():
    """Drop every cache entry the card depends on.

    cached() holds the observations for 30 minutes and the built card for
    another 30, so without this the second test in a class is handed the first
    one's card and its guard assertion passes for the wrong reason.
    """
    with app_module._cache_lock:
        for key in [k for k in list(app_module._cache)
                    if k == 'ndbc:latest_obs'
                    or k.startswith('social:highlight')
                    or k.startswith('ndbc:spec:')]:
            app_module._cache.pop(key, None)


@pytest.fixture
def obs_text(monkeypatch):
    """Install a synthetic latest_obs feed."""
    def _set(text):
        rows, newest = app_module._parse_latest_obs(text)
        monkeypatch.setattr(app_module, '_fetch_ndbc_latest_obs',
                            lambda: ({'rows': rows, 'newest': newest}
                                     if rows else None))
        _clear_caches()
    yield _set
    _clear_caches()


def _healthy_rows(now=None, count=60, wvht=4.0, dpd=18.0, wspd=22.0):
    now = now or datetime.now(timezone.utc)
    ids = _known_ids(count)
    rows = []
    for i, sid in enumerate(ids):
        rows.append(_obs_line(sid, 30 + i * 0.1, -70 - i * 0.1, now,
                              wspd=1.0, wvht=0.6, dpd=6.0))
    # One clear winner on every measure.
    rows[0] = _obs_line(ids[0], 36.3, -122.1, now,
                        wspd=wspd, gust=wspd + 8, wvht=wvht, dpd=dpd,
                        apd=9.0, mwd=280)
    return rows


class TestParsing:
    def test_columns_are_read_at_the_right_offsets(self, obs_text):
        # An off-by-two here is silent and produces plausible nonsense: the
        # first version read a 350-second dominant period and did not complain.
        now = datetime.now(timezone.utc)
        sid = _known_ids(1)[0]
        rows, newest = app_module._parse_latest_obs(
            _feed([_obs_line(sid, 10.0, -20.0, now,
                             wspd=12.5, gust=15.0, wvht=3.25, dpd=17.0,
                             apd=9.1, mwd=290, wtmp=21.5)]))
        assert len(rows) == 1
        r = rows[0]
        assert r['wspd_ms'] == 12.5 and r['gust_ms'] == 15.0
        assert r['wvht_m'] == 3.25 and r['dpd_s'] == 17.0
        assert r['apd_s'] == 9.1 and r['mwd_deg'] == 290
        assert r['wtmp_c'] == 21.5
        assert newest == now.replace(second=0, microsecond=0)

    def test_missing_values_do_not_shift_the_columns(self):
        sid = _known_ids(1)[0]
        rows, _ = app_module._parse_latest_obs(
            _feed([_obs_line(sid, 10.0, -20.0, datetime.now(timezone.utc),
                             wvht='MM', dpd='MM')]))
        assert rows[0]['wvht_m'] is None
        assert rows[0]['dpd_s'] is None
        assert rows[0]['wtmp_c'] == 20.0   # still read from the right column

    def test_unnamed_stations_are_dropped(self):
        rows, _ = app_module._parse_latest_obs(
            _feed([_obs_line('99999', 1.0, 1.0, datetime.now(timezone.utc),
                             wvht=9.0)]))
        assert rows == []


class TestGuards:
    def test_publishes_when_something_clears_the_bar(self, client, obs_text):
        obs_text(_feed(_healthy_rows()))
        r = client.get('/api/social-card/highlight')
        assert r.status_code == 200, r.get_json()
        assert r.get_json()['highlight'] in app_module.HIGHLIGHT_ROTATION

    def test_refuses_when_nothing_is_exceptional(self, client, obs_text):
        obs_text(_feed(_healthy_rows(wvht=1.0, dpd=8.0, wspd=5.0)))
        r = client.get('/api/social-card/highlight')
        assert r.status_code == 503
        assert 'below the' in r.get_json()['reason']

    def test_refuses_on_stale_observations(self, client, obs_text):
        old = datetime.now(timezone.utc) - timedelta(hours=12)
        obs_text(_feed(_healthy_rows(now=old)))
        r = client.get('/api/social-card/highlight')
        assert r.status_code == 503
        assert 'old' in r.get_json()['reason']

    def test_refuses_on_a_thin_read_of_the_network(self, client, obs_text):
        # A handful of stations reporting would otherwise crown whichever one
        # happened to answer.
        obs_text(_feed(_healthy_rows(count=5)))
        r = client.get('/api/social-card/highlight')
        assert r.status_code == 503
        assert 'stations reporting' in r.get_json()['reason']

    def test_refuses_when_observations_are_unavailable(self, client, monkeypatch):
        _clear_caches()
        monkeypatch.setattr(app_module, '_fetch_ndbc_latest_obs', lambda: None)
        r = client.get('/api/social-card/highlight')
        assert r.status_code == 503
        _clear_caches()

    def test_a_long_period_on_a_flat_sea_does_not_win(self, obs_text):
        # A 20 s reading under six inches of swell is sensor noise, not a
        # groundswell, and posting it would be a false claim.
        now = datetime.now(timezone.utc)
        ids = _known_ids(60)
        rows = [_obs_line(s, 30 + i * 0.1, -70, now, wspd=1.0, wvht=0.6, dpd=6.0)
                for i, s in enumerate(ids)]
        rows[0] = _obs_line(ids[0], 36.0, -122.0, now, wvht=0.2, dpd=22.0)
        obs = {'rows': app_module._parse_latest_obs(_feed(rows))[0],
               'newest': now.replace(second=0, microsecond=0)}
        pick, why = app_module._highlight_candidates(obs, 'longest-period')
        assert pick is None and 'below the' in why

    def test_a_station_that_stopped_reporting_cannot_win_on_a_stale_value(self, obs_text):
        now = datetime.now(timezone.utc)
        ids = _known_ids(60)
        rows = [_obs_line(s, 30 + i * 0.1, -70, now, wspd=1.0, wvht=0.6, dpd=6.0)
                for i, s in enumerate(ids)]
        # Enormous, and two days old.
        rows[0] = _obs_line(ids[0], 36.0, -122.0, now - timedelta(days=2),
                            wvht=12.0, dpd=22.0, wspd=40.0)
        obs = {'rows': app_module._parse_latest_obs(_feed(rows))[0], 'newest': now}
        for kind in app_module.HIGHLIGHT_ROTATION:
            pick, _ = app_module._highlight_candidates(obs, kind)
            assert pick is None or pick['station']['id'] != ids[0]


class TestRoutes:
    @pytest.mark.parametrize('path,size', [
        ('/social/highlight.png', (1080, 1350)),
        ('/social/highlight.jpg', (1080, 1350)),
        ('/social/highlight-story.png', (1080, 1920)),
        ('/social/highlight-story.jpg', (1080, 1920)),
    ])
    def test_dimensions_and_type(self, client, obs_text, path, size):
        from PIL import Image
        obs_text(_feed(_healthy_rows()))
        r = client.get(path)
        assert r.status_code == 200
        im = Image.open(io.BytesIO(r.data))
        assert im.size == size
        assert im.format == ('JPEG' if path.endswith('.jpg') else 'PNG')

    def test_jpeg_is_really_jpeg(self, client, obs_text):
        # Instagram accepts JPEG only and reports a PNG as an opaque media
        # error, so check the magic bytes rather than trusting the header.
        obs_text(_feed(_healthy_rows()))
        assert client.get('/social/highlight.jpg').data[:3] == b'\xff\xd8\xff'

    def test_all_routes_are_noindex(self, client, obs_text):
        obs_text(_feed(_healthy_rows()))
        for path in ('/social/highlight.png', '/social/highlight.jpg',
                     '/social/highlight-story.png', '/social/highlight-story.jpg',
                     '/api/social-card/highlight'):
            assert 'noindex' in client.get(path).headers.get('X-Robots-Tag', '')

    def test_absent_from_the_sitemap(self, client):
        # The site is crawl-budget constrained; these must stay undiscoverable.
        body = client.get('/sitemap.xml').get_data(as_text=True)
        assert 'highlight' not in body

    def test_images_refuse_rather_than_erroring(self, client, obs_text):
        obs_text(_feed(_healthy_rows(wvht=1.0, dpd=8.0, wspd=5.0)))
        for path in ('/social/highlight.jpg', '/social/highlight-story.jpg'):
            assert client.get(path).status_code == 503

    def test_forced_kind_still_obeys_its_floor(self, client, obs_text):
        obs_text(_feed(_healthy_rows(wvht=2.0, dpd=18.0, wspd=2.0)))
        assert client.get('/api/social-card/highlight?kind=longest-period').status_code == 200
        assert client.get('/api/social-card/highlight?kind=strongest-wind').status_code == 503

    def test_unknown_kind_refuses(self, client, obs_text):
        obs_text(_feed(_healthy_rows()))
        assert client.get('/api/social-card/highlight?kind=nonsense').status_code == 503


class TestCaption:
    def test_states_the_measured_number_and_the_station(self, client, obs_text):
        # 2.0 m is 6.6 ft, under the seas floor, so only the period qualifies
        # and the rotation cannot pick a different kind out from under us.
        obs_text(_feed(_healthy_rows(wvht=2.0, dpd=18.0, wspd=2.0)))
        d = client.get('/api/social-card/highlight').get_json()
        assert d['highlight'] == 'longest-period'
        assert '18-second' in d['caption']
        assert d['stats']['station']['id'] in d['caption']

    def test_no_emoji(self, client, obs_text):
        import unicodedata
        obs_text(_feed(_healthy_rows()))
        caption = client.get('/api/social-card/highlight').get_json()['caption']
        assert not [c for c in caption
                    if ord(c) > 0x2100 and unicodedata.category(c) in ('So', 'Sk')]

    def test_does_not_claim_worldwide_tides_or_buoys(self, client, obs_text):
        obs_text(_feed(_healthy_rows()))
        caption = client.get('/api/social-card/highlight').get_json()['caption'].lower()
        for claim in ('worldwide tides', 'global tides', 'global buoys',
                      'buoys worldwide', 'tides anywhere'):
            assert claim not in caption

    def test_does_not_claim_a_swell_origin_it_cannot_know(self, client, obs_text):
        # Group velocity is a fact about the reading; how far away the swell was
        # generated needs its age, which one observation cannot give.
        obs_text(_feed(_healthy_rows(wvht=2.0, dpd=18.0, wspd=2.0)))
        caption = client.get('/api/social-card/highlight').get_json()['caption']
        assert not re.search(r'\d[\d,]*\s*(miles|km|nm)\s+away', caption, re.I)


class TestLayout:
    def _drawn(self, data, story):
        """Every string the renderer draws, with its box."""
        from PIL import ImageDraw
        boxes = []
        real = ImageDraw.ImageDraw.text

        def spy(self, xy, text, *a, **kw):
            if text and str(text).strip():
                font = kw.get('font') or (a[1] if len(a) > 1 else None)
                try:
                    bb = self.textbbox(xy, str(text), font=font)
                except Exception:
                    bb = (xy[0], xy[1], xy[0], xy[1])
                boxes.append((bb, str(text)))
            return real(self, xy, text, *a, **kw)

        ImageDraw.ImageDraw.text = spy
        try:
            app_module._render_highlight_card(data, fmt='PNG', story=story)
        finally:
            ImageDraw.ImageDraw.text = real
        return boxes

    def _data(self, **over):
        d = {
            'kind': 'highlight', 'highlight': 'biggest-seas', 'date': 'Jan 14',
            'observed': '2026-01-14T12:00Z', 'n_stations': 214,
            'head': ['Biggest seas', 'on the buoy network'],
            'value': '38.4', 'unit': 'ft',
            'label': 'SIGNIFICANT WAVE HEIGHT, MEASURED',
            'note': ('Significant wave height is the average of the highest '
                     'third. Individual waves run larger.'),
            'tiles': [('19 s', 'dominant period'), ('44F', 'water temp'),
                      ('12:00Z', 'observed')],
            'station': {'id': '46001', 'lat': 56.3, 'lon': -148.0,
                        'name': 'GULF OF ALASKA 175NM SOUTH OF YAKUTAT ALASKA'},
        }
        d.update(over)
        return d

    @pytest.mark.parametrize('story', [False, True])
    def test_nothing_leaves_the_canvas(self, story):
        H = 1920 if story else 1350
        for data in (self._data(),
                     self._data(value='104.9', unit='kt'),
                     self._data(station={'id': '51000', 'lat': -8.9, 'lon': 179.4,
                                         'name': 'A Very Long Station Name That '
                                                 'Should Not Be Allowed To Run Off'})):
            for box, text in self._drawn(data, story):
                assert box[0] >= 0 and box[2] <= 1080, f'{text!r} runs off the side'
                assert box[1] >= 0 and box[3] <= H, f'{text!r} runs off the bottom'

    @pytest.mark.parametrize('story', [False, True])
    def test_no_two_strings_overlap(self, story):
        boxes = self._drawn(self._data(), story)
        for i, (a, ta) in enumerate(boxes):
            for b, tb in boxes[i + 1:]:
                if min(a[2], b[2]) - max(a[0], b[0]) > 0 and \
                        min(a[3], b[3]) - max(a[1], b[1]) > 0:
                    pytest.fail(f'{ta!r} overlaps {tb!r}')

    def test_story_clears_instagram_chrome(self):
        # Measured off a posted story: the top chrome covers y 11-91 and the
        # reply bar covers y 1847-1920.
        for box, text in self._drawn(self._data(), True):
            assert box[3] <= 11 or box[1] >= 91, f'{text!r} sits under the account name'
            assert box[3] <= 1847, f'{text!r} sits under the reply bar'

    def test_the_map_actually_draws_land(self):
        from PIL import Image
        img = Image.new('RGB', (400, 300), (0, 0, 0))
        # Off California: there must be land in frame.
        assert app_module._draw_mini_map(img, 0, 0, 400, 300, 36.3, -122.1)
        # Mid-Pacific: no land, but it must still return cleanly.
        img2 = Image.new('RGB', (400, 300), (0, 0, 0))
        app_module._draw_mini_map(img2, 0, 0, 400, 300, -30.0, -140.0)

    def test_the_map_stays_inside_its_panel(self):
        # PIL does not clip polygons to a region, so drawing the map straight
        # onto the card painted Alaska across the headline.
        from PIL import Image
        img = Image.new('RGB', (400, 300), (7, 7, 7))
        app_module._draw_mini_map(img, 100, 100, 200, 100, 36.3, -122.1)
        assert img.getpixel((5, 5)) == (7, 7, 7)
        assert img.getpixel((395, 295)) == (7, 7, 7)


class TestSpectrum:
    def _spec(self, peaks, bins=48):
        """A synthetic spectrum with Gaussian bumps at the given periods."""
        import math
        freqs = [0.033 + i * (0.30 - 0.033) / (bins - 1) for i in range(bins)]
        energy = []
        for f in freqs:
            p = 1.0 / f
            e = 0.0
            for centre, amp, width in peaks:
                e += amp * math.exp(-((p - centre) ** 2) / (2 * width ** 2))
            energy.append(e)
        return {'frequencies': freqs, 'energy': energy}

    def test_finds_the_primary_and_a_separated_secondary(self, monkeypatch):
        monkeypatch.setattr(app_module, '_fetch_ndbc_spectrum',
                            lambda sid: self._spec([(16.0, 1.0, 1.2),
                                                    (8.0, 0.7, 1.0)]))
        app_module._cache.pop('ndbc:spec:X', None)
        s = app_module._spectrum_for_card('X')
        assert abs(s['peak'][0] - 16.0) < 1.5
        assert s['second'] and abs(s['second'][0] - 8.0) < 1.5

    def test_a_nearby_bump_is_not_a_second_swell(self, monkeypatch):
        # Bins either side of one crest describe the same swell twice.
        monkeypatch.setattr(app_module, '_fetch_ndbc_spectrum',
                            lambda sid: self._spec([(14.0, 1.0, 1.2)]))
        app_module._cache.pop('ndbc:spec:Y', None)
        assert app_module._spectrum_for_card('Y')['second'] is None

    def test_a_tail_is_not_a_second_swell(self, monkeypatch):
        monkeypatch.setattr(app_module, '_fetch_ndbc_spectrum',
                            lambda sid: self._spec([(16.0, 1.0, 1.2),
                                                    (7.0, 0.05, 1.0)]))
        app_module._cache.pop('ndbc:spec:Z', None)
        assert app_module._spectrum_for_card('Z')['second'] is None

    @pytest.mark.parametrize('bad', [
        None,
        {'frequencies': [], 'energy': []},
        {'frequencies': [0.1, 0.2], 'energy': [1.0]},          # mismatched
        {'frequencies': [0.1] * 20, 'energy': [0.0] * 20},     # all zero
        {'frequencies': [0.0] * 20, 'energy': [1.0] * 20},     # zero frequency
    ])
    def test_unusable_spectra_are_dropped_not_crashed(self, monkeypatch, bad):
        monkeypatch.setattr(app_module, '_fetch_ndbc_spectrum', lambda sid: bad)
        app_module._cache.pop('ndbc:spec:B', None)
        assert app_module._spectrum_for_card('B') is None

    def test_the_plot_stays_inside_its_panel(self, monkeypatch):
        # Same trap as the map: PIL does not clip, so a curve leaving the panel
        # would be painted across the card.
        from PIL import Image
        monkeypatch.setattr(app_module, '_fetch_ndbc_spectrum',
                            lambda sid: self._spec([(16.0, 1.0, 1.2)]))
        app_module._cache.pop('ndbc:spec:C', None)
        spec = app_module._spectrum_for_card('C')
        img = Image.new('RGB', (500, 400), (7, 7, 7))
        assert app_module._draw_spectrum(img, 100, 100, 300, 200, spec)
        assert img.getpixel((5, 5)) == (7, 7, 7)
        assert img.getpixel((495, 395)) == (7, 7, 7)

    def test_card_falls_back_to_a_full_width_map(self, client, obs_text, monkeypatch):
        from PIL import Image
        monkeypatch.setattr(app_module, '_fetch_ndbc_spectrum', lambda sid: None)
        obs_text(_feed(_healthy_rows()))
        assert client.get('/api/social-card/highlight').get_json()  # still publishes
        r = client.get('/social/highlight.png')
        assert r.status_code == 200
        assert Image.open(io.BytesIO(r.data)).size == (1080, 1350)

    def test_the_plot_is_dropped_when_it_would_contradict_the_headline(
            self, client, obs_text, monkeypatch):
        # A card reading "18 s" beside a plot labelled "8s" looks like an error.
        monkeypatch.setattr(app_module, '_fetch_ndbc_spectrum',
                            lambda sid: self._spec([(8.0, 1.0, 1.0)]))
        obs_text(_feed(_healthy_rows(wvht=2.0, dpd=18.0, wspd=2.0)))
        d = client.get('/api/social-card/highlight').get_json()
        assert d['highlight'] == 'longest-period'
        card = app_module._cache['social:highlight:auto']['data']
        assert card['spectrum'] is None

    def test_an_agreeing_plot_is_kept(self, client, obs_text, monkeypatch):
        monkeypatch.setattr(app_module, '_fetch_ndbc_spectrum',
                            lambda sid: self._spec([(18.0, 1.0, 1.2),
                                                    (8.0, 0.6, 1.0)]))
        obs_text(_feed(_healthy_rows(wvht=2.0, dpd=18.0, wspd=2.0)))
        client.get('/api/social-card/highlight')
        card = app_module._cache['social:highlight:auto']['data']
        assert card['spectrum'] is not None
