"""
Tests for the weekly forecast-accuracy Instagram card.

Covers the five routes added for it (/social/accuracy.{jpg,png},
/social/accuracy-story.{jpg,png}, /api/social-card/accuracy), the publication
guards that keep a stale or thin number off the feed, and the caption's
coverage claims.

The card states a measured error figure in public, so the tests here check the
numbers actually rendered against the fixture that produced them rather than
just asserting a 200.

Run: pytest tests/test_accuracy_card.py -v
"""
import io
import json
import re
from datetime import datetime, timedelta, timezone

import pytest

import app as app_module
from app import app


M_TO_FT = 3.28084


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Fixture stats
# ---------------------------------------------------------------------------

# Metre values chosen so the ft conversion lands on an exact tenth and the
# assertions below can hardcode it -- a test that recomputes round(m*3.28084,1)
# would pass even if the renderer used the wrong unit.
MAE_M = 0.3048          # 1.0 ft
RMSE_M = 0.6096         # 2.0 ft
BIN_MAE_M = {'0-24': 0.1524, '24-48': 0.3048, '48-72': 0.4572}  # 0.5/1.0/1.5 ft
N_PAIRS = 12345
WINDOW_DAYS = 30

# Six stations, all above ACCURACY_SPOTLIGHT_MIN_N, so the spotlight has a
# real rotation to pick from and the >=5 station guard is satisfied.
STATION_IDS = ['41110', '44013', '44025', '46042', '46053', '51201']


def _agg(n, mae_m):
    return {'n': n, 'bias_m': 0.0, 'mae_m': mae_m, 'rmse_m': mae_m * 2}


def _healthy_stats(generated=None, n_pairs=N_PAIRS, stations=None):
    """A stats.json shaped like the live one, fresh by default."""
    if generated is None:
        generated = (datetime.now(timezone.utc) - timedelta(hours=2)
                     ).strftime('%Y-%m-%dT%H:%M:%SZ')
    if stations is None:
        stations = STATION_IDS
    return {
        'generated': generated,
        'window_days': WINDOW_DAYS,
        'n_pairs': n_pairs,
        'overall': {
            'all': {'n': n_pairs, 'bias_m': 0.0,
                    'mae_m': MAE_M, 'rmse_m': RMSE_M},
            'bins': {k: _agg(4000 + i * 100, v)
                     for i, (k, v) in enumerate(BIN_MAE_M.items())},
        },
        'stations': {
            sid: {
                'name': f'Test Buoy {sid}',
                'lat': 34.0 + i, 'lon': -77.0 - i,
                'region': f'Region {i}',
                'all': _agg(500 + i * 10, MAE_M),
                'bins': {k: _agg(200, v) for k, v in BIN_MAE_M.items()},
            }
            for i, sid in enumerate(stations)
        },
    }


def _clear_caches():
    """Drop both cache entries the card depends on.

    cached() has a 1h TTL on 'social:accuracy' and 6h on 'verification:stats',
    so without this a guard test would be served the previously built healthy
    card and pass for entirely the wrong reason.
    """
    with app_module._cache_lock:
        app_module._cache.pop('social:accuracy', None)
        app_module._cache.pop('verification:stats', None)
    app_module._ACCURACY_CARD_LAST_SKIP = None


@pytest.fixture
def stats_file(tmp_path):
    """Point the app at a writable local stats file. Yields a setter.

    VERIFICATION_STATS_FILE is read as a module global inside
    _fetch_verification_stats, so patching the attribute is enough.
    """
    path = tmp_path / 'stats.json'
    orig = app_module.VERIFICATION_STATS_FILE

    def _set(stats):
        if stats is None:
            # A path that does not exist: the loader's open() raises, which is
            # the "stats unavailable" branch.
            app_module.VERIFICATION_STATS_FILE = str(tmp_path / 'missing.json')
        else:
            path.write_text(json.dumps(stats))
            app_module.VERIFICATION_STATS_FILE = str(path)
        _clear_caches()

    _set(_healthy_stats())
    yield _set
    app_module.VERIFICATION_STATS_FILE = orig
    _clear_caches()


IMAGE_ROUTES = [
    ('/social/accuracy.jpg', 'image/jpeg', (1080, 1350)),
    ('/social/accuracy.png', 'image/png', (1080, 1350)),
    ('/social/accuracy-story.jpg', 'image/jpeg', (1080, 1920)),
    ('/social/accuracy-story.png', 'image/png', (1080, 1920)),
]
ALL_ROUTES = [r[0] for r in IMAGE_ROUTES] + ['/api/social-card/accuracy']


# ===================================================================
# 1. HEALTHY RENDER
# ===================================================================

class TestHealthyCard:
    """With fresh, thick-enough stats every route serves a real card."""

    @pytest.mark.parametrize('path,mime,size', IMAGE_ROUTES)
    def test_image_route_content_type_and_dimensions(self, client, stats_file,
                                                     path, mime, size):
        from PIL import Image
        r = client.get(path)
        assert r.status_code == 200, r.get_data(as_text=True)
        assert r.mimetype == mime
        img = Image.open(io.BytesIO(r.data))
        assert (img.width, img.height) == size

    @pytest.mark.parametrize('path', [p for p, _, _ in IMAGE_ROUTES])
    def test_image_is_not_blank(self, client, stats_file, path):
        from PIL import Image
        img = Image.open(io.BytesIO(client.get(path).data)).convert('RGB')
        # A card that renders as a flat rectangle is the failure mode a status
        # check cannot see.
        assert len(img.getcolors(maxcolors=200000) or []) > 20

    @pytest.mark.parametrize('path', ['/social/accuracy.jpg',
                                      '/social/accuracy-story.jpg'])
    def test_jpeg_variants_are_really_jpeg(self, client, stats_file, path):
        # Magic bytes, not the Content-Type header: the header is set by hand
        # in _serve_accuracy_card and would keep saying image/jpeg even if the
        # renderer handed back a PNG. Instagram's API rejects PNG uploads.
        data = client.get(path).data
        assert data[:3] == b'\xff\xd8\xff'
        assert data[-2:] == b'\xff\xd9'

    @pytest.mark.parametrize('path', ['/social/accuracy.png',
                                      '/social/accuracy-story.png'])
    def test_png_variants_are_really_png(self, client, stats_file, path):
        assert client.get(path).data[:8] == b'\x89PNG\r\n\x1a\n'

    def test_meta_route_shape(self, client, stats_file):
        r = client.get('/api/social-card/accuracy')
        assert r.status_code == 200
        d = r.get_json()
        assert d['kind'] == 'accuracy'
        for key in ('date', 'caption', 'image_url', 'image_url_png',
                    'story_image_url', 'stats'):
            assert key in d, key
        assert d['image_url'].endswith('/social/accuracy.jpg')
        assert d['story_image_url'].endswith('/social/accuracy-story.jpg')

    def test_meta_numbers_match_the_fixture(self, client, stats_file):
        s = client.get('/api/social-card/accuracy').get_json()['stats']
        assert s['mae_ft'] == 1.0                     # 0.3048 m
        assert s['rmse_ft'] == 2.0                    # 0.6096 m
        assert s['n_pairs'] == N_PAIRS
        assert s['n_stations'] == len(STATION_IDS)
        assert s['window_days'] == WINDOW_DAYS
        assert s['bias_ft'] == 0.0
        assert [b['mae_ft'] for b in s['bins']] == [0.5, 1.0, 1.5]
        assert [b['label'] for b in s['bins']] == ['0-24h', '24-48h', '48-72h']

    def test_region_slug_route_still_404s_on_an_unknown_slug(self, client):
        # /api/social-card/accuracy is a static rule sitting in front of
        # /api/social-card/<region_slug>; adding it must not have turned the
        # region route into a catch-all.
        assert client.get('/api/social-card/not-a-region').status_code == 404


# ===================================================================
# 2. CAPTION
# ===================================================================

# Ranges that cover the emoji planes plus the dingbat/symbol blocks that get
# rendered as emoji. The repo rule is no emoji anywhere, captions included.
_EMOJI = re.compile(
    '['
    '\u2190-\u21FF\u2300-\u23FF\u2460-\u24FF\u25A0-\u27BF'
    '\u2B00-\u2BFF\uFE0F\u2600-\u26FF'
    '\U0001F000-\U0001FAFF]'
)


class TestCaption:
    @pytest.fixture(autouse=True)
    def _caption(self, client, stats_file):
        self.caption = client.get('/api/social-card/accuracy').get_json()['caption']

    def test_states_the_headline_numbers(self):
        assert '1.0 ft' in self.caption
        assert f'{N_PAIRS:,}' in self.caption          # "12,345"
        assert f'{WINDOW_DAYS} days' in self.caption

    def test_states_the_lead_time_breakdown(self):
        assert '0.5 ft at 0-24h' in self.caption
        assert '1.0 ft at 24-48h' in self.caption
        assert '1.5 ft at 48-72h' in self.caption

    def test_names_the_spotlight_buoy_from_the_fixture(self):
        assert 'Buoy of the week:' in self.caption
        assert any(f'Test Buoy {sid}' in self.caption for sid in STATION_IDS)

    def test_no_emoji(self):
        found = _EMOJI.findall(self.caption)
        assert not found, f'emoji in caption: {found}'
        assert self.caption.isprintable() or '\n' in self.caption

    def test_scopes_buoy_verification_to_the_us(self):
        assert 'United States only' in self.caption
        assert 'NDBC' in self.caption

    def test_never_claims_worldwide_buoy_or_tide_coverage(self):
        low = self.caption.lower()
        assert 'worldwide' not in low
        # "global" is true of the forecasts and false of the buoys and tides,
        # so the only thing that matters is which sentence the word lands in.
        # Split on a period or newline followed by whitespace so the URL's
        # dots do not count as sentence ends.
        for sentence in re.split(r'(?<=[.!?])\s|\n', low):
            if re.search(r'global|every ocean|planet', sentence):
                assert 'buoy' not in sentence, f'global buoy claim: {sentence!r}'
                assert 'tide' not in sentence, f'global tide claim: {sentence!r}'

    def test_links_the_accuracy_page(self):
        assert 'freesurfforecast.com/accuracy' in self.caption

    def test_ends_with_hashtags(self):
        assert self.caption.rstrip().splitlines()[-1].startswith('#surf')


# ===================================================================
# 3. PUBLICATION GUARDS
# ===================================================================

class TestGuards:
    """Every guard must refuse with 503, and say which one tripped.

    503 rather than 500 on purpose: the posting script treats it as "nothing
    to post this week" and exits 0, while a 500 is a failed run.
    """

    def _assert_all_refuse(self, client, expect_in_reason):
        for path in ALL_ROUTES:
            r = client.get(path)
            assert r.status_code == 503, f'{path} returned {r.status_code}'
            body = r.get_data(as_text=True)
            assert expect_in_reason in body, f'{path}: {body!r}'
            assert 'noindex' in r.headers.get('X-Robots-Tag', '')

    def test_stale_stats_refuse(self, client, stats_file):
        old = (datetime.now(timezone.utc) - timedelta(days=5)
               ).strftime('%Y-%m-%dT%H:%M:%SZ')
        stats_file(_healthy_stats(generated=old))
        self._assert_all_refuse(client, 'max 48h')

    def test_thin_sample_refuses(self, client, stats_file):
        stats_file(_healthy_stats(n_pairs=app_module.ACCURACY_CARD_MIN_PAIRS - 1))
        self._assert_all_refuse(client, 'scored pairs')

    def test_unavailable_stats_refuse(self, client, stats_file):
        stats_file(None)
        self._assert_all_refuse(client, 'verification stats unavailable')

    def test_too_few_stations_refuse(self, client, stats_file):
        stats_file(_healthy_stats(stations=STATION_IDS[:2]))
        self._assert_all_refuse(client, 'stations have scored pairs')

    def test_unparseable_timestamp_refuses(self, client, stats_file):
        stats_file(_healthy_stats(generated='not-a-date'))
        self._assert_all_refuse(client, 'unparseable generated timestamp')

    def test_missing_mae_refuses(self, client, stats_file):
        stats = _healthy_stats()
        stats['overall']['all']['mae_m'] = None
        stats_file(stats)
        self._assert_all_refuse(client, 'overall mae_m missing')

    def test_future_timestamp_refuses(self, client, stats_file):
        """A stamp in the future is a broken clock or a broken writer -- the
        same failure the age check exists to catch. Testing only one side of
        the comparison left the guard permanently open instead of tripping."""
        ahead = (datetime.now(timezone.utc) + timedelta(days=30)
                 ).strftime('%Y-%m-%dT%H:%M:%SZ')
        stats_file(_healthy_stats(generated=ahead))
        self._assert_all_refuse(client, 'in the future')

    def test_small_clock_skew_still_publishes(self, client, stats_file):
        """A couple of minutes of skew between the pipeline runner and the web
        dyno is normal and must not cost a week's post."""
        skewed = (datetime.now(timezone.utc) + timedelta(minutes=2)
                  ).strftime('%Y-%m-%dT%H:%M:%SZ')
        stats_file(_healthy_stats(generated=skewed))
        assert client.get('/social/accuracy.jpg').status_code == 200

    @pytest.mark.parametrize('mangle,label', [
        (lambda s: s['overall'].update(bins=None), 'null bins'),
        (lambda s: s['stations'].update({STATION_IDS[0]: 'not-a-dict'}), 'string station'),
        (lambda s: s['overall']['all'].update(mae_m='0.152'), 'string mae'),
        (lambda s: s.update(stations='nope'), 'string stations'),
        (lambda s: s['overall'].update(all=None), 'null overall'),
    ])
    def test_unusable_stats_shapes_never_500(self, client, stats_file, mangle, label):
        """compute_stats cannot emit these, but a 500 would make the posting
        script retry three times and paint the run red over data no retry can
        fix. Every one of them has to end as a card or as a clean refusal."""
        stats = _healthy_stats()
        mangle(stats)
        stats_file(stats)
        for path in ALL_ROUTES:
            code = client.get(path).status_code
            assert code in (200, 503), f'{label} on {path} returned {code}'

    def test_meta_refusal_is_json_with_a_reason(self, client, stats_file):
        stats_file(None)
        r = client.get('/api/social-card/accuracy')
        assert r.status_code == 503
        # scripts/instagram_publish.py reads this field to print why a week
        # was skipped; without it a skipped post is invisible.
        assert r.get_json()['reason']

    def test_recovers_once_the_stats_are_healthy_again(self, client, stats_file):
        stats_file(None)
        assert client.get('/social/accuracy.jpg').status_code == 503
        stats_file(_healthy_stats())
        assert client.get('/social/accuracy.jpg').status_code == 200


# ===================================================================
# 4. SPOTLIGHT ROTATION
# ===================================================================

class TestSpotlight:
    def test_stable_within_a_week(self, client, stats_file):
        first = client.get('/api/social-card/accuracy').get_json()
        _clear_caches()
        second = client.get('/api/social-card/accuracy').get_json()
        # Same week, same inputs, cache bypassed: a re-run or a cache miss
        # must not swap the buoy out from under a half-published post.
        assert (first['stats']['spotlight']['id']
                == second['stats']['spotlight']['id'])
        assert first['caption'] == second['caption']

    def test_spotlight_is_one_of_the_fixture_stations(self, client, stats_file):
        sp = client.get('/api/social-card/accuracy').get_json()['stats']['spotlight']
        assert sp['id'] in STATION_IDS
        assert sp['mae_ft'] == 1.0
        assert sp['n'] >= app_module.ACCURACY_SPOTLIGHT_MIN_N

    def test_thin_stations_are_not_spotlighted(self, client, stats_file):
        stats = _healthy_stats()
        for i, sid in enumerate(STATION_IDS):
            stats['stations'][sid]['all']['n'] = (
                app_module.ACCURACY_SPOTLIGHT_MIN_N - 1)
        # Leave exactly one station eligible; it must be the one picked.
        stats['stations']['46042']['all']['n'] = 900
        stats_file(stats)
        sp = client.get('/api/social-card/accuracy').get_json()['stats']['spotlight']
        assert sp['id'] == '46042'

    def test_card_still_renders_with_no_eligible_spotlight(self, client, stats_file):
        stats = _healthy_stats()
        for sid in STATION_IDS:
            stats['stations'][sid]['all']['n'] = 10
        stats['overall']['all']['n'] = N_PAIRS
        stats_file(stats)
        r = client.get('/social/accuracy.jpg')
        assert r.status_code == 200
        assert client.get('/api/social-card/accuracy').get_json(
        )['stats']['spotlight'] is None

    def test_card_still_renders_with_no_lead_bins(self, client, stats_file):
        stats = _healthy_stats()
        stats['overall']['bins'] = {}
        stats_file(stats)
        assert client.get('/social/accuracy.jpg').status_code == 200
        assert client.get('/api/social-card/accuracy').get_json(
        )['stats']['bins'] == []


# ===================================================================
# 5. CRAWL SURFACE
# ===================================================================

class TestCrawlSurface:
    """The site is crawl-budget constrained: these URLs serve no searcher and
    must stay out of the sitemap and out of the index."""

    @pytest.mark.parametrize('path', ALL_ROUTES)
    def test_noindex_header(self, client, stats_file, path):
        h = client.get(path).headers.get('X-Robots-Tag', '')
        assert 'noindex' in h and 'nofollow' in h

    def test_absent_from_sitemap(self, client):
        xml = client.get('/sitemap.xml').get_data(as_text=True)
        assert '/social/accuracy' not in xml
        assert '/api/social-card' not in xml

    def test_not_linked_from_any_template(self):
        import pathlib
        root = pathlib.Path(app_module.__file__).parent / 'templates'
        for f in root.rglob('*.html'):
            assert '/social/accuracy' not in f.read_text(), f


# ===================================================================
# 6. LAYOUT
# ===================================================================

class TestLayout:
    """The card exists to state numbers, so overlapping text is a defect even
    when every string is individually inside the canvas."""

    @staticmethod
    def _drawn_boxes(data, story):
        """Render once, recording the ink box of every string drawn."""
        from PIL import ImageDraw
        boxes = []
        original = ImageDraw.ImageDraw.text

        def recording(self, xy, text, *a, **kw):
            boxes.append((self.textbbox(xy, text, font=kw.get('font')), text))
            return original(self, xy, text, *a, **kw)

        ImageDraw.ImageDraw.text = recording
        try:
            app_module._render_accuracy_card(data, fmt='PNG', story=story)
        finally:
            ImageDraw.ImageDraw.text = original
        return boxes

    @pytest.mark.parametrize('story', [False, True])
    def test_no_two_strings_overlap(self, client, stats_file, story):
        # The buoy-of-the-week name and its detail line collided by 2-4px on
        # every real card until the feed panel was given more height.
        data = client.get('/api/social-card/accuracy').get_json()
        card = app_module.cached('social:accuracy', app_module._accuracy_card_data,
                                 ttl=3600)
        assert card, data
        boxes = self._drawn_boxes(card, story)
        for i, (a, ta) in enumerate(boxes):
            for b, tb in boxes[i + 1:]:
                overlap_x = min(a[2], b[2]) - max(a[0], b[0])
                overlap_y = min(a[3], b[3]) - max(a[1], b[1])
                assert not (overlap_x > 0 and overlap_y > 0), (
                    f'{ta!r} overlaps {tb!r} by {overlap_x}x{overlap_y}px')

    @pytest.mark.parametrize('story', [False, True])
    def test_nothing_is_drawn_off_the_canvas(self, client, stats_file, story):
        card = app_module.cached('social:accuracy', app_module._accuracy_card_data,
                                 ttl=3600)
        height = 1920 if story else 1350
        for box, text in self._drawn_boxes(card, story):
            assert box[0] >= 0 and box[2] <= 1080, f'{text!r} runs off the sides'
            assert box[1] >= 0 and box[3] <= height, f'{text!r} runs off the ends'
