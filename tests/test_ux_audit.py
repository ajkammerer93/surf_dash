"""Tests for the pure helpers in scripts/ux_audit.py.

Everything here runs without a browser and without the network, because the
maths is the part that can be silently wrong: a contrast ratio off by a factor
never announces itself, it just produces a snapshot that reads fine and says
the wrong thing.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'scripts'))

import ux_audit  # noqa: E402


# --- colour parsing -------------------------------------------------------

@pytest.mark.parametrize('value,expected', [
    ('#fff', (255, 255, 255, 1.0)),
    ('#000000', (0, 0, 0, 1.0)),
    ('#6b6b6b', (107, 107, 107, 1.0)),
    ('rgb(16, 32, 48)', (16, 32, 48, 1.0)),
    ('rgba(255, 255, 255, 0.5)', (255, 255, 255, 0.5)),
    ('RGB(1,2,3)', (1, 2, 3, 1.0)),
    ('transparent', (0, 0, 0, 0.0)),
    ('rgba(0, 0, 0, 0)', (0, 0, 0, 0.0)),
    ('white', (255, 255, 255, 1.0)),
    ('#80808080', (128, 128, 128, 128 / 255.0)),
])
def test_parse_css_color(value, expected):
    got = ux_audit.parse_css_color(value)
    assert got[:3] == expected[:3]
    assert got[3] == pytest.approx(expected[3], abs=0.01)


@pytest.mark.parametrize('value', ['', None, 'chartreuse', 'color(display-p3 1 0 0)',
                                   '#12345', 'rgb(1)'])
def test_parse_css_color_gives_up_loudly(value):
    # A guess would produce a plausible-looking wrong ratio, so unknown forms
    # return None and get counted as unparsed instead.
    assert ux_audit.parse_css_color(value) is None


# --- contrast maths -------------------------------------------------------

def test_relative_luminance_endpoints():
    assert ux_audit.relative_luminance((255, 255, 255)) == pytest.approx(1.0, abs=1e-9)
    assert ux_audit.relative_luminance((0, 0, 0)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize('fg,bg,expected', [
    ((0, 0, 0), (255, 255, 255), 21.0),        # the two extremes
    ((255, 255, 255), (255, 255, 255), 1.0),   # identical colours
    ((119, 119, 119), (255, 255, 255), 4.48),  # #777 on white, just under AA
    ((107, 107, 107), (245, 245, 245), 4.89),  # --text-muted on the light --bg
    ((136, 136, 136), (245, 245, 245), 3.25),  # #888, the value it replaced
    ((0, 0, 255), (255, 255, 255), 8.59),      # pure blue on white
    ((232, 232, 232), (10, 10, 10), 16.16),    # the dark theme's body text
])
def test_contrast_ratio_known_pairs(fg, bg, expected):
    assert ux_audit.contrast_ratio(fg, bg) == pytest.approx(expected, abs=0.01)


def test_contrast_ratio_is_symmetric():
    a, b = (30, 90, 150), (240, 240, 200)
    assert ux_audit.contrast_ratio(a, b) == pytest.approx(ux_audit.contrast_ratio(b, a))


@pytest.mark.parametrize('size,weight,expected', [
    (12, 400, 4.5),
    (23.9, 400, 4.5),
    (24, 400, 3.0),
    (18.66, 700, 3.0),
    (18.66, 400, 4.5),
    (18.65, 700, 4.5),
    (30, '700', 3.0),
    (None, None, 4.5),
    ('bogus', 'bold', 4.5),
])
def test_required_contrast(size, weight, expected):
    assert ux_audit.required_contrast(size, weight) == expected


# --- compositing ----------------------------------------------------------

def test_composite_half_alpha_white_over_black():
    assert ux_audit.composite((255, 255, 255, 0.5), (0, 0, 0)) == (128, 128, 128)


def test_composite_stack_paints_back_to_front():
    # Stack arrives nearest-first: a half-alpha white sitting on solid black
    # must end up mid grey, not white.
    assert ux_audit.composite_stack(['rgba(255,255,255,0.5)', 'rgb(0,0,0)']) == (128, 128, 128)


def test_composite_stack_falls_back_to_canvas_when_empty():
    assert ux_audit.composite_stack([], fallback=(10, 10, 10)) == (10, 10, 10)


def test_composite_stack_skips_fully_transparent_layers():
    assert ux_audit.composite_stack(['rgba(0, 0, 0, 0)', '#ffffff']) == (255, 255, 255)


# --- sample evaluation ----------------------------------------------------

def _sample(**kw):
    base = {'selector': 'p.x', 'text': 'hello', 'color': '#767676',
            'bg_stack': ['#ffffff'], 'canvas': 'rgb(255, 255, 255)',
            'font_size': 12, 'font_weight': '400'}
    base.update(kw)
    return base


def test_evaluate_contrast_flags_only_sub_threshold():
    out = ux_audit.evaluate_contrast_samples([
        _sample(color='#888888'),            # 3.54 on white -> flagged
        _sample(color='#595959'),            # 7.0 on white -> fine
        _sample(color='#888888', font_size=30),  # large text, 3:1 applies
    ])
    assert out['sampled'] == 3
    assert out['low_count'] == 1
    assert out['low'][0]['fg'] == '#888888'
    assert out['low'][0]['bg'] == '#ffffff'
    assert out['low'][0]['ratio'] == pytest.approx(3.54, abs=0.01)
    assert out['low'][0]['required'] == 4.5


def test_evaluate_contrast_composites_translucent_text():
    # 50% white text on black reads as mid grey, ratio ~5.3, not 21.
    out = ux_audit.evaluate_contrast_samples([
        _sample(color='rgba(255, 255, 255, 0.5)', bg_stack=['#000000'],
                canvas='rgb(0, 0, 0)')])
    assert out['low_count'] == 0
    out2 = ux_audit.evaluate_contrast_samples([
        _sample(color='rgba(255, 255, 255, 0.18)', bg_stack=['#000000'],
                canvas='rgb(0, 0, 0)')])
    assert out2['low_count'] == 1


def test_evaluate_contrast_counts_unparsed_rather_than_guessing():
    out = ux_audit.evaluate_contrast_samples([_sample(color='color(srgb 1 0 0)')])
    assert out['unparsed'] == 1
    assert out['low_count'] == 0


def test_evaluate_contrast_calls_image_backdrops_indeterminate():
    """The map marker: a yellow star in a transparent Leaflet pane over tile
    images, where the colour walk lands on .leaflet-container's #ddd default.
    That grey is painted nowhere near the star, so the sample is counted as
    unmeasurable rather than reported at 1.26:1 every single day."""
    out = ux_audit.evaluate_contrast_samples([
        _sample(color='rgb(255, 255, 0)', bg_stack=['rgb(221, 221, 221)'],
                image_backdrop=True, bg_over_image=[], font_size=20)])
    assert out['indeterminate'] == 1
    assert out['low_count'] == 0


def test_evaluate_contrast_still_fails_text_no_image_could_rescue():
    """The cam attribution bar is 75% black over live video. The video cannot
    be read, but the bar over it can: even against the darkest frame the link
    reaches only 3.9:1, so the failure holds whatever is playing and must not
    be filed away as "cannot tell"."""
    out = ux_audit.evaluate_contrast_samples([
        _sample(color='rgb(107, 107, 107)', font_size=9, image_backdrop=True,
                bg_stack=['rgba(0, 0, 0, 0.75)', '#ffffff'],
                bg_over_image=['rgba(0, 0, 0, 0.75)'])])
    assert out['indeterminate'] == 0
    assert out['low_count'] == 1
    # The friendlier end of the bracket, so the reported figure is the best the
    # element can ever manage rather than a number picked from one frame.
    assert out['low'][0]['ratio'] == pytest.approx(3.94, abs=0.01)


def test_evaluate_contrast_ignores_colours_painted_under_the_image():
    """Layers below the imagery tell a reader nothing. Dropping them is what
    separates the star (nothing above the tiles, so any backdrop is possible)
    from the attribution bar (a known 75% black over whatever plays)."""
    over_white = ux_audit._best_case_over_image((0, 0, 0, 1.0), [])
    assert over_white[2] == (255, 255, 255)
    darkened = ux_audit._best_case_over_image((255, 255, 255, 1.0),
                                              ['rgba(0, 0, 0, 0.75)'])
    assert darkened[2] == (0, 0, 0)


def test_evaluate_contrast_caps_and_sorts_worst_first():
    samples = [_sample(color='#%02x%02x%02x' % (v, v, v))
               for v in range(120, 200)]
    out = ux_audit.evaluate_contrast_samples(samples)
    assert out['low_count'] == 80
    assert len(out['low']) == ux_audit.MAX_CONTRAST_REPORTED
    ratios = [x['ratio'] for x in out['low']]
    assert ratios == sorted(ratios)


def test_evaluate_contrast_marks_bracketed_rows_as_bounded():
    """A row whose background is an end of the black/white bracket is not a
    colour anyone can find on the page. It has to say so, or a reader
    comparing the snapshot against a screenshot concludes the audit is broken.
    """
    out = ux_audit.evaluate_contrast_samples([
        _sample(color='rgb(107, 107, 107)', font_size=9, image_backdrop=True,
                bg_stack=['rgba(0, 0, 0, 0.75)', '#ffffff'],
                bg_over_image=['rgba(0, 0, 0, 0.75)'])])
    assert out['low'][0]['bounded'] is True
    plain = ux_audit.evaluate_contrast_samples([
        _sample(color='#999999', bg_stack=['#ffffff'])])
    assert plain['low'][0]['bounded'] is False


# --- layout shift ranking -------------------------------------------------

def test_rank_shift_records_puts_the_largest_first():
    ranked = ux_audit.rank_shift_records([
        {'value': 0.001}, {'value': 0.12}, {'value': 0.03}])
    assert [r['value'] for r in ranked] == [0.12, 0.03, 0.001]


def test_rank_shift_records_keeps_the_phase_the_observer_stamped():
    """The stamp is the only witness to when a shift actually fired; ranking
    reorders the list, so an index-based split would relabel every record."""
    ranked = ux_audit.rank_shift_records(
        [{'value': 0.01, 'phase': 'scroll'}, {'value': 0.02, 'phase': 'load'}],
        load_count=1)
    assert [(r['value'], r['phase']) for r in ranked] == [
        (0.02, 'load'), (0.01, 'scroll')]


def test_rank_shift_records_falls_back_to_the_load_count():
    ranked = ux_audit.rank_shift_records(
        [{'value': 0.02}, {'value': 0.03}], load_count=1)
    by_value = {r['value']: r['phase'] for r in ranked}
    assert by_value == {0.02: 'load', 0.03: 'scroll'}


def test_rank_shift_records_leaves_the_tag_off_without_a_count():
    ranked = ux_audit.rank_shift_records([{'value': 0.02}])
    assert 'phase' not in ranked[0]


def test_rank_shift_records_truncates_and_survives_junk():
    records = [{'value': v / 1000.0} for v in range(20)] + [None, 'nonsense']
    ranked = ux_audit.rank_shift_records(records)
    assert len(ranked) == ux_audit.MAX_SHIFT_REPORTED
    assert ranked[0]['value'] == 0.019


# --- headings -------------------------------------------------------------

@pytest.mark.parametrize('levels,expected', [
    ([1, 2, 2, 3], []),
    ([1, 3], [[1, 3, 1]]),
    ([1, 2, 4, 5], [[2, 4, 2]]),
    ([2, 1, 2, 3], []),          # going back up is not a skip
    ([1, 2, 3, 6], [[3, 6, 3]]),
    ([], []),
    ([3], []),                   # first heading sets the baseline
    ([1, 2, 4, 6], [[2, 4, 2], [4, 6, 3]]),
])
def test_heading_skips(levels, expected):
    assert ux_audit.heading_skips(levels) == expected


def test_heading_skips_ignores_junk_levels():
    # Junk entries drop out of the sequence rather than resetting it, so the
    # h1 still counts as the level the h3 skipped from.
    assert ux_audit.heading_skips([1, None, 'h2', 3]) == [[1, 3, 3]]


# --- palette parser -------------------------------------------------------

PALETTE_CSS = """
    <style>
        :root {
            --bg: #0a0a0a;
            --text: #e8e8e8;
            --text-muted: #666;
            --font-body: 'SF Mono', monospace;
        }

        [data-theme="light"] {
            --bg: #f5f5f5;
            --text: #1a1a1a;
            --text-muted: #6b6b6b;  /* 4.7:1 on white */
        }

        body { background: var(--bg); }
    </style>
"""


def test_parse_palette_pairs_dark_and_light():
    out = ux_audit.parse_palette(PALETTE_CSS)
    assert out['count'] == 4
    assert out['root_is_dark'] is True
    assert out['has_light_overrides'] is True
    assert out['variables']['--bg'] == {'dark': '#0a0a0a', 'light': '#f5f5f5'}
    assert out['variables']['--text-muted']['light'] == '#6b6b6b'


def test_parse_palette_keeps_vars_with_no_light_override():
    out = ux_audit.parse_palette(PALETTE_CSS)
    assert out['variables']['--font-body']['light'] is None
    assert out['variables']['--font-body']['dark'] == "'SF Mono', monospace"


def test_parse_palette_says_so_instead_of_returning_an_empty_dict():
    # An empty dict would read as "this site has no theme", which is a very
    # different claim from "the regex found nothing".
    out = ux_audit.parse_palette('body { color: red; }')
    assert 'unavailable' in out
    assert 'variables' not in out


def test_parse_palette_reads_the_real_template():
    out = ux_audit.collect_palette()
    assert out['source'] == 'templates/index.html'
    assert 'unavailable' not in out
    assert out['count'] > 10
    assert out['variables']['--bg']['dark'] and out['variables']['--bg']['light']


# --- history row ----------------------------------------------------------

def _snapshot():
    return {
        'generated': '2026-08-20T12:00:00Z',
        'base_url': 'https://freesurfforecast.com',
        'build': {'version': 'v0.11.40', 'git_short': 'deadbeef',
                  'git_sha': 'deadbeef' * 5},
        'pages': [{
            'key': '/|mobile|dark', 'path': '/', 'viewport': 'mobile',
            'theme': 'dark', 'status': 200, 'ttfb_ms': 128, 'dcl_ms': 300,
            'load_ms': 624, 'cls': 0.0066, 'lcp_ms': 1432,
            'console': {'error': 2, 'warning': 1, 'log': 40},
            'console_samples': [{'level': 'error', 'text': 'x' * 300}],
            'failed_requests': [{'url': 'a', 'status': 404},
                                {'url': 'b', 'status': 'failed'}],
            'overflow': {'overflows': False, 'offender_count': 3,
                         'clipped_count': 3, 'offenders': [{'selector': 'div'}]},
            'tap_targets': {'total': 90, 'small_count': 56, 'small': [{'selector': 'a'}]},
            'tiny_text': {'count': 286, 'items': [{'selector': 'span'}]},
            'images': {'total': 8, 'missing_alt': 0, 'no_dimensions': 8, 'failed': 0},
            'headings': {'h1_count': 1, 'levels': [1, 3, 2], 'skips': [[1, 3, 1]]},
            'contrast': {'sampled': 400, 'low_count': 143, 'low': [{'selector': 'p'}]},
        }],
    }


def test_history_row_is_numeric_and_small():
    row = ux_audit.history_row(_snapshot())
    assert row['version'] == 'v0.11.40'
    assert row['git_sha'] == 'deadbeef'
    assert row['page_errors'] == 0
    page = row['pages']['/|mobile|dark']
    assert page['cls'] == 0.0066
    assert page['console_errors'] == 2
    assert page['failed_requests'] == 2
    assert page['low_contrast'] == 143
    assert page['heading_skips'] == 1
    assert page['small_tap_targets'] == 56
    # The whole point of the trimmed row is that ux-history.jsonl stays small
    # enough to read whole every week, so no selector lists may leak into it.
    blob = json.dumps(row)
    assert 'selector' not in blob
    assert len(blob) < 1200


def test_history_row_records_a_page_that_failed():
    snap = _snapshot()
    snap['pages'][0]['error'] = 'navigation: Timeout 45000ms exceeded.'
    row = ux_audit.history_row(snap)
    assert row['page_errors'] == 1
    assert row['pages']['/|mobile|dark']['error'].startswith('navigation:')


def test_history_row_survives_an_unavailable_run():
    # A run that could not reach the site still gets a row, so a gap in the
    # history is visible as a reason rather than as missing weeks.
    row = ux_audit.history_row({
        'generated': '2026-08-20T12:00:00Z', 'base_url': 'http://127.0.0.1:9',
        'build': {'version': 'v0.11.40', 'git_short': 'deadbeef'},
        'unavailable': 'base URL unreachable: connection refused'})
    assert row['unavailable'].startswith('base URL unreachable')
    assert row['pages'] == {}


def test_history_row_tolerates_missing_sections():
    row = ux_audit.history_row({'generated': 'now', 'pages': [
        {'key': '/x|mobile', 'status': 200}]})
    assert row['pages']['/x|mobile']['low_contrast'] is None
    assert row['pages']['/x|mobile']['h1_count'] is None


def test_history_row_counts_uncaught_exceptions_as_console_errors():
    """An uncaught exception arrives as 'pageerror', not 'error'. Counting only
    the latter undercounts a throwing page by exactly the bucket that matters,
    in the file the ux-data branch exists to preserve."""
    snap = _snapshot()
    snap['pages'][0]['console'] = {'error': 2, 'warning': 1, 'pageerror': 3}
    row = ux_audit.history_row(snap)
    assert row['pages']['/|mobile|dark']['console_errors'] == 5


def test_history_row_reports_checks_that_threw():
    """A check that throws records '<name>_error' and the page still looks
    measured. Unreported, that is a check producing no data forever."""
    snap = _snapshot()
    snap['pages'][0].pop('contrast')
    snap['pages'][0].pop('headings')
    snap['pages'][0]['contrast_error'] = 'boom'
    snap['pages'][0]['headings_error'] = 'boom'
    row = ux_audit.history_row(snap)
    page = row['pages']['/|mobile|dark']
    assert page['check_errors'] == ['contrast', 'headings']
    assert row['check_errors'] == 2
    assert page['low_contrast'] is None
    # None, not 0: "no heading skips" and "the check never ran" are different
    # answers and only one of them is good news.
    assert page['heading_skips'] is None


def test_history_row_carries_the_unparsed_colour_count():
    """oklch()/color-mix() are skipped rather than guessed, so a palette
    modernisation could zero the contrast check while every log line still
    read fine. The count has to travel next to low_contrast."""
    snap = _snapshot()
    snap['pages'][0]['contrast'] = {'sampled': 4, 'unparsed': 3, 'low_count': 0}
    row = ux_audit.history_row(snap)
    assert row['pages']['/|mobile|dark']['contrast_unparsed'] == 3


def test_history_row_carries_the_indeterminate_count():
    """Text over map tiles and cam frames is counted, not scored. Like the
    unparsed count, it has to sit next to low_contrast: a check that quietly
    stops looking at half a page must never read as a clean result."""
    snap = _snapshot()
    snap['pages'][0]['contrast'] = {'sampled': 40, 'unparsed': 0,
                                    'indeterminate': 3, 'low_count': 0}
    row = ux_audit.history_row(snap)
    assert row['pages']['/|mobile|dark']['contrast_indeterminate'] == 3


def test_history_row_separates_first_party_request_failures():
    snap = _snapshot()
    snap['pages'][0]['failed_requests'] = [
        {'url': 'https://youtube-nocookie.com/api/stats', 'status': 'failed',
         'first_party': False},
        {'url': 'https://freesurfforecast.com/missing.png', 'status': 404,
         'first_party': True},
    ]
    row = ux_audit.history_row(snap)
    page = row['pages']['/|mobile|dark']
    assert page['failed_requests'] == 2
    assert page['failed_first_party'] == 1


# --- first-party classification -------------------------------------------

@pytest.mark.parametrize('url,base,expected', [
    ('https://freesurfforecast.com/faq', 'https://freesurfforecast.com', True),
    ('https://freesurfforecast.com/a.png?v=2', 'https://freesurfforecast.com/', True),
    ('https://cdn.freesurfforecast.com/a.png', 'https://freesurfforecast.com', True),
    ('http://127.0.0.1:8123/faq', 'http://127.0.0.1:8123', True),
    ('https://www.youtube-nocookie.com/api/stats', 'https://freesurfforecast.com', False),
    # Not a suffix match on the raw string: evilfreesurfforecast.com must not
    # count as ours.
    ('https://evilfreesurfforecast.com/x', 'https://freesurfforecast.com', False),
    ('', 'https://freesurfforecast.com', False),
    (None, 'https://freesurfforecast.com', False),
    ('https://freesurfforecast.com/x', '', False),
])
def test_is_first_party(url, base, expected):
    assert ux_audit.is_first_party(url, base) is expected


# --- preflight ------------------------------------------------------------

def test_reachable_reports_a_closed_port_instead_of_claiming_success():
    """The old implementation returned "reachable" whenever requests was not
    importable, which is exactly the CI environment: playwright is the only
    dependency installed there, so the preflight was a no-op where it was
    supposed to do its job."""
    ok, detail = ux_audit.reachable('http://127.0.0.1:9', 2000)
    assert ok is False
    assert detail


def test_reachable_does_not_depend_on_requests(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == 'requests' or name.startswith('requests.'):
            raise ImportError('requests is not installed in CI')
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, '__import__', blocked)
    ok, detail = ux_audit.reachable('http://127.0.0.1:9', 2000)
    assert ok is False


# --- screenshot naming ----------------------------------------------------

@pytest.mark.parametrize('path,expected', [
    ('/', 'home'),
    ('/faq', 'faq'),
    ('/forecast/wrightsville-beach', 'forecast-wrightsville-beach'),
    ('/regions/great-lakes/', 'regions-great-lakes'),
    ('/learn/how_to_read', 'learn-how-to-read'),
])
def test_page_slug_is_stable_so_shots_overwrite(path, expected):
    assert ux_audit.page_slug(path) == expected


def test_default_pages_only_offers_a_forecast_slug_that_exists():
    pages = ux_audit.default_pages()
    assert pages[0] == '/'
    assert {'/locations', '/accuracy', '/faq', '/about'} <= set(pages)
    forecast = [p for p in pages if p.startswith('/forecast/')]
    assert len(forecast) == 1
    slug = forecast[0].split('/')[-1]
    with open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'surf_cameras.json')) as f:
        cams = json.load(f)
    import re
    known = {re.sub(r'[^a-z0-9]+', '-', c['name'].lower().strip()).strip('-')
             for c in cams if c.get('name')}
    assert slug in known
