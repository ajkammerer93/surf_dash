#!/usr/bin/env python3
"""Collect UI/UX measurements from the live pages into a single snapshot.

Deliberately dumb, in the same spirit as scripts/seo_audit.py: this script
drives a browser, writes numbers to JSON, and makes no judgement at all. No
scoring, no thresholds-with-opinions, no recommendations. A later reader
decides what a 0.3 CLS or a 3.9:1 contrast pair means; keeping that split clean
is what makes a bad reading a data problem rather than a reasoning one.

    python scripts/ux_audit.py --data-dir uxdata
    python scripts/ux_audit.py --data-dir uxdata --base-url http://127.0.0.1:5000
    python scripts/ux_audit.py --data-dir uxdata --pages /,/faq --no-shots

Writes into --data-dir:
    ux-latest.json                         the full snapshot
    ux-history.jsonl                       one trimmed numeric row per run
    shots/<page>-<viewport>[-<theme>].jpg  overwritten every run

Every collector degrades on its own. A check that throws records itself under
an "<name>_error" key and the rest of the page still gets measured; a page that
will not load records the failure and the other pages still run. If Playwright
is missing, the browser will not launch, or the base URL is unreachable, the
snapshot is still written with a top-level "unavailable" reason and the exit
code stays 0 -- a run that writes nothing leaves the reader with no snapshot to
explain itself with, which is worse than a snapshot full of holes.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = 'https://freesurfforecast.com'

VIEWPORTS = {
    'mobile': {'width': 390, 'height': 844, 'scale': 2, 'is_mobile': True},
    'desktop': {'width': 1440, 'height': 900, 'scale': 1, 'is_mobile': False},
}

# Both themes are captured for these paths only. Every extra theme doubles the
# shot count and the run time, so the rest are measured in light, which is what
# color_scheme falls back to below. Light is the cheaper half to cover: the
# dashboard reports fewer low-contrast pairs in light than in dark, so the
# untested half of the un-themed pages is the worse half -- worth knowing when
# reading a clean contrast number for /faq or /about.
THEMED_PATHS = {'/'}

# How long to sit on a loaded page before reading CLS and LCP. Layout shifts
# from late-arriving forecast data land well after load, and LCP is not final
# until the observer stops firing.
SETTLE_MS = 3500

# How long to wait after the scroll pass for lazily-initialised regions to
# render. The Swell Map builds on an IntersectionObserver and takes about ten
# seconds to draw once it is scrolled into view; without this the dashboard is
# measured and photographed with an 800px "Loading ocean data..." box where its
# largest panel belongs, and every DOM check silently excludes that panel.
LAZY_SETTLE_MS = 9000

# Written before first paint so the dashboard is measured in its returning
# visitor state. Every measurement gets a fresh context (see measure()), so
# without this the first-visit WELCOME overlay covers the page in all eight
# dashboard shots and there is nothing else to look at.
ONBOARDING_KEY = 'sd-seen-onboarding'

# Third-party hosts whose requests are dropped. The analytics beacon is the
# one that matters: sixteen automated measurements a week against a few
# hundred real pageviews would inflate the owner's own numbers, in the
# no-referrer bucket that is already the hardest one to read.
BLOCKED_HOSTS = ('cloudflareinsights.com',)

# Caps. These keep a run bounded on a 10k-line dashboard; the reader gets a
# sample plus the true total rather than an unbounded dump.
MAX_OFFENDERS = 25
MAX_TAP_TARGETS = 40
MAX_TINY_TEXT = 40
MAX_CONTRAST_SAMPLES = 400
MAX_CONTRAST_REPORTED = 40


def _utcnow():
    return datetime.now(timezone.utc)


def _stamp():
    return _utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


# --------------------------------------------------------------------------
# Pure helpers. No browser, no network -- these are what tests/test_ux_audit.py
# exercises.
# --------------------------------------------------------------------------

_HEX_RE = re.compile(r'^#([0-9a-fA-F]{3,8})$')
_FUNC_RE = re.compile(r'^rgba?\(([^)]*)\)$', re.I)


def parse_css_color(value):
    """Parse the colour forms getComputedStyle actually returns, plus hex.

    Returns (r, g, b, a) with channels 0-255 and alpha 0-1, or None. Anything
    exotic (colour functions, named colours beyond the two that matter) returns
    None rather than a guess, because a wrong colour silently produces a wrong
    contrast ratio and nobody would ever catch it.
    """
    if not value:
        return None
    v = value.strip().lower()
    if v in ('transparent', 'rgba(0, 0, 0, 0)'):
        return (0, 0, 0, 0.0)
    if v == 'white':
        return (255, 255, 255, 1.0)
    if v == 'black':
        return (0, 0, 0, 1.0)
    m = _HEX_RE.match(v)
    if m:
        h = m.group(1)
        if len(h) in (3, 4):
            h = ''.join(c * 2 for c in h)
        if len(h) not in (6, 8):
            return None
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        a = int(h[6:8], 16) / 255.0 if len(h) == 8 else 1.0
        return (r, g, b, a)
    m = _FUNC_RE.match(v)
    if m:
        parts = [p.strip() for p in m.group(1).replace('/', ' ').split(',')]
        if len(parts) == 1:
            parts = [p for p in m.group(1).replace('/', ' ').split() if p]
        if len(parts) < 3:
            return None
        try:
            chans = []
            for p in parts[:3]:
                if p.endswith('%'):
                    chans.append(round(float(p[:-1]) * 255 / 100.0))
                else:
                    chans.append(round(float(p)))
            a = 1.0
            if len(parts) >= 4:
                ap = parts[3]
                a = float(ap[:-1]) / 100.0 if ap.endswith('%') else float(ap)
        except ValueError:
            return None
        return (max(0, min(255, chans[0])), max(0, min(255, chans[1])),
                max(0, min(255, chans[2])), max(0.0, min(1.0, a)))
    return None


def composite(fg, bg):
    """Source-over composite of an rgba over an opaque rgb. Returns rgb."""
    a = fg[3]
    return tuple(round(fg[i] * a + bg[i] * (1 - a)) for i in range(3))


def composite_stack(stack, fallback=(255, 255, 255)):
    """Flatten a nearest-first stack of background colours to one opaque rgb.

    The browser hands back the stack from the text node outward; painting order
    is the reverse, so it is walked back to front over the fallback canvas.
    """
    parsed = [c for c in (parse_css_color(s) for s in stack) if c]
    out = fallback
    for colour in reversed(parsed):
        if colour[3] <= 0:
            continue
        out = composite(colour, out)
    return out


def relative_luminance(rgb):
    """WCAG 2.x relative luminance for an opaque sRGB triple."""
    chans = []
    for c in rgb[:3]:
        s = c / 255.0
        chans.append(s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4)
    return 0.2126 * chans[0] + 0.7152 * chans[1] + 0.0722 * chans[2]


def contrast_ratio(rgb_a, rgb_b):
    """WCAG contrast ratio between two opaque colours, 1.0 to 21.0."""
    la, lb = relative_luminance(rgb_a), relative_luminance(rgb_b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def required_contrast(font_size_px, font_weight):
    """The AA threshold that applies to text of this size and weight.

    Large text is >= 24px, or >= 18.66px when bold, and gets 3:1; everything
    else gets 4.5:1.
    """
    try:
        size = float(font_size_px)
    except (TypeError, ValueError):
        return 4.5
    try:
        weight = float(font_weight)
    except (TypeError, ValueError):
        weight = 400
    if size >= 24 or (size >= 18.66 and weight >= 700):
        return 3.0
    return 4.5


def heading_skips(levels):
    """Places where heading depth jumps by more than one, in document order.

    Returns a list of (from_level, to_level, index_of_the_jump). An empty list
    means no skips; it says nothing about whether the outline is any good.
    """
    skips = []
    prev = None
    for i, lvl in enumerate(levels):
        try:
            lvl = int(lvl)
        except (TypeError, ValueError):
            continue
        if prev is not None and lvl > prev + 1:
            skips.append([prev, lvl, i])
        prev = lvl
    return skips


def rgb_hex(rgb):
    return '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])


def evaluate_contrast_samples(samples):
    """Turn raw browser colour samples into the sub-threshold list.

    Every sample carries its own background stack, so compositing happens here
    where it can be tested, not in page JS where it cannot.
    """
    out = {'sampled': len(samples), 'unparsed': 0, 'low': []}
    for s in samples:
        fg_raw = parse_css_color(s.get('color'))
        if not fg_raw:
            out['unparsed'] += 1
            continue
        canvas = parse_css_color(s.get('canvas') or '') or (255, 255, 255, 1.0)
        base = canvas[:3] if canvas[3] > 0 else (255, 255, 255)
        bg = composite_stack(s.get('bg_stack') or [], fallback=base)
        fg = composite(fg_raw, bg) if fg_raw[3] < 1 else fg_raw[:3]
        ratio = contrast_ratio(fg, bg)
        need = required_contrast(s.get('font_size'), s.get('font_weight'))
        if ratio < need:
            out['low'].append({
                'selector': s.get('selector'),
                'text': s.get('text'),
                'fg': rgb_hex(fg),
                'bg': rgb_hex(bg),
                'font_size': s.get('font_size'),
                'font_weight': s.get('font_weight'),
                'ratio': round(ratio, 2),
                'required': need,
            })
    out['low_count'] = len(out['low'])
    out['low'].sort(key=lambda x: x['ratio'])
    out['low'] = out['low'][:MAX_CONTRAST_REPORTED]
    return out


_VAR_RE = re.compile(r'(--[a-zA-Z0-9_-]+)\s*:\s*([^;{}]+);')


def parse_palette(css_text):
    """Pull the design tokens out of the template.

    The dark values live on bare :root and the light ones on
    [data-theme="light"], so a reader can reason about the palette without
    re-deriving it from a screenshot. A parse that finds nothing says so --
    emitting an empty dict would read as "this site has no theme", which is a
    very different claim from "my regex missed".
    """
    blocks = {}
    for selector, key in (('root', 'dark'), ('light', 'light')):
        if selector == 'root':
            m = re.search(r':root\s*\{', css_text)
        else:
            m = re.search(r'\[data-theme\s*=\s*["\']light["\']\]\s*\{', css_text)
        if not m:
            continue
        start = m.end()
        end = css_text.find('}', start)
        if end == -1:
            continue
        blocks[key] = dict(
            (name, val.strip()) for name, val in _VAR_RE.findall(css_text[start:end]))
    dark = blocks.get('dark', {})
    light = blocks.get('light', {})
    if not dark and not light:
        return {'unavailable': 'no :root or [data-theme="light"] custom '
                               'properties matched in the template'}
    variables = {}
    for name in sorted(set(dark) | set(light)):
        variables[name] = {'dark': dark.get(name), 'light': light.get(name)}
    return {
        'count': len(variables),
        'root_is_dark': bool(dark),
        'has_light_overrides': bool(light),
        'variables': variables,
    }


def history_row(snapshot):
    """One trimmed row per run. Numbers only, no lists of selectors.

    ux-history.jsonl is read on every review, so it has to stay small enough to
    load whole; the detail lives in the per-run ux-latest.json.
    """
    row = {
        'generated': snapshot.get('generated'),
        'base_url': snapshot.get('base_url'),
        'version': (snapshot.get('build') or {}).get('version'),
        'git_sha': (snapshot.get('build') or {}).get('git_short'),
    }
    if snapshot.get('unavailable'):
        row['unavailable'] = snapshot['unavailable']
        row['pages'] = {}
        return row
    pages = {}
    for p in snapshot.get('pages') or []:
        key = p.get('key')
        if not key:
            continue
        contrast = p.get('contrast') or {}
        images = p.get('images') or {}
        headings = p.get('headings') or {}
        console = p.get('console') or {}
        failed = p.get('failed_requests') or []
        pages[key] = {
            'status': p.get('status'),
            'ttfb_ms': p.get('ttfb_ms'),
            'dcl_ms': p.get('dcl_ms'),
            'load_ms': p.get('load_ms'),
            'cls': p.get('cls'),
            'lcp_ms': p.get('lcp_ms'),
            # An uncaught exception is a 'pageerror', not an 'error': counting
            # only the latter undercounts a page that throws by exactly the
            # bucket that matters, in the file the whole branch exists for.
            'console_errors': (console.get('error') or 0) + (console.get('pageerror') or 0),
            'console_warnings': console.get('warning') or 0,
            'failed_requests': len(failed),
            'failed_first_party': sum(1 for f in failed if f.get('first_party')),
            'overflows': (p.get('overflow') or {}).get('overflows'),
            'overflow_elements': (p.get('overflow') or {}).get('offender_count'),
            'overflow_clipped': (p.get('overflow') or {}).get('clipped_count'),
            'small_tap_targets': (p.get('tap_targets') or {}).get('small_count'),
            'tiny_text': (p.get('tiny_text') or {}).get('count'),
            'img_missing_alt': images.get('missing_alt'),
            'img_no_dimensions': images.get('no_dimensions'),
            'img_failed': images.get('failed'),
            'h1_count': headings.get('h1_count'),
            # None, not 0, when the check never ran -- a missing section used
            # to read as "no heading skips", which is the failure mode where a
            # broken check reports a clean result.
            'heading_skips': (len(headings['skips'])
                              if isinstance(headings.get('skips'), list) else None),
            'low_contrast': contrast.get('low_count'),
            # Colours this build cannot parse (oklch(), color-mix()) are
            # skipped rather than guessed, so the count has to travel with
            # low_contrast: one palette modernisation would otherwise zero the
            # contrast check permanently while every log line still said fine.
            'contrast_unparsed': contrast.get('unparsed'),
            'check_errors': sorted(k[:-6] for k in p if k.endswith('_error')),
            'error': p.get('error'),
        }
    row['pages'] = pages
    row['page_errors'] = sum(1 for p in pages.values() if p.get('error'))
    row['check_errors'] = sum(len(p.get('check_errors') or []) for p in pages.values())
    return row


def is_first_party(url, base_url):
    """Is this request URL served by the site under audit?

    Every failed request in a real run so far has been third-party teardown
    noise -- aborted YouTube telemetry beacons and CARTO tiles -- so a raw
    "failed_requests: 4" line trains the reader to ignore the number a real
    404 would appear in.
    """
    try:
        from urllib.parse import urlsplit
        host = (urlsplit(url or '').hostname or '').lower()
        base = (urlsplit(base_url or '').hostname or '').lower()
    except (ValueError, TypeError):
        return False
    if not host or not base:
        return False
    return host == base or host.endswith('.' + base)


def page_slug(path):
    """Screenshot basename for a path. Stable across runs so shots overwrite."""
    s = path.strip('/')
    if not s:
        return 'home'
    s = re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')
    return s or 'home'


# --------------------------------------------------------------------------
# Cheap collectors: no browser needed, so they run even when the browser path
# is unavailable.
# --------------------------------------------------------------------------

def collect_build():
    out = {}
    try:
        out['git_sha'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
        out['git_short'] = out['git_sha'][:8]
    except (OSError, subprocess.SubprocessError) as e:
        out['git_error'] = str(e)
    try:
        with open(os.path.join(REPO_ROOT, 'VERSION')) as f:
            out['version'] = f.read().strip()
    except OSError as e:
        out['version_error'] = str(e)
    return out


def collect_palette():
    path = os.path.join(REPO_ROOT, 'templates', 'index.html')
    try:
        with open(path, encoding='utf-8') as f:
            css = f.read()
    except OSError as e:
        return {'unavailable': f'cannot read templates/index.html: {e}'}
    out = parse_palette(css)
    out['source'] = 'templates/index.html'
    return out


def default_pages():
    """The default page set, with a forecast slug that provably exists.

    Guessing a slug produces a 404 that looks like a broken page rather than a
    broken audit, so the slug comes from the app's own index, falling back to
    the camera file if importing the app is not possible here.
    """
    preferred = ['wrightsville-beach', 'virginia-beach']
    slug = None
    try:
        sys.path.insert(0, REPO_ROOT)
        from app import LOCATION_BY_SLUG, slugify  # noqa: F401
        slugs = list(LOCATION_BY_SLUG)
        slug = next((s for s in preferred if s in LOCATION_BY_SLUG), None) or (
            slugs[0] if slugs else None)
    except Exception:
        try:
            with open(os.path.join(REPO_ROOT, 'surf_cameras.json')) as f:
                cams = json.load(f)
            names = [c.get('name', '') for c in cams if c.get('name')]
            derived = [re.sub(r'[^a-z0-9]+', '-', n.lower().strip()).strip('-')
                       for n in names]
            slug = next((s for s in preferred if s in derived), None) or (
                derived[0] if derived else None)
        except (OSError, ValueError, KeyError):
            slug = None
    pages = ['/']
    if slug:
        pages.append(f'/forecast/{slug}')
    pages += ['/locations', '/accuracy', '/faq', '/about']
    return pages


# --------------------------------------------------------------------------
# Browser JS. Each snippet is evaluated on its own so that one throwing check
# does not cost us the rest of the page.
# --------------------------------------------------------------------------

JS_PRELUDE = """
const uxSelector = (el) => {
  const part = (n) => {
    let s = n.tagName.toLowerCase();
    if (n.id) return s + '#' + n.id;
    const raw = (typeof n.className === 'string') ? n.className : '';
    const cls = raw.trim().split(/\\s+/).filter(Boolean).slice(0, 2);
    if (cls.length) s += '.' + cls.join('.');
    const p = n.parentElement;
    if (p) {
      const sibs = Array.from(p.children).filter(c => c.tagName === n.tagName);
      if (sibs.length > 1) s += ':nth-of-type(' + (sibs.indexOf(n) + 1) + ')';
    }
    return s;
  };
  const parts = [];
  let n = el;
  while (n && n.nodeType === 1 && parts.length < 4) {
    parts.unshift(part(n));
    if (n.id) break;
    n = n.parentElement;
  }
  return parts.join(' > ');
};
const uxVisible = (el) => {
  const st = getComputedStyle(el);
  if (st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return false;
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0;
};
"""

# CLS and LCP have to be observed from the very first paint, so this is
# installed as an init script rather than evaluated after load.
JS_OBSERVERS = """
window.__ux = {cls: 0, shifts: 0, lcp: 0};
try {
  new PerformanceObserver((list) => {
    for (const e of list.getEntries()) {
      if (!e.hadRecentInput) { window.__ux.cls += e.value; window.__ux.shifts += 1; }
    }
  }).observe({type: 'layout-shift', buffered: true});
} catch (e) { window.__ux.cls_error = String(e); }
try {
  new PerformanceObserver((list) => {
    const es = list.getEntries();
    if (es.length) window.__ux.lcp = es[es.length - 1].startTime;
  }).observe({type: 'largest-contentful-paint', buffered: true});
} catch (e) { window.__ux.lcp_error = String(e); }
"""

JS_TIMINGS = """() => {
  const n = performance.getEntriesByType('navigation')[0];
  if (!n) return {timing_error: 'no navigation entry'};
  return {
    ttfb_ms: Math.round(n.responseStart - n.requestStart),
    dcl_ms: Math.round(n.domContentLoadedEventEnd),
    load_ms: Math.round(n.loadEventEnd),
    transfer_bytes: n.transferSize || null,
    theme_applied: document.documentElement.getAttribute('data-theme'),
    title: document.title,
  };
}"""

JS_OVERFLOW = JS_PRELUDE + """
(max) => {
  // An element can sit past the right edge and be perfectly fine -- a Leaflet
  // tile inside a clipped map pane does exactly that. The clip is recorded
  // rather than filtered, because deciding which overhangs matter is the
  // reader's job, not this script's.
  const uxClipped = (el) => {
    let n = el.parentElement;
    while (n && n !== document.documentElement) {
      const o = getComputedStyle(n);
      if (o.overflowX !== 'visible' || o.overflow === 'hidden') return true;
      n = n.parentElement;
    }
    return false;
  };
  const de = document.documentElement;
  const vw = window.innerWidth;
  const out = {
    scroll_width: de.scrollWidth,
    inner_width: vw,
    overflows: de.scrollWidth > vw + 1,
    offenders: [],
    offender_count: 0,
    clipped_count: 0,
  };
  for (const el of document.querySelectorAll('body *')) {
    if (!uxVisible(el)) continue;
    const r = el.getBoundingClientRect();
    if (r.right <= vw + 1) continue;
    out.offender_count += 1;
    if (uxClipped(el)) out.clipped_count += 1;
    if (out.offenders.length < max) {
      out.offenders.push({
        selector: uxSelector(el),
        right: Math.round(r.right),
        width: Math.round(r.width),
        position: getComputedStyle(el).position,
        clipped_by_ancestor: uxClipped(el),
      });
    }
  }
  return out;
}"""

JS_TAP_TARGETS = JS_PRELUDE + """
(max) => {
  const sel = 'a, button, input, select, [role=button]';
  const out = {total: 0, small_count: 0, small: []};
  for (const el of document.querySelectorAll(sel)) {
    if (el.tagName === 'INPUT' && el.type === 'hidden') continue;
    if (!uxVisible(el)) continue;
    out.total += 1;
    const r = el.getBoundingClientRect();
    if (r.width >= 44 && r.height >= 44) continue;
    out.small_count += 1;
    if (out.small.length < max) {
      out.small.push({
        selector: uxSelector(el),
        w: Math.round(r.width * 10) / 10,
        h: Math.round(r.height * 10) / 10,
        text: (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().slice(0, 40),
      });
    }
  }
  return out;
}"""

JS_TINY_TEXT = JS_PRELUDE + """
(args) => {
  const [max, floorPx] = args;
  const out = {count: 0, items: [], threshold_px: floorPx};
  const seen = new Set();
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const text = (node.nodeValue || '').trim();
    if (!text) continue;
    const el = node.parentElement;
    if (!el || el.closest('script, style, noscript')) continue;
    if (!uxVisible(el)) continue;
    const size = parseFloat(getComputedStyle(el).fontSize);
    if (!(size < floorPx)) continue;
    out.count += 1;
    const sel = uxSelector(el);
    if (seen.has(sel) || out.items.length >= max) continue;
    seen.add(sel);
    out.items.push({selector: sel, font_size: Math.round(size * 100) / 100,
                    text: text.slice(0, 40)});
  }
  return out;
}"""

JS_IMAGES = JS_PRELUDE + """
() => {
  const out = {total: 0, missing_alt: 0, empty_alt: 0, no_dimensions: 0,
               failed: 0, failed_src: []};
  for (const img of document.images) {
    out.total += 1;
    if (!img.hasAttribute('alt')) out.missing_alt += 1;
    else if (!img.getAttribute('alt').trim()) out.empty_alt += 1;
    if (!img.hasAttribute('width') || !img.hasAttribute('height')) out.no_dimensions += 1;
    if (img.complete && img.naturalWidth === 0) {
      out.failed += 1;
      if (out.failed_src.length < 10) out.failed_src.push((img.currentSrc || img.src || '').slice(0, 200));
    }
  }
  return out;
}"""

JS_HEADINGS = JS_PRELUDE + """
() => {
  const nodes = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'));
  const visible = nodes.filter(uxVisible);
  return {
    levels: visible.map(h => Number(h.tagName.slice(1))),
    h1_count: visible.filter(h => h.tagName === 'H1').length,
    h1_text: visible.filter(h => h.tagName === 'H1').map(h => (h.innerText || '').trim().slice(0, 120)),
    total: visible.length,
    hidden: nodes.length - visible.length,
  };
}"""

JS_CONTRAST_SAMPLES = JS_PRELUDE + """
(max) => {
  const samples = [];
  const seen = new Set();
  const canvas = getComputedStyle(document.documentElement).backgroundColor;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode()) && samples.length < max) {
    const text = (node.nodeValue || '').trim();
    if (!text) continue;
    const el = node.parentElement;
    if (!el || el.closest('script, style, noscript')) continue;
    if (!uxVisible(el)) continue;
    const st = getComputedStyle(el);
    const stack = [];
    let n = el;
    while (n && n.nodeType === 1 && stack.length < 8) {
      const bg = getComputedStyle(n).backgroundColor;
      if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
        stack.push(bg);
        if (!/rgba\\([^)]*,\\s*0?\\.\\d+\\s*\\)/.test(bg)) break;
      }
      n = n.parentElement;
    }
    const key = uxSelector(el) + '|' + st.color + '|' + stack.join(',');
    if (seen.has(key)) continue;
    seen.add(key);
    samples.push({
      selector: uxSelector(el),
      text: text.slice(0, 40),
      color: st.color,
      bg_stack: stack,
      canvas: canvas,
      font_size: parseFloat(st.fontSize),
      font_weight: st.fontWeight,
    });
  }
  return samples;
}"""


JS_SCROLL = """async (stepPause) => {
  // Lazily-initialised regions build on an IntersectionObserver, which never
  // fires for a page that is only ever screenshotted full-page: full_page
  // captures past the viewport without scrolling it. So walk the page.
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const step = Math.max(200, Math.round(window.innerHeight * 0.75));
  let y = 0;
  for (let i = 0; i < 60; i++) {
    window.scrollTo(0, y);
    await sleep(stepPause);
    if (y >= document.documentElement.scrollHeight) break;
    y += step;
  }
  window.scrollTo(0, 0);
  await sleep(stepPause);
  return {scroll_height: document.documentElement.scrollHeight, steps: Math.ceil(y / step)};
}"""


def _eval(page, entry, key, expr, arg=None):
    """Run one check, recording a throw as <key>_error instead of losing the page."""
    try:
        entry[key] = page.evaluate(expr, arg) if arg is not None else page.evaluate(expr)
    except Exception as e:
        entry[f'{key}_error'] = str(e)[:300]


def measure(browser, base_url, path, viewport, theme, shots_dir, timeout_ms):
    """Measure one page in one viewport in one theme, in its own browser context.

    A FRESH CONTEXT PER MEASUREMENT IS NOT OPTIONAL. Reusing one context across
    measurements lets the page's 5-minute Cache-Control serve a warm copy, so
    cold loads never get measured, real run-to-run variance reads as noise, and
    the numbers support a conclusion that is simply wrong -- that cost several
    rounds of chasing the wrong thing during the CLS work. Do not "optimise"
    this into a shared context.
    """
    vp = VIEWPORTS[viewport]
    key = f'{path}|{viewport}' + (f'|{theme}' if theme else '')
    entry = {
        'key': key, 'path': path, 'viewport': viewport, 'theme': theme,
        'url': base_url.rstrip('/') + path,
        'viewport_size': {'width': vp['width'], 'height': vp['height']},
    }
    console = {}
    console_samples = []
    failed_requests = []
    context = None
    try:
        context = browser.new_context(
            viewport={'width': vp['width'], 'height': vp['height']},
            device_scale_factor=vp['scale'],
            is_mobile=vp['is_mobile'],
            has_touch=vp['is_mobile'],
            color_scheme=('dark' if theme == 'dark' else 'light'),
            ignore_https_errors=False,
        )
        context.set_default_timeout(timeout_ms)
        context.set_default_navigation_timeout(timeout_ms)
        if theme:
            # The site reads localStorage['theme'] before first paint and falls
            # back to prefers-color-scheme, so the key has to be written before
            # any page script runs.
            context.add_init_script(
                "try { localStorage.setItem('theme', %s); } catch (e) {}"
                % json.dumps(theme))
        context.add_init_script(
            "try { localStorage.setItem(%s, '1'); } catch (e) {}"
            % json.dumps(ONBOARDING_KEY))
        entry['onboarding_dismissed'] = True
        context.add_init_script(JS_OBSERVERS)
        # Keep the audit out of the owner's analytics: sixteen automated
        # measurements a week would inflate a few hundred real pageviews.
        # Answered empty rather than aborted -- an abort makes the browser log
        # "Failed to load resource" on every page, and a console-error count
        # that is always 1 for a reason of our own making is exactly the kind
        # of noise that trains a reader to ignore the number.
        def _block(route):
            try:
                route.fulfill(status=200, body='',
                              content_type='application/javascript')
            except Exception:
                try:
                    route.abort()
                except Exception:
                    pass
        for host in BLOCKED_HOSTS:
            context.route(f'**://*{host}/**', _block)
        page = context.new_page()

        def on_console(msg):
            lvl = msg.type
            console[lvl] = console.get(lvl, 0) + 1
            if lvl in ('error', 'warning') and len(console_samples) < 20:
                console_samples.append({'level': lvl, 'text': msg.text[:300]})

        def on_response(resp):
            if resp.status >= 400 and len(failed_requests) < 25:
                failed_requests.append({'url': resp.url[:300], 'status': resp.status,
                                        'first_party': is_first_party(resp.url, base_url)})

        def on_request_failed(req):
            if len(failed_requests) < 25:
                failed_requests.append({'url': req.url[:300],
                                        'status': 'failed',
                                        'reason': (req.failure or '')[:120],
                                        'first_party': is_first_party(req.url, base_url)})

        page.on('console', on_console)
        page.on('response', on_response)
        def on_page_error(e):
            # Same cap as the console listener: a page throwing in a loop must
            # not turn the snapshot into a log file.
            console['pageerror'] = console.get('pageerror', 0) + 1
            if len(console_samples) < 20:
                console_samples.append({'level': 'pageerror', 'text': str(e)[:300]})

        page.on('pageerror', on_page_error)
        page.on('requestfailed', on_request_failed)

        try:
            resp = page.goto(entry['url'], wait_until='load', timeout=timeout_ms)
            entry['status'] = resp.status if resp else None
        except Exception as e:
            entry['error'] = f'navigation: {str(e)[:300]}'
            entry['console'] = console
            entry['console_samples'] = console_samples
            entry['failed_requests'] = failed_requests
            return entry

        try:
            page.wait_for_load_state('networkidle', timeout=min(timeout_ms, 15000))
        except Exception as e:
            # Not "_error": the dashboard keeps live cam streams open, so this
            # times out on almost every run. Naming it an error would put a
            # standing warning on every dashboard measurement, which is how a
            # reader learns to skim past the warnings that matter.
            entry['networkidle_timeout'] = str(e)[:200]
        page.wait_for_timeout(SETTLE_MS)

        _eval(page, entry, 'timings', JS_TIMINGS)
        for k in ('ttfb_ms', 'dcl_ms', 'load_ms', 'transfer_bytes',
                  'theme_applied', 'title'):
            if isinstance(entry.get('timings'), dict) and k in entry['timings']:
                entry[k] = entry['timings'][k]
        entry.pop('timings', None)

        try:
            ux = page.evaluate('() => window.__ux || {}')
            entry['cls'] = round(float(ux.get('cls') or 0), 4)
            entry['cls_shifts'] = ux.get('shifts')
            entry['lcp_ms'] = round(float(ux.get('lcp') or 0))
            for k in ('cls_error', 'lcp_error'):
                if ux.get(k):
                    entry[k] = ux[k]
        except Exception as e:
            entry['webvitals_error'] = str(e)[:300]

        # Walk the page before the DOM checks run. Everything below counts
        # what is in the DOM at the moment it runs, and the dashboard's
        # largest panel does not exist until it has been scrolled into view.
        _eval(page, entry, 'scroll', JS_SCROLL, 150)
        page.wait_for_timeout(LAZY_SETTLE_MS)
        try:
            ux2 = page.evaluate('() => window.__ux || {}')
            # Kept separate from `cls`: that one is the load-time figure the
            # history has always carried, this one includes shifts a reader
            # provokes by scrolling.
            entry['cls_after_scroll'] = round(float(ux2.get('cls') or 0), 4)
        except Exception as e:
            entry['cls_after_scroll_error'] = str(e)[:300]

        _eval(page, entry, 'overflow', JS_OVERFLOW, MAX_OFFENDERS)
        if vp['is_mobile']:
            # The 44px rule is a touch-target rule. Applying it to a 1440px
            # mouse-driven viewport reported 63 of 64 controls "small" every
            # week, which is a number nobody can ever act on.
            _eval(page, entry, 'tap_targets', JS_TAP_TARGETS, MAX_TAP_TARGETS)
        _eval(page, entry, 'tiny_text', JS_TINY_TEXT, [MAX_TINY_TEXT, 12])
        _eval(page, entry, 'images', JS_IMAGES)
        _eval(page, entry, 'headings', JS_HEADINGS)
        if isinstance(entry.get('headings'), dict):
            entry['headings']['skips'] = heading_skips(entry['headings'].get('levels') or [])

        try:
            samples = page.evaluate(JS_CONTRAST_SAMPLES, MAX_CONTRAST_SAMPLES)
            entry['contrast'] = evaluate_contrast_samples(samples)
        except Exception as e:
            entry['contrast_error'] = str(e)[:300]

        if shots_dir:
            # JPEG, not PNG: these are committed to a branch every week and
            # PNGs do not delta, so a year of full-page dashboard captures is
            # hundreds of megabytes of permanent history for images nobody
            # diffs byte-wise.
            name = page_slug(path) + f'-{viewport}' + (f'-{theme}' if theme else '') + '.jpg'
            dest = os.path.join(shots_dir, name)
            try:
                page.screenshot(path=dest, full_page=True, timeout=timeout_ms,
                                type='jpeg', quality=72)
                entry['screenshot'] = os.path.join('shots', name)
                entry['screenshot_bytes'] = os.path.getsize(dest)
            except Exception as e:
                entry['screenshot_error'] = str(e)[:300]
    except Exception as e:
        entry['error'] = f'context: {str(e)[:300]}'
    finally:
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
    entry['console'] = console
    entry['console_samples'] = console_samples
    entry['failed_requests'] = failed_requests
    return entry


def reachable(base_url, timeout_ms):
    """Cheap preflight so an unreachable host produces a reason, not a stack.

    urllib, not requests: CI installs playwright and nothing else, so the old
    `import requests` guard turned this into an unconditional "reachable" in
    the one environment it was written for -- sixteen navigation timeouts
    instead of one clear line, and a promise in the docstring above that was
    false wherever it mattered.
    """
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError
    req = Request(base_url, headers={'User-Agent': 'surf-dash-ux-audit'})
    try:
        with urlopen(req, timeout=max(5, timeout_ms / 1000.0)) as r:
            return True, getattr(r, 'status', None) or r.getcode()
    except HTTPError as e:
        # The host answered. A 500 on the front page is the audit's subject,
        # not a reason to skip the run.
        return True, e.code
    except (URLError, OSError, ValueError) as e:
        return False, str(e)[:300]


def run(args):
    snapshot = {
        'generated': _stamp(),
        'base_url': args.base_url,
        'build': collect_build(),
        'palette': collect_palette(),
        'config': {
            'viewports': VIEWPORTS,
            'themed_paths': sorted(THEMED_PATHS),
            'timeout_ms': args.timeout_ms,
            'settle_ms': SETTLE_MS,
            'screenshots': not args.no_shots,
        },
        'pages': [],
    }

    pages = ([p.strip() for p in args.pages.split(',') if p.strip()]
             if args.pages else default_pages())
    snapshot['config']['pages'] = pages

    ok, detail = reachable(args.base_url, args.timeout_ms)
    if not ok:
        snapshot['unavailable'] = f'base URL unreachable: {detail}'
        return snapshot
    snapshot['preflight_status'] = detail

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        snapshot['unavailable'] = f'playwright not installed: {e}'
        return snapshot

    shots_dir = None
    if not args.no_shots:
        shots_dir = os.path.join(args.data_dir, 'shots')
        os.makedirs(shots_dir, exist_ok=True)

    try:
        pw = sync_playwright().start()
    except Exception as e:
        snapshot['unavailable'] = f'playwright failed to start: {str(e)[:300]}'
        return snapshot
    try:
        try:
            browser = pw.chromium.launch(args=['--disable-dev-shm-usage'])
        except Exception as e:
            snapshot['unavailable'] = f'chromium failed to launch: {str(e)[:300]}'
            return snapshot
        try:
            for path in pages:
                themes = ['dark', 'light'] if (
                    path in THEMED_PATHS or path.startswith('/forecast/')) else [None]
                for viewport in ('mobile', 'desktop'):
                    for theme in themes:
                        snapshot['pages'].append(measure(
                            browser, args.base_url, path, viewport, theme,
                            shots_dir, args.timeout_ms))
        finally:
            try:
                browser.close()
            except Exception:
                pass
    finally:
        try:
            pw.stop()
        except Exception:
            pass

    snapshot['summary'] = {
        'measurements': len(snapshot['pages']),
        'with_errors': sum(1 for p in snapshot['pages'] if p.get('error')),
        'screenshots': sum(1 for p in snapshot['pages'] if p.get('screenshot')),
    }
    return snapshot


def main():
    ap = argparse.ArgumentParser(
        description='Collect a UI/UX snapshot (measurements only, no judgement)')
    # Not '.': a bare run would otherwise drop a JSON pair and ~8MB of
    # screenshots into the repo root, where nothing ignores them and the next
    # `git add -A` commits them.
    ap.add_argument('--data-dir', default='uxdata',
                    help='directory to write ux-latest.json, ux-history.jsonl and shots/')
    ap.add_argument('--base-url', default=SITE, help='site to measure')
    ap.add_argument('--pages', default=None,
                    help='comma-separated paths, overriding the default set')
    ap.add_argument('--no-shots', action='store_true', help='skip screenshots')
    ap.add_argument('--timeout-ms', type=int, default=45000,
                    help='per-navigation and per-action timeout')
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    snapshot = run(args)

    with open(os.path.join(args.data_dir, 'ux-latest.json'), 'w') as f:
        json.dump(snapshot, f, indent=2)
    with open(os.path.join(args.data_dir, 'ux-history.jsonl'), 'a') as f:
        f.write(json.dumps(history_row(snapshot)) + '\n')

    if snapshot.get('unavailable'):
        print(f"UX snapshot written to {args.data_dir} (unavailable: "
              f"{snapshot['unavailable']})")
        return 0
    s = snapshot['summary']
    print(f"UX snapshot written to {args.data_dir}. "
          f"{s['measurements']} measurement(s), {s['with_errors']} with errors, "
          f"{s['screenshots']} screenshot(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
