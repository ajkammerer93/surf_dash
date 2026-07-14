#!/usr/bin/env python3
"""Discover, verify, and manage YouTube live surf cams.

Only streams whose owners allow embedding are ever surfaced or approved —
the scanner filters on videoEmbeddable, and everything renders through the
official YouTube iframe player so owner branding, ads, and channel links
stay intact. Never restream, proxy, or frame-grab.

Subcommands
    scan     Search YouTube for live, embeddable surf cams and write new
             candidates to youtube_cam_candidates.json for human review.
             Requires the YOUTUBE_API_KEY environment variable (free key,
             YouTube Data API v3; a full scan costs ~100 units/query of the
             10k/day quota).
    verify   Re-check approved cams in youtube_cams.json. Uses the API when
             a key is present (live + embeddable), otherwise falls back to
             the oEmbed endpoint (public + embeddable). Dead streams get
             disabled: true, never deleted — channels often restart streams
             under new video ids, so keep an eye on candidates for the
             replacement.
    approve  Promote a candidate into youtube_cams.json. Coordinates come
             from --lat/--lon, or from the candidate's suggested spot match.
    reject   Dismiss a candidate; its id is remembered so scans don't
             re-suggest it.

Typical review loop (local):
    YOUTUBE_API_KEY=... python scripts/youtube_cam_scan.py scan
    # read youtube_cam_candidates.json, pick winners
    python scripts/youtube_cam_scan.py approve VIDEO_ID --name "Spot Name" --lat 33.19 --lon -117.38 --state CA
    python scripts/youtube_cam_scan.py reject VIDEO_ID
"""

import argparse
import json
import os
import re
import sys
from datetime import date

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS_FILE = os.path.join(ROOT, 'youtube_cams.json')
CANDIDATES_FILE = os.path.join(ROOT, 'youtube_cam_candidates.json')
SPOTS_FILE = os.path.join(ROOT, 'surf_cameras.json')

API_BASE = 'https://www.googleapis.com/youtube/v3'

DEFAULT_QUERIES = [
    'surf cam live',
    'live surf report cam',
    'beach cam live',
    'pier cam live',
]

# A candidate must mention one of these in its title to be worth reviewing
TITLE_KEYWORDS = re.compile(r'\b(surf|beach|pier|wave|ocean|cam)\b', re.I)


def _load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def _save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


def _api_key(required=True):
    key = os.environ.get('YOUTUBE_API_KEY')
    if required and not key:
        sys.exit('YOUTUBE_API_KEY is not set. Create a free YouTube Data API v3 '
                 'key in Google Cloud Console and export it first.')
    return key


def _api_get(endpoint, params):
    resp = requests.get(f'{API_BASE}/{endpoint}', params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _load_spots():
    """Spot names + coords from the main catalog, for title matching."""
    spots = []
    for item in _load_json(SPOTS_FILE, []):
        name = item.get('name', '')
        # Strip trailing state abbreviations for looser matching
        base = re.sub(r'\s+[A-Z]{2}$', '', name)
        tokens = [t for t in re.split(r'[^a-z0-9]+', base.lower()) if len(t) > 3]
        if tokens:
            spots.append({'name': name, 'lat': item['lat'], 'lon': item['lon'],
                          'state': item.get('state', ''), 'tokens': tokens})
    return spots


def _match_spot(text, spots):
    """Best-effort match of a video title/description to a known spot.

    All of a spot's name tokens must appear in the text. Longest token list
    wins so 'Nags Head - Jennettes Pier' beats 'Nags Head'.
    """
    text = text.lower()
    best = None
    for spot in spots:
        if all(t in text for t in spot['tokens']):
            if best is None or len(spot['tokens']) > len(best['tokens']):
                best = spot
    return best


def cmd_scan(args):
    key = _api_key()
    cams_data = _load_json(CAMS_FILE, {'cams': [], 'rejected_video_ids': []})
    known = {c['video_id'] for c in cams_data['cams']}
    known.update(cams_data.get('rejected_video_ids', []))
    candidates = _load_json(CANDIDATES_FILE, {'candidates': []})
    existing = {c['video_id'] for c in candidates['candidates']}
    spots = _load_spots()

    queries = args.query or DEFAULT_QUERIES
    found_ids = []
    for q in queries:
        data = _api_get('search', {
            'key': key, 'part': 'snippet', 'q': q, 'type': 'video',
            'eventType': 'live', 'videoEmbeddable': 'true',
            'safeSearch': 'strict', 'maxResults': args.limit,
            'relevanceLanguage': 'en',
        })
        for item in data.get('items', []):
            vid = item['id']['videoId']
            if vid not in found_ids:
                found_ids.append(vid)

    new_candidates = []
    # videos.list accepts up to 50 ids per call
    for i in range(0, len(found_ids), 50):
        batch = found_ids[i:i + 50]
        data = _api_get('videos', {
            'key': key, 'part': 'snippet,status,liveStreamingDetails',
            'id': ','.join(batch),
        })
        for item in data.get('items', []):
            vid = item['id']
            if vid in known or vid in existing:
                continue
            snippet = item.get('snippet', {})
            status = item.get('status', {})
            live = item.get('liveStreamingDetails', {})
            title = snippet.get('title', '')
            if not status.get('embeddable'):
                continue
            if snippet.get('liveBroadcastContent') != 'live':
                continue
            if not TITLE_KEYWORDS.search(title):
                continue
            match = _match_spot(title + ' ' + snippet.get('description', ''), spots)
            cand = {
                'video_id': vid,
                'title': title,
                'channel': snippet.get('channelTitle', ''),
                'url': f'https://www.youtube.com/watch?v={vid}',
                'concurrent_viewers': live.get('concurrentViewers'),
                'suggested_spot': match['name'] if match else None,
                'suggested_lat': match['lat'] if match else None,
                'suggested_lon': match['lon'] if match else None,
                'suggested_state': match['state'] if match else None,
                'first_seen': date.today().isoformat(),
            }
            new_candidates.append(cand)

    candidates['candidates'].extend(new_candidates)
    _save_json(CANDIDATES_FILE, candidates)

    print(f'{len(new_candidates)} new candidate(s), '
          f'{len(candidates["candidates"])} pending review in {os.path.basename(CANDIDATES_FILE)}')
    for c in new_candidates:
        spot = c['suggested_spot'] or 'no spot match'
        print(f'  - {c["video_id"]}  {c["title"][:70]}  [{spot}]  {c["url"]}')
    if args.markdown:
        def md_safe(text):
            # Titles/channels are untrusted — strip markdown link syntax
            return re.sub(r'[\[\]()`<>]', '', text or '')
        with open(args.markdown, 'w') as f:
            f.write('New YouTube surf cam candidates. Review each stream, then '
                    '`approve` or `reject` it with scripts/youtube_cam_scan.py.\n\n')
            for c in new_candidates:
                spot = md_safe(c['suggested_spot']) or '_no spot match_'
                f.write(f'- [ ] [{md_safe(c["title"])}]({c["url"]}) — {md_safe(c["channel"])} — '
                        f'suggested spot: {spot} — `{c["video_id"]}`\n')
    return 0


def _check_oembed(video_id):
    """True if the video is public and embeddable per the oEmbed endpoint."""
    resp = requests.get('https://www.youtube.com/oembed', params={
        'url': f'https://www.youtube.com/watch?v={video_id}', 'format': 'json'
    }, timeout=15)
    return resp.status_code == 200


def cmd_verify(args):
    key = _api_key(required=False)
    cams_data = _load_json(CAMS_FILE, {'cams': [], 'rejected_video_ids': []})
    cams = cams_data['cams']
    if not cams:
        print('No approved cams to verify.')
        return 0

    changed = 0
    if key:
        by_id = {}
        ids = [c['video_id'] for c in cams]
        for i in range(0, len(ids), 50):
            data = _api_get('videos', {
                'key': key, 'part': 'snippet,status',
                'id': ','.join(ids[i:i + 50]),
            })
            for item in data.get('items', []):
                by_id[item['id']] = item
        for cam in cams:
            item = by_id.get(cam['video_id'])
            ok = bool(item
                      and item.get('status', {}).get('embeddable')
                      and item.get('snippet', {}).get('liveBroadcastContent') == 'live')
            if ok:
                cam['last_verified'] = date.today().isoformat()
                if cam.get('disabled'):
                    cam['disabled'] = False
                    changed += 1
                    print(f'  re-enabled: {cam["name"]} ({cam["video_id"]})')
            elif not cam.get('disabled'):
                cam['disabled'] = True
                changed += 1
                print(f'  DISABLED (offline or embedding revoked): {cam["name"]} ({cam["video_id"]})')
    else:
        print('No YOUTUBE_API_KEY — falling back to oEmbed (checks public+embeddable, not liveness).')
        for cam in cams:
            ok = _check_oembed(cam['video_id'])
            if ok:
                cam['last_verified'] = date.today().isoformat()
            elif not cam.get('disabled'):
                cam['disabled'] = True
                changed += 1
                print(f'  DISABLED (gone or embedding revoked): {cam["name"]} ({cam["video_id"]})')

    _save_json(CAMS_FILE, cams_data)
    live = sum(1 for c in cams if not c.get('disabled'))
    print(f'{live}/{len(cams)} cams healthy; {changed} status change(s).')

    if args.legacy:
        _verify_legacy_embeds()
    return 0


def _verify_legacy_embeds():
    """Report on the hand-curated YouTube iframes in surf_cameras.json.

    Those entries double as forecast locations, so this never edits the
    file — it just names the dead ones so they can be disabled by hand
    (or replaced via the candidates flow).
    """
    print('\nLegacy YouTube iframes in surf_cameras.json:')
    dead = 0
    for cam in _load_json(SPOTS_FILE, []):
        url = cam.get('stream_url') or ''
        m = re.search(r'youtube\.com/embed/([A-Za-z0-9_-]{11})', url)
        if not m:
            continue
        if cam.get('disabled'):
            continue
        if not _check_oembed(m.group(1)):
            dead += 1
            print(f'  DEAD: {cam["name"]}  {m.group(1)}  -> set "disabled": true')
    if not dead:
        print('  all enabled legacy embeds respond to oEmbed '
              '(note: an ended stream still passes; only the API check catches those)')


def cmd_approve(args):
    cams_data = _load_json(CAMS_FILE, {'cams': [], 'rejected_video_ids': []})
    if any(c['video_id'] == args.video_id for c in cams_data['cams']):
        sys.exit(f'{args.video_id} is already approved.')

    candidates = _load_json(CANDIDATES_FILE, {'candidates': []})
    cand = next((c for c in candidates['candidates'] if c['video_id'] == args.video_id), None)

    name = args.name or (cand and cand.get('suggested_spot'))
    lat = args.lat if args.lat is not None else (cand and cand.get('suggested_lat'))
    lon = args.lon if args.lon is not None else (cand and cand.get('suggested_lon'))
    state = args.state or (cand and cand.get('suggested_state')) or ''
    channel = args.channel or (cand and cand.get('channel')) or ''
    if not (name and lat is not None and lon is not None):
        sys.exit('Need --name, --lat, and --lon (candidate had no spot suggestion to fall back on).')

    if not _check_oembed(args.video_id):
        sys.exit(f'{args.video_id} failed the oEmbed embeddability check — not approving.')

    cams_data['cams'].append({
        'video_id': args.video_id,
        'name': name,
        'lat': float(lat),
        'lon': float(lon),
        'state': state,
        'channel': channel,
        'added': date.today().isoformat(),
        'last_verified': date.today().isoformat(),
        'disabled': False,
    })
    _save_json(CAMS_FILE, cams_data)

    if cand:
        candidates['candidates'] = [c for c in candidates['candidates'] if c['video_id'] != args.video_id]
        _save_json(CANDIDATES_FILE, candidates)
    print(f'Approved {args.video_id} as "{name}" ({lat}, {lon}). '
          'Consider a courtesy email to the channel owner.')
    return 0


def cmd_reject(args):
    cams_data = _load_json(CAMS_FILE, {'cams': [], 'rejected_video_ids': []})
    rejected = cams_data.setdefault('rejected_video_ids', [])
    if args.video_id not in rejected:
        rejected.append(args.video_id)
    _save_json(CAMS_FILE, cams_data)

    candidates = _load_json(CANDIDATES_FILE, {'candidates': []})
    before = len(candidates['candidates'])
    candidates['candidates'] = [c for c in candidates['candidates'] if c['video_id'] != args.video_id]
    if len(candidates['candidates']) != before:
        _save_json(CANDIDATES_FILE, candidates)
    print(f'Rejected {args.video_id}; it will not be suggested again.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('scan', help='find new live embeddable surf cam candidates')
    p.add_argument('--query', action='append', help='override the default search queries (repeatable)')
    p.add_argument('--limit', type=int, default=25, help='max results per query (default 25)')
    p.add_argument('--markdown', help='also write a review checklist to this markdown file')
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser('verify', help='re-check approved cams; disable dead ones')
    p.add_argument('--legacy', action='store_true',
                   help='also report on hand-curated YouTube iframes in surf_cameras.json (read-only)')
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser('approve', help='promote a candidate to youtube_cams.json')
    p.add_argument('video_id')
    p.add_argument('--name')
    p.add_argument('--lat', type=float)
    p.add_argument('--lon', type=float)
    p.add_argument('--state')
    p.add_argument('--channel')
    p.set_defaults(func=cmd_approve)

    p = sub.add_parser('reject', help='dismiss a candidate permanently')
    p.add_argument('video_id')
    p.set_defaults(func=cmd_reject)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()
