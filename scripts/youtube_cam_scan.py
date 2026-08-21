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
    render-issue
             Write the pending candidate list as markdown, ranked by how many
             scans each stream has survived.
    migrate-issues
             Seed the persistent store from the old weekly review issues,
             recovering their times_seen counts.

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


# A candidate line as the weekly review issues render it. Titles are
# untrusted, so the id is read from the trailing backticked field rather than
# from anywhere a title could imitate.
ISSUE_LINE = re.compile(
    r'^- \[(?P<done>[ x])\]\s+'
    r'\[(?P<title>.*?)\]\((?P<url>[^)]*)\)'
    r'(?P<rest>.*?)'
    r'`(?P<vid>[A-Za-z0-9_-]{11})`\s*$')

# The middle of the line is deliberately not pinned down: the old weekly issues
# separated fields with em dashes and the current renderer uses hyphens, and a
# parser that only knew one of them matched nothing at all while looking like it
# worked. Only the checkbox and the trailing id are load-bearing.
ISSUE_CHANNEL = re.compile(r'^\s*[-—]\s*(?P<channel>.*?)\s*[-—]\s*suggested spot:')
ISSUE_SPOT = re.compile(r'suggested spot:\s*(?P<spot>.*?)\s*(?:[-—]\s*seen\s|[-—]\s*$|$)')


def _candidates_path(data_dir):
    """Where the persistent candidate store lives.

    Without --data-dir this is the repo-local file, which is what the local
    review loop uses. CI passes a checkout of the data branch instead: the
    file used to be written into an ephemeral workspace and thrown away, so
    every scheduled scan started blank, rediscovered the same streams and
    filed another issue. Persisting it is what makes times_seen mean anything.
    """
    if not data_dir:
        return CANDIDATES_FILE
    return os.path.join(data_dir, 'youtube_cam_candidates.json')


def _handled_ids():
    """Video ids that are already approved, rejected or embedded on the site,
    and so must never reappear as candidates."""
    cams_data = _load_json(CAMS_FILE, {'cams': [], 'rejected_video_ids': []})
    handled = {c['video_id'] for c in cams_data['cams']}
    handled.update(cams_data.get('rejected_video_ids', []))
    for spot in _load_json(SPOTS_FILE, []):
        m = re.search(r'youtube\.com/embed/([A-Za-z0-9_-]{11})',
                      spot.get('stream_url') or '')
        if m:
            handled.add(m.group(1))
    return handled


def _rank(candidates):
    """Most-persistent first.

    A stream that keeps reappearing week after week is a stable cam worth a
    human's attention; one that showed up once was probably a passing
    broadcast. Viewer count breaks ties, and the title only orders the
    remainder so the list is stable between runs.
    """
    return sorted(candidates, key=lambda c: (
        -int(c.get('times_seen') or 1),
        -int(c.get('concurrent_viewers') or 0),
        (c.get('title') or '').lower(),
    ))


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
    # Approved, rejected, and the legacy hand-curated embeds in
    # surf_cameras.json all count as handled -- never re-suggest them.
    known = _handled_ids()
    path = _candidates_path(getattr(args, 'data_dir', None))
    candidates = _load_json(path, {'candidates': [], 'dismissed': {}})
    # Ticked off in the review issue means "not this one" -- as final as a
    # reject, just expressed in the place the reviewing actually happens.
    known.update(candidates.get('dismissed', {}))
    existing = {c['video_id']: c for c in candidates['candidates']}
    spots = _load_spots()

    queries = args.query or DEFAULT_QUERIES
    found_ids = []
    for q in queries:
        # Embeddability is deliberately NOT a search filter — combining
        # videoEmbeddable with eventType=live makes search.list return few
        # or zero results. The videos.list status check below enforces it.
        data = _api_get('search', {
            'key': key, 'part': 'snippet', 'q': q, 'type': 'video',
            'eventType': 'live', 'maxResults': args.limit,
            'relevanceLanguage': 'en',
        })
        items = data.get('items', [])
        print(f'query "{q}": {len(items)} live result(s)')
        for item in items:
            vid = item['id']['videoId']
            if vid not in found_ids:
                found_ids.append(vid)

    new_candidates = []
    reseen = 0
    today = date.today().isoformat()
    skipped = {'known': 0, 'not_embeddable': 0, 'not_live': 0, 'no_keyword': 0}
    # videos.list accepts up to 50 ids per call
    for i in range(0, len(found_ids), 50):
        batch = found_ids[i:i + 50]
        data = _api_get('videos', {
            'key': key, 'part': 'snippet,status,liveStreamingDetails',
            'id': ','.join(batch),
        })
        for item in data.get('items', []):
            vid = item['id']
            if vid in known:
                skipped['known'] += 1
                continue
            if vid in existing:
                # Already pending: this sighting is the signal, not noise.
                prior = existing[vid]
                prior['times_seen'] = int(prior.get('times_seen') or 1) + 1
                prior['last_seen'] = today
                live_now = (item.get('liveStreamingDetails') or {}).get(
                    'concurrentViewers')
                if live_now:
                    prior['concurrent_viewers'] = live_now
                reseen += 1
                continue
            snippet = item.get('snippet', {})
            status = item.get('status', {})
            live = item.get('liveStreamingDetails', {})
            title = snippet.get('title', '')
            if not status.get('embeddable'):
                skipped['not_embeddable'] += 1
                continue
            if snippet.get('liveBroadcastContent') != 'live':
                skipped['not_live'] += 1
                continue
            if not TITLE_KEYWORDS.search(title):
                skipped['no_keyword'] += 1
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
                'first_seen': today,
                'last_seen': today,
                'times_seen': 1,
            }
            new_candidates.append(cand)

    candidates['candidates'].extend(new_candidates)
    # Drop anything approved or rejected since the last scan, so the store
    # never carries work that is already done.
    candidates['candidates'] = [c for c in candidates['candidates']
                                if c['video_id'] not in known]
    candidates['candidates'] = _rank(candidates['candidates'])
    candidates['last_scan'] = today
    _save_json(path, candidates)

    print(f'filtered out: {skipped}')
    print(f'{len(new_candidates)} new candidate(s), {reseen} seen again, '
          f'{len(candidates["candidates"])} pending review in {path}')
    for c in new_candidates:
        spot = c['suggested_spot'] or 'no spot match'
        print(f'  - {c["video_id"]}  {c["title"][:70]}  [{spot}]  {c["url"]}')
    if args.markdown:
        _write_issue_markdown(candidates['candidates'], args.markdown)
    return 0


def _md_safe(text):
    """Titles and channel names come from YouTube and are untrusted."""
    return re.sub(r'[\[\]()`<>]', '', text or '')


def _write_issue_markdown(candidates, path):
    """Render the pending list, most-persistent first.

    Grouped by sighting count rather than sorted flat, because the useful
    question when reviewing is "which of these have proven they stick around",
    and a bare ordering does not make that visible at a glance.
    """
    ranked = _rank(candidates)
    tiers = [
        (4, 'Seen in four or more scans - most likely to be stable cams'),
        (2, 'Seen in two or three scans'),
        (1, 'Seen once - newest, least proven'),
    ]
    with open(path, 'w') as f:
        f.write('Live YouTube surf cams found by the weekly scan and still '
                'awaiting review, ranked by how many scans each has survived. '
                'A stream that keeps reappearing is far more likely to be a '
                'permanent cam than a one-off broadcast, so work from the top.'
                '\n\nApprove or reject with `scripts/youtube_cam_scan.py`; '
                'handled entries drop off this list automatically on the next '
                'scan. This issue is rewritten in place each week rather than '
                'replaced, so it stays the single place to look.\n\n'
                'Tick the box on anything you do not want and it will not be '
                'suggested again, so there is no need to approve or reject a '
                'stream just to be rid of it.\n')
        for floor, heading in tiers:
            ceiling = tiers[tiers.index((floor, heading)) - 1][0] if floor != 4 else None
            group = [c for c in ranked
                     if (int(c.get('times_seen') or 1) >= floor
                         and (ceiling is None
                              or int(c.get('times_seen') or 1) < ceiling))]
            if not group:
                continue
            f.write(f'\n## {heading}\n\n')
            for c in group:
                spot = _md_safe(c.get('suggested_spot')) or '_no spot match_'
                seen = int(c.get('times_seen') or 1)
                viewers = c.get('concurrent_viewers')
                extra = f' - {viewers} watching' if viewers else ''
                f.write(f'- [ ] [{_md_safe(c["title"])}]({c["url"]}) - '
                        f'{_md_safe(c.get("channel"))} - suggested spot: {spot} - '
                        f'seen {seen}x{extra} - `{c["video_id"]}`\n')
        f.write(f'\n{len(ranked)} awaiting review.\n')


def cmd_absorb_checkmarks(args):
    """Treat a ticked box in the review issue as "not this one, don't ask again".

    Reviewing a stream usually ends one of three ways: approve it, reject it, or
    decide it is not worth adding and move on. Only the first two had a home,
    so the third kept coming back every week. Ticking the box in the issue is
    the natural way to express it, so that is what this reads.

    Ordering matters: the scan rewrites the issue body, which would wipe the
    ticks, so this has to run BEFORE the render or the dismissal is lost and the
    stream reappears.
    """
    import subprocess
    path = _candidates_path(args.data_dir)
    store = _load_json(path, {'candidates': [], 'dismissed': {}})
    store.setdefault('dismissed', {})

    try:
        body = subprocess.run(
            ['gh', 'issue', 'view', str(args.issue), '--json', 'body', '-q', '.body'],
            capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f'Could not read issue #{args.issue}: {e}')
        return 1

    ticked = []
    for line in body.splitlines():
        m = ISSUE_LINE.match(line.strip())
        if m and m.group('done') == 'x':
            ticked.append((m.group('vid'), m.group('title')))

    today = date.today().isoformat()
    newly = 0
    for vid, title in ticked:
        if vid not in store['dismissed']:
            store['dismissed'][vid] = {'title': title, 'dismissed_on': today}
            newly += 1
    before = len(store['candidates'])
    store['candidates'] = [c for c in store['candidates']
                           if c['video_id'] not in store['dismissed']]
    _save_json(path, store)
    print(f'{len(ticked)} ticked in issue #{args.issue}; {newly} newly dismissed, '
          f'{before - len(store["candidates"])} removed from the pending list '
          f'({len(store["dismissed"])} dismissed in total)')
    return 0


def cmd_render_issue(args):
    """Write the ranked pending list to a markdown file."""
    path = _candidates_path(args.data_dir)
    store = _load_json(path, {'candidates': [], 'dismissed': {}})
    known = _handled_ids()
    known.update(store.get('dismissed', {}))
    pending = [c for c in store['candidates'] if c['video_id'] not in known]
    _write_issue_markdown(pending, args.out)
    print(f'Wrote {len(pending)} pending candidate(s) to {args.out}')
    return 0


def cmd_migrate_issues(args):
    """Seed the persistent store from the old weekly review issues.

    Those issues already record the signal we want: a stream listed in four of
    six weekly issues has demonstrated exactly the persistence that makes it
    worth reviewing first. Reading them back recovers that history instead of
    starting the ranking from zero.
    """
    import subprocess
    path = _candidates_path(args.data_dir)
    store = _load_json(path, {'candidates': []})
    by_id = {c['video_id']: c for c in store['candidates']}
    known = _handled_ids()
    checked_off = set()
    seen_counts = {}

    for number in args.issues:
        raw = subprocess.run(
            ['gh', 'issue', 'view', str(number), '--json', 'body,createdAt'],
            capture_output=True, text=True, check=True).stdout
        payload = json.loads(raw)
        created = (payload.get('createdAt') or '')[:10]
        for line in payload['body'].splitlines():
            m = ISSUE_LINE.match(line.strip())
            if not m:
                continue
            vid = m.group('vid')
            if m.group('done') == 'x':
                # Already handled by hand in the issue; do not resurrect it.
                checked_off.add(vid)
                continue
            seen_counts[vid] = seen_counts.get(vid, 0) + 1
            rest = m.group('rest') or ''
            chan_m = ISSUE_CHANNEL.match(rest)
            spot_m = ISSUE_SPOT.search(rest)
            spot = spot_m.group('spot') if spot_m else None
            if spot in ('_no spot match_', '', None):
                spot = None
            entry = by_id.setdefault(vid, {
                'video_id': vid,
                'title': m.group('title'),
                'channel': (chan_m.group('channel') if chan_m else '') or '',
                'url': m.group('url'),
                'concurrent_viewers': None,
                'suggested_spot': spot,
                'suggested_lat': None,
                'suggested_lon': None,
                'suggested_state': None,
                'first_seen': created,
            })
            if created and created < (entry.get('first_seen') or created):
                entry['first_seen'] = created
            entry['last_seen'] = max(entry.get('last_seen') or '', created)

    for vid, count in seen_counts.items():
        by_id[vid]['times_seen'] = max(int(by_id[vid].get('times_seen') or 0), count)

    kept = [c for vid, c in by_id.items()
            if vid not in known and vid not in checked_off]
    store['candidates'] = _rank(kept)
    _save_json(path, store)

    tiers = {}
    for c in store['candidates']:
        tiers[c['times_seen']] = tiers.get(c['times_seen'], 0) + 1
    print(f'Migrated {len(store["candidates"])} pending candidate(s) from '
          f'issues {args.issues} into {path}')
    print(f'  skipped {len(checked_off)} already checked off in the issues, '
          f'{len(by_id) - len(kept)} already approved or rejected')
    for seen in sorted(tiers, reverse=True):
        print(f'  seen {seen}x: {tiers[seen]}')
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


def cmd_remove(args):
    """Un-approve a cam (e.g. approved with the wrong id or coords).

    Unlike reject, the id is NOT blacklisted — it can be re-approved or
    will resurface as a candidate on a future scan.
    """
    cams_data = _load_json(CAMS_FILE, {'cams': [], 'rejected_video_ids': []})
    before = len(cams_data['cams'])
    cams_data['cams'] = [c for c in cams_data['cams'] if c['video_id'] != args.video_id]
    if len(cams_data['cams']) == before:
        sys.exit(f'{args.video_id} is not in the approved catalog.')
    _save_json(CAMS_FILE, cams_data)
    print(f'Removed {args.video_id} from the approved catalog. '
          'It can be re-approved at any time.')
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


def cmd_sync_issue(args):
    """Check off handled candidates in a cam-review issue.

    Marks the checklist line of every approved or rejected video id as done,
    so the issue always shows what is left to review. Uses the gh CLI (must
    be authenticated as the repo account). Run after an approve/reject
    session.
    """
    import subprocess
    cams_data = _load_json(CAMS_FILE, {'cams': [], 'rejected_video_ids': []})
    handled = {c['video_id'] for c in cams_data['cams']}
    handled.update(cams_data.get('rejected_video_ids', []))

    body = subprocess.run(
        ['gh', 'issue', 'view', str(args.issue), '--json', 'body', '-q', '.body'],
        capture_output=True, text=True, check=True, cwd=ROOT).stdout
    out, checked, remaining = [], 0, 0
    for line in body.splitlines():
        if line.lstrip().startswith('- [ ]'):
            m = re.search(r'`([A-Za-z0-9_-]{11})`\s*$', line)
            if m and m.group(1) in handled:
                line = line.replace('- [ ]', '- [x]', 1)
                checked += 1
            else:
                remaining += 1
        out.append(line)

    if checked:
        subprocess.run(['gh', 'issue', 'edit', str(args.issue), '--body-file', '-'],
                       input='\n'.join(out) + '\n', text=True, check=True, cwd=ROOT)
    print(f'Checked off {checked} handled cam(s) in issue #{args.issue}; '
          f'{remaining} left to review.')
    if checked and not remaining:
        print('All candidates handled — the issue can be closed.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('scan', help='find new live embeddable surf cam candidates')
    p.add_argument('--query', action='append', help='override the default search queries (repeatable)')
    p.add_argument('--limit', type=int, default=25, help='max results per query (default 25)')
    p.add_argument('--markdown', help='also write a review checklist to this markdown file')
    p.add_argument('--data-dir',
                   help='directory holding the persistent candidate store; CI '
                        'passes a checkout of the data branch so sightings '
                        'accumulate across runs')
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

    p = sub.add_parser('remove', help='un-approve a cam without blacklisting it (fix a mistaken approve)')
    p.add_argument('video_id')
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser('reject', help='dismiss a candidate permanently')
    p.add_argument('video_id')
    p.set_defaults(func=cmd_reject)

    p = sub.add_parser('absorb-checkmarks',
                       help='treat ticked boxes in the review issue as dismissals')
    p.add_argument('issue', type=int, help='issue number to read')
    p.add_argument('--data-dir', help='directory holding the persistent candidate store')
    p.set_defaults(func=cmd_absorb_checkmarks)

    p = sub.add_parser('render-issue', help='write the ranked pending list as markdown')
    p.add_argument('--data-dir', help='directory holding the persistent candidate store')
    p.add_argument('--out', required=True, help='markdown file to write')
    p.set_defaults(func=cmd_render_issue)

    p = sub.add_parser('migrate-issues',
                       help='seed the store from the old weekly review issues')
    p.add_argument('issues', type=int, nargs='+', help='issue numbers to mine')
    p.add_argument('--data-dir', help='directory holding the persistent candidate store')
    p.set_defaults(func=cmd_migrate_issues)

    p = sub.add_parser('sync-issue', help='check off approved/rejected cams in a review issue')
    p.add_argument('issue', type=int, help='issue number (e.g. 20)')
    p.set_defaults(func=cmd_sync_issue)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()
