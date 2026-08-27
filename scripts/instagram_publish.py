#!/usr/bin/env python3
"""Publish the daily regional surf card to Instagram.

Pulls the day's card (image + caption) from the live site's social endpoints
and posts it via the Instagram Graph API. Designed to run from cron or GitHub
Actions, but also works as a manual weekly routine:

    # Print today's image URL + caption without posting (no credentials needed)
    python scripts/instagram_publish.py --dry-run

    # Print the whole week's rotation for batch scheduling in Meta Business Suite
    python scripts/instagram_publish.py --dry-run --week

    # Post today's card (requires IG_USER_ID and IG_ACCESS_TOKEN env vars)
    python scripts/instagram_publish.py

    # Post the weekly forecast-accuracy card instead of a regional one
    python scripts/instagram_publish.py --accuracy

The accuracy endpoint answers 503 when its publication guards decide the
verification data is too stale or too thin to publish. That is a deliberate
"no", so --accuracy prints the reason and exits 0 rather than retrying or
failing the run -- a skipped week is a correct outcome.

Credentials (env vars):
    IG_USER_ID       Instagram user id (from scripts/instagram_auth.py)
    IG_ACCESS_TOKEN  long-lived token with instagram_business_content_publish

Setup is handled by scripts/instagram_auth.py, which uses "Instagram API with
Instagram Login" -- no linked Facebook Page required, just a Professional
Instagram account. Tokens last 60 days and must be refreshed before then; the
token-refresh workflow watches the clock and this script reports the API error
clearly when one has expired.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import requests

BASE_URL = 'https://freesurfforecast.com'
# Instagram Login path. The Facebook Login path (graph.facebook.com) needs a
# linked Page and a Page access token; this one needs neither.
GRAPH_URL = 'https://graph.instagram.com/v25.0'

# Which schedule each region belongs to. Posting a Southern California card at
# 4am Pacific or a Hawaii card at 1am local means reporting overnight
# conditions to a sleeping audience, so the Pacific regions run on a later cron.
REGION_GROUPS = {
    'outer-banks': 'east',
    'north-carolina-coast': 'east',
    'virginia-coast': 'east',
    'southern-california': 'west',
    'northern-california': 'west',
    'great-lakes': 'east',
    'jersey-shore': 'east',
    'florida-space-coast': 'east',
    'hawaii-north-shore': 'west',
    'new-england': 'east',
}

# Not regions. The weekly card built from the forecast-verification pipeline,
# and the daily highlight card built from whatever the NDBC network is measuring
# right now. Both can decline to publish, which the region cards cannot.
ACCURACY_SLUG = 'accuracy'
HIGHLIGHT_SLUG = 'highlight'
REFUSING_SLUGS = (ACCURACY_SLUG, HIGHLIGHT_SLUG)

# The rotation is DEMAND-WEIGHTED, not one-per-weekday. It is indexed by day
# of year rather than weekday, so its length is free and a region can appear
# more than once per cycle.
#
# The weights come from what people actually read (Cloudflare top paths, 7d):
# wrightsville-beach 26, avalon-nj 20, virginia-beach 12, cape-may-cove-nj 12,
# santa-cruz-ca 10. The old seven-region rotation had no card for three of
# those top five -- the #1 page on the site had no post at all -- so the feed
# was advertising coastlines the audience does not read. north-carolina-coast
# and jersey-shore appear twice per cycle because they carry the top four
# spots between them.
REGION_ROTATION = [
    'north-carolina-coast',   # Wrightsville / Topsail -- most-read spots
    'southern-california',
    'jersey-shore',           # Avalon / Cape May
    'great-lakes',
    'virginia-coast',
    'hawaii-north-shore',
    'north-carolina-coast',
    'outer-banks',
    'jersey-shore',
    'northern-california',    # Santa Cruz
    'florida-space-coast',
    'new-england',
]

# Below this the card is a flat day: score is roughly 10 points per foot of
# face (_simple_surf_score in app.py), so 20 is about 2 ft.
FLAT_SCORE = 20.0
# How far down the rotation to look for a better card on a flat day.
FLAT_LOOKAHEAD = 4


def rotation_region(day, group=None):
    """Today's scheduled region, optionally restricted to one cron group."""
    idx = day.timetuple().tm_yday % len(REGION_ROTATION)
    if group is None:
        return REGION_ROTATION[idx]
    for step in range(len(REGION_ROTATION)):
        slug = REGION_ROTATION[(idx + step) % len(REGION_ROTATION)]
        if REGION_GROUPS.get(slug) == group:
            return slug
    return None


def card_best_score(card):
    """Best spot score on a card, or 0 when it carries none."""
    scores = [sp.get('score') or 0 for sp in (card.get('spots') or [])]
    return max(scores) if scores else 0.0


def flat_day_alternatives(slug, group):
    """Same-group regions to try when `slug` comes up flat.

    Deliberately SAME-GROUP only. Picking the best surf anywhere would send an
    east-coast audience a Hawaii card most days -- the Pacific regions are
    bigger nearly all the time, so an unrestricted "best surf wins" rule
    quietly starves exactly the coastlines this rotation exists to serve. The
    swap is a floor on quality, not a ranking.
    """
    idx = REGION_ROTATION.index(slug) if slug in REGION_ROTATION else 0
    out = []
    for step in range(1, len(REGION_ROTATION)):
        cand = REGION_ROTATION[(idx + step) % len(REGION_ROTATION)]
        if cand != slug and cand not in out and REGION_GROUPS.get(cand) == group:
            out.append(cand)
        if len(out) >= FLAT_LOOKAHEAD:
            break
    return out


def pick_region(base_url, scheduled, group, dry_run=False):
    """The region to post today: the scheduled one, unless it is flat and a
    same-group sibling is doing better.

    A flat card is worth avoiding: "1.4ft @ 7s" is not a reason to open the
    app, and the feed reads as dead. But the fix must not become "post
    whichever coast has the biggest surf", because the Pacific wins that
    contest nearly every day and the audience this rotation serves is on the
    Atlantic. So the rotation stays authoritative and this only intervenes
    when the scheduled region is genuinely flat, choosing among ITS OWN group.

    Best-effort by construction: any fetch problem here leaves the scheduled
    region in place rather than failing the run, because a flat card still
    beats no card.
    """
    if not group:
        return scheduled
    try:
        card = fetch_card(base_url, scheduled, retries=1)
        best = card_best_score(card)
    except Exception as e:
        print(f'  flat-day check skipped for {scheduled}: {e}')
        return scheduled
    if best >= FLAT_SCORE:
        return scheduled
    print(f'  {scheduled} is flat (best score {best:.0f} < {FLAT_SCORE:.0f}); '
          f'looking for a better {group} region')
    winner, winner_score = scheduled, best
    for cand in flat_day_alternatives(scheduled, group):
        try:
            alt = fetch_card(base_url, cand, retries=1)
        except Exception as e:
            print(f'    {cand}: unavailable ({e})')
            continue
        alt_score = card_best_score(alt)
        print(f'    {cand}: best score {alt_score:.0f}')
        if alt_score > winner_score:
            winner, winner_score = cand, alt_score
    if winner != scheduled:
        print(f'  posting {winner} instead of {scheduled} '
              f'({winner_score:.0f} vs {best:.0f})')
    else:
        print(f'  no better {group} region today; keeping {scheduled}')
    return winner


class CardRefused(Exception):
    """The endpoint declined to build a card on purpose, as opposed to failing
    to build one. Only the accuracy and highlight cards can refuse."""


def _refusal_reason(resp):
    """The endpoint's own reason for declining, or None if this 503 did not
    come from the endpoint.

    Only a JSON body carrying the guard marker counts. A 503 is also what a
    dead site, a CDN error page or a failed stats fetch produces, and reading
    those as "the guards said no" turns an outage into a green run with a
    silently skipped week -- a no-op that reports success. Anything that is not
    the endpoint speaking falls through to the normal retry-then-fail path.
    """
    try:
        body = resp.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    marker = str(body.get('error') or '')
    if not marker.endswith('card not published') and not body.get('reason'):
        return None
    for key in ('reason', 'error', 'message', 'detail'):
        if body.get(key):
            return str(body[key])
    return 'no reason given'


def fetch_card(base_url, slug, retries=3, wait_s=25, refuse_on_503=False):
    """Fetch caption + image URL for a card. The first request after a cold
    cache can 503 while the server gathers upstream forecasts — retry.

    refuse_on_503 turns that around for the cards that can decline, whose 503 is
    the publication guards saying no. Retrying a guard just asks the same
    question three times and delays the same answer, so raise CardRefused.
    """
    url = f'{base_url}/api/social-card/{slug}'
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 200:
                return resp.json()
            if refuse_on_503 and resp.status_code == 503:
                reason = _refusal_reason(resp)
                if reason:
                    raise CardRefused(reason)
            print(f'  {url} -> HTTP {resp.status_code} (attempt {attempt + 1}/{retries})')
        except requests.RequestException as e:
            print(f'  {url} -> {e} (attempt {attempt + 1}/{retries})')
        if attempt < retries - 1:
            time.sleep(wait_s)
    return None


def publish(ig_user_id, token, image_url, caption, story=False):
    """Two-step Graph API publish: create a media container, then publish it.

    A story uses media_type=STORIES, carries no caption at all, and expires
    after 24 hours -- which is why the story image has to repeat the call to
    action the feed caption would otherwise provide.
    """
    # The API fetches this URL itself and accepts JPEG only -- a PNG here comes
    # back as an opaque media-download error, so fail with a readable one.
    if not image_url.lower().split('?')[0].endswith(('.jpg', '.jpeg')):
        sys.exit(f'Instagram accepts JPEG only, refusing to post {image_url}. '
                 'Use the .jpg card variant.')
    fields = {'image_url': image_url, 'access_token': token}
    if story:
        fields['media_type'] = 'STORIES'
    else:
        fields['caption'] = caption
    create = requests.post(
        f'{GRAPH_URL}/{ig_user_id}/media',
        data=fields,
        timeout=60,
    )
    body = create.json()
    if 'id' not in body:
        err = body.get('error', {})
        if err.get('code') == 190:
            sys.exit('Access token expired or invalid - run '
                     '"python scripts/instagram_auth.py refresh" if it is still '
                     'within 60 days, otherwise re-authorize with the "url" and '
                     '"exchange" commands, then update the IG_ACCESS_TOKEN secret.')
        sys.exit(f'Media container creation failed: {body}')
    creation_id = body['id']

    # Containers for single images are usually ready immediately; give the
    # CDN fetch a moment anyway.
    time.sleep(5)

    pub = requests.post(
        f'{GRAPH_URL}/{ig_user_id}/media_publish',
        data={'creation_id': creation_id, 'access_token': token},
        timeout=60,
    )
    body = pub.json()
    if 'id' not in body:
        sys.exit(f'Publish failed: {body}')
    return body['id']


def main():
    parser = argparse.ArgumentParser(description='Post the daily surf card to Instagram')
    parser.add_argument('--region', help='Region slug (default: weekday rotation)')
    parser.add_argument('--base-url', default=BASE_URL)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print image URL + caption without posting')
    parser.add_argument('--week', action='store_true',
                        help="With --dry-run: print the next 7 days' rotation")
    parser.add_argument('--no-story', action='store_true',
                        help='Post only the feed card, skip the story')
    parser.add_argument('--story-only', action='store_true',
                        help='Post only the story, skip the feed card -- lets '
                             'you test stories without duplicating a feed post')
    parser.add_argument('--accuracy', action='store_true',
                        help='Post the weekly forecast-accuracy card instead '
                             'of a regional one; skips cleanly (exit 0) when '
                             'the endpoint refuses to publish')
    parser.add_argument('--highlight', action='store_true',
                        help='Post the daily highlight card (the most notable '
                             'reading on the NDBC buoy network) instead of a '
                             'regional one; skips cleanly (exit 0) when nothing '
                             'clears the bar for the day')
    parser.add_argument('--kind', choices=['biggest-seas', 'longest-period',
                                           'strongest-wind'],
                        help='With --highlight: force one kind instead of '
                             'taking the day\'s rotation')
    parser.add_argument('--group', choices=['east', 'west'],
                        help='Only post if the day\'s region is in this group; '
                             'lets one schedule serve East Coast mornings and '
                             'another serve Pacific mornings')
    args = parser.parse_args()

    if args.accuracy and args.highlight:
        sys.exit('--accuracy and --highlight post different cards; pick one.')
    special = '--accuracy' if args.accuracy else ('--highlight' if args.highlight else None)
    if special:
        # --region/--group/--week all describe the weekday region rotation,
        # which neither of these cards is part of.
        clashes = [name for name, on in (('--region', args.region),
                                         ('--group', args.group),
                                         ('--week', args.week)) if on]
        if clashes:
            sys.exit(f'{special} posts a card that is not part of the region '
                     f'rotation, so it cannot be combined with '
                     f'{", ".join(clashes)}.')
    if args.kind and not args.highlight:
        sys.exit('--kind only applies to --highlight.')
    if args.week and not args.dry_run:
        sys.exit('--week only makes sense with --dry-run')
    if args.story_only and args.no_story:
        sys.exit('--story-only and --no-story cancel each other out')

    today = datetime.now()
    if args.accuracy:
        targets = [(today.strftime('%A %b %d'), ACCURACY_SLUG)]
    elif args.highlight:
        slug = HIGHLIGHT_SLUG + (f'?kind={args.kind}' if args.kind else '')
        targets = [(today.strftime('%A %b %d'), slug)]
    elif args.week:
        days = [(today + timedelta(days=i)) for i in range(len(REGION_ROTATION))]
        targets = [(d.strftime('%A %b %d'), rotation_region(d)) for d in days]
    else:
        if args.region:
            # An explicit region is a deliberate manual post: no group filter,
            # no flat-day swap. What was asked for is what gets posted.
            region = args.region
        else:
            region = rotation_region(today)
            if args.group and REGION_GROUPS.get(region) != args.group:
                print(f"Today's region ({region}) is not in the {args.group} "
                      f"group - nothing to do on this schedule.")
                sys.exit(0)
            region = pick_region(args.base_url, region, args.group,
                                 dry_run=args.dry_run)
        targets = [(today.strftime('%A %b %d'), region)]

    failures = 0
    for label, slug in targets:
        print(f'== {label}: {slug} ==')
        try:
            card = fetch_card(args.base_url, slug,
                              refuse_on_503=slug.split('?')[0] in REFUSING_SLUGS)
        except CardRefused as e:
            # A refused week is the guards working, not a broken run. Exiting
            # non-zero here would paint the weekly job red on purpose and train
            # me to ignore it, so report and stop clean -- but annotate the run
            # in Actions, because a skipped week that shows as a plain green
            # tick is indistinguishable from a week that posted.
            print(f'  endpoint refused to publish: {e}')
            print('  nothing to post - skipping this run.')
            if os.environ.get('GITHUB_ACTIONS'):
                print(f'::warning::{slug.split("?")[0]} post skipped: {e}')
            sys.exit(0)
        if not card:
            print('  FAILED to fetch card data')
            failures += 1
            continue
        # A 200 missing its fields is a real failure, not a skip. Catch it here
        # so it reports readably instead of tracebacking mid-publish.
        missing = [k for k in ('image_url', 'caption') if not card.get(k)]
        if missing:
            print(f'  FAILED: card response is missing {", ".join(missing)}')
            failures += 1
            continue
        if args.dry_run:
            print(f"  image:   {card['image_url']}")
            print('  caption:')
            for line in card['caption'].split('\n'):
                print(f'    {line}')
            print()
            continue

        ig_user_id = os.environ.get('IG_USER_ID')
        token = os.environ.get('IG_ACCESS_TOKEN')
        if not ig_user_id or not token:
            sys.exit('IG_USER_ID and IG_ACCESS_TOKEN env vars are required to '
                     'post (or use --dry-run to print the card for manual '
                     'scheduling).')
        if not args.story_only:
            media_id = publish(ig_user_id, token, card['image_url'], card['caption'])
            print(f'  published: media id {media_id}')

        if args.no_story or not card.get('story_image_url'):
            # Say so. With --story-only and no story URL the loop would
            # otherwise end having printed nothing and posted nothing, which
            # reads exactly like a successful post.
            if not args.no_story:
                print('  no story_image_url in the card - story skipped')
        else:
            try:
                story_id = publish(ig_user_id, token, card['story_image_url'],
                                   None, story=True)
                print(f'  story:     media id {story_id}')
            except SystemExit as e:
                # On a normal run the feed post already went up, so report the
                # story failure loudly but keep the two outcomes distinct.
                print(f'  STORY FAILED: {e}')
                failures += 1

    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
