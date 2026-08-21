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

# One region per weekday (Mon..Sun). Rotating regions keeps the feed varied
# without any per-day decisions.
# Which schedule each region belongs to. Posting a Southern California card at
# 4am Pacific or a Hawaii card at 1am local means reporting overnight
# conditions to a sleeping audience, so the Pacific regions run on a later cron.
REGION_GROUPS = {
    'outer-banks': 'east',
    'southern-california': 'west',
    'great-lakes': 'east',
    'jersey-shore': 'east',
    'florida-space-coast': 'east',
    'hawaii-north-shore': 'west',
    'new-england': 'east',
}

# Not a region -- the weekly card built from the forecast-verification
# pipeline, which scores forecasts against NDBC buoys.
ACCURACY_SLUG = 'accuracy'

REGION_ROTATION = [
    'outer-banks',          # Monday
    'southern-california',  # Tuesday
    'great-lakes',          # Wednesday
    'jersey-shore',         # Thursday
    'florida-space-coast',  # Friday
    'hawaii-north-shore',   # Saturday
    'new-england',          # Sunday
]


class CardRefused(Exception):
    """The endpoint declined to build a card on purpose, as opposed to failing
    to build one. Only the accuracy card can refuse."""


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
    if body.get('error') != 'accuracy card not published' and not body.get('reason'):
        return None
    for key in ('reason', 'error', 'message', 'detail'):
        if body.get(key):
            return str(body[key])
    return 'no reason given'


def fetch_card(base_url, slug, retries=3, wait_s=25, refuse_on_503=False):
    """Fetch caption + image URL for a card. The first request after a cold
    cache can 503 while the server gathers upstream forecasts — retry.

    refuse_on_503 turns that around for the accuracy card, whose 503 is the
    publication guards saying no. Retrying a guard just asks the same question
    three times and delays the same answer, so raise CardRefused instead.
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
    parser.add_argument('--group', choices=['east', 'west'],
                        help='Only post if the day\'s region is in this group; '
                             'lets one schedule serve East Coast mornings and '
                             'another serve Pacific mornings')
    args = parser.parse_args()

    if args.accuracy:
        # --region/--group/--week all describe the weekday region rotation,
        # which the accuracy card is not part of.
        clashes = [name for name, on in (('--region', args.region),
                                         ('--group', args.group),
                                         ('--week', args.week)) if on]
        if clashes:
            sys.exit(f'--accuracy posts the weekly accuracy card, which is not '
                     f'part of the region rotation, so it cannot be combined '
                     f'with {", ".join(clashes)}.')
    if args.week and not args.dry_run:
        sys.exit('--week only makes sense with --dry-run')
    if args.story_only and args.no_story:
        sys.exit('--story-only and --no-story cancel each other out')

    today = datetime.now()
    if args.accuracy:
        targets = [(today.strftime('%A %b %d'), ACCURACY_SLUG)]
    elif args.week:
        days = [(today + timedelta(days=i)) for i in range(7)]
        targets = [(d.strftime('%A %b %d'), REGION_ROTATION[d.weekday()]) for d in days]
    else:
        region = args.region or REGION_ROTATION[today.weekday()]
        # An explicitly requested region is a deliberate manual post, so the
        # schedule's group filter should not veto it.
        if not args.region and args.group and REGION_GROUPS.get(region) != args.group:
            print(f"Today's region ({region}) is not in the {args.group} group "
                  f"- nothing to do on this schedule.")
            sys.exit(0)
        targets = [(today.strftime('%A %b %d'), region)]

    failures = 0
    for label, slug in targets:
        print(f'== {label}: {slug} ==')
        try:
            card = fetch_card(args.base_url, slug, refuse_on_503=args.accuracy)
        except CardRefused as e:
            # A refused week is the guards working, not a broken run. Exiting
            # non-zero here would paint the weekly job red on purpose and train
            # me to ignore it, so report and stop clean -- but annotate the run
            # in Actions, because a skipped week that shows as a plain green
            # tick is indistinguishable from a week that posted.
            print(f'  endpoint refused to publish: {e}')
            print('  nothing to post - skipping this week.')
            if os.environ.get('GITHUB_ACTIONS'):
                print(f'::warning::accuracy post skipped: {e}')
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
