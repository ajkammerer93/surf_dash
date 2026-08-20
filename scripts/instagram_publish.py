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
REGION_ROTATION = [
    'outer-banks',          # Monday
    'southern-california',  # Tuesday
    'great-lakes',          # Wednesday
    'jersey-shore',         # Thursday
    'florida-space-coast',  # Friday
    'hawaii-north-shore',   # Saturday
    'new-england',          # Sunday
]


def fetch_card(base_url, region, retries=3, wait_s=25):
    """Fetch caption + image URL for a region. The first request after a cold
    cache can 503 while the server gathers upstream forecasts — retry."""
    url = f'{base_url}/api/social-card/{region}'
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 200:
                return resp.json()
            print(f'  {url} -> HTTP {resp.status_code} (attempt {attempt + 1}/{retries})')
        except requests.RequestException as e:
            print(f'  {url} -> {e} (attempt {attempt + 1}/{retries})')
        if attempt < retries - 1:
            time.sleep(wait_s)
    return None


def publish(ig_user_id, token, image_url, caption):
    """Two-step Graph API publish: create a media container, then publish it."""
    # The API fetches this URL itself and accepts JPEG only -- a PNG here comes
    # back as an opaque media-download error, so fail with a readable one.
    if not image_url.lower().split('?')[0].endswith(('.jpg', '.jpeg')):
        sys.exit(f'Instagram accepts JPEG only, refusing to post {image_url}. '
                 'Use the .jpg card variant.')
    create = requests.post(
        f'{GRAPH_URL}/{ig_user_id}/media',
        data={'image_url': image_url, 'caption': caption, 'access_token': token},
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
    args = parser.parse_args()

    if args.week and not args.dry_run:
        sys.exit('--week only makes sense with --dry-run')

    today = datetime.now()
    if args.week:
        days = [(today + timedelta(days=i)) for i in range(7)]
        regions = [(d.strftime('%A %b %d'), REGION_ROTATION[d.weekday()]) for d in days]
    else:
        region = args.region or REGION_ROTATION[today.weekday()]
        regions = [(today.strftime('%A %b %d'), region)]

    failures = 0
    for label, region in regions:
        print(f'== {label}: {region} ==')
        card = fetch_card(args.base_url, region)
        if not card:
            print('  FAILED to fetch card data')
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
        media_id = publish(ig_user_id, token, card['image_url'], card['caption'])
        print(f'  published: media id {media_id}')

    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
