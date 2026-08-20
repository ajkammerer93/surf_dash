#!/usr/bin/env python3
"""One-time Instagram credential setup for the daily card pipeline.

Uses "Instagram API with Instagram Login", which talks to graph.instagram.com
and needs no linked Facebook Page -- only a Professional (Business or Creator)
Instagram account and a Meta app with the Instagram product added.

The whole setup is three commands:

    # 1. Print the authorization URL, open it, approve, copy the ?code= value
    python scripts/instagram_auth.py url --app-id 1234567890

    # 2. Trade that code for a 60-day token and the IG user id
    export IG_APP_SECRET=...        # from the App Dashboard, keep it out of argv
    python scripts/instagram_auth.py exchange --app-id 1234567890 --code AQB...

    # 3. Later, before day 60, roll the token forward another 60 days
    export IG_ACCESS_TOKEN=...
    python scripts/instagram_auth.py refresh

Step 2 prints the two values to store as repository secrets, IG_USER_ID and
IG_ACCESS_TOKEN. The authorization code from step 1 is single-use and expires
in about a minute -- if it fails, just reopen the URL from step 1.

Tokens last 60 days and are only refreshable once they are 24 hours old. A
token that goes 60 days without a refresh is dead and the flow has to start
over at step 1, so the refresh workflow watches the clock.
"""
import argparse
import os
import sys
import urllib.parse

import requests

# Business Login authorization lives on www.instagram.com; the code-for-token
# exchange lives on api.instagram.com; everything after that is graph.
AUTH_URL = 'https://www.instagram.com/oauth/authorize'
TOKEN_URL = 'https://api.instagram.com/oauth/access_token'
GRAPH = 'https://graph.instagram.com'
GRAPH_VERSIONED = f'{GRAPH}/v25.0'

SCOPES = 'instagram_business_basic,instagram_business_content_publish'

# Must exactly match an entry in the app's valid OAuth redirect URIs. Nothing
# needs to be served here -- the code comes back as a query parameter and is
# copied out of the address bar by hand.
DEFAULT_REDIRECT = 'https://freesurfforecast.com/'


def _app_secret(args):
    secret = os.environ.get('IG_APP_SECRET') or getattr(args, 'app_secret', None)
    if not secret:
        sys.exit('Set IG_APP_SECRET (preferred, keeps it out of shell history) '
                 'or pass --app-secret.')
    return secret


def cmd_url(args):
    query = urllib.parse.urlencode({
        'client_id': args.app_id,
        'redirect_uri': args.redirect_uri,
        'response_type': 'code',
        'scope': SCOPES,
    })
    print('Open this URL, approve, then copy the "code" query parameter from')
    print('the address bar you land on (it expires in about a minute):\n')
    print(f'{AUTH_URL}?{query}\n')
    print('The redirect page itself does not need to exist -- only the code matters.')
    print(f'Note: {args.redirect_uri} must be listed verbatim in the app\'s')
    print('valid OAuth redirect URIs, or Instagram rejects the request.')


def cmd_exchange(args):
    secret = _app_secret(args)

    # Authorization code -> short-lived (1 hour) token.
    resp = requests.post(TOKEN_URL, data={
        'client_id': args.app_id,
        'client_secret': secret,
        'grant_type': 'authorization_code',
        'redirect_uri': args.redirect_uri,
        'code': args.code,
    }, timeout=60)
    body = resp.json()
    if 'access_token' not in body:
        sys.exit(f'Code exchange failed: {body}\n\n'
                 'Most common causes: the code was already used, it expired '
                 '(they last about a minute), or redirect_uri does not match '
                 'the one used to generate the code.')
    short_token = body['access_token']

    # Short-lived -> long-lived (60 days).
    resp = requests.get(f'{GRAPH}/access_token', params={
        'grant_type': 'ig_exchange_token',
        'client_secret': secret,
        'access_token': short_token,
    }, timeout=60)
    body = resp.json()
    if 'access_token' not in body:
        sys.exit(f'Long-lived token exchange failed: {body}')
    long_token = body['access_token']
    days = round(body.get('expires_in', 0) / 86400)

    # Identify the account the token belongs to.
    resp = requests.get(f'{GRAPH_VERSIONED}/me', params={
        'fields': 'user_id,username',
        'access_token': long_token,
    }, timeout=60)
    me = resp.json()
    user_id = me.get('user_id')
    if not user_id:
        sys.exit(f'Could not read the Instagram user id: {me}')

    print(f'\nAuthorized @{me.get("username", "?")} -- token valid {days} days.\n')
    print('Store these as repository secrets')
    print('(Settings -> Secrets and variables -> Actions -> New repository secret):\n')
    print(f'  IG_USER_ID       {user_id}')
    print(f'  IG_ACCESS_TOKEN  {long_token}\n')
    print('Then uncomment the schedule block in .github/workflows/social-post.yml.')
    print(f'Refresh the token before it expires in {days} days -- the token-refresh')
    print('workflow will warn you, or run: python scripts/instagram_auth.py refresh')


def cmd_refresh(args):
    token = os.environ.get('IG_ACCESS_TOKEN')
    if not token:
        sys.exit('Set IG_ACCESS_TOKEN to the current long-lived token.')
    resp = requests.get(f'{GRAPH}/refresh_access_token', params={
        'grant_type': 'ig_refresh_token',
        'access_token': token,
    }, timeout=60)
    body = resp.json()
    if 'access_token' not in body:
        sys.exit(f'Refresh failed: {body}\n\n'
                 'A token must be at least 24 hours old to refresh, and one '
                 'that already expired cannot be refreshed -- start over with '
                 'the "url" command.')
    days = round(body.get('expires_in', 0) / 86400)
    if args.quiet:
        # Only the token, for piping straight into `gh secret set`.
        print(body['access_token'])
    else:
        print(f'Refreshed -- valid another {days} days.\n')
        print('Update the IG_ACCESS_TOKEN repository secret with:\n')
        print(f'  {body["access_token"]}')


def main():
    parser = argparse.ArgumentParser(
        description='Instagram credential setup for the daily card pipeline')
    sub = parser.add_subparsers(dest='command', required=True)

    p_url = sub.add_parser('url', help='print the authorization URL')
    p_url.add_argument('--app-id', required=True, help='Instagram App ID')
    p_url.add_argument('--redirect-uri', default=DEFAULT_REDIRECT)
    p_url.set_defaults(func=cmd_url)

    p_ex = sub.add_parser('exchange', help='trade the code for a 60-day token')
    p_ex.add_argument('--app-id', required=True)
    p_ex.add_argument('--code', required=True, help='the ?code= value from step 1')
    p_ex.add_argument('--redirect-uri', default=DEFAULT_REDIRECT)
    p_ex.add_argument('--app-secret', help='prefer the IG_APP_SECRET env var')
    p_ex.set_defaults(func=cmd_exchange)

    p_rf = sub.add_parser('refresh', help='extend the current token 60 more days')
    p_rf.add_argument('--quiet', action='store_true',
                      help='print only the new token')
    p_rf.set_defaults(func=cmd_refresh)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
