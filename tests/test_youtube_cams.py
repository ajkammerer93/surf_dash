"""YouTube live cam catalog and serving tests.

Pins the rules that make the feature safe to run unattended:
  - youtube_cams.json stays schema-valid (bad edits fail CI, not prod)
  - only the official youtube-nocookie embed player is ever served
  - disabled cams never reach the API response
  - the SurfChex link-only rule is untouched by the new cam type
"""

import json
import os
import re

import pytest

import app as app_module

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_ID_RE = re.compile(r'^[A-Za-z0-9_-]{11}$')


def _catalog():
    with open(os.path.join(ROOT, 'youtube_cams.json')) as f:
        return json.load(f)


class TestCatalogSchema:
    def test_catalog_loads_and_has_cams_list(self):
        data = _catalog()
        assert isinstance(data['cams'], list)
        assert isinstance(data.get('rejected_video_ids', []), list)

    def test_every_cam_is_well_formed(self):
        seen = set()
        for cam in _catalog()['cams']:
            assert VIDEO_ID_RE.match(cam['video_id']), cam
            assert cam['video_id'] not in seen, f"duplicate: {cam['video_id']}"
            seen.add(cam['video_id'])
            assert cam['name'].strip()
            assert -90 <= cam['lat'] <= 90
            assert -180 <= cam['lon'] <= 180
            assert isinstance(cam.get('disabled', False), bool)

    def test_rejected_ids_never_overlap_approved(self):
        data = _catalog()
        approved = {c['video_id'] for c in data['cams']}
        assert not approved & set(data.get('rejected_video_ids', []))


class TestServing:
    def test_youtube_cam_served_near_its_spot(self):
        # Seed cam: Oceanside CA
        cams = app_module.find_nearest_cameras(33.1959, -117.3795, count=10)
        yt = [c for c in cams if c['type'] == 'youtube']
        assert yt, 'expected the Oceanside YouTube cam to be returned'
        cam = yt[0]
        assert cam['url'].startswith('https://www.youtube-nocookie.com/embed/')
        assert 'mute=1' in cam['url']
        assert cam['page_url'].startswith('https://www.youtube.com/watch?v=')
        assert cam['distance_km'] < 5

    def test_youtube_cam_not_served_far_away(self):
        # Surf City NC is ~3,500 km from Oceanside CA
        cams = app_module.find_nearest_cameras(34.43, -77.55, count=50)
        for cam in cams:
            if cam['type'] == 'youtube':
                assert cam['distance_km'] <= app_module.SURFCHEX_MAX_DISTANCE_KM

    def test_disabled_youtube_cam_is_skipped(self, monkeypatch):
        monkeypatch.setattr(app_module, 'YOUTUBE_CAMERAS', [{
            'video_id': 'abcdefghijk', 'name': 'Dead Cam',
            'lat': 33.1959, 'lon': -117.3795, 'disabled': True,
        }])
        cams = app_module.find_nearest_cameras(33.1959, -117.3795, count=10)
        assert not [c for c in cams if c['type'] == 'youtube']

    def test_youtube_type_never_carries_raw_stream_urls(self):
        """The embed URL is the only playback surface — no HLS, no proxying."""
        cams = app_module.find_nearest_cameras(33.1959, -117.3795, count=10)
        for cam in cams:
            if cam['type'] == 'youtube':
                assert 'youtube-nocookie.com/embed/' in cam['url']
                assert '.m3u8' not in cam['url']

    def test_surfchex_cams_remain_link_only(self):
        """The June 2026 owner agreement: SurfChex streams are never embedded."""
        for cam in app_module.find_nearest_cameras(40.117, -74.036, count=50):
            urls = (cam.get('url') or '') + (cam.get('page_url') or '')
            if 'surfchex' in urls.lower():
                assert cam['type'] == 'link'
