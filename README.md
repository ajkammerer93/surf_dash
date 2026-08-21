# cam-data

Persistent YouTube cam candidate store for `.github/workflows/youtube-cam-scan.yml`.

The scan used to write its candidate list into the CI workspace, which is thrown
away at the end of the run. Every scheduled scan therefore started from nothing,
rediscovered the same streams and filed another review issue. Keeping the store
here lets sightings accumulate, so a stream that reappears week after week can be
ranked above one that turned up once.

- `youtube_cam_candidates.json` — pending candidates with `times_seen`,
  `first_seen` and `last_seen`

Approve and reject still happen locally against master; handled ids drop out of
this store on the next scan.
