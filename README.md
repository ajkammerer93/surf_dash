# verification-data

Machine-written branch. The `forecast-verification` GitHub Actions workflow
(master branch, runs every 6 hours) snapshots the live site's forecasts at
NDBC buoy locations, scores them against buoy observations, and commits the
results here:

- `snapshots.jsonl` — forecast wave series captured at each buoy location
- `pairs.jsonl` — scored (forecast, observation) pairs
- `stats.json` — rolling 30-day bias/MAE/RMSE by station and lead time;
  read by the app for the /accuracy page and forecast bias correction

Do not edit by hand and do not merge into master. The station list and the
pipeline code live on master (`data/verification/buoy_pairs.json`,
`scripts/forecast_verification.py`).
