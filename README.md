# seo-data

Audit snapshots written by `.github/workflows/seo-audit.yml`.

Master is protected, so the audit data lives here, the same way the forecast
verification data lives on `verification-data`.

- `audit-latest.json` — the most recent full snapshot
- `audit-history.jsonl` — one trimmed row per run, for trends
- `audit-state.json` — cursor for the rotating URL Inspection slice

Nothing here is served to users. The scheduled audit reads it to compare
against previous runs.
