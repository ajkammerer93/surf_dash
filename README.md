# ux-data

Snapshots from the weekly UI/UX audit (`.github/workflows/ux-audit.yml`).

Master is protected, so no scheduled job commits to it. This orphan branch is
where `scripts/ux_audit.py` publishes what it measured, the same arrangement
`verification-data`, `seo-data` and `cam-data` use. A script that writes state
into the CI workspace loses it when the run ends; keeping it on a branch is what
makes week-over-week comparison possible at all.

| File | Contents |
|---|---|
| `ux-latest.json` | The full snapshot from the most recent run |
| `ux-history.jsonl` | One trimmed numeric row per run, appended |
| `shots/` | Full-page screenshots, overwritten each run so the tree stays small |

Nothing here is authoritative about the site. The collector measures and records;
it makes no judgements. Anything that reads this decides what it means.
