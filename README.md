# Netflix viewing history: cleaned, catalog-matched, time-split

This is the base dataset for the taste model. It covers 5,000 viewing events (17 Aug to 16 Nov 2026), each cleaned and matched to a title in the November 2026 catalog. It is split by date so the model learns from the early weeks and is scored on a final week it never saw.

Rebuild everything with `python clean_events.py` (needs pandas). The script checks that kept + quarantined = 5,000.

## Result

| File (`data/clean/`) | Rows | What it is |
|---|---|---|
| `train.csv` | 4,428 | Events from 2026-08-17 to 2026-11-09. The model learns from these. |
| `holdout.csv` | 358 | Events from 2026-11-10 to 2026-11-16. Hidden final week, used for evaluation only. |
| `events_clean.csv` | 4,786 | Train and holdout together, with a `split` column. |
| `quarantine.csv` | 214 | Removed events, each with its `quarantine_reason`. |
| `cleaning_log.json` | | The count for every rule below. |

**Cutoff date: 2026-11-10.** Anything dated before it goes to train, and 10 to 16 Nov goes to holdout: exactly 7 days, Tuesday to Monday. Events only have a date, not a time, so the cut falls cleanly on a day boundary. Every holdout member also appears in train. One holdout title, TT-0195, never appears in train (see below), so it's a genuine cold-start case.

## Problems found and the fix chosen for each

| # | Problem | Events | Fix | Why |
|---|---|---|---|---|
| 1 | `title_id` not in catalog (TT-0347, TT-0412, TT-0588) | 180 | **Quarantine** | No name, runtime or ID pattern ties them to a real title. Guessing would teach the model wrong tastes. |
| 2 | Watched before the title's release date (TT-0163, TT-0195, TT-0258) | 27 | **Quarantine** | These views can't have happened. If kept, they would also leak "future" titles into train. |
| 3 | Same member, title and day logged twice | 7 | **Keep the longest watch, drop the other** | It's one interaction, so counting it twice would overweight it. |
| 4 | `watch_seconds` between 30,000 and 85,000 (8 to 24 h). All other events are ≤ 3,600 s | 46 | **Set to null**, mark `watch_seconds_flag = corrupt_over_1h`, keep the event | The view is real but the duration is broken. There's a clean gap with nothing between 3,600 and 30,000. |
| 5 | `watch_seconds` blank in the source | 67 kept | Leave null, mark `watch_seconds_flag = missing` | Better not to invent a duration. |
| 6 | `device_type` / `country` blank | 131 / 47 kept | Fill with `unknown` | Keeps the row and makes the missing value explicit. |
| 7 | Kids profile watching an R or TV-MA title | 46 | **Flag only** (`kids_profile_mature_title`) | Could be a mislabelled profile or real behaviour. Downstream can decide. |

### Completion (derived, replaces the source flag)

The source `completed` flag disagrees with watch time far too often to trust. It is renamed **`completed_raw`** and kept for reference only.

- **`completion_ratio`** = `watch_seconds / (runtime_minutes × 60)`, capped at 1.0. It is left blank when watch time or runtime is missing.
- **`completed_derived`** = `completion_ratio ≥ 0.9`. It is blank wherever the ratio is blank.
- **Series caveat:** `runtime_minutes` for a series is the length of one episode. So for series the ratio means *"finished at least one episode"*, **not** "finished the series". A series can be capped at 1.0 after a single long session.

| | Events |
|---|---|
| Completion derived | 4,570 |
| Not derivable (113 missing watch time, 104 missing runtime, 1 missing both) | 216 |
| `completed_raw` ≠ `completed_derived` | 1,799 (39.4% of derivable) |
| … raw True, derived False | 1,459 |
| … raw False, derived True | 340 |

### Checks (asserted in the script)
- Kept + quarantined = 5,000.
- Every kept event matches exactly one catalog title. The join is on `title_id` only, catalog IDs are unique, and the join neither drops nor duplicates rows.
- No holdout event is dated before 2026-11-10, and no train event is dated on or after it.

### Catalog notes (the catalog itself is unchanged)
- **Duplicate names with different IDs:** *Dead Air Frequency* (TT-0054 / TT-0155) and *The Quiet Ward* (TT-0021 / TT-0299) are different titles. Always join on `title_id`, never on the name.
- **Missing `runtime_minutes`:** 10 titles (listed in the log), so completion ratios can't be worked out for them.
- **Not yet released:** TT-0300 *Pancake Summit* comes out 2026-11-20 and has no events.

### Known noise left as is
- Most members appear in several countries and both profile types. This looks like normal account sharing or travel, not an error.
