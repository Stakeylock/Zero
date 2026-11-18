"""Clean the Netflix viewing-events export and split it into train / holdout.

Usage:  python clean_events.py
Inputs  (repo root):
    Netflix title catalog, November 2026.csv
    Netflix viewing events export, 17 Aug to 16 Nov 2026.csv
Outputs (data/clean/):
    events_clean.csv   every kept event, with catalog fields + `split` column
    train.csv          events dated before CUTOFF_DATE  (model learns from these)
    holdout.csv        events dated CUTOFF_DATE .. END_DATE (hidden final week)
    quarantine.csv     events removed, with the reason
    cleaning_log.json  counts for every rule (reproducible audit trail)
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "Netflix title catalog, November 2026.csv"
EVENTS = ROOT / "Netflix viewing events export, 17 Aug to 16 Nov 2026.csv"
OUT = ROOT / "data" / "clean"

CUTOFF_DATE = pd.Timestamp("2026-11-10")  # first day of the hidden final week
END_DATE = pd.Timestamp("2026-11-16")     # last day in the export
MAX_WATCH_SECONDS = 3600                  # every plausible event is <= 1 hour; corrupt ones are 8-24 h
MATURE = {"R", "TV-MA"}

log = {}

cat = pd.read_csv(CATALOG, dtype=str, keep_default_na=False)
ev = pd.read_csv(EVENTS, dtype=str, keep_default_na=False)
log["raw_events"] = len(ev)

# ---- types -----------------------------------------------------------------
ev["event_date"] = pd.to_datetime(ev["event_datetime"], format="%Y-%m-%d")
ev["watch_seconds"] = pd.to_numeric(ev["watch_seconds"].replace("", None))
ev["completed"] = ev["completed"].map({"True": True, "False": False})
cat["release_date"] = pd.to_datetime(cat["release_date"], format="%Y-%m-%d")
cat["runtime_minutes"] = pd.to_numeric(cat["runtime_minutes"].replace("", None))
cat["is_new_series"] = cat["is_new_series"].map({"true": True, "false": False})

quarantine = []

def drop(mask, reason):
    rows = ev[mask].copy()
    rows["quarantine_reason"] = reason
    quarantine.append(rows)
    log[f"quarantined_{reason}"] = int(mask.sum())
    return ev[~mask]

# ---- 1. title_id not in catalog -> cannot be matched to a real title -------
orphan = ~ev["title_id"].isin(cat["title_id"])
log["orphan_title_ids"] = sorted(ev.loc[orphan, "title_id"].unique().tolist())
ev = drop(orphan, "title_not_in_catalog")

# ---- 2. watched before the title was released -> impossible event ----------
ev = ev.merge(cat[["title_id", "release_date"]], on="title_id", how="left")
pre = ev["event_date"] < ev["release_date"]
log["pre_release_titles"] = sorted(ev.loc[pre, "title_id"].unique().tolist())
ev = drop(pre, "watched_before_release").drop(columns="release_date")

# ---- 3. same member + title + day logged twice -> keep one session ---------
key = ["member_id", "title_id", "event_datetime"]
ev = ev.sort_values(key + ["watch_seconds"], ascending=[True] * 3 + [False], na_position="last")
dup = ev.duplicated(key, keep="first")
ev = drop(dup, "duplicate_member_title_day")

# ---- 4. impossible watch time -> null it out, keep the interaction ---------
bad_watch = ev["watch_seconds"] > MAX_WATCH_SECONDS
ev["watch_seconds_flag"] = "ok"
ev.loc[ev["watch_seconds"].isna(), "watch_seconds_flag"] = "missing"
ev.loc[bad_watch, "watch_seconds_flag"] = "corrupt_over_1h"
ev.loc[bad_watch, "watch_seconds"] = pd.NA
log["watch_seconds_nulled_corrupt"] = int(bad_watch.sum())
log["watch_seconds_missing_in_source"] = int((ev["watch_seconds_flag"] == "missing").sum())

# ---- 5. missing categorical context -> explicit 'unknown' ------------------
for col in ["device_type", "country"]:
    blank = ev[col] == ""
    log[f"{col}_filled_unknown"] = int(blank.sum())
    ev.loc[blank, col] = "unknown"

# ---- 6. join catalog (every remaining event matches exactly one title) -----
cat_cols = ["title_id", "title_name", "title_type", "primary_genre", "secondary_genre",
            "release_date", "maturity_rating", "runtime_minutes"]
n_before = len(ev)
ev = ev.merge(cat[cat_cols], on="title_id", how="inner", validate="many_to_one")
assert len(ev) == n_before, "catalog join lost or duplicated rows"

# flag (do not drop): kids profile watching R / TV-MA
ev["kids_profile_mature_title"] = (ev["profile_type"] == "kids") & ev["maturity_rating"].isin(MATURE)
log["kids_profile_mature_title_flagged"] = int(ev["kids_profile_mature_title"].sum())

# ---- 7. time split ---------------------------------------------------------
ev["split"] = ev["event_date"].lt(CUTOFF_DATE).map({True: "train", False: "holdout"})
ev = ev.sort_values(["event_date", "event_id"]).reset_index(drop=True)
train, hold = ev[ev["split"] == "train"], ev[ev["split"] == "holdout"]

log.update({
    "cutoff_date": str(CUTOFF_DATE.date()),
    "train_range": [str(train.event_date.min().date()), str(train.event_date.max().date())],
    "holdout_range": [str(hold.event_date.min().date()), str(hold.event_date.max().date())],
    "clean_events": len(ev), "train_events": len(train), "holdout_events": len(hold),
    "quarantined_total": int(sum(len(q) for q in quarantine)),
    "holdout_members_unseen_in_train": int((~hold.member_id.isin(train.member_id)).sum()),
    "holdout_titles_unseen_in_train": sorted(set(hold.title_id) - set(train.title_id)),
    "catalog_duplicate_names": cat[cat.title_name.duplicated(keep=False)]
        .groupby("title_name").title_id.apply(list).to_dict(),
    "catalog_missing_runtime": cat.loc[cat.runtime_minutes.isna(), "title_id"].tolist(),
    "catalog_unreleased_titles": cat.loc[cat.release_date > END_DATE, "title_id"].tolist(),
})
assert log["raw_events"] == log["clean_events"] + log["quarantined_total"]

# ---- write -----------------------------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)
ev["event_date"] = ev["event_date"].dt.date
ev["release_date"] = ev["release_date"].dt.date
ev["watch_seconds"] = ev["watch_seconds"].astype("Int64")
ev["runtime_minutes"] = ev["runtime_minutes"].astype("Int64")
ev = ev.drop(columns="event_datetime")
ev.to_csv(OUT / "events_clean.csv", index=False)
ev[ev.split == "train"].to_csv(OUT / "train.csv", index=False)
ev[ev.split == "holdout"].to_csv(OUT / "holdout.csv", index=False)
q = pd.concat(quarantine).drop(columns=["event_date"], errors="ignore").sort_values("event_id")
q["watch_seconds"] = q["watch_seconds"].astype("Int64")
q.to_csv(OUT / "quarantine.csv", index=False)
(OUT / "cleaning_log.json").write_text(json.dumps(log, indent=2, default=str))
print(json.dumps(log, indent=2, default=str))
