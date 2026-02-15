# %%
"""
01_clean_targets.py

Pulls Statcast data, creates target columns (y_swing, y_whiff), assigns temporal splits,
and saves labeled data for downstream pipeline steps.

Output: data/processed/statcast_{START_DT}_{END_DT}_labeled.parquet
"""

from pathlib import Path
import pandas as pd
from statcast_fetcher import fetch_statcast

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", None)

# %%
# Configuration: date range for data pull
# For cross-year validation: pull 2023-04-01 to 2024-10-01
START_DT = "2023-04-01"
END_DT = "2024-10-01"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = RAW_DIR / f"statcast_{START_DT}_{END_DT}.parquet"
OUT_PATH = PROCESSED_DIR / f"statcast_{START_DT}_{END_DT}_labeled.parquet"

# %%
# Pull or load raw Statcast data
if RAW_PATH.exists():
    df_raw = pd.read_parquet(RAW_PATH)
    print("Loaded cached raw:", RAW_PATH, df_raw.shape)
else:
    print(f"Fetching Statcast data from {START_DT} to {END_DT}...")
    df_raw = fetch_statcast(START_DT, END_DT, cache_dir=CACHE_DIR, verbose=True)
    df_raw.to_parquet(RAW_PATH, index=False)
    print("Pulled + saved raw:", RAW_PATH, df_raw.shape)
#%%
unque_descriptions = df_raw['description'].unique()
print("Unique description values in raw data:", len(unque_descriptions))
print(unque_descriptions)

# %%
# Target outcome definitions from Statcast 'description' column
SWING_OUTCOMES = [
    "hit_into_play",
    "swinging_strike",
    "foul",
    "foul_tip",
    "swinging_strike_blocked",
    "missed_bunt",
    "bunt_foul_tip",
    "foul_bunt",
]

TAKE_OUTCOMES = [
    "ball",
    "called_strike",
    "blocked_ball",
    "pitchout",
]

WHIFF_OUTCOMES = [
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
]

CONTACT_OUTCOMES = [
    "hit_into_play",
    "foul",
    "foul_tip",
    "bunt_foul_tip",
    "foul_bunt",
]

# Dropped outcomes - not actual batter decisions, not useful for modeling
# HBP: batter hit by pitch (neither swing nor take)
# automatic_ball/strike: pitch clock violations, umpire-enforced (not batter action)
DROP_OUTCOMES = [
    "hit_by_pitch",
    "automatic_ball",
    "automatic_strike",
]

# %%
# Create working copy
df = df_raw.copy()
input_rows = len(df)
print(f"Input rows: {input_rows}")

# %%
# Ensure game_date is datetime
df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
date_nulls = df["game_date"].isna().sum()
if date_nulls > 0:
    print(f"WARNING: {date_nulls} rows with null game_date will be dropped")
    df = df.dropna(subset=["game_date"]).copy()

# %%
# Drop non-decision outcomes (HBP, automatic ball/strike)
# These are not actual batter decisions and excluded from modeling
drop_mask = df["description"].isin(DROP_OUTCOMES)
drop_count = drop_mask.sum()
print(f"Dropping {drop_count} non-decision outcomes ({drop_count/len(df)*100:.2f}%)")
print(f"  Breakdown: {df[drop_mask]['description'].value_counts().to_dict()}")
df = df[~drop_mask].copy()

# %%
# Create target columns
df["target_raw"] = df["description"].astype(str)

# y_swing: 1 for swings, 0 for takes
df["y_swing"] = pd.NA
df.loc[df["description"].isin(SWING_OUTCOMES), "y_swing"] = 1
df.loc[df["description"].isin(TAKE_OUTCOMES), "y_swing"] = 0
df["y_swing"] = df["y_swing"].astype("Int8")

# y_whiff: 1 for whiffs, 0 for contact, null for takes/HBP
# Only defined for swings
df["y_whiff"] = pd.NA
swing_mask = df["y_swing"] == 1
df.loc[swing_mask & df["description"].isin(WHIFF_OUTCOMES), "y_whiff"] = 1
df.loc[swing_mask & df["description"].isin(CONTACT_OUTCOMES), "y_whiff"] = 0
df["y_whiff"] = df["y_whiff"].astype("Int8")

# %%
# Target validation
print("\n=== TARGET VALIDATION ===")

# Row counts
rows_with_swing_target = df["y_swing"].notna().sum()
rows_without_swing_target = df["y_swing"].isna().sum()
print(f"Rows with y_swing defined: {rows_with_swing_target} ({rows_with_swing_target/len(df)*100:.1f}%)")
print(f"Rows without y_swing (HBP/other): {rows_without_swing_target} ({rows_without_swing_target/len(df)*100:.1f}%)")

# y_swing distribution
swing_counts = df["y_swing"].value_counts(dropna=False)
print(f"\ny_swing distribution:")
print(swing_counts)
swing_rate = df["y_swing"].mean()
print(f"Swing rate (among defined): {swing_rate:.3f}")

# y_whiff distribution (among swings only)
swings_only = df[df["y_swing"] == 1]
whiff_counts = swings_only["y_whiff"].value_counts(dropna=False)
print(f"\ny_whiff distribution (swings only):")
print(whiff_counts)
whiff_rate = swings_only["y_whiff"].mean()
print(f"Whiff rate (among swings): {whiff_rate:.3f}")

# Critical check: y_whiff should be null IFF y_swing != 1
takes_count = (df["y_swing"] == 0).sum()
non_swing_count = (df["y_swing"] != 1).sum()
whiff_nulls = df["y_whiff"].isna().sum()
print(f"\nNULL ALIGNMENT CHECK:")
print(f"  Takes (y_swing=0): {takes_count}")
print(f"  Non-swings (y_swing!=1): {non_swing_count}")
print(f"  y_whiff nulls: {whiff_nulls}")
assert whiff_nulls == non_swing_count, f"y_whiff null mismatch: {whiff_nulls} nulls vs {non_swing_count} non-swings"
print("  PASS: y_whiff is null for all non-swings")

# %%
# Temporal split assignment
# 2023 Apr-Aug = train, 2023 Sep-Oct = val, 2024+ = test
print("\n=== TEMPORAL SPLIT ASSIGNMENT ===")

df["split"] = "test"  # default for 2024+
df.loc[df["game_date"] < "2023-09-01", "split"] = "train"
df.loc[
    (df["game_date"] >= "2023-09-01") & (df["game_date"] < "2024-01-01"),
    "split"
] = "val"

# Split validation
split_counts = df["split"].value_counts()
print(f"\nSplit distribution:")
for split_name in ["train", "val", "test"]:
    count = split_counts.get(split_name, 0)
    pct = count / len(df) * 100
    print(f"  {split_name}: {count:,} rows ({pct:.1f}%)")

# Date ranges per split
print("\nDate ranges per split:")
for split_name in ["train", "val", "test"]:
    split_df = df[df["split"] == split_name]
    if len(split_df) > 0:
        min_date = split_df["game_date"].min().strftime("%Y-%m-%d")
        max_date = split_df["game_date"].max().strftime("%Y-%m-%d")
        print(f"  {split_name}: {min_date} to {max_date}")
    else:
        print(f"  {split_name}: NO DATA")

# Verify no date leakage
train_max = df[df["split"] == "train"]["game_date"].max()
val_min = df[df["split"] == "val"]["game_date"].min()
val_max = df[df["split"] == "val"]["game_date"].max()
test_min = df[df["split"] == "test"]["game_date"].min()

if pd.notna(train_max) and pd.notna(val_min):
    assert train_max < val_min, f"Date leakage: train max {train_max} >= val min {val_min}"
if pd.notna(val_max) and pd.notna(test_min):
    assert val_max < test_min, f"Date leakage: val max {val_max} >= test min {test_min}"
print("\nDATE LEAKAGE CHECK: PASS")

# Assert splits are non-empty (warn if any missing)
for split_name in ["train", "val", "test"]:
    if split_counts.get(split_name, 0) == 0:
        print(f"WARNING: {split_name} split is empty!")

# %%
# Final validation summary
print("\n=== FINAL VALIDATION SUMMARY ===")
print(f"Input rows: {input_rows}")
print(f"Output rows: {len(df)} ({len(df)/input_rows*100:.1f}% retained)")

# Check description values not in known outcomes (after dropping non-decisions)
known_outcomes = set(SWING_OUTCOMES + TAKE_OUTCOMES)
unknown_desc = df[~df["description"].isin(known_outcomes)]["description"].unique()
if len(unknown_desc) > 0:
    print(f"\nWARNING: {len(unknown_desc)} unknown description values:")
    for desc in unknown_desc[:10]:
        count = (df["description"] == desc).sum()
        print(f"  '{desc}': {count} rows")
    if len(unknown_desc) > 10:
        print(f"  ... and {len(unknown_desc) - 10} more")

# %%
# Save labeled data
df.to_parquet(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:20]}...")

# %%
