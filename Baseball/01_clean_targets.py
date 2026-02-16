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
# =============================================
# CONFIGURATION - Modify these for different date ranges
# =============================================

# Date range to pull from Statcast
START_DT = "2023-04-01"
END_DT = "2024-11-05"

# Split boundaries (must be in chronological order)
# - Train: START_DT to TRAIN_END (exclusive)
# - Val: TRAIN_END to VAL_END (exclusive)
# - Test: VAL_END to END_DT
TRAIN_END = "2023-09-01"  # First day of validation
VAL_END = "2024-01-01"    # First day of test

# =============================================


# =============================================
# CONFIGURATION VALIDATION - Prevents accidental leakage
# =============================================

def validate_split_config():
    """Validate split configuration to prevent data leakage."""
    errors = []

    # Parse dates
    start = pd.to_datetime(START_DT)
    end = pd.to_datetime(END_DT)
    train_end = pd.to_datetime(TRAIN_END)
    val_end = pd.to_datetime(VAL_END)

    # Check chronological order
    if not (start < train_end < val_end <= end):
        errors.append(
            f"Dates must be chronological: START_DT ({START_DT}) < TRAIN_END ({TRAIN_END}) "
            f"< VAL_END ({VAL_END}) <= END_DT ({END_DT})"
        )

    # Check reasonable split sizes (warn if any period < 30 days)
    train_days = (train_end - start).days
    val_days = (val_end - train_end).days
    test_days = (end - val_end).days

    if train_days < 30:
        errors.append(f"Train period too short: {train_days} days (< 30)")
    if val_days < 14:
        errors.append(f"Validation period too short: {val_days} days (< 14)")
    if test_days < 30:
        errors.append(f"Test period too short: {test_days} days (< 30)")

    if errors:
        print("=" * 60)
        print("CONFIGURATION ERROR - Fix before proceeding:")
        print("=" * 60)
        for e in errors:
            print(f"  - {e}")
        raise ValueError("Invalid split configuration. See errors above.")

    # Print config summary
    print("=" * 60)
    print("SPLIT CONFIGURATION")
    print("=" * 60)
    print(f"  Data range: {START_DT} to {END_DT}")
    print(f"  Train: {START_DT} to {TRAIN_END} ({train_days} days)")
    print(f"  Val:   {TRAIN_END} to {VAL_END} ({val_days} days)")
    print(f"  Test:  {VAL_END} to {END_DT} ({test_days} days)")
    print("=" * 60)


# Run validation immediately
validate_split_config()

# =============================================

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
    # Check if an older raw file exists that we can extend instead of re-fetching everything
    OLD_RAW_PATH = RAW_DIR / "statcast_2023-04-01_2024-10-01.parquet"
    if OLD_RAW_PATH.exists():
        print(f"Found older raw file: {OLD_RAW_PATH}")
        print(f"Loading existing data and fetching additional range 2024-10-01 to {END_DT}...")
        df_old = pd.read_parquet(OLD_RAW_PATH)
        print(f"  Old raw: {df_old.shape}")

        # Fetch the additional date range (all game types to capture playoffs)
        df_new = fetch_statcast("2024-10-01", END_DT, cache_dir=CACHE_DIR, verbose=True)
        print(f"  New fetch: {df_new.shape}")

        # Align dtypes: old parquet may have typed columns (e.g. date) while
        # freshly fetched CSV has strings.  Convert shared columns in df_new to
        # match df_old dtypes where possible.
        for col in df_new.columns:
            if col in df_old.columns and df_old[col].dtype != df_new[col].dtype:
                try:
                    df_new[col] = df_new[col].astype(df_old[col].dtype)
                except (ValueError, TypeError):
                    # If conversion fails, cast both to object so concat works
                    df_old[col] = df_old[col].astype(object)
                    df_new[col] = df_new[col].astype(object)

        # Combine and deduplicate
        df_raw = pd.concat([df_old, df_new], ignore_index=True)
        if "game_pk" in df_raw.columns and "at_bat_number" in df_raw.columns and "pitch_number" in df_raw.columns:
            before = len(df_raw)
            df_raw = df_raw.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
            after = len(df_raw)
            if before != after:
                print(f"  Removed {before - after:,} duplicates")
        df_raw.to_parquet(RAW_PATH, index=False)
        print(f"Saved combined raw: {RAW_PATH}, {df_raw.shape}")
        del df_old, df_new
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
# Filter to regular season only (exclude playoffs)
# game_type: 'R' = Regular, 'F' = Wild Card, 'D' = Division, 'L' = LCS, 'W' = World Series
if "game_type" in df.columns:
    playoff_mask = df["game_type"] != "R"
    playoff_count = playoff_mask.sum()
    if playoff_count > 0:
        print(f"Removing {playoff_count} playoff rows ({playoff_count/len(df)*100:.2f}%)")
        print(f"  Breakdown: {df[playoff_mask]['game_type'].value_counts().to_dict()}")
        df = df[~playoff_mask].copy()
    else:
        print("No playoff games found in data")
else:
    print("WARNING: game_type column not found, cannot filter playoffs")

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
# Uses config: TRAIN_END and VAL_END boundaries
print("\n=== TEMPORAL SPLIT ASSIGNMENT ===")

df["split"] = "test"  # default
df.loc[df["game_date"] < TRAIN_END, "split"] = "train"
df.loc[
    (df["game_date"] >= TRAIN_END) & (df["game_date"] < VAL_END),
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

# Verify no date leakage (post-assignment check)
train_dates = df[df["split"] == "train"]["game_date"]
val_dates = df[df["split"] == "val"]["game_date"]
test_dates = df[df["split"] == "test"]["game_date"]

if len(train_dates) > 0 and len(val_dates) > 0:
    assert train_dates.max() < val_dates.min(), \
        f"LEAKAGE: Train max ({train_dates.max()}) >= Val min ({val_dates.min()})"

if len(val_dates) > 0 and len(test_dates) > 0:
    assert val_dates.max() < test_dates.min(), \
        f"LEAKAGE: Val max ({val_dates.max()}) >= Test min ({test_dates.min()})"

print("\nLEAKAGE CHECK: PASS - No temporal overlap between splits")

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
