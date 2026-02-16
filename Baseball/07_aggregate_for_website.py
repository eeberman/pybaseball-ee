# %%
"""
07_aggregate_for_website.py

Generates JSON data files for the whiff prediction website.
Processes 2024 regular season data to create:

1. pitchers_overall.json    - Main leaderboard (250 pitch min)
2. pitchers_by_pitch_type.json - Pitch type breakdowns (20 pitch min per type)
3. pitchers_by_zone.json    - 3x3 zone grid (10 pitch min per zone)
4. metadata.json            - Website metadata

Data flow:
  step1 (all pitches)  -> pitch counts per pitcher, pitch type counts, zone counts
  step2 (swings only)  -> Model 2 predictions -> whiff skill metrics
  raw data             -> team extraction (home_team/away_team + inning_topbot)
  MLB API              -> pitcher names (cached to name_cache.json)

Note: The step1/step2 datasets were created by 01_clean_targets.py which already
filtered to game_type == 'R' (regular season only). No additional game_type
filtering is needed here.
"""

from __future__ import annotations

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import requests

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", None)

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

# Thresholds
MIN_PITCHES_RS = 250       # Minimum total pitches for RS qualification
MIN_PITCHES_PITCH_TYPE = 20  # Minimum pitches per pitch type
MIN_PITCHES_ZONE = 10      # Minimum pitches per zone

# Paths - inputs
STEP1_PATH = Path("data/processed") / "statcast_step1_swing_take.parquet"
STEP2_PATH = Path("data/processed") / "statcast_step2_whiff_contact.parquet"
RAW_PATH = Path("data/raw") / "statcast_2023-04-01_2024-11-05.parquet"

MODEL_DIR = Path("artifacts/models")
MODEL2_PATH = MODEL_DIR / "model2_whiff_contact.json"
CALIBRATOR_PATH = MODEL_DIR / "model2_calibrator.joblib"
CONFIG_PATH = MODEL_DIR / "model_config.json"

# Paths - outputs
WEBSITE_DATA_DIR = Path("website/data")
WEBSITE_DATA_DIR.mkdir(parents=True, exist_ok=True)

OVERALL_PATH = WEBSITE_DATA_DIR / "pitchers_overall.json"
BY_PITCH_TYPE_PATH = WEBSITE_DATA_DIR / "pitchers_by_pitch_type.json"
BY_ZONE_PATH = WEBSITE_DATA_DIR / "pitchers_by_zone.json"
METADATA_PATH = WEBSITE_DATA_DIR / "metadata.json"
NAME_CACHE_PATH = WEBSITE_DATA_DIR / "name_cache.json"

# %%
# =============================================================================
# LOAD DATA AND MODELS
# =============================================================================
print("=== LOADING DATA AND MODELS ===")

# Load step1 (all pitches) - for total pitch counts
step1 = pd.read_parquet(STEP1_PATH)
step1["game_date"] = pd.to_datetime(step1["game_date"], errors="coerce")
print(f"Step1 (all pitches): {len(step1):,} rows")

# Load step2 (swings only) - for whiff predictions
step2 = pd.read_parquet(STEP2_PATH)
step2["game_date"] = pd.to_datetime(step2["game_date"], errors="coerce")
print(f"Step2 (swings only): {len(step2):,} rows")

# Filter to 2024 test set
step1_2024 = step1[step1["split"] == "test"].copy()
step2_2024 = step2[step2["split"] == "test"].copy()
print(f"\n2024 test set:")
print(f"  Step1: {len(step1_2024):,} pitches")
print(f"  Step2: {len(step2_2024):,} swings")
print(f"  Date range: {step1_2024['game_date'].min()} to {step1_2024['game_date'].max()}")

# Load Model 2 and calibrator
model2 = xgb.Booster()
model2.load_model(MODEL2_PATH)
calibrator = joblib.load(CALIBRATOR_PATH)

with open(CONFIG_PATH) as f:
    config = json.load(f)

NUMERIC_FEATURE_COLS = config["numeric_feature_cols"]
CAT_FEATURE_COLS = config["categorical_cols"]
EXPECTED_FEATURES = config["feature_cols_model2"]

print(f"\nLoaded Model 2: {MODEL2_PATH}")
print(f"Loaded calibrator: {CALIBRATOR_PATH}")

# %%
# =============================================================================
# TEAM EXTRACTION FROM RAW DATA
# =============================================================================
print("\n=== EXTRACTING TEAM INFO ===")

# Load minimal columns from raw data for team lookup
team_cols = ["pitcher", "game_date", "game_type", "inning_topbot"]
optional_team_cols = ["home_team", "away_team", "player_name"]

# Read raw data - only load columns we need
import pyarrow.parquet as pq

raw_schema = pq.read_schema(RAW_PATH)
raw_cols_available = [f.name for f in raw_schema]
load_cols = [c for c in team_cols + optional_team_cols if c in raw_cols_available]
print(f"Loading raw data columns: {load_cols}")

raw_team = pd.read_parquet(RAW_PATH, columns=load_cols)
raw_team["game_date"] = pd.to_datetime(raw_team["game_date"], errors="coerce")

# Filter to 2024 regular season
if "game_type" in raw_team.columns:
    raw_team = raw_team[raw_team["game_type"] == "R"].copy()
    print(f"Filtered to regular season: {len(raw_team):,} rows")

raw_2024 = raw_team[raw_team["game_date"] >= "2024-01-01"].copy()
print(f"Filtered to 2024: {len(raw_2024):,} rows")

# Extract team for each pitcher from their most recent game
has_team_cols = "home_team" in raw_2024.columns and "away_team" in raw_2024.columns

if has_team_cols:
    # Determine pitcher's team from inning_topbot:
    # Top of inning = visiting team batting, home team pitching
    # Bot of inning = home team batting, visiting team pitching
    raw_2024 = raw_2024.copy()
    raw_2024["pitcher_team"] = np.where(
        raw_2024["inning_topbot"] == "Top",
        raw_2024["home_team"],
        raw_2024["away_team"]
    )

    # Get last team for each pitcher (handles mid-season trades)
    pitcher_teams = (
        raw_2024.sort_values("game_date")
        .groupby("pitcher")["pitcher_team"]
        .last()
        .to_dict()
    )
    print(f"Extracted teams for {len(pitcher_teams)} pitchers")
else:
    print("WARNING: home_team/away_team not in raw data, will use MLB API for teams")
    pitcher_teams = {}

# Also try to extract names from raw data (if player_name column exists)
pitcher_names_raw = {}
if "player_name" in raw_2024.columns:
    pitcher_names_raw = (
        raw_2024.sort_values("game_date")
        .groupby("pitcher")["player_name"]
        .last()
        .to_dict()
    )
    print(f"Extracted raw names for {len(pitcher_names_raw)} pitchers")

# %%
# =============================================================================
# RUN MODEL 2 INFERENCE
# =============================================================================
print("\n=== RUNNING MODEL 2 INFERENCE ===")


def sanitize_feature_name(name: str) -> str:
    """Sanitize feature names for XGBoost (no [, ], or <)."""
    return name.replace("[", "_").replace("]", "_").replace("<", "_lt_")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature matrix with one-hot encoded categoricals."""
    X = df[NUMERIC_FEATURE_COLS].copy()

    # Convert nullable int to float
    for c in X.columns:
        if X[c].dtype == "Int8":
            X[c] = X[c].astype("float32")

    # One-hot encode categoricals
    for cat_col in CAT_FEATURE_COLS:
        if cat_col in df.columns and df[cat_col].notna().any():
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, dummy_na=True)
            X = pd.concat([X, dummies], axis=1)

    # Sanitize feature names
    X.columns = [sanitize_feature_name(c) for c in X.columns]
    return X


def align_columns(X: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """Align columns to match expected, adding missing as 0."""
    X_aligned = X.copy()
    for col in expected_cols:
        if col not in X_aligned.columns:
            X_aligned[col] = 0
    return X_aligned[expected_cols]


# Prepare features and predict
X_test = prepare_features(step2_2024)
X_test = align_columns(X_test, EXPECTED_FEATURES)

dtest = xgb.DMatrix(X_test, feature_names=EXPECTED_FEATURES)
pred_raw = model2.predict(dtest)
pred_calibrated = np.clip(calibrator.predict(pred_raw), 0, 1)

step2_2024["pred_whiff_calibrated"] = pred_calibrated

print(f"Predictions generated: {len(pred_calibrated):,}")
print(f"  Mean predicted whiff rate: {pred_calibrated.mean():.4f}")
print(f"  Actual whiff rate: {step2_2024['y_whiff'].mean():.4f}")

# %%
# =============================================================================
# ZONE ASSIGNMENT
# =============================================================================
print("\n=== ASSIGNING ZONES ===")


def assign_zone(plate_x_batter: float, plate_z: float) -> dict:
    """
    Assign 3x3 zone ID based on plate location.

    Uses plate_x_batter (batter-relative) for horizontal:
      Inside: x < -0.27
      Middle: -0.27 <= x < 0.27
      Away:   x >= 0.27

    Uses plate_z for vertical:
      Low:    z < 2.0
      Middle: 2.0 <= z < 3.0
      High:   z >= 3.0
    """
    # Horizontal (batter-relative)
    if plate_x_batter < -0.27:
        x_zone = "inside"
        x_min, x_max = -0.83, -0.27
    elif plate_x_batter < 0.27:
        x_zone = "middle"
        x_min, x_max = -0.27, 0.27
    else:
        x_zone = "away"
        x_min, x_max = 0.27, 0.83

    # Vertical
    if plate_z < 2.0:
        z_zone = "low"
        z_min, z_max = 1.5, 2.0
    elif plate_z < 3.0:
        z_zone = "middle"
        z_min, z_max = 2.0, 3.0
    else:
        z_zone = "high"
        z_min, z_max = 3.0, 4.0

    return {
        "zone_id": f"{z_zone}_{x_zone}",
        "x_min": x_min,
        "x_max": x_max,
        "z_min": z_min,
        "z_max": z_max,
    }


# Assign zones to step1 (all pitches) and step2 (swings)
# Use vectorized approach for performance
def assign_zones_vectorized(df: pd.DataFrame) -> pd.Series:
    """Assign zone IDs vectorized."""
    x = df["plate_x_batter"]
    z = df["plate_z"]

    x_zone = pd.Series("middle", index=df.index)
    x_zone[x < -0.27] = "inside"
    x_zone[x >= 0.27] = "away"

    z_zone = pd.Series("middle", index=df.index)
    z_zone[z < 2.0] = "low"
    z_zone[z >= 3.0] = "high"

    return z_zone + "_" + x_zone


step1_2024["zone_id"] = assign_zones_vectorized(step1_2024)
step2_2024["zone_id"] = assign_zones_vectorized(step2_2024)

print(f"Zone distribution (step1):")
print(step1_2024["zone_id"].value_counts().to_string())

# %%
# =============================================================================
# AGGREGATE: OVERALL PITCHER STATS
# =============================================================================
print("\n=== AGGREGATING OVERALL PITCHER STATS ===")

# Total pitches per pitcher from step1
pitcher_pitch_counts = (
    step1_2024.groupby("pitcher")
    .size()
    .reset_index(name="n_pitches")
)

# Whiff stats per pitcher from step2
pitcher_swing_stats = (
    step2_2024.groupby("pitcher")
    .agg(
        n_swings=("y_whiff", "count"),
        actual_whiff_rate=("y_whiff", "mean"),
        expected_whiff_rate=("pred_whiff_calibrated", "mean"),
    )
    .reset_index()
)

# Merge
overall = pitcher_pitch_counts.merge(pitcher_swing_stats, on="pitcher", how="inner")
overall["whiff_skill"] = overall["actual_whiff_rate"] - overall["expected_whiff_rate"]

# Apply qualification filter
overall_qualified = overall[overall["n_pitches"] >= MIN_PITCHES_RS].copy()
overall_qualified = overall_qualified[overall_qualified["n_swings"] > 0].copy()

# Percentile ranking (100 = best whiff skill)
overall_qualified["percentile"] = (
    overall_qualified["whiff_skill"].rank(pct=True, method="average") * 100
)
overall_qualified["percentile"] = overall_qualified["percentile"].round().astype(int)

# Sort: primary by whiff_skill desc, secondary by n_pitches desc (for ties)
overall_qualified = overall_qualified.sort_values(
    ["whiff_skill", "n_pitches"], ascending=[False, False]
)

print(f"Total pitchers in 2024 RS: {len(overall)}")
print(f"Qualified (>= {MIN_PITCHES_RS} pitches): {len(overall_qualified)}")

# %%
# =============================================================================
# AGGREGATE: BY PITCH TYPE
# =============================================================================
print("\n=== AGGREGATING BY PITCH TYPE ===")

# Total pitches per pitcher per pitch type from step1
pitcher_type_pitches = (
    step1_2024.groupby(["pitcher", "pitch_type_mode"])
    .size()
    .reset_index(name="n_pitches")
)

# Whiff stats per pitcher per pitch type from step2
pitcher_type_swings = (
    step2_2024.groupby(["pitcher", "pitch_type_mode"])
    .agg(
        n_swings=("y_whiff", "count"),
        whiff_rate=("y_whiff", "mean"),
        expected=("pred_whiff_calibrated", "mean"),
    )
    .reset_index()
)

# Merge
pitch_type_stats = pitcher_type_pitches.merge(
    pitcher_type_swings, on=["pitcher", "pitch_type_mode"], how="left"
)
pitch_type_stats["skill"] = pitch_type_stats["whiff_rate"] - pitch_type_stats["expected"]

# Filter: minimum pitches per type AND only for qualified pitchers
qualified_pitcher_ids = set(overall_qualified["pitcher"].values)
pitch_type_stats = pitch_type_stats[
    (pitch_type_stats["n_pitches"] >= MIN_PITCHES_PITCH_TYPE)
    & (pitch_type_stats["pitcher"].isin(qualified_pitcher_ids))
    & (pitch_type_stats["n_swings"] > 0)
].copy()

print(f"Pitch type entries (after filtering): {len(pitch_type_stats)}")
print(f"Pitch types represented: {pitch_type_stats['pitch_type_mode'].nunique()}")

# %%
# =============================================================================
# AGGREGATE: BY ZONE
# =============================================================================
print("\n=== AGGREGATING BY ZONE ===")

# All 9 zone IDs
ZONE_IDS = [
    "high_inside", "high_middle", "high_away",
    "middle_inside", "middle_middle", "middle_away",
    "low_inside", "low_middle", "low_away",
]

# Zone boundary definitions
ZONE_BOUNDS = {
    "high_inside":   {"x_min": -0.83, "x_max": -0.27, "z_min": 3.0, "z_max": 4.0},
    "high_middle":   {"x_min": -0.27, "x_max": 0.27,  "z_min": 3.0, "z_max": 4.0},
    "high_away":     {"x_min": 0.27,  "x_max": 0.83,  "z_min": 3.0, "z_max": 4.0},
    "middle_inside": {"x_min": -0.83, "x_max": -0.27, "z_min": 2.0, "z_max": 3.0},
    "middle_middle": {"x_min": -0.27, "x_max": 0.27,  "z_min": 2.0, "z_max": 3.0},
    "middle_away":   {"x_min": 0.27,  "x_max": 0.83,  "z_min": 2.0, "z_max": 3.0},
    "low_inside":    {"x_min": -0.83, "x_max": -0.27, "z_min": 1.5, "z_max": 2.0},
    "low_middle":    {"x_min": -0.27, "x_max": 0.27,  "z_min": 1.5, "z_max": 2.0},
    "low_away":      {"x_min": 0.27,  "x_max": 0.83,  "z_min": 1.5, "z_max": 2.0},
}

# Total pitches per pitcher per zone from step1
pitcher_zone_pitches = (
    step1_2024.groupby(["pitcher", "zone_id"])
    .size()
    .reset_index(name="n_pitches")
)

# Whiff stats per pitcher per zone from step2
pitcher_zone_swings = (
    step2_2024.groupby(["pitcher", "zone_id"])
    .agg(
        n_swings=("y_whiff", "count"),
        whiff_rate=("y_whiff", "mean"),
        expected=("pred_whiff_calibrated", "mean"),
    )
    .reset_index()
)

# Merge
zone_stats = pitcher_zone_pitches.merge(
    pitcher_zone_swings, on=["pitcher", "zone_id"], how="left"
)
zone_stats["skill"] = zone_stats["whiff_rate"] - zone_stats["expected"]

# Only keep qualified pitchers
zone_stats = zone_stats[zone_stats["pitcher"].isin(qualified_pitcher_ids)].copy()

print(f"Zone entries (qualified pitchers): {len(zone_stats)}")

# %%
# =============================================================================
# MLB API NAME LOOKUPS
# =============================================================================
print("\n=== FETCHING PITCHER NAMES ===")

# Load existing cache
if NAME_CACHE_PATH.exists():
    with open(NAME_CACHE_PATH) as f:
        name_cache = json.load(f)
    print(f"Loaded name cache: {len(name_cache)} entries")
else:
    name_cache = {}
    print("No existing name cache found")

# Pitcher IDs that need lookup
pitcher_ids_needed = set(overall_qualified["pitcher"].values)
pitcher_ids_to_fetch = [
    pid for pid in pitcher_ids_needed
    if str(int(pid)) not in name_cache
]

print(f"Pitchers needing lookup: {len(pitcher_ids_to_fetch)}")


def fetch_pitcher_name(pitcher_id: int, cache: dict) -> str:
    """Fetch pitcher name from MLB API with caching."""
    cache_key = str(int(pitcher_id))

    if cache_key in cache:
        return cache[cache_key]["name"]

    # Check raw data names first
    if pitcher_id in pitcher_names_raw:
        raw_name = pitcher_names_raw[pitcher_id]
        if pd.notna(raw_name) and raw_name.strip():
            # Statcast names are "Last, First" - convert to "First Last"
            parts = raw_name.split(", ")
            if len(parts) == 2:
                name = f"{parts[1]} {parts[0]}"
            else:
                name = raw_name
            cache[cache_key] = {"name": name}
            return name

    # Fall back to MLB API
    try:
        response = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{int(pitcher_id)}",
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        name = data["people"][0]["fullName"]
        cache[cache_key] = {"name": name}
        return name
    except Exception as e:
        print(f"  API failed for {pitcher_id}: {e}")
        cache[cache_key] = {"name": f"Pitcher {int(pitcher_id)}"}
        return cache[cache_key]["name"]


# Fetch names (with rate limiting for API calls)
api_call_count = 0
for i, pid in enumerate(pitcher_ids_to_fetch):
    name = fetch_pitcher_name(pid, name_cache)

    # Only count actual API calls (not cache/raw hits)
    cache_key = str(int(pid))
    if cache_key in name_cache and pid not in pitcher_names_raw:
        api_call_count += 1
        # Rate limit: 1 request per 0.1s to be respectful
        if api_call_count % 50 == 0:
            print(f"  Fetched {api_call_count} names from API...")
            time.sleep(1)
        else:
            time.sleep(0.1)

    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(pitcher_ids_to_fetch)} lookups...")

# Save cache
with open(NAME_CACHE_PATH, "w") as f:
    json.dump(name_cache, f, indent=2)
print(f"Saved name cache: {len(name_cache)} entries")

# %%
# =============================================================================
# BUILD JSON: pitchers_overall.json
# =============================================================================
print("\n=== BUILDING pitchers_overall.json ===")

pitchers_list = []
for _, row in overall_qualified.iterrows():
    pid = int(row["pitcher"])
    cache_key = str(pid)
    name = name_cache.get(cache_key, {}).get("name", f"Pitcher {pid}")
    team = pitcher_teams.get(pid, "UNK")

    pitchers_list.append({
        "pitcher_id": pid,
        "name": name,
        "team": team,
        "n_pitches": int(row["n_pitches"]),
        "n_swings": int(row["n_swings"]),
        "actual_whiff_rate": round(float(row["actual_whiff_rate"]), 4),
        "expected_whiff_rate": round(float(row["expected_whiff_rate"]), 4),
        "whiff_skill": round(float(row["whiff_skill"]), 4),
        "percentile": int(row["percentile"]),
    })

overall_json = {
    "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "season": "2024",
    "min_pitches": MIN_PITCHES_RS,
    "pitchers": pitchers_list,
}

with open(OVERALL_PATH, "w") as f:
    json.dump(overall_json, f)

print(f"Saved: {OVERALL_PATH}")
print(f"  {len(pitchers_list)} qualified pitchers")

# %%
# =============================================================================
# BUILD JSON: pitchers_by_pitch_type.json
# =============================================================================
print("\n=== BUILDING pitchers_by_pitch_type.json ===")

pitch_type_json = {}
for pid in overall_qualified["pitcher"].values:
    pid_int = int(pid)
    pitcher_types = pitch_type_stats[pitch_type_stats["pitcher"] == pid]

    if len(pitcher_types) == 0:
        continue

    type_dict = {}
    for _, row in pitcher_types.iterrows():
        pt = row["pitch_type_mode"]
        if pd.isna(pt) or pt == "<NA>":
            continue

        type_dict[pt] = {
            "n_pitches": int(row["n_pitches"]),
            "n_swings": int(row["n_swings"]),
            "whiff_rate": round(float(row["whiff_rate"]), 4),
            "expected": round(float(row["expected"]), 4),
            "skill": round(float(row["skill"]), 4),
        }

    if type_dict:
        pitch_type_json[str(pid_int)] = type_dict

with open(BY_PITCH_TYPE_PATH, "w") as f:
    json.dump(pitch_type_json, f)

print(f"Saved: {BY_PITCH_TYPE_PATH}")
print(f"  {len(pitch_type_json)} pitchers with pitch type data")

# %%
# =============================================================================
# BUILD JSON: pitchers_by_zone.json
# =============================================================================
print("\n=== BUILDING pitchers_by_zone.json ===")

# Compact format: zone order is fixed (ZONE_IDS), so store as arrays
# Each pitcher gets 9 entries in fixed order: [n_pitches, n_swings, whiff_rate, expected, skill]
# null values for zones below threshold
zone_json = {
    "zone_order": ZONE_IDS,
    "zone_bounds": ZONE_BOUNDS,
    "pitchers": {},
}

for pid in overall_qualified["pitcher"].values:
    pid_int = int(pid)
    pitcher_zones = zone_stats[zone_stats["pitcher"] == pid]

    zones_list = []
    for zone_id in ZONE_IDS:
        zone_row = pitcher_zones[pitcher_zones["zone_id"] == zone_id]

        if len(zone_row) == 0:
            zones_list.append([0, 0, None, None, None])
        else:
            zr = zone_row.iloc[0]
            n_pitches = int(zr["n_pitches"])
            n_swings = int(zr["n_swings"]) if pd.notna(zr["n_swings"]) else 0

            if n_pitches < MIN_PITCHES_ZONE or n_swings == 0:
                zones_list.append([n_pitches, n_swings, None, None, None])
            else:
                zones_list.append([
                    n_pitches,
                    n_swings,
                    round(float(zr["whiff_rate"]), 4),
                    round(float(zr["expected"]), 4),
                    round(float(zr["skill"]), 4),
                ])

    zone_json["pitchers"][str(pid_int)] = zones_list

with open(BY_ZONE_PATH, "w") as f:
    json.dump(zone_json, f)

print(f"Saved: {BY_ZONE_PATH}")
print(f"  {len(zone_json['pitchers'])} pitchers with zone data")

# %%
# =============================================================================
# BUILD JSON: metadata.json
# =============================================================================
print("\n=== BUILDING metadata.json ===")

metadata = {
    "last_updated": pd.Timestamp.now().isoformat(),
    "seasons_included": ["2024"],
    "total_pitchers_rs": len(overall_qualified),
    "total_pitches_rs": int(step1_2024[step1_2024["pitcher"].isin(qualified_pitcher_ids)]["pitcher"].count()),
    "total_swings_rs": int(step2_2024[step2_2024["pitcher"].isin(qualified_pitcher_ids)]["pitcher"].count()),
    "model_version": "v1.0",
    "model_trained_on": "regular_season_only",
    "thresholds": {
        "rs_min_pitches": MIN_PITCHES_RS,
        "pitch_type_min": MIN_PITCHES_PITCH_TYPE,
        "zone_min": MIN_PITCHES_ZONE,
    },
    "data_source": "Baseball Savant (Statcast)",
    "methodology": "Whiff skill = actual_whiff_rate - expected_whiff_rate. "
                    "Expected rate from XGBoost Model 2 with isotonic calibration. "
                    "Positive skill = pitcher generates more whiffs than expected.",
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved: {METADATA_PATH}")

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("WEBSITE DATA GENERATION COMPLETE")
print("=" * 60)
print(f"\nOutputs:")
print(f"  {OVERALL_PATH} ({len(pitchers_list)} pitchers)")
print(f"  {BY_PITCH_TYPE_PATH} ({len(pitch_type_json)} pitchers)")
print(f"  {BY_ZONE_PATH} ({len(zone_json['pitchers'])} pitchers)")
print(f"  {METADATA_PATH}")
print(f"  {NAME_CACHE_PATH} ({len(name_cache)} cached names)")

print(f"\nTop 10 pitchers by whiff skill:")
for i, p in enumerate(pitchers_list[:10], 1):
    print(f"  {i:2d}. {p['name']:25s} ({p['team']}) skill={p['whiff_skill']:+.4f}  "
          f"({p['n_pitches']} pitches, {p['n_swings']} swings)")

print(f"\nBottom 5 pitchers by whiff skill:")
for i, p in enumerate(pitchers_list[-5:], len(pitchers_list) - 4):
    print(f"  {i:2d}. {p['name']:25s} ({p['team']}) skill={p['whiff_skill']:+.4f}")
# %%
