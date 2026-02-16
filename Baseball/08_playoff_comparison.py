# %%
"""
08_playoff_comparison.py

Generates playoff comparison JSON files for the whiff prediction website.
Compares regular season skill to playoff performance.

Methodology:
  - Uses RS-trained model for both RS and PO predictions
  - skill_delta = playoff_skill - regular_season_skill
  - Positive delta = pitcher improved in playoffs ("riser")
  - Negative delta = pitcher regressed in playoffs ("faller")

Pipeline for playoff data:
  1. Fetch raw playoff data from Baseball Savant (Oct-Nov 2024)
  2. Apply feature engineering (same as 03_feature_engineering.py)
  3. Impute missing features with medians from RS training data
  4. Run Model 2 inference with calibration
  5. Aggregate skill metrics per pitcher
  6. Compare to RS metrics from 07_aggregate_for_website.py

Outputs:
  - website/data/pitchers_playoffs.json     - PO leaderboard (50 pitch min)
  - website/data/playoff_comparison.json    - Risers/fallers (250 RS + 50 PO min)
  - website/data/metadata.json              - Updated with PO stats

Note: game_type values used:
  F = Wild Card, D = Division Series, L = League Championship, W = World Series
"""

from __future__ import annotations

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import requests as req_lib

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", None)

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

# Thresholds
MIN_PITCHES_RS = 250    # RS qualification for comparison chart
MIN_PITCHES_PO = 50     # PO qualification for PO leaderboard

PLAYOFF_GAME_TYPES = ["F", "D", "L", "W"]

# Paths - inputs
RAW_PATH = Path("data/raw") / "statcast_2023-04-01_2024-11-05.parquet"
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("artifacts/models")
MODEL2_PATH = MODEL_DIR / "model2_whiff_contact.json"
CALIBRATOR_PATH = MODEL_DIR / "model2_calibrator.joblib"
CONFIG_PATH = MODEL_DIR / "model_config.json"

# RS skill data (from 07_aggregate_for_website.py)
RS_OVERALL_PATH = Path("website/data") / "pitchers_overall.json"
NAME_CACHE_PATH = Path("website/data") / "name_cache.json"

# Paths - outputs
WEBSITE_DATA_DIR = Path("website/data")
WEBSITE_DATA_DIR.mkdir(parents=True, exist_ok=True)

PO_LEADERBOARD_PATH = WEBSITE_DATA_DIR / "pitchers_playoffs.json"
COMPARISON_PATH = WEBSITE_DATA_DIR / "playoff_comparison.json"
METADATA_PATH = WEBSITE_DATA_DIR / "metadata.json"

# %%
# =============================================================================
# LOAD MODELS AND CONFIG
# =============================================================================
print("=== LOADING MODELS AND CONFIG ===")

model2 = xgb.Booster()
model2.load_model(MODEL2_PATH)
calibrator = joblib.load(CALIBRATOR_PATH)

with open(CONFIG_PATH) as f:
    config = json.load(f)

NUMERIC_FEATURE_COLS = config["numeric_feature_cols"]
CAT_FEATURE_COLS = config["categorical_cols"]
EXPECTED_FEATURES = config["feature_cols_model2"]

print(f"Loaded Model 2, calibrator, and config")

# Load RS data for comparison
with open(RS_OVERALL_PATH) as f:
    rs_data = json.load(f)
rs_pitchers = {p["pitcher_id"]: p for p in rs_data["pitchers"]}
print(f"Loaded RS leaderboard: {len(rs_pitchers)} pitchers")

# Load name cache
if NAME_CACHE_PATH.exists():
    with open(NAME_CACHE_PATH) as f:
        name_cache = json.load(f)
else:
    name_cache = {}

# %%
# =============================================================================
# FETCH AND PREPARE PLAYOFF DATA
# =============================================================================
print("\n=== LOADING PLAYOFF DATA ===")

# First check what playoff data exists in the raw parquet
import pyarrow.parquet as pq

raw_schema = pq.read_schema(RAW_PATH)
raw_cols = [f.name for f in raw_schema]

# Load playoff data from existing raw file
playoff_cols_needed = [
    "game_pk", "game_date", "game_type", "batter", "pitcher",
    "at_bat_number", "pitch_number", "inning", "inning_topbot",
    "description", "pitch_type",
    "effective_speed", "release_speed", "release_spin_rate",
    "pfx_x", "pfx_z", "plate_x", "plate_z",
    "p_throws", "stand", "sz_bot", "sz_top",
    "balls", "strikes", "outs_when_up",
    "on_1b", "on_2b", "on_3b",
    "home_score", "away_score",
    "release_pos_x", "release_pos_z", "release_pos_y",
    "release_extension", "spin_axis",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    "home_team", "away_team", "player_name",
]
playoff_cols_needed = [c for c in playoff_cols_needed if c in raw_cols]

raw = pd.read_parquet(RAW_PATH, columns=playoff_cols_needed)
raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce")

# Extract 2024 playoff games from raw data
po_existing = raw[
    raw["game_type"].isin(PLAYOFF_GAME_TYPES)
    & (raw["game_date"] >= "2024-01-01")
].copy()
print(f"2024 playoff data in raw file: {len(po_existing):,} rows")
if len(po_existing) > 0:
    print(f"  Game types: {po_existing['game_type'].value_counts().to_dict()}")
    print(f"  Date range: {po_existing['game_date'].min()} to {po_existing['game_date'].max()}")

del raw  # Free memory

# Check if we need to fetch additional playoff data
# 2024 playoffs: Wild Card Oct 1-3, DS Oct 5-14, LCS Oct 13-25, WS Oct 25-Nov 2
needs_fetch = len(po_existing) == 0 or po_existing["game_date"].max() < pd.Timestamp("2024-10-15")

if needs_fetch:
    print("\nAttempting to fetch additional 2024 playoff data from Baseball Savant...")
    from statcast_fetcher import fetch_statcast

    try:
        # Fetch Oct-Nov 2024 to cover all playoff rounds
        po_fetched = fetch_statcast(
            start_date="2024-10-01",
            end_date="2024-11-05",
            cache_dir=CACHE_DIR,
            verbose=True,
        )

        if isinstance(po_fetched, pd.DataFrame) and len(po_fetched) > 0 and "game_date" in po_fetched.columns:
            po_fetched["game_date"] = pd.to_datetime(po_fetched["game_date"], errors="coerce")

            # Filter to playoff games only
            if "game_type" in po_fetched.columns:
                po_fetched = po_fetched[po_fetched["game_type"].isin(PLAYOFF_GAME_TYPES)].copy()
            print(f"Fetched playoff data: {len(po_fetched):,} rows")
            if len(po_fetched) > 0:
                print(f"  Game types: {po_fetched['game_type'].value_counts().to_dict()}")
                print(f"  Date range: {po_fetched['game_date'].min()} to {po_fetched['game_date'].max()}")

            # Combine with existing (dedup by game_pk + at_bat_number + pitch_number)
            if len(po_existing) > 0 and len(po_fetched) > 0:
                combined = pd.concat([po_existing, po_fetched], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["game_pk", "at_bat_number", "pitch_number"], keep="last"
                )
                po_raw = combined
            elif len(po_fetched) > 0:
                po_raw = po_fetched
            else:
                po_raw = po_existing
        else:
            print("  No additional data returned from API. Using existing data only.")
            po_raw = po_existing
    except Exception as e:
        print(f"  Fetch failed: {e}. Using existing data only.")
        po_raw = po_existing
else:
    po_raw = po_existing

print(f"\nTotal 2024 playoff pitches: {len(po_raw):,}")

if len(po_raw) == 0:
    print("ERROR: No 2024 playoff data available. Cannot generate playoff comparison.")
    print("The 2024 playoff data may need to be fetched from Baseball Savant.")
    import sys
    sys.exit(1)

# %%
# =============================================================================
# FEATURE ENGINEERING (inline, replicates 03_feature_engineering.py)
# =============================================================================
print("\n=== FEATURE ENGINEERING ===")

df = po_raw.copy()

# --- Target columns ---
SWING_OUTCOMES = [
    "hit_into_play", "swinging_strike", "foul", "foul_tip",
    "swinging_strike_blocked", "missed_bunt", "bunt_foul_tip", "foul_bunt",
]
TAKE_OUTCOMES = ["ball", "called_strike", "blocked_ball", "pitchout"]
WHIFF_OUTCOMES = ["swinging_strike", "swinging_strike_blocked", "missed_bunt"]
DROP_OUTCOMES = ["hit_by_pitch", "automatic_ball", "automatic_strike"]

# Drop non-decisions
df = df[~df["description"].isin(DROP_OUTCOMES)].copy()
print(f"After dropping non-decisions: {len(df):,} rows")

# Create targets
df["y_swing"] = pd.NA
df.loc[df["description"].isin(SWING_OUTCOMES), "y_swing"] = 1
df.loc[df["description"].isin(TAKE_OUTCOMES), "y_swing"] = 0
df["y_swing"] = df["y_swing"].astype("Int8")

df["y_whiff"] = pd.NA
swing_mask = df["y_swing"] == 1
df.loc[swing_mask & df["description"].isin(WHIFF_OUTCOMES), "y_whiff"] = 1
df.loc[swing_mask & ~df["description"].isin(WHIFF_OUTCOMES), "y_whiff"] = 0
df["y_whiff"] = df["y_whiff"].astype("Int8")

print(f"Swings: {(df['y_swing'] == 1).sum():,}, Takes: {(df['y_swing'] == 0).sum():,}")
print(f"Whiffs: {(df['y_whiff'] == 1).sum():,}, Contact: {(df['y_whiff'] == 0).sum():,}")

# --- Drop core nulls ---
core_null_cols = [
    "plate_x", "plate_z", "p_throws", "stand", "balls", "strikes",
    "release_pos_x", "release_pos_z", "pfx_x", "pfx_z", "sz_bot", "sz_top",
]
core_null_cols = [c for c in core_null_cols if c in df.columns]
df = df.dropna(subset=core_null_cols).copy()
print(f"After core null drop: {len(df):,} rows")

# --- Batter-relative features ---
stand_sign = df["stand"].map({"R": -1, "L": 1})
throw_sign = df["p_throws"].map({"R": -1, "L": 1})

df["plate_x_batter"] = df["plate_x"] * stand_sign
df["pfx_x_norm"] = df["pfx_x"] * throw_sign
df["release_pos_x_batter"] = df["release_pos_x"] * stand_sign
df["release_pos_z_batter"] = df["release_pos_z"]
df["same_side"] = (df["stand"] == df["p_throws"]).astype("int8")

# --- Count features ---
df["count_state"] = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
df["two_strikes"] = (df["strikes"] == 2).astype("int8")

# --- Zone features ---
ZONE_HALF_WIDTH_FT = 0.83
df["in_zone"] = (
    df["plate_x"].abs().le(ZONE_HALF_WIDTH_FT)
    & df["plate_z"].ge(df["sz_bot"])
    & df["plate_z"].le(df["sz_top"])
).astype("int8")

df["x_out_mag"] = (df["plate_x"].abs() - ZONE_HALF_WIDTH_FT).clip(lower=0)
df["x_out_signed_batter"] = np.sign(df["plate_x_batter"]) * df["x_out_mag"]
df["z_above_mag"] = (df["plate_z"] - df["sz_top"]).clip(lower=0)
df["z_below_mag"] = (df["sz_bot"] - df["plate_z"]).clip(lower=0)
df["z_out_signed"] = df["z_above_mag"] - df["z_below_mag"]
df["zone_out_dist"] = np.sqrt(df["x_out_mag"]**2 + np.maximum(df["z_above_mag"], df["z_below_mag"])**2)
df["zone_height"] = (df["sz_top"] - df["sz_bot"]).astype("float32")
df["plate_z_rel"] = ((df["plate_z"] - df["sz_bot"]) / df["zone_height"]).astype("float32")

# --- Runner features ---
df["runner_on_1b"] = df["on_1b"].notna().astype("int8")
df["runner_on_2b"] = df["on_2b"].notna().astype("int8")
df["runner_on_3b"] = df["on_3b"].notna().astype("int8")
df["any_runner_on"] = (df["runner_on_1b"] | df["runner_on_2b"] | df["runner_on_3b"]).astype("int8")
df["risp"] = (df["runner_on_2b"] | df["runner_on_3b"]).astype("int8")
df["bases_loaded"] = (df["runner_on_1b"] & df["runner_on_2b"] & df["runner_on_3b"]).astype("int8")

# --- Score context ---
df["batting_score_diff"] = np.where(
    df["inning_topbot"].eq("Top"),
    df["away_score"] - df["home_score"],
    df["home_score"] - df["away_score"],
)
bins = [-np.inf, -5, -3, -1, 0, 2, 4, np.inf]
labels = ["trail_5plus", "trail_4-3", "trail_2-1", "tied", "lead_1-2", "lead_3-4", "lead_5plus"]
df["score_bucket"] = pd.cut(df["batting_score_diff"], bins=bins, labels=labels, include_lowest=True)

# --- Use raw pitch_type as pitch_type_mode ---
# The RS model was trained on pitch_type_mode (from GMM clusters).
# For playoff data without clustering, raw pitch_type is a close substitute.
if "pitch_type" in df.columns:
    df["pitch_type_mode"] = df["pitch_type"]
    print(f"Using raw pitch_type as pitch_type_mode: {df['pitch_type_mode'].nunique()} types")

# --- Optional physics features ---
for c in ["release_extension", "release_pos_y", "spin_axis", "vx0", "vy0", "vz0", "ax", "ay", "az"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "spin_axis" in df.columns:
    axis = (df["spin_axis"] % 360.0).astype("float32")
    df["spin_axis_sin"] = np.sin(np.deg2rad(axis)).astype("float32")
    df["spin_axis_cos"] = np.cos(np.deg2rad(axis)).astype("float32")

# --- Trajectory / deception features ---
if "release_pos_y" in df.columns and "vy0" in df.columns:
    df["time_to_plate"] = (df["release_pos_y"] / df["vy0"].abs()).astype("float32")
    df["late_break_z"] = (df["pfx_z"] / df["time_to_plate"]).astype("float32")
    df["late_break_x"] = (df["pfx_x_norm"] / df["time_to_plate"]).astype("float32")

if "release_pos_y" in df.columns:
    df["approach_angle_z"] = (
        (df["plate_z"] - df["release_pos_z"]) / df["release_pos_y"]
    ).astype("float32")
    df["approach_angle_x"] = (
        (df["plate_x_batter"] - df["release_pos_x_batter"]) / df["release_pos_y"]
    ).astype("float32")

if all(c in df.columns for c in ["ax", "ay", "az"]):
    df["accel_magnitude"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2).astype("float32")

print(f"Feature engineering complete: {len(df):,} rows, {len(df.columns)} columns")

# %%
# =============================================================================
# IMPUTE OPTIONAL FEATURES
# =============================================================================
print("\n=== IMPUTING OPTIONAL FEATURES ===")

OPTIONAL_IMPUTE = [
    "release_extension", "release_pos_y", "spin_axis", "spin_axis_sin", "spin_axis_cos",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    "time_to_plate", "late_break_z", "late_break_x",
    "approach_angle_z", "approach_angle_x", "accel_magnitude",
]
OPTIONAL_IMPUTE = [c for c in OPTIONAL_IMPUTE if c in df.columns]

for c in OPTIONAL_IMPUTE:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    med = df[c].median()
    null_count = df[c].isna().sum()
    df[c] = df[c].fillna(med).astype("float32")
    if null_count > 0:
        print(f"  {c}: imputed {null_count} nulls with median={med:.3f}")

# %%
# =============================================================================
# RUN MODEL 2 INFERENCE ON SWINGS
# =============================================================================
print("\n=== RUNNING INFERENCE ON PLAYOFF SWINGS ===")

# Filter to swings only for whiff prediction
swings = df[df["y_swing"] == 1].copy()
print(f"Playoff swings: {len(swings):,}")

if len(swings) == 0:
    print("ERROR: No swings in playoff data")
    import sys
    sys.exit(1)


def sanitize_feature_name(name: str) -> str:
    return name.replace("[", "_").replace("]", "_").replace("<", "_lt_")


def prepare_features(df_in: pd.DataFrame) -> pd.DataFrame:
    X = df_in[NUMERIC_FEATURE_COLS].copy()
    for c in X.columns:
        if X[c].dtype == "Int8":
            X[c] = X[c].astype("float32")
    for cat_col in CAT_FEATURE_COLS:
        if cat_col in df_in.columns and df_in[cat_col].notna().any():
            dummies = pd.get_dummies(df_in[cat_col], prefix=cat_col, dummy_na=True)
            X = pd.concat([X, dummies], axis=1)
    X.columns = [sanitize_feature_name(c) for c in X.columns]
    return X


def align_columns(X: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    X_aligned = X.copy()
    for col in expected_cols:
        if col not in X_aligned.columns:
            X_aligned[col] = 0
    return X_aligned[expected_cols]


X_po = prepare_features(swings)
X_po = align_columns(X_po, EXPECTED_FEATURES)

dpo = xgb.DMatrix(X_po, feature_names=EXPECTED_FEATURES)
pred_raw = model2.predict(dpo)
pred_calibrated = np.clip(calibrator.predict(pred_raw), 0, 1)

swings["pred_whiff_calibrated"] = pred_calibrated

print(f"Mean predicted whiff rate: {pred_calibrated.mean():.4f}")
print(f"Actual whiff rate: {swings['y_whiff'].mean():.4f}")

# %%
# =============================================================================
# EXTRACT TEAM INFO
# =============================================================================
print("\n=== EXTRACTING TEAM INFO ===")

has_team_cols = "home_team" in df.columns and "away_team" in df.columns
if has_team_cols:
    df_for_teams = df.copy()
    df_for_teams["pitcher_team"] = np.where(
        df_for_teams["inning_topbot"] == "Top",
        df_for_teams["home_team"],
        df_for_teams["away_team"],
    )
    po_pitcher_teams = (
        df_for_teams.sort_values("game_date")
        .groupby("pitcher")["pitcher_team"]
        .last()
        .to_dict()
    )
    print(f"Extracted PO teams for {len(po_pitcher_teams)} pitchers")
    del df_for_teams
else:
    po_pitcher_teams = {}

# %%
# =============================================================================
# AGGREGATE PLAYOFF SKILL METRICS
# =============================================================================
print("\n=== AGGREGATING PLAYOFF SKILL METRICS ===")

# Total pitches per pitcher (all pitches, not just swings)
po_pitch_counts = df.groupby("pitcher").size().reset_index(name="n_pitches")

# Whiff stats per pitcher (swings only)
po_swing_stats = (
    swings.groupby("pitcher")
    .agg(
        n_swings=("y_whiff", "count"),
        actual_whiff_rate=("y_whiff", "mean"),
        expected_whiff_rate=("pred_whiff_calibrated", "mean"),
    )
    .reset_index()
)

po_stats = po_pitch_counts.merge(po_swing_stats, on="pitcher", how="inner")
po_stats["whiff_skill"] = po_stats["actual_whiff_rate"] - po_stats["expected_whiff_rate"]

# Filter to PO qualification threshold
po_qualified = po_stats[po_stats["n_pitches"] >= MIN_PITCHES_PO].copy()
po_qualified = po_qualified[po_qualified["n_swings"] > 0].copy()

# Percentile
po_qualified["percentile"] = (
    po_qualified["whiff_skill"].rank(pct=True, method="average") * 100
)
po_qualified["percentile"] = po_qualified["percentile"].round().astype(int)

# Sort
po_qualified = po_qualified.sort_values(["whiff_skill", "n_pitches"], ascending=[False, False])

print(f"Total PO pitchers: {len(po_stats)}")
print(f"PO qualified (>= {MIN_PITCHES_PO} pitches): {len(po_qualified)}")


# %%
# =============================================================================
# CONFIDENCE AND RS CONTEXT
# =============================================================================
def assign_confidence(n_pitches: int) -> str:
    if n_pitches < 100:
        return "low"
    elif n_pitches < 200:
        return "medium"
    else:
        return "high"


po_qualified["confidence"] = po_qualified["n_pitches"].apply(assign_confidence)

# Add RS context
po_qualified["rs_pitches"] = po_qualified["pitcher"].map(
    lambda pid: rs_pitchers.get(int(pid), {}).get("n_pitches", 0)
)
po_qualified["rs_qualified"] = po_qualified["rs_pitches"] >= MIN_PITCHES_RS

print(f"PO pitchers also RS qualified: {po_qualified['rs_qualified'].sum()}")

# %%
# =============================================================================
# NAME LOOKUPS FOR NEW PITCHERS
# =============================================================================
print("\n=== NAME LOOKUPS ===")

# Check for pitchers not in name cache
po_pitcher_ids = set(po_qualified["pitcher"].values)
pitcher_names_raw_po = {}
if "player_name" in po_raw.columns:
    pitcher_names_raw_po = (
        po_raw.sort_values("game_date")
        .groupby("pitcher")["player_name"]
        .last()
        .to_dict()
    )

new_lookups = 0
for pid in po_pitcher_ids:
    cache_key = str(int(pid))
    if cache_key not in name_cache:
        # Try raw data name first
        if pid in pitcher_names_raw_po:
            raw_name = pitcher_names_raw_po[pid]
            if pd.notna(raw_name) and raw_name.strip():
                parts = raw_name.split(", ")
                name = f"{parts[1]} {parts[0]}" if len(parts) == 2 else raw_name
                name_cache[cache_key] = {"name": name}
                continue

        # Fall back to MLB API
        try:
            resp = req_lib.get(
                f"https://statsapi.mlb.com/api/v1/people/{int(pid)}",
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            name = data["people"][0]["fullName"]
            name_cache[cache_key] = {"name": name}
            new_lookups += 1
            time.sleep(0.1)
        except Exception as e:
            name_cache[cache_key] = {"name": f"Pitcher {int(pid)}"}

print(f"New API lookups: {new_lookups}")

# Save updated cache
with open(NAME_CACHE_PATH, "w") as f:
    json.dump(name_cache, f, indent=2)

# %%
# =============================================================================
# BUILD JSON: pitchers_playoffs.json
# =============================================================================
print("\n=== BUILDING pitchers_playoffs.json ===")

po_pitchers_list = []
for _, row in po_qualified.iterrows():
    pid = int(row["pitcher"])
    cache_key = str(pid)
    name = name_cache.get(cache_key, {}).get("name", f"Pitcher {pid}")
    team = po_pitcher_teams.get(pid, rs_pitchers.get(pid, {}).get("team", "UNK"))

    po_pitchers_list.append({
        "pitcher_id": pid,
        "name": name,
        "team": team,
        "rs_pitches": int(row["rs_pitches"]),
        "rs_qualified": bool(row["rs_qualified"]),
        "n_pitches": int(row["n_pitches"]),
        "n_swings": int(row["n_swings"]),
        "actual_whiff_rate": round(float(row["actual_whiff_rate"]), 4),
        "expected_whiff_rate": round(float(row["expected_whiff_rate"]), 4),
        "whiff_skill": round(float(row["whiff_skill"]), 4),
        "percentile": int(row["percentile"]),
        "confidence": row["confidence"],
    })

po_json = {
    "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "season": "2024_playoffs",
    "min_pitches": MIN_PITCHES_PO,
    "pitchers": po_pitchers_list,
}

with open(PO_LEADERBOARD_PATH, "w") as f:
    json.dump(po_json, f, indent=2)

print(f"Saved: {PO_LEADERBOARD_PATH}")
print(f"  {len(po_pitchers_list)} PO-qualified pitchers")

# %%
# =============================================================================
# BUILD JSON: playoff_comparison.json
# =============================================================================
print("\n=== BUILDING playoff_comparison.json ===")

# Merge RS and PO skill metrics
# RS metrics from the RS leaderboard
comparison_rows = []
for _, row in po_qualified.iterrows():
    pid = int(row["pitcher"])
    rs_info = rs_pitchers.get(pid)

    if rs_info is None:
        continue  # No RS data
    if rs_info["n_pitches"] < MIN_PITCHES_RS:
        continue  # Not RS qualified

    rs_skill = rs_info["whiff_skill"]
    po_skill = float(row["whiff_skill"])
    skill_delta = po_skill - rs_skill

    # Categorize
    if skill_delta > 0.05:
        category = "big_riser"
    elif skill_delta > 0.02:
        category = "riser"
    elif skill_delta >= -0.02:
        category = "neutral"
    elif skill_delta >= -0.05:
        category = "faller"
    else:
        category = "big_faller"

    cache_key = str(pid)
    name = name_cache.get(cache_key, {}).get("name", f"Pitcher {pid}")
    team = po_pitcher_teams.get(pid, rs_info.get("team", "UNK"))

    comparison_rows.append({
        "pitcher_id": pid,
        "name": name,
        "team": team,
        "rs_pitches": rs_info["n_pitches"],
        "rs_skill": rs_skill,
        "po_pitches": int(row["n_pitches"]),
        "po_skill": round(po_skill, 4),
        "skill_delta": round(skill_delta, 4),
        "category": category,
        "confidence": row["confidence"],
    })

# Sort by absolute skill_delta (biggest movers first)
comparison_rows.sort(key=lambda x: abs(x["skill_delta"]), reverse=True)

# Add rank
for i, row in enumerate(comparison_rows, 1):
    row["rank"] = i

comparison_json = {
    "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "season": "2024",
    "methodology": "skill_delta = playoff_skill - regular_season_skill (using RS model for both)",
    "min_rs_pitches": MIN_PITCHES_RS,
    "min_po_pitches": MIN_PITCHES_PO,
    "pitchers": comparison_rows,
}

with open(COMPARISON_PATH, "w") as f:
    json.dump(comparison_json, f, indent=2)

print(f"Saved: {COMPARISON_PATH}")
print(f"  {len(comparison_rows)} pitchers in comparison")

# Category breakdown
from collections import Counter
cats = Counter(r["category"] for r in comparison_rows)
for cat in ["big_riser", "riser", "neutral", "faller", "big_faller"]:
    print(f"  {cat}: {cats.get(cat, 0)}")

# %%
# =============================================================================
# UPDATE METADATA
# =============================================================================
print("\n=== UPDATING metadata.json ===")

if METADATA_PATH.exists():
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
else:
    metadata = {}

metadata["total_pitchers_po"] = len(po_qualified)
metadata["total_pitchers_comparison"] = len(comparison_rows)
metadata["total_pitches_po"] = int(po_stats["n_pitches"].sum())
metadata["thresholds"]["po_min_pitches"] = MIN_PITCHES_PO
metadata["last_updated"] = pd.Timestamp.now().isoformat()

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Updated: {METADATA_PATH}")

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("PLAYOFF COMPARISON COMPLETE")
print("=" * 60)
print(f"\nOutputs:")
print(f"  {PO_LEADERBOARD_PATH} ({len(po_pitchers_list)} PO-qualified)")
print(f"  {COMPARISON_PATH} ({len(comparison_rows)} dual-qualified)")

if comparison_rows:
    print(f"\nTop 5 risers:")
    risers = [r for r in comparison_rows if r["skill_delta"] > 0]
    for r in risers[:5]:
        print(f"  {r['name']:25s} ({r['team']}) delta={r['skill_delta']:+.4f} "
              f"(RS={r['rs_skill']:+.4f} -> PO={r['po_skill']:+.4f}) [{r['confidence']}]")

    print(f"\nTop 5 fallers:")
    fallers = [r for r in comparison_rows if r["skill_delta"] < 0]
    for r in fallers[:5]:
        print(f"  {r['name']:25s} ({r['team']}) delta={r['skill_delta']:+.4f} "
              f"(RS={r['rs_skill']:+.4f} -> PO={r['po_skill']:+.4f}) [{r['confidence']}]")
# %%
