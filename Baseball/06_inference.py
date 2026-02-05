# %%
"""
06_inference.py

Load saved models and run cascade inference on any date range.
Outputs predictions with IDs for downstream sabermetric tools.

This script can either:
1. Run on pre-processed data (if pipeline has been run for the date range)
2. Pull and process raw data on-the-fly

Input:
  - artifacts/models/model1_swing_take.json
  - artifacts/models/model2_whiff_contact.json
  - artifacts/models/model_config.json

Output:
  - data/predictions/predictions_{START_DT}_{END_DT}.parquet
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", None)

# %%
# Configuration: date range for inference
# Change these to run inference on different date ranges
START_DT = "2024-04-01"
END_DT = "2024-10-01"

# Paths
MODEL_DIR = Path("artifacts/models")
OUTPUT_DIR = Path("data/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL1_PATH = MODEL_DIR / "model1_swing_take.json"
MODEL2_PATH = MODEL_DIR / "model2_whiff_contact.json"
CONFIG_PATH = MODEL_DIR / "model_config.json"

OUTPUT_PATH = OUTPUT_DIR / f"predictions_{START_DT}_{END_DT}.parquet"

# Check if we have pre-processed data for this date range
PROCESSED_PATH = Path("data/processed") / f"statcast_{START_DT}_{END_DT}_features.parquet"
STEP1_PATH = Path("data/processed") / "statcast_step1_swing_take.parquet"

# %%
# Load models and config
print("=== LOADING MODELS ===")

if not MODEL1_PATH.exists():
    raise FileNotFoundError(f"Model 1 not found: {MODEL1_PATH}. Run 05_train_and_validate.py first.")
if not MODEL2_PATH.exists():
    raise FileNotFoundError(f"Model 2 not found: {MODEL2_PATH}. Run 05_train_and_validate.py first.")
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config not found: {CONFIG_PATH}. Run 05_train_and_validate.py first.")

model1 = xgb.Booster()
model1.load_model(MODEL1_PATH)
print(f"Loaded: {MODEL1_PATH}")

model2 = xgb.Booster()
model2.load_model(MODEL2_PATH)
print(f"Loaded: {MODEL2_PATH}")

with open(CONFIG_PATH) as f:
    config = json.load(f)

FEATURE_COLS_M1 = config["feature_cols_model1"]
FEATURE_COLS_M2 = config["feature_cols_model2"]
NUMERIC_FEATURE_COLS = config["numeric_feature_cols"]
CATEGORICAL_COLS = config["categorical_cols"]
THRESHOLD_1 = config["threshold_swing"]
THRESHOLD_2 = config["threshold_whiff"]

print(f"Thresholds: swing={THRESHOLD_1}, whiff={THRESHOLD_2}")
print(f"Model 1 features: {len(FEATURE_COLS_M1)}")
print(f"Model 2 features: {len(FEATURE_COLS_M2)}")

# %%
# Load or prepare data
print("\n=== LOADING DATA ===")

# Option 1: Use pre-processed step1 data if available (recommended)
if STEP1_PATH.exists():
    print(f"Loading pre-processed data: {STEP1_PATH}")
    df = pd.read_parquet(STEP1_PATH)

    # Filter to requested date range
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df[(df["game_date"] >= START_DT) & (df["game_date"] <= END_DT)].copy()
    print(f"Filtered to {START_DT} - {END_DT}: {len(df)} rows")

# Option 2: Check for feature-engineered data
elif PROCESSED_PATH.exists():
    print(f"Loading feature data: {PROCESSED_PATH}")
    df = pd.read_parquet(PROCESSED_PATH)
    print(f"Loaded: {len(df)} rows")

# Option 3: Pull and process from scratch (requires full pipeline)
else:
    print(f"No pre-processed data found for {START_DT} - {END_DT}")
    print("Options:")
    print("  1. Run the full pipeline (01-04) for this date range")
    print("  2. Use existing step1 data and filter by date")

    # Try to use existing step1 data with different dates
    all_step1 = Path("data/processed")
    step1_files = list(all_step1.glob("statcast_step1*.parquet"))
    if step1_files:
        print(f"\nFound step1 files: {[f.name for f in step1_files]}")
        print("Loading first available and filtering...")
        df = pd.read_parquet(step1_files[0])
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df[(df["game_date"] >= START_DT) & (df["game_date"] <= END_DT)].copy()
        print(f"Filtered to {START_DT} - {END_DT}: {len(df)} rows")
    else:
        raise FileNotFoundError("No processed data available. Run pipeline steps 01-04 first.")

if len(df) == 0:
    raise ValueError(f"No data found for date range {START_DT} - {END_DT}")

# %%
# Prepare features
print("\n=== PREPARING FEATURES ===")


def prepare_features_for_inference(
    df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    cat_cols: list[str]
) -> pd.DataFrame:
    """Prepare feature matrix matching training format."""
    X = pd.DataFrame(index=df.index)

    # Add numeric features
    for c in numeric_cols:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.dtype == "Int8":
                vals = vals.astype("float32")
            X[c] = vals
        else:
            X[c] = 0.0

    # One-hot encode categoricals
    for cat_col in cat_cols:
        if cat_col in df.columns:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, dummy_na=True)
            X = pd.concat([X, dummies], axis=1)

    # Align to expected feature columns (add missing, remove extra)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0

    X = X[feature_cols]
    return X


# Prepare for Model 1
X_m1 = prepare_features_for_inference(df, FEATURE_COLS_M1, NUMERIC_FEATURE_COLS, CATEGORICAL_COLS)
print(f"Model 1 feature matrix: {X_m1.shape}")

# %%
# Run cascade inference
print("\n=== RUNNING CASCADE INFERENCE ===")

# Model 1: Swing probability
dm1 = xgb.DMatrix(X_m1, feature_names=FEATURE_COLS_M1)
df["pred_swing_prob"] = model1.predict(dm1)
df["pred_swing"] = (df["pred_swing_prob"] > THRESHOLD_1).astype(int)

print(f"Model 1 predictions:")
print(f"  Predicted swings: {(df['pred_swing'] == 1).sum():,}")
print(f"  Predicted takes: {(df['pred_swing'] == 0).sum():,}")

# Model 2: Whiff probability (only for predicted swings)
pred_swing_mask = df["pred_swing"] == 1
df["pred_whiff_prob"] = np.nan
df["pred_whiff"] = np.nan

if pred_swing_mask.sum() > 0:
    swing_subset = df[pred_swing_mask].copy()
    X_m2 = prepare_features_for_inference(swing_subset, FEATURE_COLS_M2, NUMERIC_FEATURE_COLS, CATEGORICAL_COLS)
    dm2 = xgb.DMatrix(X_m2, feature_names=FEATURE_COLS_M2)

    df.loc[pred_swing_mask, "pred_whiff_prob"] = model2.predict(dm2)
    df.loc[pred_swing_mask, "pred_whiff"] = (df.loc[pred_swing_mask, "pred_whiff_prob"] > THRESHOLD_2).astype(int)

    print(f"\nModel 2 predictions (on predicted swings):")
    print(f"  Predicted whiffs: {(df['pred_whiff'] == 1).sum():,}")
    print(f"  Predicted contact: {(df['pred_whiff'] == 0).sum():,}")

# %%
# Validation against actual outcomes (if available)
print("\n=== VALIDATION (IF ACTUALS AVAILABLE) ===")

has_actuals = "y_swing" in df.columns and df["y_swing"].notna().any()

if has_actuals:
    actual_swing_mask = df["y_swing"] == 1

    # Model 1 accuracy
    m1_correct = (df["pred_swing"] == df["y_swing"]).sum()
    m1_accuracy = m1_correct / len(df)
    print(f"Model 1 accuracy: {m1_accuracy:.4f}")

    # Cascade breakdown
    tp = (pred_swing_mask & actual_swing_mask).sum()
    fp = (pred_swing_mask & ~actual_swing_mask).sum()
    tn = (~pred_swing_mask & ~actual_swing_mask).sum()
    fn = (~pred_swing_mask & actual_swing_mask).sum()

    print(f"  True positives: {tp:,}")
    print(f"  False positives: {fp:,}")
    print(f"  True negatives: {tn:,}")
    print(f"  False negatives: {fn:,}")

    # Model 2 on reachable swings
    reachable = pred_swing_mask & actual_swing_mask
    if reachable.sum() > 0 and "y_whiff" in df.columns:
        reachable_df = df[reachable]
        m2_correct = (reachable_df["pred_whiff"] == reachable_df["y_whiff"]).sum()
        m2_accuracy = m2_correct / len(reachable_df)
        print(f"\nModel 2 accuracy (on reachable swings): {m2_accuracy:.4f}")
else:
    print("No actual outcomes available for validation")

# %%
# Prepare output with IDs for downstream sabermetric tools
print("\n=== PREPARING OUTPUT ===")

ID_COLS = [
    "game_pk",
    "game_date",
    "batter",
    "pitcher",
    "at_bat_number",
    "pitch_number",
]

CONTEXT_COLS = [
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "on_1b",
    "on_2b",
    "on_3b",
    "home_score",
    "away_score",
    "p_throws",
    "stand",
]

PREDICTION_COLS = [
    "pred_swing_prob",
    "pred_swing",
    "pred_whiff_prob",
    "pred_whiff",
]

ACTUAL_COLS = ["y_swing", "y_whiff"]

# Build output dataframe
output_cols = []
for col in ID_COLS + CONTEXT_COLS + PREDICTION_COLS + ACTUAL_COLS:
    if col in df.columns:
        output_cols.append(col)

output_df = df[output_cols].copy()

# Ensure proper types for output
if "game_date" in output_df.columns:
    output_df["game_date"] = pd.to_datetime(output_df["game_date"])

print(f"Output columns: {output_df.columns.tolist()}")
print(f"Output shape: {output_df.shape}")

# %%
# Save predictions
output_df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")

# %%
# Summary statistics
print("\n=== OUTPUT SUMMARY ===")
print(f"Date range: {START_DT} to {END_DT}")
print(f"Total pitches: {len(output_df):,}")
print(f"Predicted swings: {(output_df['pred_swing'] == 1).sum():,} ({(output_df['pred_swing'] == 1).mean()*100:.1f}%)")
print(f"Predicted takes: {(output_df['pred_swing'] == 0).sum():,} ({(output_df['pred_swing'] == 0).mean()*100:.1f}%)")

swing_preds = output_df[output_df["pred_swing"] == 1]
if len(swing_preds) > 0:
    whiff_rate = (swing_preds["pred_whiff"] == 1).mean()
    print(f"Predicted whiff rate (among pred swings): {whiff_rate*100:.1f}%")

# Distribution by count
if "balls" in output_df.columns and "strikes" in output_df.columns:
    output_df["count"] = output_df["balls"].astype(int).astype(str) + "-" + output_df["strikes"].astype(int).astype(str)
    print("\nPredicted swing rate by count:")
    swing_by_count = output_df.groupby("count")["pred_swing"].mean().sort_values(ascending=False)
    for count, rate in swing_by_count.items():
        print(f"  {count}: {rate*100:.1f}%")

# %%
print("\n" + "="*60)
print("INFERENCE COMPLETE")
print("="*60)
print(f"\nOutput saved to: {OUTPUT_PATH}")
print("\nOutput schema for sabermetric tools:")
print("  - ID columns: game_pk, game_date, batter, pitcher, at_bat_number, pitch_number")
print("  - Context: balls, strikes, outs, inning, runners, score, handedness")
print("  - Predictions: pred_swing_prob, pred_swing, pred_whiff_prob, pred_whiff")
print("  - Actuals (if available): y_swing, y_whiff")
#%%