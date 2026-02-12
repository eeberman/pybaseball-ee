# %%
"""
05_train_and_validate.py

Trains XGBoost models for the two-stage cascade:
  - Model 1: Swing vs. Take (all pitches)
  - Model 2: Whiff vs. Contact (swings only)

Includes cascade evaluation metrics and model persistence.

Input:
  - data/processed/statcast_step1_swing_take.parquet
  - data/processed/statcast_step2_whiff_contact.parquet

Output:
  - artifacts/models/model1_swing_take.json
  - artifacts/models/model2_whiff_contact.json
  - artifacts/models/model_config.json
  - artifacts/training/training_metrics.json
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", None)

# %%
# Paths
STEP1_PATH = Path("data/processed") / "statcast_step1_swing_take.parquet"
STEP2_PATH = Path("data/processed") / "statcast_step2_whiff_contact.parquet"

MODEL_DIR = Path("artifacts/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_DIR = Path("artifacts/training")
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

MODEL1_PATH = MODEL_DIR / "model1_swing_take.json"
MODEL2_PATH = MODEL_DIR / "model2_whiff_contact.json"
CONFIG_PATH = MODEL_DIR / "model_config.json"
METRICS_PATH = TRAINING_DIR / "training_metrics.json"

# %%
# Load data
df_step1 = pd.read_parquet(STEP1_PATH)
df_step2 = pd.read_parquet(STEP2_PATH)

print("Loaded Step 1:", STEP1_PATH, df_step1.shape)
print("Loaded Step 2:", STEP2_PATH, df_step2.shape)

# %%
# === INPUT VALIDATION ===
print("\n=== INPUT VALIDATION ===")

assert "split" in df_step1.columns, "Missing split column in step1"
assert "split" in df_step2.columns, "Missing split column in step2"
assert "y_swing" in df_step1.columns, "Missing y_swing in step1"
assert "y_whiff" in df_step2.columns, "Missing y_whiff in step2"

print(f"Step 1 splits: {df_step1['split'].value_counts().to_dict()}")
print(f"Step 2 splits: {df_step2['split'].value_counts().to_dict()}")

# %%
# Define feature columns
# Exclude ID, target, and split columns
ID_COLS = ["game_pk", "game_date", "batter", "pitcher", "at_bat_number", "pitch_number", "inning", "inning_topbot"]
TARGET_COLS = ["target_raw", "y_swing", "y_whiff"]
SPLIT_COLS = ["split"]
EXCLUDE_COLS = ID_COLS + TARGET_COLS + SPLIT_COLS

# Categorical columns that need special handling
# NOTE: pitch_cluster_name excluded - too high cardinality (3700+ unique values)
#       It's pitcher-specific and would cause memory issues with one-hot encoding
CATEGORICAL_COLS = ["p_throws", "stand", "count_state", "score_bucket", "pitch_type_mode"]
HIGH_CARDINALITY_EXCLUDE = ["pitch_cluster_name"]  # Documented for future reference

# Get feature columns
all_cols = df_step1.columns.tolist()
FEATURE_COLS = [c for c in all_cols if c not in EXCLUDE_COLS]

# Filter to columns that exist and are usable
NUMERIC_FEATURE_COLS = []
CAT_FEATURE_COLS = []

for c in FEATURE_COLS:
    if c in CATEGORICAL_COLS:
        CAT_FEATURE_COLS.append(c)
    elif df_step1[c].dtype in ["float64", "float32", "int64", "int32", "int8", "Int8"]:
        NUMERIC_FEATURE_COLS.append(c)

print(f"\nNumeric features ({len(NUMERIC_FEATURE_COLS)}): {NUMERIC_FEATURE_COLS[:10]}...")
print(f"Categorical features ({len(CAT_FEATURE_COLS)}): {CAT_FEATURE_COLS}")

# %%
# Prepare features - one-hot encode categoricals
def sanitize_feature_name(name: str) -> str:
    """Sanitize feature names for XGBoost (no [, ], or <)."""
    return name.replace("[", "_").replace("]", "_").replace("<", "_lt_")


def prepare_features(df: pd.DataFrame, fit_encoder: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Prepare feature matrix with one-hot encoded categoricals."""
    X = df[NUMERIC_FEATURE_COLS].copy()

    # Convert any nullable int to float
    for c in X.columns:
        if X[c].dtype == "Int8":
            X[c] = X[c].astype("float32")

    # One-hot encode categoricals
    for cat_col in CAT_FEATURE_COLS:
        if cat_col in df.columns and df[cat_col].notna().any():
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, dummy_na=True)
            X = pd.concat([X, dummies], axis=1)

    # Sanitize feature names for XGBoost
    X.columns = [sanitize_feature_name(c) for c in X.columns]

    feature_names = X.columns.tolist()
    return X, feature_names


# %%
# Split by temporal column
print("\n=== SPLITTING DATA ===")

train1 = df_step1[df_step1["split"] == "train"]
val1 = df_step1[df_step1["split"] == "val"]
test1 = df_step1[df_step1["split"] == "test"]

train2 = df_step2[df_step2["split"] == "train"]
val2 = df_step2[df_step2["split"] == "val"]
test2 = df_step2[df_step2["split"] == "test"]

print(f"Step 1 - Train: {len(train1)}, Val: {len(val1)}, Test: {len(test1)}")
print(f"Step 2 - Train: {len(train2)}, Val: {len(val2)}, Test: {len(test2)}")

# %%
# Prepare feature matrices
X_train1, feature_names1 = prepare_features(train1)
X_val1, _ = prepare_features(val1)
X_test1, _ = prepare_features(test1)

X_train2, feature_names2 = prepare_features(train2)
X_val2, _ = prepare_features(val2)
X_test2, _ = prepare_features(test2)

# Align columns across splits (some dummy columns may be missing)
def align_columns(X_ref: pd.DataFrame, X_target: pd.DataFrame) -> pd.DataFrame:
    """Align X_target columns to match X_ref, adding missing columns as 0."""
    missing = set(X_ref.columns) - set(X_target.columns)
    extra = set(X_target.columns) - set(X_ref.columns)

    X_aligned = X_target.copy()
    for col in missing:
        X_aligned[col] = 0

    # Keep only columns in reference, in same order
    X_aligned = X_aligned[X_ref.columns]
    return X_aligned

X_val1 = align_columns(X_train1, X_val1)
X_test1 = align_columns(X_train1, X_test1)
X_val2 = align_columns(X_train2, X_val2)
X_test2 = align_columns(X_train2, X_test2)

y_train1 = train1["y_swing"].values
y_val1 = val1["y_swing"].values
y_test1 = test1["y_swing"].values

y_train2 = train2["y_whiff"].values
y_val2 = val2["y_whiff"].values
y_test2 = test2["y_whiff"].values

print(f"\nFeature matrix shapes:")
print(f"  Step 1: train={X_train1.shape}, val={X_val1.shape}, test={X_test1.shape}")
print(f"  Step 2: train={X_train2.shape}, val={X_val2.shape}, test={X_test2.shape}")

# %%
# Create DMatrix objects
dtrain1 = xgb.DMatrix(X_train1, label=y_train1, feature_names=X_train1.columns.tolist())
dval1 = xgb.DMatrix(X_val1, label=y_val1, feature_names=X_train1.columns.tolist())
dtest1 = xgb.DMatrix(X_test1, label=y_test1, feature_names=X_train1.columns.tolist())

dtrain2 = xgb.DMatrix(X_train2, label=y_train2, feature_names=X_train2.columns.tolist())
dval2 = xgb.DMatrix(X_val2, label=y_val2, feature_names=X_train2.columns.tolist())
dtest2 = xgb.DMatrix(X_test2, label=y_test2, feature_names=X_train2.columns.tolist())

# %%
# XGBoost parameters
params1 = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

# Model 2 parameters with class imbalance correction and regularization
# Class ratio: ~76.5% contact, ~23.5% whiff -> scale_pos_weight ~= 3.26
# Reduced depth and added regularization to combat overfitting
params2 = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "max_depth": 4,           # Reduced from 6 for better generalization
    "eta": 0.1,
    "subsample": 0.7,         # Reduced from 0.8
    "colsample_bytree": 0.8,
    "min_child_weight": 5,    # New: prevents overfitting to small partitions
    "gamma": 0.1,             # New: minimum loss reduction for splits
    "scale_pos_weight": 3.26, # Corrects for class imbalance (contacts/whiffs)
    "seed": 42,
}

# %%
# Train Model 1 (swing/take)
print("\n=== TRAINING MODEL 1 (SWING/TAKE) ===")

evals1 = [(dtrain1, "train"), (dval1, "val")]
evals_result1 = {}

model1 = xgb.train(
    params1,
    dtrain1,
    num_boost_round=500,
    evals=evals1,
    evals_result=evals_result1,
    early_stopping_rounds=50,
    verbose_eval=50,
)

print(f"Best iteration: {model1.best_iteration}")
print(f"Best val AUC: {evals_result1['val']['auc'][model1.best_iteration]:.4f}")

# %%
# Train Model 2 (whiff/contact)
print("\n=== TRAINING MODEL 2 (WHIFF/CONTACT) ===")

evals2 = [(dtrain2, "train"), (dval2, "val")]
evals_result2 = {}

model2 = xgb.train(
    params2,
    dtrain2,
    num_boost_round=500,
    evals=evals2,
    evals_result=evals_result2,
    early_stopping_rounds=50,
    verbose_eval=50,
)

print(f"Best iteration: {model2.best_iteration}")
print(f"Best val AUC: {evals_result2['val']['auc'][model2.best_iteration]:.4f}")

# %%
# === EVALUATE MODEL 1 INDEPENDENTLY ===
print("\n=== MODEL 1 EVALUATION (SWING/TAKE) ===")

# Predictions
pred1_train_prob = model1.predict(dtrain1)
pred1_val_prob = model1.predict(dval1)
pred1_test_prob = model1.predict(dtest1)

# Default threshold
THRESHOLD_1 = 0.5
pred1_train = (pred1_train_prob > THRESHOLD_1).astype(int)
pred1_val = (pred1_val_prob > THRESHOLD_1).astype(int)
pred1_test = (pred1_test_prob > THRESHOLD_1).astype(int)

def print_metrics(y_true, y_pred, y_prob, label: str):
    """Print classification metrics including Brier score for calibration."""
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{label}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:,} FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,} TP={cm[1,1]:,}")

    return {"accuracy": acc, "roc_auc": roc_auc, "pr_auc": pr_auc, "brier_score": brier}

m1_train = print_metrics(y_train1, pred1_train, pred1_train_prob, "Train")
m1_val = print_metrics(y_val1, pred1_val, pred1_val_prob, "Validation (2023 Sep-Oct)")
m1_test = print_metrics(y_test1, pred1_test, pred1_test_prob, "Test (2024)")

# %%
# === EVALUATE MODEL 2 INDEPENDENTLY ===
print("\n=== MODEL 2 EVALUATION (WHIFF/CONTACT) ===")

pred2_train_prob = model2.predict(dtrain2)
pred2_val_prob = model2.predict(dval2)
pred2_test_prob = model2.predict(dtest2)

THRESHOLD_2 = 0.5
pred2_train = (pred2_train_prob > THRESHOLD_2).astype(int)
pred2_val = (pred2_val_prob > THRESHOLD_2).astype(int)
pred2_test = (pred2_test_prob > THRESHOLD_2).astype(int)

m2_train = print_metrics(y_train2, pred2_train, pred2_train_prob, "Train")
m2_val = print_metrics(y_val2, pred2_val, pred2_val_prob, "Validation (2023 Sep-Oct)")
m2_test = print_metrics(y_test2, pred2_test, pred2_test_prob, "Test (2024)")

# %%
# === MODEL 2 ZONE-AWARE DIAGNOSTICS ===
print("\n" + "="*60)
print("=== MODEL 2 PERFORMANCE BY ZONE ===")
print("="*60)

# Breaking ball types for diagnostic purposes
BREAKING_TYPES = ["SL", "CU", "ST", "KC", "SV", "CB"]

# Store zone metrics for later
zone_metrics = {}

# In-zone predictions (test set)
if "in_zone" in test2.columns:
    in_zone_mask_test = test2["in_zone"] == 1
    out_zone_mask_test = test2["in_zone"] == 0

    print(f"\nTest set breakdown:")
    print(f"  In-zone: {in_zone_mask_test.sum():,} ({in_zone_mask_test.mean()*100:.1f}%)")
    print(f"  Out-of-zone: {out_zone_mask_test.sum():,} ({out_zone_mask_test.mean()*100:.1f}%)")

    # In-zone metrics
    if in_zone_mask_test.sum() > 100:
        y_test2_in = y_test2[in_zone_mask_test]
        pred2_test_in = pred2_test[in_zone_mask_test]
        pred2_test_prob_in = pred2_test_prob[in_zone_mask_test]

        print(f"\n  In-zone whiff rate: {y_test2_in.mean()*100:.1f}%")
        m2_in_zone = print_metrics(y_test2_in, pred2_test_in, pred2_test_prob_in, "In-Zone (test)")
        zone_metrics["in_zone"] = m2_in_zone

    # Out-of-zone metrics
    if out_zone_mask_test.sum() > 100:
        y_test2_out = y_test2[out_zone_mask_test]
        pred2_test_out = pred2_test[out_zone_mask_test]
        pred2_test_prob_out = pred2_test_prob[out_zone_mask_test]

        print(f"\n  Out-of-zone whiff rate: {y_test2_out.mean()*100:.1f}%")
        m2_out_zone = print_metrics(y_test2_out, pred2_test_out, pred2_test_prob_out, "Out-of-Zone (test)")
        zone_metrics["out_of_zone"] = m2_out_zone

    # Breaking ball out-of-zone (the chaotic subset)
    if "pitch_type_mode" in test2.columns:
        breaking_ooz_mask = out_zone_mask_test & test2["pitch_type_mode"].isin(BREAKING_TYPES)
        if breaking_ooz_mask.sum() > 100:
            y_test2_breaking = y_test2[breaking_ooz_mask]
            pred2_test_breaking = pred2_test[breaking_ooz_mask]
            pred2_test_prob_breaking = pred2_test_prob[breaking_ooz_mask]

            print(f"\n  Out-of-zone breaking balls: {breaking_ooz_mask.sum():,} ({breaking_ooz_mask.sum()/len(test2)*100:.1f}% of test)")
            print(f"  OOZ breaking ball whiff rate: {y_test2_breaking.mean()*100:.1f}%")
            m2_breaking_ooz = print_metrics(y_test2_breaking, pred2_test_breaking, pred2_test_prob_breaking, "Out-of-Zone Breaking Balls (chaotic)")
            zone_metrics["ooz_breaking"] = m2_breaking_ooz

        # Non-breaking out-of-zone for comparison
        non_breaking_ooz_mask = out_zone_mask_test & ~test2["pitch_type_mode"].isin(BREAKING_TYPES)
        if non_breaking_ooz_mask.sum() > 100:
            y_test2_non_breaking = y_test2[non_breaking_ooz_mask]
            pred2_test_non_breaking = pred2_test[non_breaking_ooz_mask]
            pred2_test_prob_non_breaking = pred2_test_prob[non_breaking_ooz_mask]

            print(f"\n  Out-of-zone non-breaking: {non_breaking_ooz_mask.sum():,}")
            print(f"  OOZ non-breaking whiff rate: {y_test2_non_breaking.mean()*100:.1f}%")
            m2_non_breaking_ooz = print_metrics(y_test2_non_breaking, pred2_test_non_breaking, pred2_test_prob_non_breaking, "Out-of-Zone Non-Breaking")
            zone_metrics["ooz_non_breaking"] = m2_non_breaking_ooz
else:
    print("WARNING: in_zone column not found in test2, skipping zone diagnostics")

# %%
# === MODEL 2 PROBABILITY CALIBRATION ===
print("\n=== MODEL 2 PROBABILITY CALIBRATION (ISOTONIC) ===")

# Fit isotonic calibration on validation set
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(pred2_val_prob, y_val2)

# Apply calibration to get calibrated probabilities
pred2_val_calibrated = calibrator.predict(pred2_val_prob)
pred2_test_calibrated = calibrator.predict(pred2_test_prob)

# Evaluate calibration improvement
brier_val_uncal = brier_score_loss(y_val2, pred2_val_prob)
brier_val_cal = brier_score_loss(y_val2, pred2_val_calibrated)
brier_test_uncal = brier_score_loss(y_test2, pred2_test_prob)
brier_test_cal = brier_score_loss(y_test2, pred2_test_calibrated)

print(f"\nBrier Score (lower is better):")
print(f"  Val uncalibrated:  {brier_val_uncal:.4f}")
print(f"  Val calibrated:    {brier_val_cal:.4f} ({(brier_val_uncal - brier_val_cal)/brier_val_uncal*100:.1f}% improvement)")
print(f"  Test uncalibrated: {brier_test_uncal:.4f}")
print(f"  Test calibrated:   {brier_test_cal:.4f} ({(brier_test_uncal - brier_test_cal)/brier_test_uncal*100:.1f}% improvement)")

# Generate calibration curve plot
CALIBRATION_PLOT_PATH = TRAINING_DIR / "calibration_curve_model2.png"

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Compute calibration curves
n_bins = 10
prob_true_uncal, prob_pred_uncal = calibration_curve(y_test2, pred2_test_prob, n_bins=n_bins)
prob_true_cal, prob_pred_cal = calibration_curve(y_test2, pred2_test_calibrated, n_bins=n_bins)

ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
ax.plot(prob_pred_uncal, prob_true_uncal, "s-", label=f"Uncalibrated (Brier={brier_test_uncal:.4f})")
ax.plot(prob_pred_cal, prob_true_cal, "o-", label=f"Calibrated (Brier={brier_test_cal:.4f})")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives (whiffs)")
ax.set_title("Model 2 Calibration Curve (Test Set)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(CALIBRATION_PLOT_PATH, dpi=150)
plt.close()
print(f"\nSaved calibration curve: {CALIBRATION_PLOT_PATH}")

# Store calibration metrics for later
m2_calibration = {
    "brier_val_uncalibrated": float(brier_val_uncal),
    "brier_val_calibrated": float(brier_val_cal),
    "brier_test_uncalibrated": float(brier_test_uncal),
    "brier_test_calibrated": float(brier_test_cal),
}

# %%
# === CASCADE EVALUATION ===
print("\n" + "="*60)
print("=== CASCADE EVALUATION (THE KEY PART) ===")
print("="*60)

# Run on full test set (all pitches)
test_all = df_step1[df_step1["split"] == "test"].copy()
print(f"\nTest set size: {len(test_all)} pitches")

# Prepare features for full test set
X_test_all, _ = prepare_features(test_all)
X_test_all = align_columns(X_train1, X_test_all)
dtest_all = xgb.DMatrix(X_test_all, feature_names=X_train1.columns.tolist())

# Model 1 predictions
test_all["pred_swing_prob"] = model1.predict(dtest_all)
test_all["pred_swing"] = (test_all["pred_swing_prob"] > THRESHOLD_1).astype(int)

# Model 2 predictions (only for predicted swings)
pred_swing_mask = test_all["pred_swing"] == 1
test_all["pred_whiff_prob"] = np.nan

if pred_swing_mask.sum() > 0:
    # Prepare features for predicted swings
    swing_subset = test_all[pred_swing_mask].copy()
    X_swing_subset, _ = prepare_features(swing_subset)
    X_swing_subset = align_columns(X_train2, X_swing_subset)
    dswing_subset = xgb.DMatrix(X_swing_subset, feature_names=X_train2.columns.tolist())

    test_all.loc[pred_swing_mask, "pred_whiff_prob"] = model2.predict(dswing_subset)

test_all["pred_whiff"] = np.where(
    test_all["pred_whiff_prob"].notna(),
    (test_all["pred_whiff_prob"] > THRESHOLD_2).astype(int),
    np.nan
)

# %%
# CASCADE METRICS
print("\n=== CASCADE METRICS ===")

actual_swing_mask = test_all["y_swing"] == 1
pred_swing_mask = test_all["pred_swing"] == 1

true_positives_m1 = (pred_swing_mask & actual_swing_mask).sum()
false_positives_m1 = (pred_swing_mask & ~actual_swing_mask).sum()
true_negatives_m1 = (~pred_swing_mask & ~actual_swing_mask).sum()
false_negatives_m1 = (~pred_swing_mask & actual_swing_mask).sum()

print(f"\nModel 1 predictions on test set:")
print(f"  Predicted swings: {pred_swing_mask.sum():,}")
print(f"  Predicted takes: {(~pred_swing_mask).sum():,}")
print(f"  True positives (correctly predicted swings): {true_positives_m1:,}")
print(f"  False positives (takes predicted as swings): {false_positives_m1:,}")
print(f"  True negatives (correctly predicted takes): {true_negatives_m1:,}")
print(f"  False negatives (swings predicted as takes): {false_negatives_m1:,}")
print(f"  Precision: {true_positives_m1 / pred_swing_mask.sum():.4f}")
print(f"  Recall: {true_positives_m1 / actual_swing_mask.sum():.4f}")

# %%
# Model 2 metrics ONLY on reachable ground truth
# (predicted swing AND actual swing → has y_whiff)
reachable = pred_swing_mask & actual_swing_mask
test_reachable = test_all[reachable].copy()

print(f"\n=== MODEL 2 ON REACHABLE SWINGS ===")
print(f"Reachable swings (predicted swing AND actual swing): {len(test_reachable):,}")
print(f"Coverage of actual swings: {len(test_reachable) / actual_swing_mask.sum():.4f}")

if len(test_reachable) > 0:
    y_whiff_reachable = test_reachable["y_whiff"].values
    pred_whiff_reachable = test_reachable["pred_whiff"].values
    pred_whiff_prob_reachable = test_reachable["pred_whiff_prob"].values

    m2_reachable = print_metrics(
        y_whiff_reachable.astype(int),
        pred_whiff_reachable.astype(int),
        pred_whiff_prob_reachable,
        "Model 2 on Reachable Swings"
    )

# %%
# End-to-end cascade accuracy
print("\n=== END-TO-END CASCADE ACCURACY ===")

# For actual swings: did we predict swing AND get whiff/contact right?
# For actual takes: did we predict take?

# Swings correctly handled (predicted swing AND correct whiff prediction)
swing_correct = (
    actual_swing_mask &
    pred_swing_mask &
    (test_all["pred_whiff"] == test_all["y_whiff"])
).sum()

# Takes correctly handled (predicted take)
take_correct = (~actual_swing_mask & ~pred_swing_mask).sum()

cascade_accuracy = (swing_correct + take_correct) / len(test_all)

print(f"Swings correctly handled: {swing_correct:,}")
print(f"Takes correctly handled: {take_correct:,}")
print(f"Total correct: {swing_correct + take_correct:,} / {len(test_all):,}")
print(f"Cascade accuracy: {cascade_accuracy:.4f}")

# %%
# Unreachable predictions (false positives - no ground truth for whiff)
unreachable = pred_swing_mask & ~actual_swing_mask
print(f"\n=== UNREACHABLE PREDICTIONS ===")
print(f"Unreachable predictions (M1 false positives): {unreachable.sum():,}")
print(f"These have Model 2 predictions but no y_whiff ground truth")

# Distribution of whiff predictions on false positives
if unreachable.sum() > 0:
    unreachable_whiff_pred = test_all.loc[unreachable, "pred_whiff"].value_counts()
    print(f"Whiff predictions on unreachable: {unreachable_whiff_pred.to_dict()}")

# %%
# === FEATURE IMPORTANCE (MODEL 2) ===
print("\n=== MODEL 2 TOP FEATURE IMPORTANCES ===")

# Get feature importance scores
importance_scores = model2.get_score(importance_type="gain")

# Sort by importance
sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

# Print top 20 features
print("\nTop 20 features by gain:")
for i, (feat, score) in enumerate(sorted_importance[:20], 1):
    print(f"  {i:2d}. {feat}: {score:.2f}")

# Check if cluster features are being used
cluster_features = [f for f, _ in sorted_importance if "cluster_" in f.lower()]
print(f"\nCluster-related features in model: {len(cluster_features)}")
if cluster_features:
    print(f"  {cluster_features[:10]}...")

# Save feature importance to file
IMPORTANCE_PATH = TRAINING_DIR / "feature_importance_model2.json"
with open(IMPORTANCE_PATH, "w") as f:
    json.dump(dict(sorted_importance), f, indent=2)
print(f"Saved feature importance: {IMPORTANCE_PATH}")

# %%
# === YEAR-OVER-YEAR COMPARISON ===
print("\n=== YEAR-OVER-YEAR COMPARISON ===")
print("Comparing validation (2023 Sep-Oct) vs test (2024):")
print(f"\nModel 1 (Swing/Take):")
print(f"  Val ROC-AUC: {m1_val['roc_auc']:.4f}")
print(f"  Test ROC-AUC: {m1_test['roc_auc']:.4f}")
print(f"  Difference: {m1_test['roc_auc'] - m1_val['roc_auc']:.4f}")

print(f"\nModel 2 (Whiff/Contact):")
print(f"  Val ROC-AUC: {m2_val['roc_auc']:.4f}")
print(f"  Test ROC-AUC: {m2_test['roc_auc']:.4f}")
print(f"  Difference: {m2_test['roc_auc'] - m2_val['roc_auc']:.4f}")

# %%
# === SAVE MODELS ===
print("\n=== SAVING MODELS ===")

model1.save_model(MODEL1_PATH)
model2.save_model(MODEL2_PATH)
print(f"Saved: {MODEL1_PATH}")
print(f"Saved: {MODEL2_PATH}")

# Save calibrator for Model 2
import joblib
CALIBRATOR_PATH = MODEL_DIR / "model2_calibrator.joblib"
joblib.dump(calibrator, CALIBRATOR_PATH)
print(f"Saved: {CALIBRATOR_PATH}")

# %%
# Save model config for inference
config = {
    "feature_cols_model1": X_train1.columns.tolist(),
    "feature_cols_model2": X_train2.columns.tolist(),
    "numeric_feature_cols": NUMERIC_FEATURE_COLS,
    "categorical_cols": CAT_FEATURE_COLS,
    "threshold_swing": THRESHOLD_1,
    "threshold_whiff": THRESHOLD_2,
    "training_date_range": {
        "train": "2023-04-01 to 2023-08-31",
        "val": "2023-09-01 to 2023-10-01",
        "test": "2024-04-01 to 2024-10-01",
    },
    "model1_best_iteration": int(model1.best_iteration),
    "model2_best_iteration": int(model2.best_iteration),
}

with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)
print(f"Saved: {CONFIG_PATH}")

# %%
# Save training metrics
metrics = {
    "model1": {
        "train": m1_train,
        "val": m1_val,
        "test": m1_test,
        "best_iteration": int(model1.best_iteration),
    },
    "model2": {
        "train": m2_train,
        "val": m2_val,
        "test": m2_test,
        "best_iteration": int(model2.best_iteration),
        "calibration": m2_calibration,
        "params": {k: v for k, v in params2.items() if k not in ["objective", "eval_metric"]},
        "zone_diagnostics": zone_metrics if zone_metrics else None,
    },
    "cascade": {
        "test_size": int(len(test_all)),
        "predicted_swings": int(pred_swing_mask.sum()),
        "true_positives_m1": int(true_positives_m1),
        "false_positives_m1": int(false_positives_m1),
        "reachable_swings": int(len(test_reachable)),
        "cascade_accuracy": float(cascade_accuracy),
        "m2_reachable_roc_auc": float(m2_reachable["roc_auc"]) if len(test_reachable) > 0 else None,
    },
    "year_over_year": {
        "model1_val_roc_auc": m1_val["roc_auc"],
        "model1_test_roc_auc": m1_test["roc_auc"],
        "model1_diff": m1_test["roc_auc"] - m1_val["roc_auc"],
        "model2_val_roc_auc": m2_val["roc_auc"],
        "model2_test_roc_auc": m2_test["roc_auc"],
        "model2_diff": m2_test["roc_auc"] - m2_val["roc_auc"],
    },
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved: {METRICS_PATH}")

# %%
print("\n" + "="*60)
print("TRAINING AND VALIDATION COMPLETE")
print("="*60)
print(f"\nModels saved to: {MODEL_DIR}")
print(f"Metrics saved to: {METRICS_PATH}")
print(f"\nNext: Run 06_inference.py to generate predictions on new data")
#%%