# %%
"""
06_player_skill_metrics.py

Evaluates batter contact skill and pitcher whiff skill by comparing actual
outcomes to model-predicted expected rates.

Core concept: Skill = Actual - Expected
- Batter contact skill = expected_whiff - actual_whiff (positive = good contact)
- Pitcher whiff skill = actual_whiff - expected_whiff (positive = good at whiffs)

Input:
  - data/processed/statcast_step2_whiff_contact.parquet (swings only)
  - artifacts/models/model2_whiff_contact.json
  - artifacts/models/model2_calibrator.joblib
  - artifacts/models/model_config.json

Output:
  - artifacts/skill/batter_contact_skill.parquet
  - artifacts/skill/pitcher_whiff_skill.parquet
  - artifacts/skill/skill_scatter.png
  - artifacts/skill/skill_leaderboards.png
  - artifacts/skill/skill_summary.json
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", None)

# %%
# Paths
STEP2_PATH = Path("data/processed") / "statcast_step2_whiff_contact.parquet"

MODEL_DIR = Path("artifacts/models")
MODEL2_PATH = MODEL_DIR / "model2_whiff_contact.json"
CALIBRATOR_PATH = MODEL_DIR / "model2_calibrator.joblib"
CONFIG_PATH = MODEL_DIR / "model_config.json"

SKILL_DIR = Path("artifacts/skill")
SKILL_DIR.mkdir(parents=True, exist_ok=True)

BATTER_SKILL_PATH = SKILL_DIR / "batter_contact_skill.parquet"
PITCHER_SKILL_PATH = SKILL_DIR / "pitcher_whiff_skill.parquet"
SCATTER_PATH = SKILL_DIR / "skill_scatter.png"
LEADERBOARD_PATH = SKILL_DIR / "skill_leaderboards.png"
SUMMARY_PATH = SKILL_DIR / "skill_summary.json"

# %%
# Configuration
MIN_SWINGS_BATTER = 50   # Minimum swings for batter to be included
MIN_SWINGS_PITCHER = 100 # Minimum swings for pitcher to be included

# %%
# Load data and models
print("Loading data and models...")

df = pd.read_parquet(STEP2_PATH)
print(f"Loaded step2 data: {df.shape}")

model2 = xgb.Booster()
model2.load_model(MODEL2_PATH)
print(f"Loaded model: {MODEL2_PATH}")

calibrator = joblib.load(CALIBRATOR_PATH)
print(f"Loaded calibrator: {CALIBRATOR_PATH}")

with open(CONFIG_PATH) as f:
    config = json.load(f)
print(f"Loaded config: {CONFIG_PATH}")

# %%
# Filter to test set (2024) for skill evaluation
# This ensures we're using held-out data
print("\n=== FILTERING TO TEST SET ===")

df_test = df[df["split"] == "test"].copy()
print(f"Test set (2024): {len(df_test):,} swings")
print(f"Date range: {df_test['game_date'].min()} to {df_test['game_date'].max()}")

# Player counts
n_batters = df_test["batter"].nunique()
n_pitchers = df_test["pitcher"].nunique()
print(f"Unique batters: {n_batters:,}")
print(f"Unique pitchers: {n_pitchers:,}")

# %%
# Prepare feature matrix (same as training script)
print("\n=== PREPARING FEATURES ===")

NUMERIC_FEATURE_COLS = config["numeric_feature_cols"]
CAT_FEATURE_COLS = config["categorical_cols"]
EXPECTED_FEATURES = config["feature_cols_model2"]


def sanitize_feature_name(name: str) -> str:
    """Sanitize feature names for XGBoost (no [, ], or <)."""
    return name.replace("[", "_").replace("]", "_").replace("<", "_lt_")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
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

    return X


def align_columns(X: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """Align columns to match expected, adding missing columns as 0."""
    missing = set(expected_cols) - set(X.columns)
    extra = set(X.columns) - set(expected_cols)

    X_aligned = X.copy()
    for col in missing:
        X_aligned[col] = 0

    # Keep only expected columns in order
    X_aligned = X_aligned[expected_cols]
    return X_aligned


# Prepare features
X_test = prepare_features(df_test)
X_test = align_columns(X_test, EXPECTED_FEATURES)

print(f"Feature matrix shape: {X_test.shape}")

# %%
# Generate predictions
print("\n=== GENERATING PREDICTIONS ===")

dtest = xgb.DMatrix(X_test, feature_names=EXPECTED_FEATURES)
pred_raw = model2.predict(dtest)

# Apply isotonic calibration
pred_calibrated = calibrator.predict(pred_raw)

# Add predictions to dataframe
df_test["pred_whiff_raw"] = pred_raw
df_test["pred_whiff_calibrated"] = pred_calibrated

print(f"Raw predictions - mean: {pred_raw.mean():.4f}, std: {pred_raw.std():.4f}")
print(f"Calibrated predictions - mean: {pred_calibrated.mean():.4f}, std: {pred_calibrated.std():.4f}")
print(f"Actual whiff rate: {df_test['y_whiff'].mean():.4f}")

# %%
# Aggregate by batter
print("\n=== BATTER CONTACT SKILL ===")

batter_stats = df_test.groupby("batter").agg(
    n_swings=("y_whiff", "count"),
    actual_whiff_rate=("y_whiff", "mean"),
    expected_whiff_rate=("pred_whiff_calibrated", "mean"),
    expected_whiff_raw=("pred_whiff_raw", "mean"),
).reset_index()

# Contact skill: positive = better contact (fewer whiffs than expected)
batter_stats["contact_skill"] = batter_stats["expected_whiff_rate"] - batter_stats["actual_whiff_rate"]

# Standard error (binomial approximation)
# SE = sqrt(p*(1-p)/n)
batter_stats["std_error"] = np.sqrt(
    batter_stats["actual_whiff_rate"] * (1 - batter_stats["actual_whiff_rate"]) / batter_stats["n_swings"]
)

# Apply minimum sample filter
batter_stats_filtered = batter_stats[batter_stats["n_swings"] >= MIN_SWINGS_BATTER].copy()

print(f"Total batters: {len(batter_stats):,}")
print(f"Batters with >= {MIN_SWINGS_BATTER} swings: {len(batter_stats_filtered):,}")

# Add percentile ranking (100 = best contact)
batter_stats_filtered["percentile"] = batter_stats_filtered["contact_skill"].rank(pct=True) * 100
batter_stats_filtered["percentile"] = batter_stats_filtered["percentile"].round().astype(int)

# Sort by contact skill (descending - best contact first)
batter_stats_filtered = batter_stats_filtered.sort_values("contact_skill", ascending=False)

print(f"\nTop 10 contact skill batters:")
print(batter_stats_filtered[["batter", "n_swings", "actual_whiff_rate", "expected_whiff_rate", "contact_skill", "percentile"]].head(10).to_string(index=False))

print(f"\nBottom 10 contact skill batters:")
print(batter_stats_filtered[["batter", "n_swings", "actual_whiff_rate", "expected_whiff_rate", "contact_skill", "percentile"]].tail(10).to_string(index=False))

# %%
# Aggregate by pitcher
print("\n=== PITCHER WHIFF SKILL ===")

pitcher_stats = df_test.groupby("pitcher").agg(
    n_swings=("y_whiff", "count"),
    actual_whiff_rate=("y_whiff", "mean"),
    expected_whiff_rate=("pred_whiff_calibrated", "mean"),
    expected_whiff_raw=("pred_whiff_raw", "mean"),
).reset_index()

# Whiff skill: positive = better at generating whiffs (more whiffs than expected)
pitcher_stats["whiff_skill"] = pitcher_stats["actual_whiff_rate"] - pitcher_stats["expected_whiff_rate"]

# Standard error
pitcher_stats["std_error"] = np.sqrt(
    pitcher_stats["actual_whiff_rate"] * (1 - pitcher_stats["actual_whiff_rate"]) / pitcher_stats["n_swings"]
)

# Apply minimum sample filter
pitcher_stats_filtered = pitcher_stats[pitcher_stats["n_swings"] >= MIN_SWINGS_PITCHER].copy()

print(f"Total pitchers: {len(pitcher_stats):,}")
print(f"Pitchers with >= {MIN_SWINGS_PITCHER} swings: {len(pitcher_stats_filtered):,}")

# Add percentile ranking (100 = best whiff skill)
pitcher_stats_filtered["percentile"] = pitcher_stats_filtered["whiff_skill"].rank(pct=True) * 100
pitcher_stats_filtered["percentile"] = pitcher_stats_filtered["percentile"].round().astype(int)

# Sort by whiff skill (descending - best whiff generation first)
pitcher_stats_filtered = pitcher_stats_filtered.sort_values("whiff_skill", ascending=False)

print(f"\nTop 10 whiff skill pitchers:")
print(pitcher_stats_filtered[["pitcher", "n_swings", "actual_whiff_rate", "expected_whiff_rate", "whiff_skill", "percentile"]].head(10).to_string(index=False))

print(f"\nBottom 10 whiff skill pitchers:")
print(pitcher_stats_filtered[["pitcher", "n_swings", "actual_whiff_rate", "expected_whiff_rate", "whiff_skill", "percentile"]].tail(10).to_string(index=False))

# %%
# Sanity checks
print("\n=== SANITY CHECKS ===")

# Mean skill should be ~0 (by construction)
mean_batter_skill = batter_stats_filtered["contact_skill"].mean()
mean_pitcher_skill = pitcher_stats_filtered["whiff_skill"].mean()

print(f"Mean batter contact skill: {mean_batter_skill:.4f} (should be ~0)")
print(f"Mean pitcher whiff skill: {mean_pitcher_skill:.4f} (should be ~0)")

# Correlation between expected and actual
batter_corr = batter_stats_filtered["expected_whiff_rate"].corr(batter_stats_filtered["actual_whiff_rate"])
pitcher_corr = pitcher_stats_filtered["expected_whiff_rate"].corr(pitcher_stats_filtered["actual_whiff_rate"])

print(f"\nExpected vs actual correlation (batters): {batter_corr:.4f}")
print(f"Expected vs actual correlation (pitchers): {pitcher_corr:.4f}")

# Skill standard deviation
print(f"\nBatter contact skill std: {batter_stats_filtered['contact_skill'].std():.4f}")
print(f"Pitcher whiff skill std: {pitcher_stats_filtered['whiff_skill'].std():.4f}")

# %%
# Save skill metrics
print("\n=== SAVING SKILL METRICS ===")

batter_stats_filtered.to_parquet(BATTER_SKILL_PATH, index=False)
print(f"Saved: {BATTER_SKILL_PATH} ({len(batter_stats_filtered)} batters)")

pitcher_stats_filtered.to_parquet(PITCHER_SKILL_PATH, index=False)
print(f"Saved: {PITCHER_SKILL_PATH} ({len(pitcher_stats_filtered)} pitchers)")

# %%
# Generate scatter plots
print("\n=== GENERATING VISUALIZATIONS ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Batter scatter plot
ax1 = axes[0]
ax1.scatter(
    batter_stats_filtered["expected_whiff_rate"],
    batter_stats_filtered["actual_whiff_rate"],
    alpha=0.4,
    s=10,
    c="blue"
)
# Diagonal line (league average)
ax1.plot([0, 0.6], [0, 0.6], "k--", alpha=0.5, label="Expected = Actual")
ax1.set_xlabel("Expected Whiff Rate")
ax1.set_ylabel("Actual Whiff Rate")
ax1.set_title(f"Batter Contact Skill (n={len(batter_stats_filtered)})")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 0.6)
ax1.set_ylim(0, 0.6)

# Add annotations for best/worst batters
best_batter = batter_stats_filtered.iloc[0]
worst_batter = batter_stats_filtered.iloc[-1]
ax1.annotate(f"Best: {best_batter['batter']}\n(skill={best_batter['contact_skill']:.3f})",
             xy=(best_batter["expected_whiff_rate"], best_batter["actual_whiff_rate"]),
             fontsize=8, ha="left")
ax1.annotate(f"Worst: {worst_batter['batter']}\n(skill={worst_batter['contact_skill']:.3f})",
             xy=(worst_batter["expected_whiff_rate"], worst_batter["actual_whiff_rate"]),
             fontsize=8, ha="right")

# Pitcher scatter plot
ax2 = axes[1]
ax2.scatter(
    pitcher_stats_filtered["expected_whiff_rate"],
    pitcher_stats_filtered["actual_whiff_rate"],
    alpha=0.4,
    s=10,
    c="red"
)
# Diagonal line
ax2.plot([0, 0.6], [0, 0.6], "k--", alpha=0.5, label="Expected = Actual")
ax2.set_xlabel("Expected Whiff Rate")
ax2.set_ylabel("Actual Whiff Rate")
ax2.set_title(f"Pitcher Whiff Skill (n={len(pitcher_stats_filtered)})")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 0.6)
ax2.set_ylim(0, 0.6)

# Add annotations for best/worst pitchers
best_pitcher = pitcher_stats_filtered.iloc[0]
worst_pitcher = pitcher_stats_filtered.iloc[-1]
ax2.annotate(f"Best: {best_pitcher['pitcher']}\n(skill={best_pitcher['whiff_skill']:.3f})",
             xy=(best_pitcher["expected_whiff_rate"], best_pitcher["actual_whiff_rate"]),
             fontsize=8, ha="left")
ax2.annotate(f"Worst: {worst_pitcher['pitcher']}\n(skill={worst_pitcher['whiff_skill']:.3f})",
             xy=(worst_pitcher["expected_whiff_rate"], worst_pitcher["actual_whiff_rate"]),
             fontsize=8, ha="right")

plt.tight_layout()
plt.savefig(SCATTER_PATH, dpi=150)
plt.close()
print(f"Saved: {SCATTER_PATH}")

# %%
# Generate leaderboard visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 10))

# Top 20 batters
ax1 = axes[0]
top_batters = batter_stats_filtered.head(20)
colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in top_batters["contact_skill"]]
bars = ax1.barh(range(len(top_batters)), top_batters["contact_skill"], color=colors)
ax1.set_yticks(range(len(top_batters)))
ax1.set_yticklabels([f"{row['batter']} (n={row['n_swings']})" for _, row in top_batters.iterrows()], fontsize=8)
ax1.set_xlabel("Contact Skill (expected - actual whiff rate)")
ax1.set_title("Top 20 Batters by Contact Skill")
ax1.axvline(x=0, color="black", linewidth=0.5)
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis="x")

# Add value labels
for i, (_, row) in enumerate(top_batters.iterrows()):
    ax1.text(row["contact_skill"] + 0.002, i, f"{row['contact_skill']:.3f}", va="center", fontsize=7)

# Top 20 pitchers
ax2 = axes[1]
top_pitchers = pitcher_stats_filtered.head(20)
colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in top_pitchers["whiff_skill"]]
bars = ax2.barh(range(len(top_pitchers)), top_pitchers["whiff_skill"], color=colors)
ax2.set_yticks(range(len(top_pitchers)))
ax2.set_yticklabels([f"{row['pitcher']} (n={row['n_swings']})" for _, row in top_pitchers.iterrows()], fontsize=8)
ax2.set_xlabel("Whiff Skill (actual - expected whiff rate)")
ax2.set_title("Top 20 Pitchers by Whiff Skill")
ax2.axvline(x=0, color="black", linewidth=0.5)
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis="x")

# Add value labels
for i, (_, row) in enumerate(top_pitchers.iterrows()):
    ax2.text(row["whiff_skill"] + 0.002, i, f"{row['whiff_skill']:.3f}", va="center", fontsize=7)

plt.tight_layout()
plt.savefig(LEADERBOARD_PATH, dpi=150)
plt.close()
print(f"Saved: {LEADERBOARD_PATH}")

# %%
# Save summary JSON
summary = {
    "data": {
        "test_set_size": len(df_test),
        "date_range": f"{df_test['game_date'].min()} to {df_test['game_date'].max()}",
        "unique_batters_raw": n_batters,
        "unique_pitchers_raw": n_pitchers,
    },
    "filters": {
        "min_swings_batter": MIN_SWINGS_BATTER,
        "min_swings_pitcher": MIN_SWINGS_PITCHER,
    },
    "batter_stats": {
        "n_batters": len(batter_stats_filtered),
        "mean_contact_skill": float(mean_batter_skill),
        "std_contact_skill": float(batter_stats_filtered["contact_skill"].std()),
        "min_contact_skill": float(batter_stats_filtered["contact_skill"].min()),
        "max_contact_skill": float(batter_stats_filtered["contact_skill"].max()),
        "expected_vs_actual_correlation": float(batter_corr),
        "top_5_batters": batter_stats_filtered[["batter", "n_swings", "contact_skill"]].head(5).to_dict(orient="records"),
        "bottom_5_batters": batter_stats_filtered[["batter", "n_swings", "contact_skill"]].tail(5).to_dict(orient="records"),
    },
    "pitcher_stats": {
        "n_pitchers": len(pitcher_stats_filtered),
        "mean_whiff_skill": float(mean_pitcher_skill),
        "std_whiff_skill": float(pitcher_stats_filtered["whiff_skill"].std()),
        "min_whiff_skill": float(pitcher_stats_filtered["whiff_skill"].min()),
        "max_whiff_skill": float(pitcher_stats_filtered["whiff_skill"].max()),
        "expected_vs_actual_correlation": float(pitcher_corr),
        "top_5_pitchers": pitcher_stats_filtered[["pitcher", "n_swings", "whiff_skill"]].head(5).to_dict(orient="records"),
        "bottom_5_pitchers": pitcher_stats_filtered[["pitcher", "n_swings", "whiff_skill"]].tail(5).to_dict(orient="records"),
    },
}

with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"Saved: {SUMMARY_PATH}")

# %%
print("\n" + "=" * 60)
print("PLAYER SKILL EVALUATION COMPLETE")
print("=" * 60)
print(f"\nOutputs:")
print(f"  - {BATTER_SKILL_PATH}")
print(f"  - {PITCHER_SKILL_PATH}")
print(f"  - {SCATTER_PATH}")
print(f"  - {LEADERBOARD_PATH}")
print(f"  - {SUMMARY_PATH}")
print("\nNext steps:")
print("  - Look up player names using MLB API or pybaseball")
print("  - Add zone breakdowns (in-zone vs out-of-zone skill)")
print("  - Compare year-over-year skill consistency")
# %%
