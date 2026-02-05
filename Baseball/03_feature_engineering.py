# %%
"""
03_feature_engineering.py

Creates batter-relative, zone, and game state features from clustered Statcast data.

Input: data/processed/statcast_{dates}_clustered.parquet
Output: data/processed/statcast_{dates}_features.parquet
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", None)

# %%
START_DT = "2023-04-01"
END_DT = "2024-10-01"

IN_PATH = Path("data/processed") / f"statcast_{START_DT}_{END_DT}_clustered.parquet"
OUT_PATH = Path("data/processed") / f"statcast_{START_DT}_{END_DT}_features.parquet"

ART_DIR = Path("artifacts/features")
ART_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SPEC_PATH = ART_DIR / "feature_spec.json"

# %%
df_raw = pd.read_parquet(IN_PATH)
print("Loaded:", IN_PATH, df_raw.shape)
input_rows = len(df_raw)

# %%
# === INPUT VALIDATION ===
print("\n=== INPUT VALIDATION ===")

REQUIRED_COLS = ["target_raw", "y_swing", "y_whiff", "split", "game_date", "pitcher"]
missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
print(f"Required columns present: {REQUIRED_COLS}")

# Check split integrity
split_counts_in = df_raw["split"].value_counts()
print(f"Split distribution (input): {split_counts_in.to_dict()}")

# Track nulls before feature engineering
null_before = df_raw.isna().sum()

# %%
df = df_raw.copy()

df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
df = df.dropna(subset=["game_date"]).copy()
print(f"After game_date null drop: {len(df)} rows ({len(df)/input_rows*100:.1f}%)")

# %%
# Required columns for your core feature set
REQ_CORE = [
    "effective_speed", "release_spin_rate", "pfx_x", "pfx_z", "plate_x", "plate_z",
    "p_throws", "stand", "sz_bot", "sz_top",
    "balls", "strikes", "inning_topbot", "home_score", "away_score", "outs_when_up",
    "on_1b", "on_2b", "on_3b",
    "release_pos_x", "release_pos_z",
    "target_raw", "y_swing", "y_whiff",
]

missing = [c for c in REQ_CORE if c not in df.columns]
if missing:
    raise ValueError(f"Missing required core columns: {missing}")

# Hard-drop only truly required geometry/state nulls
rows_before_core_drop = len(df)
df = df.dropna(subset=[
    "plate_x", "plate_z", "p_throws", "stand", "balls", "strikes",
    "release_pos_x", "release_pos_z", "pfx_x", "pfx_z", "sz_bot", "sz_top",
]).copy()
print(f"After core null drop: {len(df)} rows ({len(df)/rows_before_core_drop*100:.1f}%)")

# %%
# Batter-relative engineering

df_fe = df.copy()

stand_sign = df_fe["stand"].map({"R": -1, "L": 1})
throw_sign = df_fe["p_throws"].map({"R": -1, "L": 1})

df_fe["plate_x_batter"] = df_fe["plate_x"] * stand_sign

df_fe["pfx_x_norm"] = df_fe["pfx_x"] * throw_sign
df_fe["release_pos_x_batter"] = df_fe["release_pos_x"] * stand_sign
df_fe["release_pos_z_batter"] = df_fe["release_pos_z"]

df_fe["same_side"] = (df_fe["stand"] == df_fe["p_throws"]).astype("int8")

df_fe["count_state"] = df_fe["balls"].astype(int).astype(str) + "-" + df_fe["strikes"].astype(int).astype(str)
df_fe["two_strikes"] = (df_fe["strikes"] == 2).astype("int8")

ZONE_HALF_WIDTH_FT = 0.83

df_fe["in_zone"] = (
    df_fe["plate_x"].abs().le(ZONE_HALF_WIDTH_FT) &
    df_fe["plate_z"].ge(df_fe["sz_bot"]) &
    df_fe["plate_z"].le(df_fe["sz_top"])
).astype("int8")

df_fe["x_out_mag"] = (df_fe["plate_x"].abs() - ZONE_HALF_WIDTH_FT).clip(lower=0)
df_fe["x_out_signed_batter"] = np.sign(df_fe["plate_x_batter"]) * df_fe["x_out_mag"]

df_fe["z_above_mag"] = (df_fe["plate_z"] - df_fe["sz_top"]).clip(lower=0)
df_fe["z_below_mag"] = (df_fe["sz_bot"] - df_fe["plate_z"]).clip(lower=0)
df_fe["z_out_signed"] = df_fe["z_above_mag"] - df_fe["z_below_mag"]

df_fe["zone_out_dist"] = np.sqrt(df_fe["x_out_mag"]**2 + np.maximum(df_fe["z_above_mag"], df_fe["z_below_mag"])**2)

# Useful zone-relative vertical placement (you already found it matters)
df_fe["zone_height"] = (df_fe["sz_top"] - df_fe["sz_bot"]).astype("float32")
df_fe["plate_z_rel"] = ((df_fe["plate_z"] - df_fe["sz_bot"]) / df_fe["zone_height"]).astype("float32")

# %%
# Runner flags

df_fe["runner_on_1b"] = df_fe["on_1b"].notna().astype("int8")
df_fe["runner_on_2b"] = df_fe["on_2b"].notna().astype("int8")
df_fe["runner_on_3b"] = df_fe["on_3b"].notna().astype("int8")

df_fe["any_runner_on"] = (df_fe["runner_on_1b"] | df_fe["runner_on_2b"] | df_fe["runner_on_3b"]).astype("int8")
df_fe["risp"] = (df_fe["runner_on_2b"] | df_fe["runner_on_3b"]).astype("int8")
df_fe["bases_loaded"] = (df_fe["runner_on_1b"] & df_fe["runner_on_2b"] & df_fe["runner_on_3b"]).astype("int8")

# %%
# Score context

df_fe["batting_score_diff"] = np.where(
    df_fe["inning_topbot"].eq("Top"),
    df_fe["away_score"] - df_fe["home_score"],
    df_fe["home_score"] - df_fe["away_score"]
)

bins = [-np.inf, -5, -3, -1, 0, 2, 4, np.inf]
labels = ["trail_5plus", "trail_4-3", "trail_2-1", "tied", "lead_1-2", "lead_3-4", "lead_5plus"]

df_fe["score_bucket"] = pd.cut(df_fe["batting_score_diff"], bins=bins, labels=labels, include_lowest=True)

# %%
# Optional new physics/flight/spin features (keep if present, do not hard-drop)

OPTIONAL_RAW = [
    "release_extension",
    "release_pos_y",
    "spin_axis",
    "vx0", "vy0", "vz0",
    "ax", "ay", "az",
    "pitch_cluster_name",
    "pitch_type_mode",
]

for c in OPTIONAL_RAW:
    if c in df_fe.columns:
        if c not in ["pitch_cluster_name", "pitch_type_mode"]:
            df_fe[c] = pd.to_numeric(df_fe[c], errors="coerce")

if "spin_axis" in df_fe.columns:
    axis = (df_fe["spin_axis"] % 360.0).astype("float32")
    df_fe["spin_axis_sin"] = np.sin(np.deg2rad(axis)).astype("float32")
    df_fe["spin_axis_cos"] = np.cos(np.deg2rad(axis)).astype("float32")

# %%
# ID columns (for downstream inference output)
ID_COLS = [
    "game_pk",
    "game_date",
    "batter",
    "pitcher",
    "at_bat_number",
    "pitch_number",
    "inning",
    "inning_topbot",
]

# Context columns (useful for inference output)
CONTEXT_COLS = [
    "balls",
    "strikes",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "home_score",
    "away_score",
    "p_throws",
    "stand",
]

# %%
# FINAL columns
TARGET_COLS = ["target_raw", "y_swing", "y_whiff"]
SPLIT_COLS = ["split"]

RAW_FEATURE_COLS = [
    "game_date",
    "effective_speed",
    "release_spin_rate",
    "pfx_z",
    "plate_z",
    "p_throws",
    "stand",
    "sz_bot",
    "sz_top",
    "balls",
    "strikes",
    "outs_when_up",
]

ENGINEERED_COLS = [
    "plate_x_batter",
    "release_pos_x_batter",
    "release_pos_z_batter",
    "same_side",
    "pfx_x_norm",
    "count_state",
    "two_strikes",
    "in_zone",
    "x_out_mag",
    "x_out_signed_batter",
    "z_out_signed",
    "zone_out_dist",
    "zone_height",
    "plate_z_rel",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "any_runner_on",
    "risp",
    "bases_loaded",
    "batting_score_diff",
    "score_bucket",
]

OPTIONAL_FINAL = [
    "release_extension",
    "release_pos_y",
    "spin_axis",
    "spin_axis_sin",
    "spin_axis_cos",
    "vx0", "vy0", "vz0",
    "ax", "ay", "az",
    "pitch_cluster_name",
    "pitch_type_mode",
]

# Include ID columns for downstream use
FINAL_COLS = (
    [c for c in ID_COLS if c in df_fe.columns] +
    SPLIT_COLS +
    TARGET_COLS +
    RAW_FEATURE_COLS +
    ENGINEERED_COLS +
    [c for c in OPTIONAL_FINAL if c in df_fe.columns]
)
FINAL_COLS = [c for c in FINAL_COLS if c in df_fe.columns]

df_fe_final = df_fe[FINAL_COLS].copy()

print(f"\ndf_fe_final shape: {df_fe_final.shape}")
print(f"Columns kept: {df_fe_final.columns.tolist()}")

# %%
# === NULL PROPAGATION TRACKING ===
print("\n=== NULL PROPAGATION TRACKING ===")

null_after = df_fe_final.isna().sum()

# Compare nulls (only for columns that exist in both)
common_cols = null_before.index.intersection(null_after.index)
null_before_common = null_before.reindex(null_after.index, fill_value=0)
null_introduced = null_after - null_before_common
null_introduced = null_introduced[null_introduced > 0]

if len(null_introduced):
    print("WARNING: Nulls introduced during feature engineering:")
    for col, count in null_introduced.items():
        pct = count / len(df_fe_final) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")
else:
    print("No additional nulls introduced during feature engineering")

# %%
# Null audit (do not drop here, just report)

null_counts = df_fe_final.isna().sum().sort_values(ascending=False)
null_counts = null_counts[null_counts > 0]

if len(null_counts) == 0:
    print("\nNull check: no missing values in df_fe_final")
else:
    print("\nNull check: columns with missing values in df_fe_final")
    print(null_counts.head(50))
    print((null_counts / len(df_fe_final)).head(50))

# %%
# === SPLIT INTEGRITY CHECK ===
print("\n=== SPLIT INTEGRITY CHECK ===")

split_counts_out = df_fe_final["split"].value_counts()
print(f"Split distribution (output): {split_counts_out.to_dict()}")

# Verify all splits still present
for split_name in ["train", "val", "test"]:
    if split_name in split_counts_in.index and split_name not in split_counts_out.index:
        print(f"WARNING: {split_name} split lost during feature engineering!")

# %%
# === OUTPUT VALIDATION ===
print("\n=== OUTPUT VALIDATION ===")
print(f"Input rows: {input_rows}")
print(f"Output rows: {len(df_fe_final)} ({len(df_fe_final)/input_rows*100:.1f}% retained)")

# Target preservation check
for target_col in TARGET_COLS:
    if target_col in df_fe_final.columns:
        non_null = df_fe_final[target_col].notna().sum()
        print(f"  {target_col} non-null: {non_null}")

# %%
# Save features dataset (single source of truth for modeling)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_fe_final.to_parquet(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")

# %%
# Save deterministic feature spec

spec = {
    "input_path": str(IN_PATH),
    "output_path": str(OUT_PATH),
    "final_cols": df_fe_final.columns.tolist(),
    "id_cols": [c for c in ID_COLS if c in df_fe_final.columns],
    "target_cols": TARGET_COLS,
    "split_cols": SPLIT_COLS,
    "raw_feature_cols": [c for c in RAW_FEATURE_COLS if c in df_fe_final.columns],
    "engineered_cols": [c for c in ENGINEERED_COLS if c in df_fe_final.columns],
    "optional_cols": [c for c in OPTIONAL_FINAL if c in df_fe_final.columns],
    "shape": list(df_fe_final.shape),
    "target_counts": df_fe_final["target_raw"].value_counts().to_dict(),
    "null_counts_nonzero": null_counts.to_dict(),
    "split_counts": split_counts_out.to_dict(),
}

with open(FEATURE_SPEC_PATH, "w") as f:
    json.dump(spec, f, indent=2)

print("Saved feature spec:", FEATURE_SPEC_PATH)
#%%