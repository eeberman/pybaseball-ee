# %%
"""
04_build_model_datasets.py

Builds final model-ready datasets from feature-engineered data.
Imputes optional features, drops HBP, creates step1 (swing/take) and step2 (whiff/contact) datasets.

Input: data/processed/statcast_{dates}_features.parquet
Output:
  - data/processed/statcast_model_base.parquet
  - data/processed/statcast_step1_swing_take.parquet
  - data/processed/statcast_step2_whiff_contact.parquet
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", None)

# %%
START_DT = "2023-04-01"
END_DT = "2024-10-01"

IN_PATH = Path("data/processed") / f"statcast_{START_DT}_{END_DT}_features.parquet"

BASE_PATH = Path("data/processed") / "statcast_model_base.parquet"
STEP1_PATH = Path("data/processed") / "statcast_step1_swing_take.parquet"
STEP2_PATH = Path("data/processed") / "statcast_step2_whiff_contact.parquet"
STEP2_FILTERED_PATH = Path("data/processed") / "statcast_step2_whiff_contact_filtered.parquet"

ART_DIR = Path("artifacts/datasets")
ART_DIR.mkdir(parents=True, exist_ok=True)

DATASET_SPEC_PATH = ART_DIR / "dataset_spec.json"

# %%
df = pd.read_parquet(IN_PATH)
print("Loaded:", IN_PATH, df.shape)
input_rows = len(df)

# %%
# === INPUT VALIDATION ===
print("\n=== INPUT VALIDATION ===")

REQUIRED_COLS = ["target_raw", "y_swing", "y_whiff", "split"]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
print(f"Required columns present: {REQUIRED_COLS}")

# Split distribution
split_counts_in = df["split"].value_counts()
print(f"Split distribution (input): {split_counts_in.to_dict()}")

# %%
TARGET_COLS = ["target_raw", "y_swing", "y_whiff"]

# Columns that must have non-null values for modeling
CORE_DROP_COLS = [
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
    "batting_score_diff",
    "score_bucket",
    "zone_height",
    "plate_z_rel",
]

CORE_DROP_COLS = [c for c in CORE_DROP_COLS if c in df.columns]

rows_before_core_drop = len(df)
df2 = df.dropna(subset=CORE_DROP_COLS).copy()
print(f"\n=== ROW FILTERING AUDIT ===")
print(f"After dropping CORE nulls: {len(df2)} rows ({len(df2)/rows_before_core_drop*100:.1f}%)")

# %%
# Optional impute cols (numeric only)
OPTIONAL_IMPUTE = [
    "release_extension",
    "release_pos_y",
    "spin_axis",
    "spin_axis_sin",
    "spin_axis_cos",
    "vx0", "vy0", "vz0",
    "ax", "ay", "az",
    # Cluster physical statistics (may have nulls for unseen clusters in val/test)
    "cluster_mean_velocity",
    "cluster_mean_eff_velocity",
    "cluster_mean_spin",
    "cluster_mean_pfx_x",
    "cluster_mean_pfx_z",
    "cluster_mean_extension",
    # Trajectory / deception features
    "time_to_plate",
    "late_break_z",
    "late_break_x",
    "approach_angle_z",
    "approach_angle_x",
    "accel_magnitude",
    # Deviation from cluster mean
    "velocity_vs_cluster",
    "spin_vs_cluster",
    "pfx_z_vs_cluster",
]

OPTIONAL_IMPUTE = [c for c in OPTIONAL_IMPUTE if c in df2.columns]

print(f"\nImputing {len(OPTIONAL_IMPUTE)} optional columns with median...")
for c in OPTIONAL_IMPUTE:
    df2[c] = pd.to_numeric(df2[c], errors="coerce")
    med = df2[c].median()
    null_count = df2[c].isna().sum()
    df2[c] = df2[c].fillna(med).astype("float32")
    if null_count > 0:
        print(f"  {c}: imputed {null_count} nulls with median={med:.3f}")

# %%
# Drop HBP (neither swing nor take)
rows_before_hbp = len(df2)
df2 = df2[df2["target_raw"] != "hit_by_pitch"].copy()
print(f"After dropping HBP: {len(df2)} rows ({len(df2)/rows_before_hbp*100:.1f}%)")

# %%
# Final null check
null_counts = df2.isna().sum().sort_values(ascending=False)
null_counts = null_counts[null_counts > 0]
if len(null_counts) == 0:
    print("\nNull check: no missing values in model base")
else:
    print("\nNull check: columns with missing values in model base")
    print(null_counts.head(50))
    print((null_counts / len(df2)).head(50))

# %%
# Build datasets
df_model_base = df2.copy()

# Step 1: All pitches with y_swing defined (swings and takes)
df_step1 = df_model_base.dropna(subset=["y_swing"]).copy()
df_step1["y_swing"] = df_step1["y_swing"].astype(int)

# Step 2: Swings only with y_whiff defined
df_step2 = df_model_base.dropna(subset=["y_whiff"]).copy()
df_step2["y_whiff"] = df_step2["y_whiff"].astype(int)

# %%
# === FILTERED STEP 2: EXCLUDE OUT-OF-ZONE BREAKING BALLS ===
# These pitches have ~50% whiff rate (coin flip) and add noise to the model

# Breaking ball types (sliders, curveballs, sweepers, etc.)
BREAKING_TYPES = ["SL", "CU", "ST", "KC", "SV", "CB"]

print("\n=== CREATING FILTERED STEP 2 DATASET ===")

if "pitch_type_mode" in df_step2.columns and "in_zone" in df_step2.columns:
    # Identify chaotic pitches: out-of-zone breaking balls
    is_out_of_zone = df_step2["in_zone"] == 0
    is_breaking_ball = df_step2["pitch_type_mode"].isin(BREAKING_TYPES)
    is_chaotic = is_out_of_zone & is_breaking_ball

    # Create filtered dataset
    df_step2_filtered = df_step2[~is_chaotic].copy()

    print(f"Total Step 2 rows: {len(df_step2):,}")
    print(f"Out-of-zone breaking balls removed: {is_chaotic.sum():,} ({is_chaotic.mean()*100:.1f}%)")
    print(f"Filtered Step 2 rows: {len(df_step2_filtered):,}")

    # Compare whiff rates
    whiff_rate_full = df_step2["y_whiff"].mean()
    whiff_rate_filtered = df_step2_filtered["y_whiff"].mean()
    whiff_rate_chaotic = df_step2.loc[is_chaotic, "y_whiff"].mean() if is_chaotic.sum() > 0 else 0

    print(f"\nWhiff rates:")
    print(f"  Full dataset: {whiff_rate_full*100:.1f}%")
    print(f"  Filtered dataset: {whiff_rate_filtered*100:.1f}%")
    print(f"  Removed (chaotic): {whiff_rate_chaotic*100:.1f}%")

    # Split distribution for filtered
    print(f"\nFiltered dataset split distribution: {df_step2_filtered['split'].value_counts().to_dict()}")
else:
    print("WARNING: pitch_type_mode or in_zone not available, skipping filtered dataset")
    df_step2_filtered = None

# %%
# === STEP 1 -> STEP 2 TRANSITION VALIDATION ===
print("\n=== STEP 1 -> STEP 2 TRANSITION VALIDATION ===")

print(f"\nModel base: {len(df_model_base)} rows")

print(f"\nStep 1 dataset (all pitches with y_swing):")
print(f"  Rows: {len(df_step1)}")
print(f"  y_swing distribution: {df_step1['y_swing'].value_counts(normalize=True).to_dict()}")
print(f"  Split distribution: {df_step1['split'].value_counts().to_dict()}")

print(f"\nStep 2 dataset (swings only with y_whiff):")
print(f"  Rows: {len(df_step2)}")
print(f"  y_whiff distribution: {df_step2['y_whiff'].value_counts(normalize=True).to_dict()}")
print(f"  Split distribution: {df_step2['split'].value_counts().to_dict()}")

# %%
# CRITICAL CHECK: y_whiff nulls should equal non-swings
takes_count = (df_model_base["y_swing"] == 0).sum()
non_swing_count = (df_model_base["y_swing"] != 1).sum()
whiff_nulls = df_model_base["y_whiff"].isna().sum()

print(f"\n=== Y_WHIFF NULL ALIGNMENT CHECK ===")
print(f"Takes (y_swing=0): {takes_count}")
print(f"Non-swings (y_swing!=1): {non_swing_count}")
print(f"y_whiff nulls: {whiff_nulls}")

# Non-swings include takes AND rows with null y_swing (shouldn't be any after HBP drop)
y_swing_nulls = df_model_base["y_swing"].isna().sum()
print(f"y_swing nulls: {y_swing_nulls}")

if y_swing_nulls == 0:
    # After HBP drop, all remaining rows should have y_swing defined
    assert takes_count == whiff_nulls, f"y_whiff null mismatch: {whiff_nulls} nulls vs {takes_count} takes"
    print("PASS: y_whiff is null exactly for takes")
else:
    print(f"WARNING: {y_swing_nulls} rows have null y_swing (unexpected after HBP drop)")
    # Check if whiff nulls = takes + y_swing nulls
    expected_nulls = takes_count + y_swing_nulls
    assert whiff_nulls == expected_nulls, f"y_whiff null mismatch: {whiff_nulls} vs expected {expected_nulls}"
    print(f"PASS: y_whiff nulls ({whiff_nulls}) = takes ({takes_count}) + y_swing nulls ({y_swing_nulls})")

# %%
# Step 2 should be subset of Step 1 swings
step1_swings = (df_step1["y_swing"] == 1).sum()
print(f"\nStep 1 swings: {step1_swings}")
print(f"Step 2 rows: {len(df_step2)}")
assert len(df_step2) == step1_swings, f"Step 2 size mismatch: {len(df_step2)} vs {step1_swings} Step 1 swings"
print("PASS: Step 2 size equals Step 1 swing count")

# %%
# === SPLIT PRESERVATION CHECK ===
print("\n=== SPLIT PRESERVATION CHECK ===")

for split_name in ["train", "val", "test"]:
    base_count = (df_model_base["split"] == split_name).sum()
    step1_count = (df_step1["split"] == split_name).sum()
    step2_count = (df_step2["split"] == split_name).sum()
    print(f"{split_name}: base={base_count}, step1={step1_count}, step2={step2_count}")

# %%
# === OUTPUT VALIDATION ===
print("\n=== OUTPUT VALIDATION ===")
print(f"Input rows: {input_rows}")
print(f"Model base rows: {len(df_model_base)} ({len(df_model_base)/input_rows*100:.1f}%)")
print(f"Step 1 rows: {len(df_step1)} ({len(df_step1)/input_rows*100:.1f}%)")
print(f"Step 2 rows: {len(df_step2)} ({len(df_step2)/input_rows*100:.1f}%)")

# %%
# Save
BASE_PATH.parent.mkdir(parents=True, exist_ok=True)

df_model_base.to_parquet(BASE_PATH, index=False)
df_step1.to_parquet(STEP1_PATH, index=False)
df_step2.to_parquet(STEP2_PATH, index=False)

print("\nSaved:")
print(f"  - {BASE_PATH}")
print(f"  - {STEP1_PATH}")
print(f"  - {STEP2_PATH}")

# Save filtered dataset if created
if df_step2_filtered is not None:
    df_step2_filtered.to_parquet(STEP2_FILTERED_PATH, index=False)
    print(f"  - {STEP2_FILTERED_PATH}")

# %%
# Save dataset spec
spec = {
    "input_path": str(IN_PATH),
    "output_paths": {
        "model_base": str(BASE_PATH),
        "step1": str(STEP1_PATH),
        "step2": str(STEP2_PATH),
        "step2_filtered": str(STEP2_FILTERED_PATH) if df_step2_filtered is not None else None,
    },
    "row_counts": {
        "input": input_rows,
        "model_base": len(df_model_base),
        "step1": len(df_step1),
        "step2": len(df_step2),
        "step2_filtered": len(df_step2_filtered) if df_step2_filtered is not None else None,
    },
    "step1_y_swing_distribution": df_step1["y_swing"].value_counts(normalize=True).to_dict(),
    "step2_y_whiff_distribution": df_step2["y_whiff"].value_counts(normalize=True).to_dict(),
    "step2_filtered_y_whiff_distribution": df_step2_filtered["y_whiff"].value_counts(normalize=True).to_dict() if df_step2_filtered is not None else None,
    "step1_split_counts": df_step1["split"].value_counts().to_dict(),
    "step2_split_counts": df_step2["split"].value_counts().to_dict(),
    "step2_filtered_split_counts": df_step2_filtered["split"].value_counts().to_dict() if df_step2_filtered is not None else None,
    "columns": {
        "model_base": df_model_base.columns.tolist(),
        "step1": df_step1.columns.tolist(),
        "step2": df_step2.columns.tolist(),
    },
    "filtering": {
        "breaking_types_excluded": BREAKING_TYPES,
        "chaotic_rows_removed": int(is_chaotic.sum()) if df_step2_filtered is not None else None,
    },
}

with open(DATASET_SPEC_PATH, "w") as f:
    json.dump(spec, f, indent=2)

print(f"\nSaved dataset spec: {DATASET_SPEC_PATH}")
#%%