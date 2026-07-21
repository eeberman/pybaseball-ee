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
END_DT = "2023-10-01"

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
# Work directly on df_raw to avoid memory copies
df = df_raw  # Reference, not copy
del df_raw  # Free the reference name

df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
df = df.dropna(subset=["game_date"])
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
])
print(f"After core null drop: {len(df)} rows ({len(df)/rows_before_core_drop*100:.1f}%)")

# %%
# Batter-relative engineering (work in-place on df)

df_fe = df  # Reference, not copy
del df  # Free the reference name

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
# === CLUSTER PHYSICAL STATISTICS ===
# Encode clusters by their physical characteristics (computed from training data only)
# This captures "what kind of pitch is this" without using outcome data

if "pitch_cluster_name" in df_fe.columns and df_fe["pitch_cluster_name"].notna().any():
    print("\n=== COMPUTING CLUSTER PHYSICAL STATISTICS ===")

    # Only compute stats from training data to avoid leakage
    train_mask = df_fe["split"] == "train"
    train_data = df_fe[train_mask]

    # Physical columns to aggregate per cluster
    CLUSTER_PHYS_COLS = [
        ("release_speed", "cluster_mean_velocity"),
        ("effective_speed", "cluster_mean_eff_velocity"),
        ("release_spin_rate", "cluster_mean_spin"),
        ("pfx_x", "cluster_mean_pfx_x"),
        ("pfx_z", "cluster_mean_pfx_z"),
        ("release_extension", "cluster_mean_extension"),
    ]

    # Compute cluster means from training data
    cluster_stats = {}
    for src_col, dst_col in CLUSTER_PHYS_COLS:
        if src_col in train_data.columns:
            stat = train_data.groupby("pitch_cluster_name")[src_col].mean()
            cluster_stats[dst_col] = stat
            print(f"  Computed {dst_col} for {len(stat)} clusters")

    # Map cluster stats back to all rows (train, val, test)
    for dst_col, stat_series in cluster_stats.items():
        df_fe[dst_col] = df_fe["pitch_cluster_name"].map(stat_series).astype("float32")

    # Report coverage
    n_mapped = df_fe["cluster_mean_velocity"].notna().sum() if "cluster_mean_velocity" in df_fe.columns else 0
    print(f"  Cluster stats mapped to {n_mapped} rows ({n_mapped/len(df_fe)*100:.1f}%)")

    # Clusters in val/test but not in train will have NaN - report these
    if "cluster_mean_velocity" in df_fe.columns:
        unseen_mask = ~train_mask & df_fe["pitch_cluster_name"].notna() & df_fe["cluster_mean_velocity"].isna()
        n_unseen = unseen_mask.sum()
        if n_unseen > 0:
            print(f"  WARNING: {n_unseen} rows have clusters not seen in training")

    del train_data, cluster_stats

# %%
# === TRAJECTORY / DECEPTION FEATURES ===
# Features capturing "how late" movement happens and approach angles

print("\n=== COMPUTING TRAJECTORY / DECEPTION FEATURES ===")

# Time to plate (approximate, in seconds) from release
# release_pos_y is typically ~55 ft, vy0 is negative (toward home plate)
if "release_pos_y" in df_fe.columns and "vy0" in df_fe.columns:
    df_fe["time_to_plate"] = (df_fe["release_pos_y"] / df_fe["vy0"].abs()).astype("float32")
    print(f"  time_to_plate: computed for {df_fe['time_to_plate'].notna().sum()} rows")

    # Late break intensity (movement per second) - higher = more deceptive
    df_fe["late_break_z"] = (df_fe["pfx_z"] / df_fe["time_to_plate"]).astype("float32")
    df_fe["late_break_x"] = (df_fe["pfx_x_norm"] / df_fe["time_to_plate"]).astype("float32")
    print(f"  late_break_z/x: computed")

# Approach angles (trajectory slope as seen by batter)
if "release_pos_y" in df_fe.columns:
    # Vertical approach angle - "flat" (high angle) vs "steep" (low angle)
    df_fe["approach_angle_z"] = (
        (df_fe["plate_z"] - df_fe["release_pos_z"]) / df_fe["release_pos_y"]
    ).astype("float32")

    # Horizontal approach angle (batter-relative)
    df_fe["approach_angle_x"] = (
        (df_fe["plate_x_batter"] - df_fe["release_pos_x_batter"]) / df_fe["release_pos_y"]
    ).astype("float32")
    print(f"  approach_angle_z/x: computed")

# Acceleration magnitude (Magnus force intensity)
if all(c in df_fe.columns for c in ["ax", "ay", "az"]):
    df_fe["accel_magnitude"] = np.sqrt(
        df_fe["ax"]**2 + df_fe["ay"]**2 + df_fe["az"]**2
    ).astype("float32")
    print(f"  accel_magnitude: computed for {df_fe['accel_magnitude'].notna().sum()} rows")

# %%
# === TUNNEL DISTANCE (pitch sequencing) ===
# Distance between this pitch and the previous pitch (same pitcher, same at-bat)
# at the "tunnel point" ~23.8 ft from the plate, where the batter must commit.
# NaN for the first pitch by a pitcher in an at-bat (no prior pitch to tunnel from);
# these NaNs are structural and deliberately NOT median-imputed downstream —
# XGBoost's native missing handling deals with them better than a fabricated median.

TUNNEL_DISTANCE_FROM_PLATE_FT = 23.8

TUNNEL_REQ_COLS = [
    "release_pos_x", "release_pos_y", "release_pos_z",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    "game_pk", "at_bat_number", "pitch_number", "pitcher",
]
if all(c in df_fe.columns for c in TUNNEL_REQ_COLS):
    print("\n=== COMPUTING TUNNEL DISTANCE ===")

    # Plain float64 (nullable Float64/pd.NA breaks boolean masks in np.where)
    phys = {
        c: pd.to_numeric(df_fe[c], errors="coerce").astype("float64")
        for c in ["release_pos_x", "release_pos_y", "release_pos_z",
                  "vx0", "vy0", "vz0", "ax", "ay", "az"]
    }

    # Time to reach the tunnel point: solve 0.5*ay*t^2 + vy0*t + dy = 0 where
    # dy = release_pos_y - tunnel_y. Quadratic (not the linear time_to_plate
    # shortcut) because the dropped ay term differs systematically by pitch
    # type, which would bias exactly the fastball/breaking-ball comparisons
    # this feature is meant to capture. Note: vy0/ay are measured at y=50ft,
    # not at release — a pre-existing approximation shared with time_to_plate
    # that largely cancels when differencing two pitches from the same pitcher.
    dy = phys["release_pos_y"] - TUNNEL_DISTANCE_FROM_PLATE_FT
    a = 0.5 * phys["ay"]
    b = phys["vy0"]  # negative (toward plate)
    disc = b**2 - 4.0 * a * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t_quad = (-b - np.sqrt(disc)) / (2.0 * a)
    t_linear = dy / phys["vy0"].abs()
    t_tunnel = pd.Series(
        np.where((disc > 0) & (t_quad > 0), t_quad, t_linear),
        index=df_fe.index,
    )

    x_tunnel = phys["release_pos_x"] + phys["vx0"] * t_tunnel + 0.5 * phys["ax"] * t_tunnel**2
    z_tunnel = phys["release_pos_z"] + phys["vz0"] * t_tunnel + 0.5 * phys["az"] * t_tunnel**2

    # Previous pitch's tunnel position (same pitcher within the at-bat) via a
    # sorted copy, mapped back by index so df_fe's row order is untouched.
    tun = pd.DataFrame({
        "game_pk": df_fe["game_pk"],
        "at_bat_number": df_fe["at_bat_number"],
        "pitcher": df_fe["pitcher"],
        "pitch_number": df_fe["pitch_number"],
        "x_tunnel": x_tunnel,
        "z_tunnel": z_tunnel,
    })
    tun_sorted = tun.sort_values(["game_pk", "at_bat_number", "pitcher", "pitch_number"])
    grp = tun_sorted.groupby(["game_pk", "at_bat_number", "pitcher"], sort=False)
    x_prev = grp["x_tunnel"].shift(1).reindex(df_fe.index)
    z_prev = grp["z_tunnel"].shift(1).reindex(df_fe.index)

    df_fe["tunnel_distance"] = np.sqrt(
        (x_tunnel - x_prev)**2 + (z_tunnel - z_prev)**2
    ).astype("float32")
    df_fe["is_first_pitch_for_pitcher_in_ab"] = x_prev.isna().astype("int8")

    n_valid = df_fe["tunnel_distance"].notna().sum()
    n_first = df_fe["is_first_pitch_for_pitcher_in_ab"].sum()
    print(f"  tunnel_distance: computed for {n_valid} rows ({n_valid/len(df_fe)*100:.1f}%)")
    print(f"  first pitch (no prior pitch) rows: {n_first} ({n_first/len(df_fe)*100:.1f}%)")
    print(f"  tunnel_distance median: {df_fe['tunnel_distance'].median():.3f} ft")

    del tun, tun_sorted, grp, phys
else:
    missing = [c for c in TUNNEL_REQ_COLS if c not in df_fe.columns]
    print(f"\nSkipping tunnel distance (missing columns: {missing})")

# %%
# === PREVIOUS PITCH TYPE (pitch sequencing) ===
# Type of the previous pitch by the same pitcher in the same at-bat, using the
# model's pitch-type representation (pitch_type_mode). NaN on the first pitch
# by a pitcher in an AB — one-hot encoding in 05 uses dummy_na, so "no prior
# pitch" gets its own indicator rather than a fabricated fill value.

SEQ_ID_COLS = ["game_pk", "at_bat_number", "pitch_number", "pitcher"]
if "pitch_type_mode" in df_fe.columns and all(c in df_fe.columns for c in SEQ_ID_COLS):
    print("\n=== COMPUTING PREVIOUS PITCH TYPE ===")

    seq = df_fe[SEQ_ID_COLS + ["pitch_type_mode"]].sort_values(
        ["game_pk", "at_bat_number", "pitcher", "pitch_number"]
    )
    prev_type = (
        seq.groupby(["game_pk", "at_bat_number", "pitcher"], sort=False)["pitch_type_mode"]
        .shift(1)
        .reindex(df_fe.index)
    )
    df_fe["prev_pitch_type_mode"] = prev_type
    df_fe["same_pitch_type_as_prev"] = (
        (prev_type.notna() & (df_fe["pitch_type_mode"] == prev_type))
        .fillna(False)
        .astype("int8")
    )

    n_valid = df_fe["prev_pitch_type_mode"].notna().sum()
    n_same = df_fe["same_pitch_type_as_prev"].sum()
    print(f"  prev_pitch_type_mode: computed for {n_valid} rows ({n_valid/len(df_fe)*100:.1f}%)")
    print(f"  same as previous pitch: {n_same} rows ({n_same/max(n_valid,1)*100:.1f}% of sequenced)")

    del seq, prev_type
else:
    print("\nSkipping previous pitch type (pitch_type_mode or ID columns missing)")

# %%
# === VELOCITY DIFFERENTIAL FROM PREVIOUS PITCH (pitch sequencing) ===
# release_speed minus the previous pitch's (same pitcher, same at-bat).
# Negative = slower than the prior pitch (e.g. changeup off a fastball).
# NaN on first pitch by a pitcher in an AB — structural, not imputed (same
# reasoning as tunnel_distance; XGBoost handles the NaNs natively).

if "release_speed" in df_fe.columns and all(c in df_fe.columns for c in SEQ_ID_COLS):
    print("\n=== COMPUTING VELOCITY DIFFERENTIAL ===")

    speed = pd.to_numeric(df_fe["release_speed"], errors="coerce").astype("float64")
    vseq = pd.DataFrame({
        "game_pk": df_fe["game_pk"],
        "at_bat_number": df_fe["at_bat_number"],
        "pitcher": df_fe["pitcher"],
        "pitch_number": df_fe["pitch_number"],
        "release_speed": speed,
    }).sort_values(["game_pk", "at_bat_number", "pitcher", "pitch_number"])
    prev_speed = (
        vseq.groupby(["game_pk", "at_bat_number", "pitcher"], sort=False)["release_speed"]
        .shift(1)
        .reindex(df_fe.index)
    )
    df_fe["velo_diff_from_prev"] = (speed - prev_speed).astype("float32")

    n_valid = df_fe["velo_diff_from_prev"].notna().sum()
    print(f"  velo_diff_from_prev: computed for {n_valid} rows ({n_valid/len(df_fe)*100:.1f}%)")
    print(f"  median {df_fe['velo_diff_from_prev'].median():.2f} mph, "
          f"p5 {df_fe['velo_diff_from_prev'].quantile(0.05):.2f}, "
          f"p95 {df_fe['velo_diff_from_prev'].quantile(0.95):.2f}")

    del speed, vseq, prev_speed
else:
    print("\nSkipping velocity differential (release_speed or ID columns missing)")

# %%
# Deviation from cluster mean (how unusual is this pitch for its cluster?)
if "cluster_mean_velocity" in df_fe.columns:
    df_fe["velocity_vs_cluster"] = (
        df_fe["effective_speed"] - df_fe["cluster_mean_velocity"]
    ).astype("float32")
    print(f"  velocity_vs_cluster: computed")

if "cluster_mean_spin" in df_fe.columns:
    df_fe["spin_vs_cluster"] = (
        df_fe["release_spin_rate"] - df_fe["cluster_mean_spin"]
    ).astype("float32")
    print(f"  spin_vs_cluster: computed")

if "cluster_mean_pfx_z" in df_fe.columns:
    df_fe["pfx_z_vs_cluster"] = (
        df_fe["pfx_z"] - df_fe["cluster_mean_pfx_z"]
    ).astype("float32")
    print(f"  pfx_z_vs_cluster: computed")

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
    # "game_date" is already in ID_COLS, don't duplicate
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
    # Cluster physical statistics (encoded from training data)
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
    # Pitch sequencing (NaN on first pitch by pitcher in AB — structural, not imputed)
    "tunnel_distance",
    "is_first_pitch_for_pitcher_in_ab",
    "prev_pitch_type_mode",
    "same_pitch_type_as_prev",
    "velo_diff_from_prev",
    # Deviation from cluster mean
    "velocity_vs_cluster",
    "spin_vs_cluster",
    "pfx_z_vs_cluster",
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

# Select final columns (copy needed here to free memory from unused columns)
df_fe_final = df_fe[FINAL_COLS]
del df_fe  # Free memory from full dataframe

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