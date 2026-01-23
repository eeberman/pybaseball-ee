# %% 
# Imports

from pybaseball import statcast, cache
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", None)
cache.enable()


# %% Pull Data

START_DT = "2023-04-01"
END_DT   = "2023-10-01"

RAW_PATH = Path("data/raw") / f"statcast_{START_DT}_{END_DT}.parquet"

if RAW_PATH.exists():
    df_raw = pd.read_parquet(RAW_PATH)
    print("Loaded cached raw:", RAW_PATH, df_raw.shape)
else:
    df_raw = statcast(START_DT, END_DT).copy()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_parquet(RAW_PATH, index=False)
    print("Pulled + saved raw:", RAW_PATH, df_raw.shape)



#%%
print(RAW_PATH)


#%%
print("Raw shape:", df_raw.shape)

df_raw["game_date"] = pd.to_datetime(df_raw["game_date"], errors="coerce")
df_raw = df_raw.dropna(subset=["game_date"]).copy()
print("Share after dropping NA game_date:", df_raw.shape)

#%% Column list
print(df_raw.columns.tolist())
#%% Clean target labels

df = df_raw.copy()

# Drop automatic_ball
df = df[df["description"].ne("automatic_ball")].copy()

# Consolidate to your canonical raw outcomes first
df["target_raw"] = df["description"].replace({
    "swinging_strike": "any_swinging_strike",
    "swinging_strike_blocked": "any_swinging_strike",
    "ball": "any_ball",
    "blocked_ball": "any_ball",
})

# Keep a clean, realistic set (no bunts / pitchout etc.)
TARGETS_KEEP = [
    "any_ball",
    "called_strike",
    "any_swinging_strike",
    "foul",
    "hit_into_play",
    "hit_by_pitch",
]

df = df[df["target_raw"].isin(TARGETS_KEEP)].copy()

print("Target_raw counts:")
print(df["target_raw"].value_counts())
print("Shape:", df.shape)

# Step 1: swing vs take
SWING_CLASSES = {"any_swinging_strike", "foul", "hit_into_play"}
TAKE_CLASSES  = {"any_ball", "called_strike"} 

df["y_swing"] = np.where(df["target_raw"].isin(SWING_CLASSES), 1,
                np.where(df["target_raw"].isin(TAKE_CLASSES), 0, np.nan))

# Step 2: conditional on swing only → whiff vs contact
df["y_whiff"] = np.where(df["target_raw"] == "any_swinging_strike", 1,
                 np.where(df["target_raw"].isin({"foul", "hit_into_play"}), 0, np.nan))

print("\nStep 1 (y_swing) distribution (excluding NaNs):")
print(df["y_swing"].dropna().value_counts())

print("\nStep 2 (y_whiff) distribution (excluding NaNs):")
print(df["y_whiff"].dropna().value_counts())

#%% Raw Feature collection

REQ_COLS = REQ_COLS = [
    # physics / location
    "effective_speed", "release_spin_rate", "pfx_x", "pfx_z",
    "plate_x", "plate_z",

    # handedness
    "p_throws", "stand",

    # strike zone bounds (player-specific)
    "sz_bot", "sz_top",

    # count + game state
    "balls", "strikes",
    "inning_topbot", "home_score", "away_score",
    "outs_when_up",

    # baserunners (IDs, but used as presence flags)
    "on_1b", "on_2b", "on_3b",

    # release position
    "release_pos_x", "release_pos_z",

    # time split
    "game_date",
]
missing = [c for c in REQ_COLS if c not in df.columns]
if missing:
    print("Missing required columns:", missing)
    raise ValueError(f"Missing required columns: {missing}")
else:
    print("All required columns present.")
df = df.dropna(subset = ["plate_x","plate_z","p_throws","stand","balls","strikes",
    "release_pos_x","release_pos_z","pfx_x","pfx_z","sz_bot","sz_top"]).copy()

df = df.dropna(subset=["sz_bot", "sz_top"]).copy()

print("Shape after dropping required nulls:", df.shape)

#%% Batter relative feature engineering

# plate_x and release_pos_x need to be flipped for lefties
# We will do this by making the values relative to the batter
# Inside to the batter postive, outside will be negative
# We will accomplish this by multiplying by -1 for righties and 1 for lefties

# %% Cell B — Finish feature engineering: batter-relative x, strike-zone, x/z out-of-zone splits, count_state

df_fe = df.copy()

# Ensure required basics exist
df_fe["game_date"] = pd.to_datetime(df_fe["game_date"], errors="coerce")
df_fe = df_fe.dropna(subset=["game_date", "plate_x", "plate_z", "stand", "p_throws", "balls", "strikes"]).copy()

# Drop zone bounds nulls (you saw ~271)
df_fe = df_fe.dropna(subset=["sz_bot", "sz_top"]).copy()

# Batter-relative sign (inside-to-batter positive)
stand_sign = df_fe["stand"].map({"R": -1, "L": 1})
df_fe["plate_x_batter"] = df_fe["plate_x"] * stand_sign


# Batter-relative pitching metrics (x flips, z does not)
throw_sign = df_fe["p_throws"].map({"R": -1, "L": 1})
df_fe["pfx_x_norm"] = df_fe["pfx_x"] * throw_sign

df_fe["release_pos_x_batter"] = df_fe["release_pos_x"] * stand_sign
df_fe["release_pos_z_batter"] = df_fe["release_pos_z"]

# Handedness matchup
df_fe["same_side"] = (df_fe["stand"] == df_fe["p_throws"]).astype("int8")

# Count state categorical (e.g., "2-1")
df_fe["count_state"] = df_fe["balls"].astype(int).astype(str) + "-" + df_fe["strikes"].astype(int).astype(str)
df_fe["two_strikes"] = (df_fe["strikes"] == 2).astype("int8")

# In-zone and zone distance splits
ZONE_HALF_WIDTH_FT = 0.83

df_fe["in_zone"] = (
    df_fe["plate_x"].abs().le(ZONE_HALF_WIDTH_FT) &
    df_fe["plate_z"].ge(df_fe["sz_bot"]) &
    df_fe["plate_z"].le(df_fe["sz_top"])
).astype("int8")

# Horizontal distance outside zone (magnitude)
df_fe["x_out_mag"] = (df_fe["plate_x"].abs() - ZONE_HALF_WIDTH_FT).clip(lower=0)

# Horizontal signed direction OUTSIDE zone
# + means "outside to batter", - means "inside to batter"
df_fe["x_out_signed_batter"] = (
    np.sign(df_fe["plate_x_batter"]) * df_fe["x_out_mag"]
)

# Vertical distances outside zone (magnitude split into above vs below)
df_fe["z_above_mag"] = (df_fe["plate_z"] - df_fe["sz_top"]).clip(lower=0)
df_fe["z_below_mag"] = (df_fe["sz_bot"] - df_fe["plate_z"]).clip(lower=0)

# One signed vertical out-of-zone number:
# + = above zone, - = below zone, 0 = inside vertical bounds
df_fe["z_out_signed"] = df_fe["z_above_mag"] - df_fe["z_below_mag"]

# Total Euclidean distance outside zone (still useful)
df_fe["zone_out_dist"] = np.sqrt(df_fe["x_out_mag"]**2 + np.maximum(df_fe["z_above_mag"], df_fe["z_below_mag"])**2)

print("Shape:", df_fe.shape)
display(df_fe[[
    "plate_x","plate_z","stand","plate_x_batter",
    "in_zone","x_out_mag","x_out_signed_batter","z_out_signed",
    "count_state","same_side"
]].head())

# %% Quick EDA: does distance + count_state create anything meaningful?

eda = df_fe[df_fe["target_raw"].isin(["any_ball", "any_swinging_strike"])].copy()
eda["is_swstr"] = (eda["target_raw"] == "any_swinging_strike").astype("int8")

# Bin horizontal and vertical OUTSIDE-ZONE splits (batter view)
# X binning (inside vs outside relative to batter)
EPS = 1e-6

x_bins = [
    -np.inf, -0.60, -0.35, -0.20, -0.10, -0.05,
    -EPS, EPS,   # <- zone bin around 0
    0.05, 0.10, 0.20, 0.35, 0.60, np.inf
]

x_labels = [
    "in_0.60+",
    "in_0.35-0.60",
    "in_0.20-0.35",
    "in_0.10-0.20",
    "in_0.05-0.10",
    "in_0-0.05",
    "zone",               
    "out_0-0.05",
    "out_0.05-0.10",
    "out_0.10-0.20",
    "out_0.20-0.35",
    "out_0.35-0.60",
    "out_0.60+",
]

eda["x_out_bin"] = pd.cut(
    eda["x_out_signed_batter"],
    bins=x_bins,
    labels=x_labels,
    include_lowest=True
)



z_bins = [
    -np.inf, -1.0, -0.60, -0.35, -0.20, -0.10,
    -EPS, EPS,     # <- zone bin around 0
    0.10, 0.20, 0.35, 0.60, 1.0, np.inf
]

z_labels = [
    "below_1.0+",
    "below_0.60-1.0",
    "below_0.35-0.60",
    "below_0.20-0.35",
    "below_0.10-0.20",
    "below_0-0.10",
    "zone",               
    "above_0-0.10",
    "above_0.10-0.20",
    "above_0.20-0.35",
    "above_0.35-0.60",
    "above_0.60-1.0",
    "above_1.0+",
]

eda["z_out_bin"] = pd.cut(
    eda["z_out_signed"],
    bins=z_bins,
    labels=z_labels,
    include_lowest=True
)


# Table: swstr rate by count_state x x_out_bin (filter small n)
tab = (
    eda.groupby(["count_state", "x_out_bin"], observed=True)
       .agg(n=("is_swstr","size"), swstr_rate=("is_swstr","mean"))
       .reset_index()
)

MIN_N = 150
tab2 = tab[tab["n"] >= MIN_N].copy()

pivot = tab2.pivot_table(index="count_state", columns="x_out_bin", values="swstr_rate")
print("SWSTR rate by count_state x batter-relative horizontal out-of-zone bin (n>=150)")
display(pivot)

# Quick: overall swstr rate by z_out_bin
tabz = (
    eda.groupby(["count_state", "z_out_bin"], observed=True)
       .agg(n=("is_swstr","size"), swstr_rate=("is_swstr","mean"))
       .reset_index()
)
pivot2 = tabz.pivot_table(index="count_state", columns="z_out_bin", values="swstr_rate")
print("\nSWSTR rate by vertical out-of-zone bin (split by count)")
display(pivot2)

#%% Runner presence
df_fe["runner_on_1b"] = df_fe["on_1b"].notna().astype("int8")
df_fe["runner_on_2b"] = df_fe["on_2b"].notna().astype("int8")
df_fe["runner_on_3b"] = df_fe["on_3b"].notna().astype("int8")

df_fe["any_runner_on"] = (
    (df_fe["runner_on_1b"] | df_fe["runner_on_2b"] | df_fe["runner_on_3b"])
).astype("int8")

df_fe["risp"] = ((df_fe["runner_on_2b"] | df_fe["runner_on_3b"])).astype("int8")
df_fe["bases_loaded"] = (
    (df_fe["runner_on_1b"] & df_fe["runner_on_2b"] & df_fe["runner_on_3b"])
).astype("int8")
#%% Score differential (batter relative, hitting team - fielding team)
df_fe["batting_score_diff"] = np.where(
    df_fe["inning_topbot"].eq("Top"),
    df_fe["away_score"] - df_fe["home_score"],
    df_fe["home_score"] - df_fe["away_score"]
)

# Bucket
bins = [-np.inf, -5, -3, -1, 0, 2, 4, np.inf]
labels = [
    "trail_5plus",
    "trail_4-3",
    "trail_2-1",
    "tied",
    "lead_1-2",
    "lead_3-4",
    "lead_5plus",
]
df_fe["score_bucket"] = pd.cut(
    df_fe["batting_score_diff"], bins=bins, labels=labels, include_lowest=True
)

#%% Pairing down to only modeling features + targets

# targets
TARGET_COLS = [
    "target_raw",
    "y_swing",
    "y_whiff",
]

# raw features you decided to keep (no engineering needed)
RAW_FEATURE_COLS = [
    "game_date",

    # physics/location
    "effective_speed",
    "release_spin_rate",
    "pfx_z",
    "plate_z",

    # handedness inputs (needed for interpretation + engineered vars)
    "p_throws",
    "stand",

    # strike zone bounds (needed for in_zone + out-of-zone features)
    "sz_bot",
    "sz_top",

    # count / state
    "balls",
    "strikes",
    "outs_when_up",
]

# engineered features
ENGINEERED_COLS = [
    # batter-relative geometry
    "plate_x_batter",
    "release_pos_x_batter",
    "release_pos_z_batter",

    # handedness normalization / matchup
    "same_side",
    "pfx_x_norm",

    # count features
    "count_state",
    "two_strikes",

    # zone / distance
    "in_zone",
    "x_out_mag",
    "x_out_signed_batter",
    "z_out_signed",
    "zone_out_dist",

    # baserunner presence flags
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "any_runner_on",
    "risp",
    "bases_loaded",

    # score context
    "batting_score_diff",
    "score_bucket",
]


FINAL_COLS = TARGET_COLS + RAW_FEATURE_COLS + ENGINEERED_COLS

# sanity check: show anything missing before selecting
missing_final = [c for c in FINAL_COLS if c not in df_fe.columns]
if missing_final:
    print("Missing FINAL_COLS (check upstream feature engineering):")
    print(missing_final)

# subset to only final cols
df_fe_final = df_fe[FINAL_COLS].copy()

print("df_fe_final shape:", df_fe_final.shape)
print("Columns kept:", df_fe_final.columns.tolist())
print(df_fe_final.head())
 
 #%% Null checks 
null_counts = df_fe_final.isna().sum().sort_values(ascending=False)
null_counts = null_counts[null_counts > 0]

if len(null_counts) == 0:
    print("\n Null check: no missing values in df_fe_final")
else:
    print("\n Null check: columns with missing values")
    display(null_counts.to_frame("null_count"))
    display((null_counts / len(df_fe_final)).to_frame("null_share"))

#%% Drop meaningless nulls
FEATURE_COLS = [c for c in df_fe_final.columns if c not in ["target_raw", "y_swing", "y_whiff"]]

# Feature nulls: 
df_model_base = df_fe_final.dropna(subset=FEATURE_COLS).copy()
print("After dropping feature nulls:", df_model_base.shape)

# Drop HBPs
df_model_base = df_model_base[df_model_base["target_raw"] != "hit_by_pitch"].copy()
print("After dropping HBPs:", df_model_base.shape)

# recheck nulls
null_counts2 = df_model_base.isna().sum().sort_values(ascending=False)
null_counts2 = null_counts2[null_counts2 > 0]

if len(null_counts2) == 0:
    print("\n Null check: no missing values in df_model_base")
else:
    print("\n Null check: columns with missing values in df_model_base")
    display(null_counts2.to_frame("null_count"))
    display((null_counts2 / len(df_model_base)).to_frame("null_share"))

#%% build test datasets
# Swing or take dataset
df_step1 = df_model_base.dropna(subset=["y_swing"]).copy()
df_step1["y_swing"] = df_step1["y_swing"].astype(int)

print("\nStep 1 shape:", df_step1.shape)
print("Step 1 label distribution:")
print(df_step1["y_swing"].value_counts(normalize=True))

# Whiff or contact dataset (NA's have been dropped to account for non-swings)
df_step2 = df_model_base.dropna(subset=["y_whiff"]).copy()
df_step2["y_whiff"] = df_step2["y_whiff"].astype(int)

print("\nStep 2 shape:", df_step2.shape)
print("Step 2 label distribution:")
print(df_step2["y_whiff"].value_counts(normalize=True))

#%%  Processed data saving for future reference
 
BASE_PATH  = Path("data/processed") / "statcast_2023_model_base.parquet"
STEP1_PATH = Path("data/processed") / "statcast_2023_step1_swing_take.parquet"
STEP2_PATH = Path("data/processed") / "statcast_2023_step2_whiff_contact.parquet"

df_model_base.to_parquet(BASE_PATH, index=False)
df_step1.to_parquet(STEP1_PATH, index=False)
df_step2.to_parquet(STEP2_PATH, index=False)

print("\n Saved:")
print(" -", BASE_PATH)
print(" -", STEP1_PATH)
print(" -", STEP2_PATH)

#%% Final checks
print("Model_base:\n", df_model_base.shape, "\n", df_model_base.head())
print("Model pt 1:\n", df_step1.shape, "\n", df_step1.head())
print("Model pt 2\n:", df_step2.shape, "\n", df_step2.head())



#%% Feature spec to make sure this is determinsitic 
FEATURE_SPEC_PATH = Path("data/processed") / "feature_spec.txt"

nulls_final = df_fe_final.isna().sum().sort_values(ascending=False)
nulls_final = nulls_final[nulls_final > 0]

with open(FEATURE_SPEC_PATH, "w") as f:
    f.write("RAW TARGET COUNTS (target_raw)\n")
    f.write(df_fe_final["target_raw"].value_counts().to_string())
    f.write("\n\n")

    f.write(" FINAL COLUMNS (df_fe_final)\n")
    f.write("\n".join(df_fe_final.columns.tolist()))
    f.write("\n\n")

    f.write(" NULL AUDIT (df_fe_final, nonzero only)\n")
    if len(nulls_final) == 0:
        f.write("No nulls in df_fe_final\n\n")
    else:
        f.write(nulls_final.to_string())
        f.write("\n\n")

    f.write(" SHAPES\n")
    f.write(f"df_fe_final:    {df_fe_final.shape}\n")
    f.write(f"df_model_base:  {df_model_base.shape}\n")
    f.write(f"df_step1:       {df_step1.shape}\n")
    f.write(f"df_step2:       {df_step2.shape}\n")
# %%
