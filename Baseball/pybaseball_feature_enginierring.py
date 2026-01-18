# %% Imports

from pybaseball import statcast, cache
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", None)
cache.enable()


#%% Pull Data

START_DT = '2023-04-01'
END_DT = '2023-10-01'

df_raw = statcast(START_DT, END_DT).copy
print("Raw shape:", df_raw.shape)

df_raw["game_date"] = pd.to_datetime(df_raw["game_date"])
df_raw = df_raw.dropna(subset=["game_date"].min(), "→", df_raw["game_date"].max())
print("Share after dropping NA game_date:", df_raw.shape)

#%% Clean target labels

df = df_raw[df_raw["description"].ne("automatic_ball")].copy()
df["target"] = df["description"].replace({
    "swinging_strike": "any_swinging_strike",
    "swinging_strike_blocked": "any_swinging_strike",
    "ball": "any_ball",
    "blocked_ball": "any_ball",
})

TARGETS_V2 = [
    "any_ball",
    "called_strike",
    "any_swinging_strike",
    "foul",
    "hit_into_play",
    "hit_by_pitch",
]

df = df[df["target"].isin(TARGETS_V2)].copy()

print("Target value counts:")
print(df["target"].value_counts().reindex(TARGETS_V2))
print("Shape:", df.shape)

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
else:
    print("All required columns present.")
df = df.dropna(subset = ["plate_x", "plate_z", "p_throw", "stand", "balls", "strikes"]).copy()

df = df.dropna(subset=["sz_bot", "sz_top"]).copy()

print("Shape after dropping required nulls:", df.shape)

#%% Batter relative feature engineering

# plate_x and release_pos_x need to be flipped for lefties
# We will do this by making the values relative to the batter
# Inside to the batter postive, outside will be negative
# We will accomplish this by multiplying by -1 for righties and 1 for lefties

df_fe = df.copy()

stand_sign = df_fe["stand"].map({"R": -1, "L": 1})
df_fe["plate_x_batter"] = df_fe["plate_x"] * stand_sign

df_fe["release_pos_x_batter"] = df_fe["release_pos_x"] * stand_sign
df_fe["release_pos_z_batter"] = df_fe["release_pos_z"]  # z doesn't flip

# in_zone based on player-specific sz_top/bot + fixed half-width
ZONE_HALF_WIDTH_FT = 0.83

df_fe["in_zone"] = (
    df_fe["plate_x"].abs().le(ZONE_HALF_WIDTH_FT) &
    df_fe["plate_z"].ge(df_fe["sz_bot"]) &
    df_fe["plate_z"].le(df_fe["sz_top"])
).astype("int8")

# zone_out_dist (continuous distance outside zone; 0 if inside)
x_out = (df_fe["plate_x"].abs() - ZONE_HALF_WIDTH_FT).clip(lower=0)
z_below = (df_fe["sz_bot"] - df_fe["plate_z"]).clip(lower=0)
z_above = (df_fe["plate_z"] - df_fe["sz_top"]).clip(lower=0)
z_out = np.maximum(z_below, z_above)

df_fe["zone_out_dist"] = np.sqrt(x_out**2 + z_out**2)

print("In-zone rate:", df_fe["in_zone"].mean())
display(df_fe[["plate_x","stand","plate_x_batter","release_pos_x","release_pos_x_batter","in_zone","zone_out_dist"]].head())

# Engineer context features
# same-side matchup
df_fe["same_side"] = (df_fe["stand"] == df_fe["p_throws"]).astype("int8")

# two strikes flag
df_fe["two_strikes"] = (df_fe["strikes"] == 2).astype("int8")

# base occupancy flags
df_fe["on_1b_flag"] = df_fe["on_1b"].notna().astype("int8")
df_fe["on_2b_flag"] = df_fe["on_2b"].notna().astype("int8")
df_fe["on_3b_flag"] = df_fe["on_3b"].notna().astype("int8")

df_fe["any_on"] = ((df_fe["on_1b_flag"] | df_fe["on_2b_flag"] | df_fe["on_3b_flag"]) > 0).astype("int8")
df_fe["risp"]   = ((df_fe["on_2b_flag"] | df_fe["on_3b_flag"]) > 0).astype("int8")
df_fe["loaded"] = ((df_fe["on_1b_flag"] & df_fe["on_2b_flag"] & df_fe["on_3b_flag"]) > 0).astype("int8")

# mutually exclusive base_state
df_fe["base_state"] = np.select(
    [df_fe["loaded"] == 1, df_fe["risp"] == 1, df_fe["any_on"] == 1],
    ["bases_loaded", "RISP", "any_on"],
    default="none_on"
)

# batting score diff (batting - fielding)
batting_diff = np.where(
    df_fe["inning_topbot"].eq("Top"),
    df_fe["away_score"] - df_fe["home_score"],
    df_fe["home_score"] - df_fe["away_score"]
)
df_fe["batting_score_diff"] = batting_diff

# bucket score diff (your bins)
bins = [-np.inf, -5, -3, -1, 0, 2, 4, np.inf]
labels = ["trail_5plus", "trail_4-3", "trail_2-1", "tied", "lead_1-2", "lead_3-4", "lead_5plus"]
df_fe["score_bucket"] = pd.cut(df_fe["batting_score_diff"], bins=bins, labels=labels, include_lowest=True)

display(df_fe[["same_side","base_state","score_bucket","balls","strikes","outs_when_up"]].head())

#%% Focus dataset to 