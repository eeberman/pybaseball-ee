# %% [markdown]
# Plan is to isloate the most likely variables to predict a swing and a miss

# %%
from pybaseball import statcast
import pandas as pd
import numpy as np


# %%
statcast(start_dt="2019-06-24", end_dt="2019-06-25").columns

# %%
statcast(start_dt="2019-06-24", end_dt="2019-06-25").head()

# %% [markdown]
# ## EDA

# %%
statcast_df = statcast(start_dt="2019-06-24", end_dt="2019-07-25")
distinct_descriptions = statcast_df["description"].unique()
print(distinct_descriptions)
description_counts = statcast_df["description"].value_counts()
print(description_counts)


# %%
print(statcast_df.shape)


# %%
## Drop unhelpful columns

# %%
coverage = (1 - statcast_df.isna().mean()).sort_values(ascending=False)
low_coverage = coverage[coverage < 0.70]
low_coverage.head(42)
low_coverage_cols = low_coverage.index.tolist()
low_coverage_cols

# %%
# Add useful columns to prevent removal
useful_cols = ['on_2b', 'on_3b', 'on_1b']

drop_cols = sorted(set(low_coverage_cols) - set(useful_cols))
(drop_cols)


# %%
sc_df_reduced = statcast_df.drop(columns=drop_cols)
sc_df_reduced.shape


# %%
sc_df_reduced.dtypes.value_counts()

# %%
int_cols = sc_df_reduced.select_dtypes(include=['int64', 'float64']).columns
int_cols

# %%
cat_cols = sc_df_reduced.select_dtypes(include=['object', 'category', 'bool']).columns
cat_cols

# %%
import pandas as pd

drop_cols_2 = [
    'player_name', 'des', 'type', 'delta_run_exp', 'delta_pitcher_run_exp',
    'post_home_score', 'post_away_score', 'post_bat_score', 'post_fld_score',
    'home_team', 'away_team', 'delta_home_win_exp'
]
deprecated_cols = [
    c for c in sc_df_reduced.columns
    if "deprecated" in str(c).lower()
]

cols_to_drop = list(set(drop_cols_2).union(deprecated_cols))

sc_df_reduced = sc_df_reduced.drop(columns=cols_to_drop, errors="ignore")

print("Dropped (explicit):", [c for c in drop_cols_2 if c in cols_to_drop])
print("Dropped (deprecated):", deprecated_cols)
print("New shape:", sc_df_reduced.shape)


# %%
int_cols_2 = sc_df_reduced.select_dtypes(include=['int64', 'float64']).columns
int_cols_2

# %%
cat_cols_2 = sc_df_reduced.select_dtypes(include=['object', 'category', 'bool']).columns
cat_cols_2

# %%
sc_df_reduced.shape

# %%
sc_df_reduced.describe(include="all").T


# %% [markdown]
# Check to see if data has been pre-normalized to account for handed ness

# %%
sc_df_reduced["p_throws"].value_counts()

# %%
import numpy as np
import pandas as pd

# df = your statcast pitch-by-pitch dataframe
# Must include: p_throws (L/R), plus whichever x-fields you want to sanity-check

X_FIELDS = [
    "plate_x",        # horizontal plate location (catcher view)
    "pfx_x",          # horizontal movement (catcher view)
    "release_pos_x",  # horizontal release position (catcher view)
]

def sanity_check_handedness_mirroring(sc_df_reduced: pd.DataFrame, x_fields=X_FIELDS) -> pd.DataFrame:
    """
    Sanity check: in Savant/Statcast data (catcher's perspective), x-fields should look mirrored by p_throws.
    We compute:
      - mean/median/std by handedness
      - correlation between value and handedness-coded sign
      - percent of values > 0 by handedness
    """
    out_rows = []

    # Clean to L/R only
    d = sc_df_reduced.copy()
    d = d[d["p_throws"].isin(["L", "R"])]

    # Encode handedness: R=+1, L=-1 (useful for sign tests)
    throw_sign = d["p_throws"].map({"R": 1, "L": -1}).astype("int8")

    for col in x_fields:
        if col not in d.columns:
            continue

        s = pd.to_numeric(d[col], errors="coerce")
        mask = s.notna()
        s = s[mask]
        ts = throw_sign[mask]

        # Summary stats by handedness
        grp = pd.DataFrame({"p_throws": d.loc[mask, "p_throws"], col: s}).groupby("p_throws")[col]
        stats = grp.agg(["count", "mean", "median", "std"]).reset_index()

        # Extra diagnostics:
        # 1) percent positive by handedness
        pct_pos = grp.apply(lambda x: (x > 0).mean()).rename("pct_gt_0").reset_index()

        # 2) "mirror score": mean(value * throw_sign)
        #    If L is roughly the mirror of R, value*throw_sign tends to align (e.g. arm-side positive).
        mirror_score = float(np.nanmean(s.to_numpy() * ts.to_numpy()))

        # 3) simple correlation between value and throw_sign
        corr = float(np.corrcoef(s.to_numpy(), ts.to_numpy())[0, 1]) if len(s) > 1 else np.nan

        # Combine into one tidy table per column
        merged = stats.merge(pct_pos, on="p_throws", how="left")
        for _, r in merged.iterrows():
            out_rows.append({
                "field": col,
                "p_throws": r["p_throws"],
                "count": int(r["count"]),
                "mean": float(r["mean"]),
                "median": float(r["median"]),
                "std": float(r["std"]) if not pd.isna(r["std"]) else np.nan,
                "pct_gt_0": float(r["pct_gt_0"]),
                "mirror_score_mean(value*throw_sign)": mirror_score,
                "corr(value, throw_sign)": corr,
            })

    return pd.DataFrame(out_rows).sort_values(["field", "p_throws"])


# ---- Example usage ----
# from pybaseball import statcast
# df = statcast(start_dt="2024-04-01", end_dt="2024-04-30")

report = sanity_check_handedness_mirroring(sc_df_reduced, X_FIELDS)
print(report.to_string(index=False))


# ---- Optional: quick visual sanity check (one field at a time) ----
import matplotlib.pyplot as plt

def plot_handedness_hist(df: pd.DataFrame, field: str, bins=60):
    d = df[df["p_throws"].isin(["L", "R"])].copy()
    d[field] = pd.to_numeric(d[field], errors="coerce")
    d = d.dropna(subset=[field])

    plt.figure()
    plt.hist(d.loc[d["p_throws"] == "R", field], bins=bins, alpha=0.6, label="RHP")
    plt.hist(d.loc[d["p_throws"] == "L", field], bins=bins, alpha=0.6, label="LHP")
    plt.title(f"{field} distribution by p_throws (expect mirrored shift if unnormalized)")
    plt.xlabel(field)
    plt.ylabel("count")
    plt.legend()
    plt.show()

# plot_handedness_hist(df, "plate_x")



# %%
sc_df_reduced.shape

# %%
FLIP_COLS = ["release_pos_x", "vx0", "ax"]

def flip_x_for_lefties(df: pd.DataFrame,
                       flip_cols=FLIP_COLS,
                       handed_col: str = "p_throws",
                       left_value: str = "L") -> pd.DataFrame:
    out = df.copy()

    if handed_col not in out.columns:
        raise KeyError(f"Missing handedness column: {handed_col}")

    mask_L = out[handed_col].eq(left_value)

    # Flip only columns that exist
    cols_present = [c for c in flip_cols if c in out.columns]
    cols_missing = [c for c in flip_cols if c not in out.columns]

    for c in cols_present:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out.loc[mask_L, c] = -out.loc[mask_L, c]

    print("Flipped columns (LHP only):", cols_present)
    if cols_missing:
        print("WARNING: missing columns (not flipped):", cols_missing)

    return out



# %%
sc_df_reduced.shape


# %%

def handedness_report(df: pd.DataFrame, cols, handed_col="p_throws") -> pd.DataFrame:
    d = df[df[handed_col].isin(["L", "R"])].copy()
    rows = []
    for c in cols:
        if c not in d.columns:
            continue
        s = pd.to_numeric(d[c], errors="coerce")
        g = pd.DataFrame({handed_col: d[handed_col], c: s}).dropna().groupby(handed_col)[c]
        stats = g.agg(count="count", mean="mean", median="median", std="std")
        stats["pct_gt_0"] = g.apply(lambda x: (x > 0).mean())
        stats["field"] = c
        rows.append(stats.reset_index())
    return pd.concat(rows, ignore_index=True)[["field", handed_col, "count", "mean", "median", "std", "pct_gt_0"]]


print("\n=== BEFORE ===")
print(handedness_report(sc_df_reduced, FLIP_COLS).to_string(index=False))

sc_df_norm = flip_x_for_lefties(sc_df_reduced, FLIP_COLS)

print("\n=== AFTER ===")
print(handedness_report(sc_df_norm, FLIP_COLS).to_string(index=False))


# %%
sc_df_norm.shape

# %%
import matplotlib.pyplot as plt


top_desc = sc_df_norm["description"].value_counts().head(10).index
plot_df = sc_df_norm[sc_df_norm["description"].isin(top_desc)]

plt.figure(figsize=(6, 6))
for desc, group in plot_df.groupby("description"):
    plt.scatter(group["pfx_x"], group["pfx_z"], alpha=0.25, s=12, label=desc)



plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel("pfx_x")
plt.ylabel("pfx_z")
plt.title("Pitch Movement For All Pitches (Catcher's Perspective)")
plt.legend(loc="best", fontsize=8)
plt.show()


# %%
import matplotlib.pyplot as plt

mask = sc_df_norm["description"].isin(["swinging_strike", "swinging_strike_blocked"])
plot_df = sc_df_norm[mask]

plt.figure(figsize=(6, 6))
for desc, group in plot_df.groupby("description"):
    plt.scatter(group["plate_x"], group["plate_z"], alpha=0.25, s=12, label=desc)

# Strike zone
plt.plot([-0.83, 0.83], [1.5, 1.5], color="red")
plt.plot([-0.83, 0.83], [3.5, 3.5], color="red")
plt.plot([-0.83, -0.83], [1.5, 3.5], color="red")
plt.plot([0.83, 0.83], [1.5, 3.5], color="red")

plt.xlim(-3, 3)
plt.ylim(0, 5)
plt.xlabel("plate_x")
plt.ylabel("plate_z")
plt.title("Swinging Strikes Only (Catcher's Perspective)")
plt.legend(loc="upper right", fontsize=8)
plt.show()


# %%
top_desc = sc_df_norm["description"].value_counts().head(10).index
plot_df = sc_df_norm[sc_df_norm["description"].isin(top_desc)]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()  # Flatten to 1D array for easier indexing

for idx, desc in enumerate(top_desc):
    group = plot_df[plot_df["description"] == desc]
    axes[idx].scatter(group["plate_x"], group["plate_z"], alpha=0.25, s=12)
    axes[idx].set_xlim(-6, 6)
    axes[idx].set_ylim(0, 6)
    axes[idx].set_xlabel("plate_x")
    axes[idx].set_ylabel("plate_z")
    axes[idx].set_title(desc, fontsize=10)

plt.tight_layout()
plt.show()

# %%
top_desc = sc_df_norm["description"].value_counts().head(10).index
plot_df = sc_df_norm[sc_df_norm["description"].isin(top_desc)]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()  # Flatten to 1D array for easier indexing

for idx, desc in enumerate(top_desc):
    group = plot_df[plot_df["description"] == desc]
    axes[idx].scatter(group["pfx_x"], group["pfx_z"], alpha=0.25, s=12)
    axes[idx].set_xlim(-3, 3)
    axes[idx].set_ylim(-3, 3)
    axes[idx].set_xlabel("pfx_x")
    axes[idx].set_ylabel("pfx_z")
    axes[idx].set_title(desc, fontsize=10)

plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

top_desc = sc_df_norm["description"].value_counts().head(10).index
plot_df = sc_df_norm[sc_df_norm["description"].isin(top_desc)]

fig = plt.figure(figsize=(16, 10))

for idx, desc in enumerate(top_desc, 1):
    ax = fig.add_subplot(2, 5, idx, projection="3d")
    group = plot_df[plot_df["description"] == desc]
    
    ax.scatter(group["vx0"], group["vy0"], group["vz0"], alpha=0.5, s=20)
    ax.set_xlabel("vx0", fontsize=8)
    ax.set_ylabel("vy0", fontsize=8)
    ax.set_zlabel("vz0", fontsize=8)
    ax.set_title(desc, fontsize=10)

plt.tight_layout()
plt.show()



# %%
top_desc = sc_df_norm["description"].value_counts().head(10).index
plot_df = sc_df_norm[sc_df_norm["description"].isin(top_desc)]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()  # Flatten to 1D array for easier indexing

for idx, desc in enumerate(top_desc):
    group = plot_df[plot_df["description"] == desc]
    axes[idx].scatter(group["effective_speed"], group["release_spin_rate"], alpha=0.25, s=12)
    axes[idx].set_xlabel("effective_speed")
    axes[idx].set_ylabel("release_spin_rate")
    axes[idx].set_title(desc, fontsize=10)

plt.tight_layout()
plt.show()


# %%
mask = sc_df_norm["description"].isin(["swinging_strike", "swinging_strike_blocked"])
masked_df = sc_df_norm[mask]

plt.figure(figsize=(6, 6))
for desc, group in masked_df.groupby("description"):
    plt.scatter(group["effective_speed"], group["release_spin_rate"], alpha=0.25, s=12, label=desc)

plt.xlabel("effective_speed")
plt.ylabel("release_spin_rate")
plt.title("Swinging Stikes Effective Speed vs. Release Spin Rate")
plt.legend(loc="upper right", fontsize=8)
plt.show()


# %%
sc_df_norm.columns

# %%
sc_df_norm["description"].value_counts()

# %%
# drop intentional walks
sc_df_norm = sc_df_norm[sc_df_norm["description"].ne("automatic_ball")].copy()

# Combine labels to group balls and stikes
sc_df_norm["description"] = sc_df_norm["description"].replace({
    "swinging_strike": "any_swinging_strike",
    "swinging_strike_blocked": "any_swinging_strike",
    "ball": "any_ball",
    "blocked_ball": "any_ball"
})

print(sc_df_norm["description"].value_counts())
print(sc_df_norm.shape)

# %% [markdown]
# Basic Model!

# %%
# Cell 1 — Target cleanup / consolidation (match your desired target set)

import pandas as pd

df = sc_df_norm.copy()

# Drop automatic_ball rows entirely
df = df[df["description"].ne("automatic_ball")].copy()

# Consolidate labels into your target taxonomy
df["target"] = df["description"].replace({
    "swinging_strike": "any_swinging_strike",
    "swinging_strike_blocked": "any_swinging_strike",
    "ball": "any_ball",
    "blocked_ball": "any_ball",
})

# Keep only the targets you listed (protects against unexpected/extra labels)
TARGETS = [
    "any_ball",
    "foul",
    "hit_into_play",
    "called_strike",
    "any_swinging_strike",
    "foul_tip",
    "hit_by_pitch",
    "foul_bunt",
    "missed_bunt",
    "pitchout",
    "bunt_foul_tip",
]

df = df[df["target"].isin(TARGETS)].copy()

print(df["target"].value_counts().reindex(TARGETS))
print("Shape:", df.shape)


# %%
# Cell 2 — Pull data (cached) + build target + time-based split (holdout = Sept 2023)

import pandas as pd
from pybaseball import statcast, cache

cache.enable()  # prevents re-pulling the same requests repeatedly

# Features
FEATURES_NUM = ["effective_speed", "release_spin_rate", "pfx_x", "pfx_z", "plate_x", "plate_z"]
FEATURES_CAT = ["p_throws", "stand"]
FEATURES = FEATURES_NUM + FEATURES_CAT

# Pull statcast data
df_2023 = statcast(start_dt="2023-04-01", end_dt="2023-10-01").copy()

# Ensure game_date is datetime
df_2023["game_date"] = pd.to_datetime(df_2023["game_date"], errors="coerce")
df_2023 = df_2023.dropna(subset=["game_date"]).copy()

# ---- Build target (your taxonomy) ----
df_2023 = df_2023[df_2023["description"].ne("automatic_ball")].copy()

df_2023["target"] = df_2023["description"].replace({
    "swinging_strike": "any_swinging_strike",
    "swinging_strike_blocked": "any_swinging_strike",
    "ball": "any_ball",
    "blocked_ball": "any_ball",
})

TARGETS = [
    "any_ball",
    "foul",
    "hit_into_play",
    "called_strike",
    "any_swinging_strike",
    "foul_tip",
    "hit_by_pitch",
    "foul_bunt",
    "missed_bunt",
    "pitchout",
    "bunt_foul_tip",
]

df_2023 = df_2023[df_2023["target"].isin(TARGETS)].copy()

print("Target counts (all pulled data):")
print(df_2023["target"].value_counts().reindex(TARGETS))
print("Shape:", df_2023.shape)



# %%

# ---- Time split ----
train_end = pd.Timestamp("2023-08-31")
test_start = pd.Timestamp("2023-09-01")
test_end = pd.Timestamp("2023-09-30")

train_df = df_2023[df_2023["game_date"] <= train_end].copy()
test_df  = df_2023[(df_2023["game_date"] >= test_start) & (df_2023["game_date"] <= test_end)].copy()

# Drop rows missing features/target
train_df = train_df.dropna(subset=FEATURES + ["target"]).copy()
test_df  = test_df.dropna(subset=FEATURES + ["target"]).copy()

print("\nTrain date range:", train_df["game_date"].min(), "→", train_df["game_date"].max(), "| rows:", len(train_df))
print("Test  date range:", test_df["game_date"].min(),  "→", test_df["game_date"].max(),  "| rows:", len(test_df))

print("\nTrain target counts:\n", train_df["target"].value_counts().reindex(TARGETS))
print("\nTest target counts:\n", test_df["target"].value_counts().reindex(TARGETS))

# %%
train_df.shape

# %%
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

X_train = train_df[FEATURES]
y_train = train_df["target"]

X_test = test_df[FEATURES]
y_test = test_df["target"]

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), FEATURES_NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore'), FEATURES_CAT)
    ],
    remainder='drop',
)

clf = LogisticRegression(
    multi_class='multinomial',
    penalty='l2',
    C=1.0,
    solver="lbfgs",
    max_iter=200,
    n_jobs=-1
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("classifier", clf)
])

model.fit(X_train, y_train)

print("Classes:", model.named_steps["classifier"].classes_)

# %%
X_train.shape 

# %%
from sklearn.metrics import classification_report, confusion_matrix, log_loss, accuracy_score, roc_auc_score 

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test, y_proba, labels=model.named_steps["classifier"].classes_)
auc_ovr = roc_auc_score(y_test, y_proba, multi_class='ovr')
auc_ovo = roc_auc_score(y_test, y_proba, multi_class='ovo')

print("Accuracy:", acc)
print("LogLoss:", ll)
print("ROC AUC (ovr):", auc_ovr)
print("ROC AUC (ovo):", auc_ovo)

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

print("\nConfusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_test, y_pred, labels=model.named_steps["classifier"].classes_)
print(pd.DataFrame(cm, index=model.named_steps["classifier"].classes_, columns=model.named_steps["classifier"].classes_))

# %% [markdown]
# # Model, redux

# %%
# Cell 6 — Diagnostics: majority-class baseline + predicted class distribution

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# 1) Majority-class baseline on the test set
majority_class = y_test.value_counts().idxmax()
y_pred_majority = np.full(shape=len(y_test), fill_value=majority_class, dtype=object)

acc_majority = accuracy_score(y_test, y_pred_majority)
print("Majority class:", majority_class)
print("Majority-class baseline accuracy:", acc_majority)

# 2) Predicted class distribution from your current model
y_pred = model.predict(X_test)
pred_dist = pd.Series(y_pred).value_counts(normalize=True).rename("predicted_share")
true_dist = y_test.value_counts(normalize=True).rename("true_share")

dist = pd.concat([true_dist, pred_dist], axis=1).fillna(0.0).sort_values("true_share", ascending=False)
print("\nTrue vs Predicted class shares:")
print(dist)


# %%
# Cell 7 — Refit with class_weight="balanced" (minimal change)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ],
    remainder="drop",
)

clf_balanced = LogisticRegression(
    multi_class="multinomial",
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=300,
    class_weight="balanced"
)

model_balanced = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", clf_balanced)
])

model_balanced.fit(X_train, y_train)

print("Classes:", model_balanced.named_steps["clf"].classes_)


# %%
from sklearn.metrics import classification_report, confusion_matrix, log_loss, accuracy_score, roc_auc_score 

y_pred = model_balanced.predict(X_test)
y_proba = model_balanced.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test, y_proba, labels=model_balanced.named_steps["clf"].classes_)
auc_ovr = roc_auc_score(y_test, y_proba, multi_class='ovr')
auc_ovo = roc_auc_score(y_test, y_proba, multi_class='ovo')

print("Accuracy:", acc)
print("LogLoss:", ll)
print("ROC AUC (ovr):", auc_ovr)
print("ROC AUC (ovo):", auc_ovo)

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

# print("\nConfusion matrix (rows=true, cols=pred):")
#cm = confusion_matrix(y_test, y_pred, labels=model.named_steps["clf_balanced"].classes_)
# print(pd.DataFrame(cm, index=model.named_steps["clf_balanced"].classes_, columns=model.named_steps["clf_balanced"].classes_))

# %%
# Post Balence -  Diagnostics: majority-class baseline + predicted class distribution

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# 1) Majority-class baseline on the test set
majority_class = y_test.value_counts().idxmax()
y_pred_majority = np.full(shape=len(y_test), fill_value=majority_class, dtype=object)

acc_majority = accuracy_score(y_test, y_pred_majority)
print("Majority class:", majority_class)
print("Majority-class baseline accuracy:", acc_majority)

# 2) Predicted class distribution from your current model
y_pred = model_balanced.predict(X_test)
pred_dist = pd.Series(y_pred).value_counts(normalize=True).rename("predicted_share")
true_dist = y_test.value_counts(normalize=True).rename("true_share")

dist = pd.concat([true_dist, pred_dist], axis=1).fillna(0.0).sort_values("true_share", ascending=False)
print("\nTrue vs Predicted class shares:")
print(dist)


# %%
# Cell 9 — Optional: simplify target by dropping ultra-rare classes and re-run baseline quickly
# (Keeps the "core" outcomes; avoids classes with near-zero support)

CORE_TARGETS = [
    "any_ball",
    "called_strike",
    "any_swinging_strike",
    "foul",
    "hit_into_play",
    "foul_tip",
    "hit_by_pitch",
]

train_core = train_df[train_df["target"].isin(CORE_TARGETS)].copy()
test_core  = test_df[test_df["target"].isin(CORE_TARGETS)].copy()

X_train_c = train_core[FEATURES]
y_train_c = train_core["target"]
X_test_c  = test_core[FEATURES]
y_test_c  = test_core["target"]

model_core = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", LogisticRegression(
        multi_class="multinomial",
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=300,
        class_weight="balanced"
    ))
])

model_core.fit(X_train_c, y_train_c)

y_pred_c = model_core.predict(X_test_c)
y_proba_c = model_core.predict_proba(X_test_c)

auc_ovr = roc_auc_score(y_test_c, y_proba_c, multi_class='ovr')
auc_ovo = roc_auc_score(y_test_c, y_proba_c, multi_class='ovo')

print("Core Accuracy:", accuracy_score(y_test_c, y_pred_c))
print("Core LogLoss:", log_loss(y_test_c, y_proba_c, labels=model_core.named_steps["clf"].classes_))
print("auc_ovo:", auc_ovo)
print("auc_ovr:", auc_ovr)
print("\nCore classification report:")
print(classification_report(y_test_c, y_pred_c, digits=3, zero_division=0))


# %%
train_df.shape
test_core.shape

# %%
df_model = df_2023.copy()

# drop strategy based labels
DROP_LABELS = {"foul_bunt", "missed_bunt", "pitchout", "bunt_foul_tip"}
df_model = df_model[~df_model["target"].isin(DROP_LABELS)].copy()

# Merge foul and foul_tip
df_model["target"] = df_model["target"].replace({
    "foul_tip": "foul"
})



# %% [markdown]
# Filtered down and condensed target set

# %%
# Set final targets
TARGETS_V2 = [
    "any_ball",
    "called_strike",
    "any_swinging_strike",
    "foul",
    "hit_into_play",
    "hit_by_pitch"
    ]

df_model = df_model[df_model["target"].isin(TARGETS_V2)].copy()

print(df_model["target"].value_counts())
print(df_model.shape)

# %%
# Cell B — Time split (train <= 2023-08-31, test = Sept 2023) + drop missing features

import pandas as pd

FEATURES_NUM = ["effective_speed", "release_spin_rate", "pfx_x", "pfx_z", "plate_x", "plate_z"]
FEATURES_CAT = ["p_throws", "stand"]
FEATURES = FEATURES_NUM + FEATURES_CAT

# Ensure datetime
df_model["game_date"] = pd.to_datetime(df_model["game_date"], errors="coerce")
df_model = df_model.dropna(subset=["game_date"]).copy()

train_end = pd.Timestamp("2023-08-31")
test_start = pd.Timestamp("2023-09-01")
test_end = pd.Timestamp("2023-09-30")

train_df_v2 = df_model[df_model["game_date"] <= train_end].copy()
test_df_v2  = df_model[(df_model["game_date"] >= test_start) & (df_model["game_date"] <= test_end)].copy()

train_df_v2 = train_df_v2.dropna(subset=FEATURES + ["target"]).copy()
test_df_v2  = test_df_v2.dropna(subset=FEATURES + ["target"]).copy()

X_train_v2 = train_df_v2[FEATURES]
y_train_v2 = train_df_v2["target"]
X_test_v2  = test_df_v2[FEATURES]
y_test_v2  = test_df_v2["target"]

print("Train rows:", len(train_df_v2), "| Test rows:", len(test_df_v2))
print("\nTrain target counts:\n", y_train_v2.value_counts().reindex(TARGETS_V2))
print("\nTest target counts:\n", y_test_v2.value_counts().reindex(TARGETS_V2))
print("Train Shape:\n" , X_train_v2.shape)
print("Test Shape:\n" , X_test_v2.shape)


# %%
# Cell C — Fit multinomial logistic regression (L2, scaled numerics, one-hot p_throws)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

preprocess_v2 = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ],
    remainder="drop",
)

clf_v2 = LogisticRegression(
    multi_class="multinomial",
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=300
)

model_v2 = Pipeline(steps=[
    ("preprocess", preprocess_v2),
    ("clf", clf_v2),
])

model_v2.fit(X_train_v2, y_train_v2)
print("Classes:", model_v2.named_steps["clf"].classes_)


# %%
# Cell D — Scoring (Accuracy, LogLoss, AUC ovr/ovo, report, confusion matrix)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

y_pred_v2 = model_v2.predict(X_test_v2)
y_proba_v2 = model_v2.predict_proba(X_test_v2)

classes_v2 = model_v2.named_steps["clf"].classes_

acc_v2 = accuracy_score(y_test_v2, y_pred_v2)
ll_v2 = log_loss(y_test_v2, y_proba_v2, labels=classes_v2)

# AUC (needs binarized y)
y_test_bin = label_binarize(y_test_v2, classes=classes_v2)
auc_ovr_v2 = roc_auc_score(y_test_bin, y_proba_v2, average="macro", multi_class="ovr")
auc_ovo_v2 = roc_auc_score(y_test_bin, y_proba_v2, average="macro", multi_class="ovo")

print("Accuracy:", acc_v2)
print("LogLoss:", ll_v2)
print("ROC AUC (ovr):", auc_ovr_v2)
print("ROC AUC (ovo):", auc_ovo_v2)

print("\nClassification report:")
print(classification_report(y_test_v2, y_pred_v2, digits=3, zero_division=0))

print("\nConfusion matrix (rows=true, cols=pred):")
cm_v2 = confusion_matrix(y_test_v2, y_pred_v2, labels=classes_v2)
print(pd.DataFrame(cm_v2, index=classes_v2, columns=classes_v2))


# %%
# Cell E — Compare true vs predicted class shares + majority baseline accuracy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# True vs predicted shares
true_share = y_test_v2.value_counts(normalize=True).rename("true_share")
pred_share = pd.Series(y_pred_v2).value_counts(normalize=True).rename("pred_share")

share_cmp = (
    pd.concat([true_share, pred_share], axis=1)
    .fillna(0.0)
    .reindex(TARGETS_V2)
)

print("True vs Predicted shares (test set):")
print(share_cmp)

# Majority baseline accuracy
majority_class = y_test_v2.value_counts().idxmax()
y_pred_majority = np.full(len(y_test_v2), majority_class, dtype=object)
acc_majority = accuracy_score(y_test_v2, y_pred_majority)

print("\nMajority class:", majority_class)
print("Majority baseline accuracy:", acc_majority)


# %% [markdown]
# Switch to xgboost model and include a "in_zone" feature

# %%
bzone_nulls = df_model["sz_bot"].isna().sum()
tzone_nulls = df_model["sz_top"].isna().sum()
print(f"sz_bot nulls: {bzone_nulls}, sz_top nulls: {tzone_nulls}")

# %%
# Cell 1 — Add plate_x_batter + in_zone, and DROP rows with null sz_bot/sz_top

df_xgb = df_model.copy()

# Drop rows where strike-zone bounds are missing (needed for in_zone)
df_xgb = df_xgb.dropna(subset=["sz_bot", "sz_top"]).copy()

# Batter-standardized horizontal axis (inside-to-batter is positive)
stand_sign = df_xgb["stand"].map({"R": -1, "L": 1})
df_xgb["plate_x_batter"] = df_xgb["plate_x"] * stand_sign

# Strike zone feature
ZONE_HALF_WIDTH_FT = 0.83
df_xgb["in_zone"] = (
    df_xgb["plate_x"].abs().le(ZONE_HALF_WIDTH_FT) &
    df_xgb["plate_z"].ge(df_xgb["sz_bot"]) &
    df_xgb["plate_z"].le(df_xgb["sz_top"])
).astype("int8")

print("Dropped rows due to null sz_bot/sz_top. New shape:", df_xgb.shape)
print(df_xgb[["plate_x", "plate_z", "stand", "plate_x_batter", "in_zone", "sz_bot", "sz_top"]].head())
print("\nIn-zone rate:", df_xgb["in_zone"].mean())


# %%
# Cell 2 — Time split (train <= 2023-08-31, test = Sept 2023) + feature sets for BOTH runs

import pandas as pd

df_xgb["game_date"] = pd.to_datetime(df_xgb["game_date"], errors="coerce")
df_xgb = df_xgb.dropna(subset=["game_date"]).copy()

train_end = pd.Timestamp("2023-08-31")
test_start = pd.Timestamp("2023-09-01")
test_end = pd.Timestamp("2023-09-30")

train_df = df_xgb[df_xgb["game_date"] <= train_end].copy()
test_df  = df_xgb[(df_xgb["game_date"] >= test_start) & (df_xgb["game_date"] <= test_end)].copy()

# Baseline XGB features (NO in_zone)
FEATURES_NUM_BASE = ["effective_speed", "release_spin_rate", "pfx_x", "pfx_z", "plate_x_batter", "plate_z"]
FEATURES_CAT = ["p_throws", "stand"]  # include stand to make plate_x_batter meaningful
FEATURES_BASE = FEATURES_NUM_BASE + FEATURES_CAT

# With in_zone
FEATURES_NUM_ZONE = FEATURES_NUM_BASE + ["in_zone"]
FEATURES_ZONE = FEATURES_NUM_ZONE + FEATURES_CAT

# Drop missing
train_df = train_df.dropna(subset=FEATURES_ZONE + ["target"]).copy()
test_df  = test_df.dropna(subset=FEATURES_ZONE + ["target"]).copy()

print("Train rows:", len(train_df), "Test rows:", len(test_df))
print("\nTrain target counts:\n", train_df["target"].value_counts())
print("\nTest  target counts:\n", test_df["target"].value_counts())
print("Train Shape:\n" , train_df[FEATURES_ZONE].shape)
print("Test Shape:\n" , test_df[FEATURES_ZONE].shape)

# %%
# Cell 3 (fixed) — XGBoost baseline model (NO in_zone) with label encoding

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# X / y
X_train_base = train_df[FEATURES_BASE]
X_test_base  = test_df[FEATURES_BASE]

le = LabelEncoder()
y_train_enc = le.fit_transform(train_df["target"])
y_test_enc  = le.transform(test_df["target"])

preprocess_base = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ],
    remainder="passthrough"
)

xgb_base = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    n_estimators=300,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

model_xgb_base = Pipeline(steps=[
    ("preprocess", preprocess_base),
    ("xgb", xgb_base)
])

model_xgb_base.fit(X_train_base, y_train_enc)

print("Label classes:", list(le.classes_))


# %%
# Cell 4 (fixed) — Evaluate baseline XGB (decode predictions back to strings)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

y_pred_enc = model_xgb_base.predict(X_test_base)
y_proba = model_xgb_base.predict_proba(X_test_base)

y_pred = le.inverse_transform(y_pred_enc)
y_test = test_df["target"].values  # string labels

acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test_enc, y_proba, labels=np.arange(len(le.classes_)))  # logloss on encoded labels

# AUC (binarize encoded y)
y_test_bin = label_binarize(y_test_enc, classes=np.arange(len(le.classes_)))
auc_ovr = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
auc_ovo = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovo")

print("=== XGB (no in_zone) ===")
print("Accuracy:", acc)
print("LogLoss:", ll)
print("ROC AUC (ovr):", auc_ovr)
print("ROC AUC (ovo):", auc_ovo)

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))

print("\nConfusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

true_share = pd.Series(y_test).value_counts(normalize=True).rename("true_share")
pred_share = pd.Series(y_pred).value_counts(normalize=True).rename("pred_share")
share_cmp = pd.concat([true_share, pred_share], axis=1).fillna(0.0).sort_values("true_share", ascending=False)

print("\nTrue vs Predicted shares:")
print(share_cmp)


# %%
# EDA Cell — Swinging-strike vs ball rate by zone_out_dist bucket x count

import numpy as np
import pandas as pd

eda = df_xgb.copy()  # df_xgb has plate_x, plate_z, sz_top, sz_bot, target, balls, strikes, stand, etc.

# Keep just the two labels of interest
eda = eda[eda["target"].isin(["any_ball", "any_swinging_strike"])].copy()

# Drop missing zone bounds (already done in df_xgb, but safe)
eda = eda.dropna(subset=["sz_bot", "sz_top", "plate_x", "plate_z", "balls", "strikes"]).copy()

# Continuous distance outside the zone (0 if in-zone)
ZONE_HALF_WIDTH_FT = 0.83
x_out = (eda["plate_x"].abs() - ZONE_HALF_WIDTH_FT).clip(lower=0)
z_out_low = (eda["sz_bot"] - eda["plate_z"]).clip(lower=0)
z_out_high = (eda["plate_z"] - eda["sz_top"]).clip(lower=0)
eda["zone_out_dist"] = np.sqrt(x_out**2 + np.maximum(z_out_low, z_out_high)**2)

# Bin distance (tweak bins if you want finer)
bins = [-1e-9, 0, 0.05, 0.10, 0.20, 0.35, 0.60, 1.00, 10.0]
labels = ["in_zone", "0-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.35", "0.35-0.60", "0.60-1.0", "1.0+"]

eda["dist_bin"] = pd.cut(eda["zone_out_dist"], bins=bins, labels=labels)

# Indicator for swinging strike
eda["is_swstr"] = (eda["target"] == "any_swinging_strike").astype(int)

# Group and compute rates
g = (
    eda.groupby(["balls", "strikes", "dist_bin"], observed=True)
       .agg(n=("is_swstr", "size"), swstr_rate=("is_swstr", "mean"))
       .reset_index()
)

# Optional: filter out tiny groups to reduce noise
MIN_N = 200
g_filt = g[g["n"] >= MIN_N].copy()

# Create a pivot table: rows = (balls,strikes), columns = dist_bin
pivot_rate = g_filt.pivot_table(index=["balls", "strikes"], columns="dist_bin", values="swstr_rate")
pivot_n    = g_filt.pivot_table(index=["balls", "strikes"], columns="dist_bin", values="n")

# Display nicely
print("Swinging-strike rate (among {any_ball, any_swinging_strike}) by count x distance-from-zone bin")
display(pivot_rate)

print(f"\nSample size n per cell (only showing groups with n >= {MIN_N})")
display(pivot_n)


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

eda = df_xgb.copy()
eda = eda[eda["target"].isin(["any_ball", "any_swinging_strike"])].copy()

# binary base occupancy flags
eda["on_1b_flag"] = eda["on_1b"].notna().astype(int)
eda["on_2b_flag"] = eda["on_2b"].notna().astype(int)
eda["on_3b_flag"] = eda["on_3b"].notna().astype(int)

eda["any_on"] = ((eda["on_1b_flag"] | eda["on_2b_flag"] | eda["on_3b_flag"]) > 0).astype(int)
eda["risp"]   = ((eda["on_2b_flag"] | eda["on_3b_flag"]) > 0).astype(int)
eda["loaded"] = ((eda["on_1b_flag"] & eda["on_2b_flag"] & eda["on_3b_flag"]) > 0).astype(int)

# mutually exclusive bucket (simple + interpretable)
eda["base_state"] = np.select(
    [eda["loaded"] == 1, eda["risp"] == 1, eda["any_on"] == 1],
    ["bases_loaded", "RISP", "any_on"],
    default="none_on"
)

eda["two_strikes"] = (eda["strikes"] == 2).astype(int)
eda["is_swstr"] = (eda["target"] == "any_swinging_strike").astype(int)

# compute rates
rate = (
    eda.groupby(["two_strikes", "base_state"], observed=True)
       .agg(n=("is_swstr", "size"), swstr_rate=("is_swstr", "mean"))
       .reset_index()
)

# plot
order = ["none_on", "any_on", "RISP", "bases_loaded"]
fig, ax = plt.subplots()

for ts in [0, 1]:
    sub = rate[rate["two_strikes"] == ts].set_index("base_state").reindex(order)
    ax.plot(order, sub["swstr_rate"], marker="o", label=f"two_strikes={ts}")

ax.set_title("Swinging-strike rate by base state (conditioned on ball vs swinging-strike)")
ax.set_ylabel("swstr_rate")
ax.set_xlabel("base_state")
ax.legend()
plt.xticks(rotation=20)
plt.show()

display(rate.sort_values(["two_strikes","base_state"]))


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

eda = df_xgb.copy()
eda = eda[eda["target"].isin(["any_ball", "any_swinging_strike"])].copy()

eda["is_swstr"] = (eda["target"] == "any_swinging_strike").astype(int)
eda["two_strikes"] = (eda["strikes"] == 2).astype(int)

# batting team score diff (batting - fielding)
# Top = away batting, Bottom = home batting
batting_diff = np.where(
    eda["inning_topbot"].eq("Top"),
    eda["away_score"] - eda["home_score"],
    eda["home_score"] - eda["away_score"]
)
eda["batting_score_diff"] = batting_diff

# bucket it
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
eda["score_bucket"] = pd.cut(
    eda["batting_score_diff"], bins=bins, labels=labels, include_lowest=True
)


rate = (
    eda.groupby(["two_strikes", "score_bucket"], observed=True)
       .agg(n=("is_swstr","size"), swstr_rate=("is_swstr","mean"))
       .reset_index()
)

order = labels

fig, ax = plt.subplots()
for ts in [0, 1]:
    sub = rate[rate["two_strikes"] == ts].set_index("score_bucket").reindex(order)
    ax.plot(order, sub["swstr_rate"], marker="o", label=f"two_strikes={ts}")

ax.set_title("Swinging-strike rate by batting score differential bucket")
ax.set_ylabel("swstr_rate")
ax.set_xlabel("score_bucket")
ax.legend()
plt.xticks(rotation=20)
plt.show()

display(rate.sort_values(["two_strikes","score_bucket"]))


# %% [markdown]
# Model with base state, count, score, handedness matchup, batter-relative release position added

# %%
# Cell 1 — Add engineered features (batter-relative release_pos_x & release_pos_z, same_side, base_state, score_bucket, zone_out_dist)

import numpy as np
import pandas as pd

df_model2 = df_xgb.copy()  # df_xgb already has plate_x_batter + dropped null sz_top/sz_bot

# --- matchup ---
df_model2["same_side"] = (df_model2["stand"] == df_model2["p_throws"]).astype("int8")

# --- batter-relative release position (x + z) ---
stand_sign = df_model2["stand"].map({"R": -1, "L": 1})
df_model2["release_pos_x_batter"] = df_model2["release_pos_x"] * stand_sign
df_model2["release_pos_z_batter"] = df_model2["release_pos_z"]  # z doesn't flip with stand

# --- edge distance outside zone (continuous) ---
ZONE_HALF_WIDTH_FT = 0.83
x_out = (df_model2["plate_x"].abs() - ZONE_HALF_WIDTH_FT).clip(lower=0)
z_below = (df_model2["sz_bot"] - df_model2["plate_z"]).clip(lower=0)
z_above = (df_model2["plate_z"] - df_model2["sz_top"]).clip(lower=0)
z_out = np.maximum(z_below, z_above)

df_model2["zone_out_dist"] = np.sqrt(x_out**2 + z_out**2)

# --- base state buckets ---
df_model2["on_1b_flag"] = df_model2["on_1b"].notna().astype("int8")
df_model2["on_2b_flag"] = df_model2["on_2b"].notna().astype("int8")
df_model2["on_3b_flag"] = df_model2["on_3b"].notna().astype("int8")

df_model2["any_on"] = ((df_model2["on_1b_flag"] | df_model2["on_2b_flag"] | df_model2["on_3b_flag"]) > 0).astype("int8")
df_model2["risp"]   = ((df_model2["on_2b_flag"] | df_model2["on_3b_flag"]) > 0).astype("int8")
df_model2["loaded"] = ((df_model2["on_1b_flag"] & df_model2["on_2b_flag"] & df_model2["on_3b_flag"]) > 0).astype("int8")

df_model2["base_state"] = np.select(
    [df_model2["loaded"] == 1, df_model2["risp"] == 1, df_model2["any_on"] == 1],
    ["bases_loaded", "RISP", "any_on"],
    default="none_on"
)

# --- batting score differential bucket (your bins) ---
batting_diff = np.where(
    df_model2["inning_topbot"].eq("Top"),
    df_model2["away_score"] - df_model2["home_score"],
    df_model2["home_score"] - df_model2["away_score"]
)
df_model2["batting_score_diff"] = batting_diff

bins = [-np.inf, -5, -3, -1, 0, 2, 4, np.inf]
labels = ["trail_5plus", "trail_4-3", "trail_2-1", "tied", "lead_1-2", "lead_3-4", "lead_5plus"]
df_model2["score_bucket"] = pd.cut(df_model2["batting_score_diff"], bins=bins, labels=labels, include_lowest=True)

print("df_model2 shape:", df_model2.shape)
print(df_model2[["same_side","release_pos_x_batter","release_pos_z_batter","zone_out_dist","base_state","score_bucket"]].head())


# %%
# Cell 2 — Features list (same originals + add batter-relative release x/z)

FEATURES_NUM2 = [
    # original numeric set
    "effective_speed", "release_spin_rate", "pfx_x", "pfx_z",
    "plate_x_batter", "plate_z",

    # new engineered numeric
    "zone_out_dist",
    "release_pos_x_batter",
    "release_pos_z_batter",
]

FEATURES_CAT2 = [
    # original cats
    "p_throws", "stand",

    # new cats
    "base_state", "score_bucket",
]

FEATURES2 = FEATURES_NUM2 + FEATURES_CAT2

print("Numeric:", FEATURES_NUM2)
print("Categorical:", FEATURES_CAT2)


# %%
# Cell 3 — Rebuild train/test split (same date logic) and label encode

from sklearn.preprocessing import LabelEncoder

# Use the same date split you chose
train_end  = pd.Timestamp("2023-08-31")
test_start = pd.Timestamp("2023-09-01")
test_end   = pd.Timestamp("2023-09-30")

# Ensure game_date exists + is datetime
df_model2["game_date"] = pd.to_datetime(df_model2["game_date"], errors="coerce")
df_model2 = df_model2.dropna(subset=["game_date"]).copy()

train_df2 = df_model2[df_model2["game_date"] <= train_end].copy()
test_df2  = df_model2[(df_model2["game_date"] >= test_start) & (df_model2["game_date"] <= test_end)].copy()

# Drop rows missing features/target
train_df2 = train_df2.dropna(subset=FEATURES2 + ["target"]).copy()
test_df2  = test_df2.dropna(subset=FEATURES2 + ["target"]).copy()

# Label encode
le2 = LabelEncoder()
y_train2_enc = le2.fit_transform(train_df2["target"])
y_test2_enc  = le2.transform(test_df2["target"])

print("Train rows:", len(train_df2), "Test rows:", len(test_df2))
print("Label classes:", list(le2.classes_))
print("\nTrain target counts:\n", train_df2["target"].value_counts())
print("\nTest target counts:\n", test_df2["target"].value_counts())


# %%
# Cell 4 — XGBoost fit (same hyperparams as before)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

X_train2 = train_df2[FEATURES2]
X_test2  = test_df2[FEATURES2]

preprocess2 = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT2),
    ],
    remainder="passthrough"
)

xgb2 = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le2.classes_),
    eval_metric="mlogloss",
    n_estimators=600,
    learning_rate=0.07,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

model_xgb2 = Pipeline(steps=[
    ("preprocess", preprocess2),
    ("xgb", xgb2)
])

model_xgb2.fit(X_train2, y_train2_enc)

print("Label classes:", list(le2.classes_))


# %%
# Cell 5 — Score + confusion + share comparison (same as before)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

y_pred2_enc = model_xgb2.predict(X_test2)
y_proba2 = model_xgb2.predict_proba(X_test2)

y_pred2 = le2.inverse_transform(y_pred2_enc)
y_test2_str = test_df2["target"].values

acc2 = accuracy_score(y_test2_str, y_pred2)
ll2  = log_loss(y_test2_enc, y_proba2, labels=np.arange(len(le2.classes_)))

y_test2_bin = label_binarize(y_test2_enc, classes=np.arange(len(le2.classes_)))
auc2_ovr = roc_auc_score(y_test2_bin, y_proba2, average="macro", multi_class="ovr")
auc2_ovo = roc_auc_score(y_test2_bin, y_proba2, average="macro", multi_class="ovo")

print("=== XGB (added game state + edge distance + batter-rel release) ===")
print("Accuracy:", acc2)
print("LogLoss:", ll2)
print("ROC AUC (ovr):", auc2_ovr)
print("ROC AUC (ovo):", auc2_ovo)

print("\nClassification report:")
print(classification_report(y_test2_str, y_pred2, digits=3, zero_division=0))

print("\nConfusion matrix (rows=true, cols=pred):")
cm2 = confusion_matrix(y_test2_str, y_pred2, labels=le2.classes_)
print(pd.DataFrame(cm2, index=le2.classes_, columns=le2.classes_))

true_share2 = pd.Series(y_test2_str).value_counts(normalize=True).rename("true_share")
pred_share2 = pd.Series(y_pred2).value_counts(normalize=True).rename("pred_share")
share_cmp2 = pd.concat([true_share2, pred_share2], axis=1).fillna(0.0).sort_values("true_share", ascending=False)

print("\nTrue vs Predicted shares:")
print(share_cmp2)


# %%
# Feature importance (gain + permutation; optional SHAP)
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

# ----- 1) Get post-transform feature names from ColumnTransformer -----
pre = model_xgb2.named_steps["preprocess"]
xgb = model_xgb2.named_steps["xgb"]

# ColumnTransformer has: OneHotEncoder on FEATURES_CAT2, passthrough for numeric
ohe = pre.named_transformers_["cat"]

cat_feature_names = ohe.get_feature_names_out(FEATURES_CAT2)
num_feature_names = np.array(FEATURES_NUM2)  # passthrough numeric order is preserved

feature_names = np.concatenate([cat_feature_names, num_feature_names])

# ----- 2) XGBoost built-in importance ("gain") -----
gain = xgb.feature_importances_  # corresponds to transformed columns
imp_gain = (
    pd.DataFrame({"feature": feature_names, "gain_importance": gain})
      .sort_values("gain_importance", ascending=False)
      .reset_index(drop=True)
)

print("Top 30 features by XGB gain importance:")
display(imp_gain.head(30))

# Optional: aggregate OHE back to original columns (so you can see if e.g. base_state dominates)
def root_name(f):
    # OHE names look like "base_state_RISP" etc -> root is "base_state"
    return f.split("_")[0] if f.split("_")[0] in FEATURES_CAT2 else f

imp_gain["root_feature"] = imp_gain["feature"].map(root_name)
imp_gain_root = (
    imp_gain.groupby("root_feature", as_index=False)["gain_importance"]
            .sum()
            .sort_values("gain_importance", ascending=False)
            .reset_index(drop=True)
)

print("\nAggregated gain importance by original feature:")
display(imp_gain_root)

# ----- 3) Permutation importance on test set (uses original X_test2; pipeline handles transforms) -----
# This is slower; keep repeats small at first
perm = permutation_importance(
    model_xgb2,
    X_test2,
    y_test2_enc,
    n_repeats=3,
    random_state=42,
    scoring="neg_log_loss",  # aligns with your objective
    n_jobs=-1,
)

imp_perm = (
    pd.DataFrame({"feature": FEATURES2, "perm_importance_mean": perm.importances_mean,
                  "perm_importance_std": perm.importances_std})
      .sort_values("perm_importance_mean", ascending=False)
      .reset_index(drop=True)
)

print("\nPermutation importance (neg_log_loss): higher = more important")
display(imp_perm)

