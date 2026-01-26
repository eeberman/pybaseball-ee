# %%
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

import xgboost as xgb


# %%
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TRAIN = 0.20

EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND = 2000

DATA_STEP1_PATH = Path("data/processed") / "statcast_2023_step1_swing_take.parquet"
ARTIFACT_DIR = Path("artifacts") / "step1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "xgb_step1_booster.json"
RUN_METRICS_PATH = ARTIFACT_DIR / "run_metrics.json"


# %%
if not DATA_STEP1_PATH.exists():
    raise FileNotFoundError(
        f"Missing {DATA_STEP1_PATH}. Run feature engineering script first to produce it."
    )

df = pd.read_parquet(DATA_STEP1_PATH)
print("Loaded:", DATA_STEP1_PATH, df.shape)
print("y_swing distribution:\n", df["y_swing"].value_counts(normalize=True))


# %%
# Targets / columns
TARGET_COL = "y_swing"

DROP_COLS = [
    "target_raw",
    "y_whiff",
]

# game_date is a timestamp and not needed as a feature (and will break xgboost if left as Timestamp)
if "game_date" in df.columns:
    DROP_COLS.append("game_date")

# Basic sanity
for col in [TARGET_COL]:
    if col not in df.columns:
        raise ValueError(f"Expected target column '{col}' not found in dataset.")


# %%
# Build X/y
df_model = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()

y = df_model[TARGET_COL].astype(int)
X = df_model.drop(columns=[TARGET_COL])

print("X shape:", X.shape, "y shape:", y.shape)
print("Feature columns:", list(X.columns))


# %%
# Identify categorical vs numeric columns
# Treat these as categorical (string / limited-set)
CATEGORICAL_COLS = [
    "p_throws",
    "stand",
    "count_state",
    "score_bucket",
]

# Keep only those that exist
CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in X.columns]

NUMERIC_COLS = [c for c in X.columns if c not in CATEGORICAL_COLS]

print("Categorical:", CATEGORICAL_COLS)
print("Numeric count:", len(NUMERIC_COLS))


# %%
# Split train/test, then train/val (for early stopping)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=VAL_SIZE_WITHIN_TRAIN,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)


# %%
# Preprocess: OneHotEncode categoricals, passthrough numeric
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ],
    remainder="passthrough",
)

X_tr_p = preprocess.fit_transform(X_train)
X_val_p = preprocess.transform(X_val)
X_test_p = preprocess.transform(X_test)

# xgboost DMatrix works best with CSR/CSC or numpy arrays; ColumnTransformer output is typically sparse.
dtrain = xgb.DMatrix(X_tr_p, label=y_train.to_numpy())
dval = xgb.DMatrix(X_val_p, label=y_val.to_numpy())
dtest = xgb.DMatrix(X_test_p, label=y_test.to_numpy())


# %%
# XGBoost params
# %%
import time
import xgboost as xgb

t0 = time.time()

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "lambda": 1.0,
    "alpha": 0.0,
    "gamma": 0.0,
    "seed": RANDOM_STATE,
    "verbosity": 1,
}

print("Building DMatrix...")
t1 = time.time()
dtrain = xgb.DMatrix(X_tr_p, label=y_train.to_numpy())
dval   = xgb.DMatrix(X_val_p, label=y_val.to_numpy())
dtest  = xgb.DMatrix(X_test_p, label=y_test.to_numpy())
print(f"DMatrix built in {time.time() - t1:,.1f}s")

evals = [(dtrain, "train"), (dval, "val")]

print("Training...")
t2 = time.time()
evals_result = {}

booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=600,
    evals=evals,
    early_stopping_rounds=30,
    evals_result=evals_result,
    verbose_eval=25,
)

print(f"Training done in {time.time() - t2:,.1f}s")
print("Total elapsed:", f"{time.time() - t0:,.1f}s")
print("Best iteration:", booster.best_iteration + 1)
print("Best val logloss:", booster.best_score)

evals = [(dtrain, "train"), (dval, "val")]





# %%
# Predict with best iteration
best_iter = getattr(booster, "best_iteration", None)
if best_iter is not None:
    proba = booster.predict(dtest, iteration_range=(0, best_iter + 1))
else:
    proba = booster.predict(dtest)

pred = (proba >= 0.5).astype(int)

acc = accuracy_score(y_test, pred)
ll = log_loss(y_test, proba)
auc = roc_auc_score(y_test, proba)

print("\nXGB Step 1 (swing vs take) — Early Stopping")
print("Accuracy:", acc)
print("LogLoss:", ll)
print("ROC AUC:", auc)
print("\nReport:\n", classification_report(y_test, pred, digits=3))

cm = confusion_matrix(y_test, pred)
cm_df = pd.DataFrame(
    cm,
    index=["true_take(0)", "true_swing(1)"],
    columns=["pred_take(0)", "pred_swing(1)"],
)
print("\nConfusion matrix:\n", cm_df)
print("Best iteration used:", (best_iter + 1) if best_iter is not None else "unknown")


# %%
# Save model + metrics artifact
booster.save_model(MODEL_PATH)

run_metrics = {
    "data_path": str(DATA_STEP1_PATH),
    "model_path": str(MODEL_PATH),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "val_size_within_train": VAL_SIZE_WITHIN_TRAIN,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    "num_boost_round": NUM_BOOST_ROUND,
    "best_iteration_used": (best_iter + 1) if best_iter is not None else None,
    "accuracy": float(acc),
    "logloss": float(ll),
    "roc_auc": float(auc),
    "n_train": int(len(y_train)),
    "n_val": int(len(y_val)),
    "n_test": int(len(y_test)),
    "categorical_cols": CATEGORICAL_COLS,
    "numeric_cols_count": int(len(NUMERIC_COLS)),
    "params": params,
}

with open(RUN_METRICS_PATH, "w") as f:
    json.dump(run_metrics, f, indent=2)

print("\nSaved:")
print(" -", MODEL_PATH)
print(" -", RUN_METRICS_PATH)
