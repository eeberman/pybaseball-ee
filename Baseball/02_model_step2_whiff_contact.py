# %%
from __future__ import annotations

# %%
# Imports + settings

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    log_loss,
)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", None)

# %%
# Config

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TRAIN = 0.20

NUM_BOOST_ROUND = 4000
EARLY_STOPPING_ROUNDS = 150

DATA_PATH = Path("data/processed") / "statcast_2023_step2_whiff_contact.parquet"

STEP1_ART_DIR = Path("artifacts/step1")
STEP2_ART_DIR = Path("artifacts/step2")
STEP2_ART_DIR.mkdir(parents=True, exist_ok=True)

STEP1_METRICS_PATH = STEP1_ART_DIR / "run_metrics.json"
STEP1_BOOSTER_PATH = STEP1_ART_DIR / "xgb_step1_booster.json"

STEP2_BOOSTER_PATH = STEP2_ART_DIR / "xgb_step2_booster.json"
STEP2_METRICS_PATH = STEP2_ART_DIR / "run_metrics.json"
STEP2_SWEEP_PATH = STEP2_ART_DIR / "threshold_sweep_val.csv"

THRESHOLD_POLICY = "recall_at_precision"  # "best_f1" | "recall_at_precision" | "precision_at_recall"
TARGET_PRECISION = 0.70
TARGET_RECALL = 0.50


# %%
# Helpers

def load_metrics(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def infer_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    target_cols = {"target_raw", "y_swing", "y_whiff"}

    if "y_whiff" not in df.columns:
        raise ValueError("Expected column y_whiff in step2 dataset but it is missing.")

    feature_cols = [c for c in df.columns if c not in target_cols]

    if "game_date" in feature_cols:
        feature_cols.remove("game_date")

    cat_cols = [c for c in feature_cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    return feature_cols, cat_cols, num_cols


def build_design_matrix(
    X: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    categories_fit: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, dict[str, list[str]], list[str]]:
    Xc = X.copy()

    for c in cat_cols:
        Xc[c] = Xc[c].astype("string")

    if categories_fit is None:
        categories_fit = {c: sorted(Xc[c].dropna().unique().tolist()) for c in cat_cols}

    for c in cat_cols:
        Xc[c] = pd.Categorical(Xc[c], categories=categories_fit[c])

    X_cat = pd.get_dummies(Xc[cat_cols], dummy_na=False) if cat_cols else pd.DataFrame(index=Xc.index)
    X_num = Xc[num_cols].astype("float32") if num_cols else pd.DataFrame(index=Xc.index)

    X_out = pd.concat([X_num, X_cat], axis=1).astype("float32")
    feature_names = X_out.columns.tolist()
    return X_out.to_numpy(), categories_fit, feature_names


def train_xgb(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[xgb.Booster, dict]:
    evals_result: dict = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=25,
    )
    return booster, evals_result


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    thresholds = np.append(thresholds, 1.0)

    f1 = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0.0
    )

    out = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })
    return out


def pick_threshold(sweep: pd.DataFrame) -> tuple[float, dict]:
    sweep2 = sweep.dropna(subset=["threshold", "precision", "recall", "f1"]).copy()
    if sweep2.empty:
        raise ValueError("Threshold sweep is empty after dropping NaNs.")

    if THRESHOLD_POLICY == "best_f1":
        idx = sweep2["f1"].idxmax()
        row = sweep2.loc[idx].to_dict()
        return float(row["threshold"]), row

    if THRESHOLD_POLICY == "recall_at_precision":
        eligible = sweep2[sweep2["precision"] >= TARGET_PRECISION]
        if eligible.empty:
            idx = sweep2["f1"].idxmax()
            row = sweep2.loc[idx].to_dict()
            return float(row["threshold"]), {**row, "note": f"no threshold met precision>={TARGET_PRECISION}; fell back to best_f1"}

        idx = eligible["recall"].idxmax()
        row = eligible.loc[idx].to_dict()
        return float(row["threshold"]), row

    if THRESHOLD_POLICY == "precision_at_recall":
        eligible = sweep2[sweep2["recall"] >= TARGET_RECALL]
        if eligible.empty:
            idx = sweep2["f1"].idxmax()
            row = sweep2.loc[idx].to_dict()
            return float(row["threshold"]), {**row, "note": f"no threshold met recall>={TARGET_RECALL}; fell back to best_f1"}

        idx = eligible["precision"].idxmax()
        row = eligible.loc[idx].to_dict()
        return float(row["threshold"]), row

    raise ValueError(f"Unknown THRESHOLD_POLICY={THRESHOLD_POLICY}")


# %%
# Load data

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing step2 dataset: {DATA_PATH}")

df = pd.read_parquet(DATA_PATH)
print(f"Loaded: {DATA_PATH} {df.shape}")

# %%
# Clean + target

df = df.dropna(subset=["y_whiff"]).copy()
df["y_whiff"] = df["y_whiff"].astype(int)

print("y_whiff distribution:")
print(df["y_whiff"].value_counts(normalize=True).rename("proportion"))

# %%
# Feature columns (auto-detect)

feature_cols, cat_cols, num_cols = infer_feature_columns(df)
X = df[feature_cols].copy()
y = df["y_whiff"].copy()

print("X shape:", X.shape, "y shape:", y.shape)
print("Feature columns:", feature_cols)
print("Categorical:", cat_cols)
print("Numeric count:", len(num_cols))

# %%
# Split: train/val/test (stratified)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=VAL_SIZE_WITHIN_TRAIN,
    stratify=y_train,
    random_state=RANDOM_STATE,
)

print("Train:", X_tr.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# %%
# Build design matrices (one-hot cats)

t0 = time.time()

X_tr_mat, cats_fit, feat_names = build_design_matrix(X_tr, cat_cols, num_cols, categories_fit=None)
X_val_mat, _, _ = build_design_matrix(X_val, cat_cols, num_cols, categories_fit=cats_fit)
X_test_mat, _, _ = build_design_matrix(X_test, cat_cols, num_cols, categories_fit=cats_fit)

dtrain = xgb.DMatrix(X_tr_mat, label=y_tr.to_numpy(), feature_names=feat_names)
dval = xgb.DMatrix(X_val_mat, label=y_val.to_numpy(), feature_names=feat_names)
dtest = xgb.DMatrix(X_test_mat, label=y_test.to_numpy(), feature_names=feat_names)

print(f"Design matrices built in {time.time() - t0:.1f}s")
print("n_features_after_onehot:", dtrain.num_col())

# %%
# XGB params
# Note: early stopping uses the LAST metric in eval_metric by default; we want AUC-PR.
# This optimizes ranking for the positive class (whiffs), which is what you want.

params = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "aucpr"],
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "seed": RANDOM_STATE,
    "nthread": -1,
}

# %%
# Train (xgb.train + early stopping)

print("Training Step 2 booster...")
t1 = time.time()

booster, evals_result = train_xgb(
    dtrain=dtrain,
    dval=dval,
    params=params,
    num_boost_round=NUM_BOOST_ROUND,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
)

train_time = time.time() - t1

best_iter = int(booster.best_iteration) if hasattr(booster, "best_iteration") else None
best_score = float(booster.best_score) if hasattr(booster, "best_score") else None

print(f"Training done in {train_time:.1f}s")
print("Best iteration:", (best_iter + 1) if best_iter is not None else "unknown")
print("Best val aucpr (early stopping metric):", best_score)

# %%
# Probabilities (use best_iteration range for safety)

iter_range = (0, booster.best_iteration + 1) if hasattr(booster, "best_iteration") else None

val_proba = booster.predict(dval, iteration_range=iter_range) if iter_range else booster.predict(dval)
test_proba = booster.predict(dtest, iteration_range=iter_range) if iter_range else booster.predict(dtest)

# Core ranking metrics (threshold-free)
val_pr_auc = float(average_precision_score(y_val, val_proba))
test_pr_auc = float(average_precision_score(y_test, test_proba))

val_roc_auc = float(roc_auc_score(y_val, val_proba))
test_roc_auc = float(roc_auc_score(y_test, test_proba))

print("\nRanking metrics (threshold-free)")
print("VAL  PR AUC:", val_pr_auc)
print("VAL  ROC AUC:", val_roc_auc)
print("TEST PR AUC:", test_pr_auc)
print("TEST ROC AUC:", test_roc_auc)

# %%
# Threshold sweep (VAL) to pick a lower threshold for whiffs

sweep = threshold_sweep(y_val.to_numpy(), val_proba)
sweep.to_csv(STEP2_SWEEP_PATH, index=False)
print("\nSaved threshold sweep:", STEP2_SWEEP_PATH)

chosen_threshold, chosen_row = pick_threshold(sweep)

print("\nChosen threshold policy:", THRESHOLD_POLICY)
if THRESHOLD_POLICY == "recall_at_precision":
    print("Target precision:", TARGET_PRECISION)
if THRESHOLD_POLICY == "precision_at_recall":
    print("Target recall:", TARGET_RECALL)
print("Chosen threshold:", chosen_threshold)
print("VAL point @ threshold:", {k: float(v) for k, v in chosen_row.items() if k in ["precision", "recall", "f1", "threshold"] or k == "note"})

# %%
# Evaluate on TEST at chosen threshold

test_pred = (test_proba >= chosen_threshold).astype(int)

acc = float(accuracy_score(y_test, test_pred))
ll = float(log_loss(y_test, test_proba))
auc = float(roc_auc_score(y_test, test_proba))

report = classification_report(y_test, test_pred, digits=3)

cm = confusion_matrix(y_test, test_pred)
cm_df = pd.DataFrame(
    cm,
    index=["true_contact(0)", "true_whiff(1)"],
    columns=["pred_contact(0)", "pred_whiff(1)"],
)

print("\nXGB Step 2 (whiff vs contact) — Early Stopping + Threshold policy")
print("Threshold:", chosen_threshold)
print("Accuracy:", acc)
print("LogLoss:", ll)
print("ROC AUC:", auc)
print("PR AUC (Average Precision):", test_pr_auc)
print("\nReport:\n", report)
print("\nConfusion matrix:\n", cm_df)

# %%
# Save booster

booster.save_model(str(STEP2_BOOSTER_PATH))
print("\nSaved booster:", STEP2_BOOSTER_PATH)

# %%
# Save run metrics (include step1 snapshot)

step1_metrics = load_metrics(STEP1_METRICS_PATH)

run_metrics = {
    "task": "step2_whiff_vs_contact",
    "data_path": str(DATA_PATH),
    "n_rows_total": int(len(df)),
    "splits": {
        "train_rows": int(len(X_tr)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "test_size": TEST_SIZE,
        "val_size_within_train": VAL_SIZE_WITHIN_TRAIN,
        "random_state": RANDOM_STATE,
        "stratify": True,
    },
    "labels": {
        "positive_class": "whiff(1)",
        "negative_class": "contact(0)",
        "y_distribution": df["y_whiff"].value_counts(normalize=True).to_dict(),
    },
    "features": {
        "feature_cols_pre_onehot": feature_cols,
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols,
        "onehot_categories": {k: v for k, v in cats_fit.items()},
        "n_features_after_onehot": int(dtrain.num_col()),
    },
    "xgb_params": {
        **params,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    },
    "training": {
        "best_iteration": (best_iter + 1) if best_iter is not None else None,
        "best_val_aucpr": best_score,
        "train_time_seconds": train_time,
        "evals_result_tail": {
            "train_logloss_last": float(evals_result["train"]["logloss"][-1]) if "train" in evals_result else None,
            "val_logloss_last": float(evals_result["val"]["logloss"][-1]) if "val" in evals_result else None,
            "train_aucpr_last": float(evals_result["train"]["aucpr"][-1]) if "train" in evals_result and "aucpr" in evals_result["train"] else None,
            "val_aucpr_last": float(evals_result["val"]["aucpr"][-1]) if "val" in evals_result and "aucpr" in evals_result["val"] else None,
        },
    },
    "ranking_metrics": {
        "val_pr_auc": val_pr_auc,
        "test_pr_auc": test_pr_auc,
        "val_roc_auc": val_roc_auc,
        "test_roc_auc": test_roc_auc,
    },
    "thresholding": {
        "policy": THRESHOLD_POLICY,
        "target_precision": TARGET_PRECISION if THRESHOLD_POLICY == "recall_at_precision" else None,
        "target_recall": TARGET_RECALL if THRESHOLD_POLICY == "precision_at_recall" else None,
        "chosen_threshold": float(chosen_threshold),
        "val_point": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in chosen_row.items()},
        "sweep_csv_path": str(STEP2_SWEEP_PATH),
    },
    "test_metrics_at_threshold": {
        "threshold": float(chosen_threshold),
        "accuracy": acc,
        "logloss": ll,
        "roc_auc": auc,
        "confusion_matrix": cm_df.to_dict(),
        "classification_report_text": report,
    },
    "references": {
        "step1_metrics_path": str(STEP1_METRICS_PATH),
        "step1_booster_path": str(STEP1_BOOSTER_PATH),
        "step1_metrics_loaded": bool(step1_metrics),
        "step1_metrics_snapshot": step1_metrics if step1_metrics else None,
    },
}

with open(STEP2_METRICS_PATH, "w") as f:
    json.dump(run_metrics, f, indent=2)

print("\nSaved metrics:", STEP2_METRICS_PATH)

# %%
# Done

print("\nSaved:")
print(" -", STEP2_BOOSTER_PATH)
print(" -", STEP2_METRICS_PATH)
print(" -", STEP2_SWEEP_PATH)

# %% Diagosing where the low recall for whiffs is coming from 

# %%
# Feature importance 

import json
from pathlib import Path

IMP_PATH = Path("artifacts/step2") / "feature_importance.json"

score_gain = booster.get_score(importance_type="gain")
score_weight = booster.get_score(importance_type="weight")
score_cover = booster.get_score(importance_type="cover")

def to_sorted_list(d):
    return sorted(
        [{"feature": k, "value": float(v)} for k, v in d.items()],
        key=lambda x: x["value"],
        reverse=True,
    )

imp = {
    "gain": to_sorted_list(score_gain),
    "weight": to_sorted_list(score_weight),
    "cover": to_sorted_list(score_cover),
}

with open(IMP_PATH, "w") as f:
    json.dump(imp, f, indent=2)

print("Saved feature importance:", IMP_PATH)
print("\nTop 20 by gain:")
for row in imp["gain"][:20]:
    print(row)


# %%
# Permutation importance for PR AUC 

from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
from pathlib import Path

PERM_PATH = Path("artifacts/step2") / "perm_importance_pr_auc.csv"

rng = np.random.default_rng(42)

def pr_auc_from_mat(X_mat):
    d = xgb.DMatrix(X_mat, feature_names=feat_names)
    p = booster.predict(d)
    return float(average_precision_score(y_test.to_numpy(), p))

baseline_pr = pr_auc_from_mat(X_test_mat)
print("Baseline TEST PR AUC:", baseline_pr)

n_repeats = 3
drops = []

X_work = X_test_mat.copy()

for j, fname in enumerate(feat_names):
    scores = []
    for _ in range(n_repeats):
        saved = X_work[:, j].copy()
        X_work[:, j] = rng.permutation(X_work[:, j])
        scores.append(pr_auc_from_mat(X_work))
        X_work[:, j] = saved
    mean_score = float(np.mean(scores))
    drop = baseline_pr - mean_score
    drops.append({"feature": fname, "pr_auc_drop": drop, "pr_auc_perm": mean_score})

perm_df = pd.DataFrame(drops).sort_values("pr_auc_drop", ascending=False)
perm_df.to_csv(PERM_PATH, index=False)

print("Saved permutation importance:", PERM_PATH)
print("\nTop 25 features by PR AUC drop:")
print(perm_df.head(25))

# %%
# Segment diagnostics 

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

threshold = float(chosen_threshold) if "chosen_threshold" in globals() else 0.5

if "test_proba" not in globals():
    raise NameError("Expected test_proba to exist. Make sure you ran the 'Probabilities' cell above.")

test_pred = (test_proba >= threshold).astype(int)

test_df = X_test.copy().reset_index(drop=True)
test_df["y_true"] = y_test.to_numpy()
test_df["proba"] = np.asarray(test_proba)
test_df["pred"] = np.asarray(test_pred)

def seg_row(df_seg: pd.DataFrame, name: str) -> dict:
    if len(df_seg) == 0:
        return {"segment": name, "n": 0, "base_rate_whiff": np.nan, "precision": np.nan, "recall": np.nan}
    y_true = df_seg["y_true"].to_numpy()
    y_pred = df_seg["pred"].to_numpy()
    return {
        "segment": name,
        "n": int(len(df_seg)),
        "base_rate_whiff": float(y_true.mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

rows = [seg_row(test_df, "ALL")]

for col in ["two_strikes", "in_zone", "same_side", "risp", "bases_loaded"]:
    if col in test_df.columns:
        rows.append(seg_row(test_df[test_df[col] == 1], f"{col}=1"))
        rows.append(seg_row(test_df[test_df[col] == 0], f"{col}=0"))

seg = pd.DataFrame(rows).sort_values(["segment"]).reset_index(drop=True)

print("\nSegment diagnostics @ threshold =", threshold)
print(seg)

# %%
# Count_state 

if "count_state" in test_df.columns:
    by_count = (
        test_df.groupby("count_state")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "base_rate_whiff": g["y_true"].mean(),
            "precision": precision_score(g["y_true"], g["pred"], zero_division=0),
            "recall": recall_score(g["y_true"], g["pred"], zero_division=0),
        }))
        .sort_values("n", ascending=False)
    )
    print("\nBy count_state (top 15 by n):")
    print(by_count.head(15))

# %%
# velocity bins

if "effective_speed" in test_df.columns:
    velo_bins = [0, 80, 90, 95, 100, 105, 120]
    test_df["velo_bin"] = pd.cut(test_df["effective_speed"], bins=velo_bins, include_lowest=True)

    by_velo = (
        test_df.groupby("velo_bin", observed=True)
        .apply(lambda g: pd.Series({
            "n": len(g),
            "base_rate_whiff": g["y_true"].mean(),
            "precision": precision_score(g["y_true"], g["pred"], zero_division=0),
            "recall": recall_score(g["y_true"], g["pred"], zero_division=0),
        }))
    )
    print("\nBy velocity bin:")
    print(by_velo)


# %% in-zone vs out-of-zone performance

test_df = X_test.copy().reset_index(drop=True)
test_df["y_true"] = y_test.to_numpy()
test_df["proba"] = test_proba  # from booster.predict(dtest, ...)
test_df["in_zone"] = X_test["in_zone"].to_numpy()

for z in [0, 1]:
    g = test_df[test_df["in_zone"] == z]
    ap = average_precision_score(g["y_true"], g["proba"])
    auc = roc_auc_score(g["y_true"], g["proba"])
    print(f"in_zone={z} | n={len(g)} | base={g['y_true'].mean():.3f} | PR-AUC={ap:.3f} | ROC-AUC={auc:.3f}")

# %%
