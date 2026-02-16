# %%
"""
02_cluster_pitch_types.py

Pitcher-specific GMM clustering to identify pitch types.
Uses BIC minimization to select optimal k per pitcher.

Input: data/processed/statcast_{dates}_labeled.parquet
Output: data/processed/statcast_{dates}_clustered.parquet
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", None)

# %%
START_DT = "2023-04-01"
END_DT = "2024-10-01"

IN_PATH = Path("data/processed") / f"statcast_{START_DT}_{END_DT}_labeled.parquet"
OUT_PATH = Path("data/processed") / f"statcast_{START_DT}_{END_DT}_clustered.parquet"

ART_DIR = Path("artifacts/clustering")
ART_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = ART_DIR / "clustering_metrics.json"

# %%
# Load data
df = pd.read_parquet(IN_PATH)
print("Loaded:", IN_PATH, df.shape)
input_rows = len(df)

# %%
# === INPUT VALIDATION ===
print("\n=== INPUT VALIDATION ===")

# Required columns from 01_clean_targets.py
REQUIRED_COLS = ["pitcher", "game_date", "target_raw", "y_swing", "y_whiff", "split"]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns from 01_clean_targets.py: {missing}")
print(f"Required columns present: {REQUIRED_COLS}")

# Check target columns have expected values
swing_defined = df["y_swing"].notna().sum()
print(f"y_swing defined: {swing_defined} / {len(df)} ({swing_defined/len(df)*100:.1f}%)")

# Check split column
split_counts = df["split"].value_counts()
print(f"Split distribution: {split_counts.to_dict()}")

# %%
# Pitcher id is essential for pitcher-specific clustering
if "pitcher" not in df.columns:
    raise ValueError("Missing `pitcher` column in Statcast pull; needed for pitcher-specific clustering.")

# We will validate vs observed pitch_type if present, but do not require it
has_pitch_type = "pitch_type" in df.columns

# %%
# Features used for clustering (only those that exist)
CLUSTER_COL_CANDIDATES = [
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "spin_axis",
    "release_extension",
    "release_pos_x",
    "release_pos_z",
    "release_pos_y",
    "vx0", "vy0", "vz0",
    "ax", "ay", "az",
]

cluster_cols = [c for c in CLUSTER_COL_CANDIDATES if c in df.columns]
print("Clustering cols:", cluster_cols)

if len(cluster_cols) < 4:
    raise ValueError(f"Too few clustering columns found ({len(cluster_cols)}). Found: {cluster_cols}")

# %%
# Clean: keep only rows with enough data to cluster (memory-optimized)
# Convert cluster columns to numeric in-place on df
for c in cluster_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

need = ["pitcher"] + cluster_cols
rows_before_null_drop = len(df)

# Create mask for valid rows instead of copying
valid_mask = df[need].notna().all(axis=1)
valid_indices = df.index[valid_mask].to_numpy()
print(f"Valid rows for clustering: {len(valid_indices)} ({len(valid_indices)/rows_before_null_drop*100:.1f}% retained)")

# %%
# Pitcher-specific clustering via GMM with BIC-based k selection
# For each pitcher:
# - if too few pitches, skip and set cluster as NaN
# - else: scale, try k=2..KMAX, pick lowest BIC
KMIN = 2
KMAX = 6
MIN_PITCHES_PER_PITCHER = 200
N_JOBS = 4  # Use 4 cores to balance speed vs memory (each worker gets copy of data)


def cluster_pitcher(pitcher_id, idx, X_data, pitch_types=None):
    """Cluster a single pitcher's pitches using GMM with BIC selection."""
    idx = np.array(idx, dtype=int)

    if len(idx) < MIN_PITCHES_PER_PITCHER:
        return None  # Skip this pitcher

    X = X_data[idx]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best_k = None
    best_model = None
    best_bic = np.inf

    for k in range(KMIN, KMAX + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42,
            reg_covar=1e-6,
            max_iter=200,
        )
        gmm.fit(Xs)
        bic = gmm.bic(Xs)
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_model = gmm

    labels = best_model.predict(Xs).astype(int)
    cluster_names = [f"P{int(pitcher_id)}_C{int(l)}" for l in labels]

    rec = {
        "pitcher": int(pitcher_id),
        "n_pitches": int(len(idx)),
        "k_chosen": int(best_k),
        "bic": float(best_bic),
    }

    # Optional validation vs pitch_type within pitcher
    if pitch_types is not None:
        pt = pitch_types[idx]
        ari = adjusted_rand_score(pt, labels)
        nmi = normalized_mutual_info_score(pt, labels)
        rec["ari_vs_pitch_type"] = float(ari)
        rec["nmi_vs_pitch_type"] = float(nmi)

    return {
        "idx": idx,
        "labels": labels,
        "cluster_names": cluster_names,
        "record": rec,
    }


# Prepare data for parallel processing (memory-optimized)
# Extract only the data we need as numpy arrays
X_data = df.loc[valid_indices, cluster_cols].to_numpy(dtype=np.float32)  # float32 saves 50% memory
pitcher_ids = df.loc[valid_indices, "pitcher"].to_numpy()
pitch_types = None
if has_pitch_type:
    pitch_types = df.loc[valid_indices, "pitch_type"].astype("string").fillna("NA").to_numpy()

# Build pitcher -> local indices mapping (indices into X_data, not df)
from collections import defaultdict
pitcher_to_local_idx = defaultdict(list)
for local_idx, pid in enumerate(pitcher_ids):
    pitcher_to_local_idx[pid].append(local_idx)
groups = {pid: np.array(idxs, dtype=int) for pid, idxs in pitcher_to_local_idx.items()}
del pitcher_to_local_idx  # free memory

print(f"\nClustering {len(groups)} pitchers using {N_JOBS} CPU cores...")
print("(N_JOBS=-1 means all available cores)")

# Run clustering in parallel
results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(cluster_pitcher)(pitcher_id, idx, X_data, pitch_types)
    for pitcher_id, idx in groups.items()
)

# Aggregate results (memory-optimized: write directly to df)
# Initialize columns with NaN
df["pitch_cluster_id"] = np.nan
df["pitch_cluster_name"] = None

# Arrays to hold results for valid rows
local_cluster_id = np.full(len(valid_indices), fill_value=-1, dtype=np.int32)
local_cluster_name = np.full(len(valid_indices), fill_value="", dtype=object)

records = []
pitchers_clustered = 0
pitchers_skipped = 0

for res in results:
    if res is None:
        pitchers_skipped += 1
    else:
        local_cluster_id[res["idx"]] = res["labels"]
        local_cluster_name[res["idx"]] = res["cluster_names"]
        records.append(res["record"])
        pitchers_clustered += 1

# Write results back to df at valid_indices positions
df.loc[valid_indices, "pitch_cluster_id"] = local_cluster_id
df.loc[valid_indices, "pitch_cluster_name"] = local_cluster_name
df.loc[df["pitch_cluster_id"] < 0, ["pitch_cluster_id", "pitch_cluster_name"]] = [np.nan, None]

# Free memory
del local_cluster_id, local_cluster_name, X_data, pitcher_ids
if pitch_types is not None:
    del pitch_types

print(f"Clustered pitchers: {pitchers_clustered}")
print(f"Skipped pitchers (<{MIN_PITCHES_PER_PITCHER} pitches): {pitchers_skipped}")

# %%
# For interpretability: mode pitch_type per cluster (if available)
if has_pitch_type:
    # Compute mode without creating copies - use df directly
    clustered_mask = df["pitch_cluster_name"].notna()
    mode_map = (
        df.loc[clustered_mask, ["pitcher", "pitch_cluster_name", "pitch_type"]]
        .assign(pitch_type=lambda x: x["pitch_type"].astype("string"))
        .groupby(["pitcher", "pitch_cluster_name"])["pitch_type"]
        .agg(lambda s: s.value_counts().index[0] if len(s) else "NA")
        .reset_index()
        .rename(columns={"pitch_type": "pitch_type_mode"})
    )
    df = df.merge(mode_map, on=["pitcher", "pitch_cluster_name"], how="left")
    del mode_map
else:
    df["pitch_type_mode"] = np.nan

# df is now our output (no separate df_out needed)
df_out = df

# %%
# === OUTPUT VALIDATION ===
print("\n=== OUTPUT VALIDATION ===")
print(f"Input rows: {input_rows}")
print(f"Output rows: {len(df_out)} ({len(df_out)/input_rows*100:.1f}% retained)")

# Cluster coverage
cluster_assigned = df_out["pitch_cluster_id"].notna().sum()
print(f"Rows with cluster assignment: {cluster_assigned} ({cluster_assigned/len(df_out)*100:.1f}%)")

# Verify required columns preserved
for col in REQUIRED_COLS:
    assert col in df_out.columns, f"Required column {col} lost during merge!"
print(f"All required columns preserved: {REQUIRED_COLS}")

# Verify split column integrity
split_counts_out = df_out["split"].value_counts()
print(f"Split distribution (output): {split_counts_out.to_dict()}")

# %%
df_out.to_parquet(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")
print(f"Shape: {df_out.shape}")

# %%
# Save clustering metrics
summary = {
    "input_path": str(IN_PATH),
    "output_path": str(OUT_PATH),
    "cluster_cols": cluster_cols,
    "kmin": KMIN,
    "kmax": KMAX,
    "min_pitches_per_pitcher": MIN_PITCHES_PER_PITCHER,
    "n_pitchers_clustered": int(pitchers_clustered),
    "n_pitchers_skipped": int(pitchers_skipped),
    "pitcher_rows_clustered": int(cluster_assigned),
    "pitcher_records": records[:200],
}

with open(METRICS_PATH, "w") as f:
    json.dump(summary, f, indent=2)

print("Saved metrics:", METRICS_PATH)
#%%