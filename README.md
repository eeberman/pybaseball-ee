# Pitcher Whiff Skill

A Statcast-based machine learning system that measures how much a pitcher over- or under-generates whiffs relative to what their pitch quality predicts.

**Live site**: [eeberman.github.io/pybaseball-ee](https://eeberman.github.io/pybaseball-ee/)

---

## What It Does

For every swing in the 2024 MLB regular season, the model asks: *given the pitch physics and game situation, what whiff rate would we expect from an average batter?* The difference between the actual rate and that expectation is **whiff skill** — a pitcher-specific ability estimate stripped of difficulty-of-schedule noise.

Key features:
- **Two-stage cascade**: Model 1 predicts swing vs. take; Model 2 predicts whiff vs. contact on swings
- **Pitch-only features**: No pitcher identity — the model only sees pitch physics, trajectory, and game state
- **Isotonic calibration**: Probability outputs are well-calibrated (25% Brier score improvement)
- **Playoff comparison**: RS model applied to 2024 postseason to detect performance changes

---

## Data Pipeline

```
Baseball Savant (Statcast)
    │
    ├─ 01_clean_targets.py        Pull & cache 2023-2024 Statcast data
    ├─ 02_cluster_pitch_types.py  Pitcher-specific GMM clustering (k=2-6, BIC)
    ├─ 03_feature_engineering.py  Batter-relative coords, zone, trajectory features
    ├─ 04_build_model_datasets.py Imputation, train-ready datasets
    ├─ 05_train_and_validate.py   XGBoost training + isotonic calibration
    ├─ 06_player_skill_metrics.py Batter contact skill & pitcher whiff skill
    ├─ 07_aggregate_for_website.py Regular season leaderboard JSON
    └─ 08_playoff_comparison.py   Playoff comparison JSON
```

Each script uses Jupyter percent-format (`# %%` cells) and can be run directly:
```bash
python Baseball/01_clean_targets.py
python Baseball/02_cluster_pitch_types.py
# ... continue in order
```

---

## Model Performance (2024 Test Set)

### Model 1 — Swing vs. Take

| Split | ROC-AUC | PR-AUC | Brier | Accuracy |
|-------|---------|--------|-------|----------|
| Train | 0.8667  | 0.8508 | 0.1491 | 77.9% |
| Val   | 0.8440  | 0.8287 | 0.1611 | 75.8% |
| Test  | 0.8426  | 0.8292 | 0.1619 | 75.8% |

### Model 2 — Whiff vs. Contact (on swings)

| Split | ROC-AUC | PR-AUC | Brier (uncal) | Brier (cal) |
|-------|---------|--------|---------------|-------------|
| Train | 0.7919  | 0.6020 | 0.1815        | —           |
| Val   | 0.7681  | 0.5733 | 0.1901        | —           |
| Test  | 0.7682  | 0.5675 | 0.1899        | **0.1420**  |

### Zone Diagnostics (Test Set)

| Zone              | ROC-AUC | PR-AUC | Whiff Rate |
|-------------------|---------|--------|------------|
| In-zone           | 0.6797  | 0.2681 | 14.4%      |
| Out-of-zone       | 0.7578  | 0.7049 | 40.1%      |
| OOZ Breaking Balls| 0.7744  | 0.8059 | 52.6%      |

---

## Key Design Decisions

**No pitcher identity features**: Cluster-derived statistics (mean velocity, spin, movement per pitch cluster) are deliberately excluded so that "expected" reflects pitch physics alone, not who threw it.

**Temporal holdout**: 2023 data for training/validation; 2024 data for test. No data leakage across seasons.

**Calibration matters**: When the model predicts 90% whiff probability on out-of-zone breaking balls, the actual whiff rate is 91.8%. Probability outputs are genuinely informative.

See the [Methodology page](https://eeberman.github.io/pybaseball-ee/methodology.html) for full details.

---

## Website

Static site deployed to GitHub Pages — no build step, no npm.

```
Baseball/website/
├── index.html       Pitcher RS leaderboard
├── pitcher.html     Individual pitcher detail (pitch type chart + zone heatmap)
├── playoffs.html    Playoff leaderboard + risers/fallers chart
├── methodology.html Model methodology documentation
├── styles.css       Dark theme
├── app.js / pitcher.js / playoffs.js
└── data/            Pre-computed JSON (~530KB total)
```

Deployed automatically on push to `main` via `.github/workflows/pages.yml`.

---

## Local Setup

```bash
pip install pandas numpy pyarrow xgboost scikit-learn matplotlib joblib requests
```

Data is fetched automatically by `01_clean_targets.py` from Baseball Savant. Raw parquet files are cached in `Baseball/data/cache/` after the first run.

To regenerate website data only (if models already exist):
```bash
python Baseball/07_aggregate_for_website.py
python Baseball/08_playoff_comparison.py
```

---

## Data Source

Baseball Savant / MLB Statcast via direct CSV API (`baseballsavant.mlb.com`).
All data is publicly available. 2024 season: April–October.
