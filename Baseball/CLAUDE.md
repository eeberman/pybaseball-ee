# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Baseball Statcast analysis project using a multi-stage ML pipeline:
- **Pitch clustering**: Pitcher-specific GMM clustering to identify pitch types
- **Two-stage classification**: Swing vs. take prediction, then whiff vs. contact for swings

## Running the Pipeline

Scripts use Jupyter percent-format (`# %%` cells) and run sequentially:

```bash
# Step 1: Pull and cache raw Statcast data
python 01_clean_targets.py

# Step 2: Pitcher-specific pitch clustering (GMM with BIC selection)
python 02_cluster_pitch_types.py

# Step 3: Feature engineering (includes cluster physical statistics + trajectory features)
python 03_feature_engineering.py

# Step 4: Build final model datasets
python 04_build_model_datasets.py

# Step 5: Train and validate models (includes zone diagnostics)
python 05_train_and_validate.py

# Step 6: Player skill evaluation (contact/whiff skill metrics)
python 06_player_skill_metrics.py

# Step 7: Generate website JSON data (regular season leaderboard)
python 07_aggregate_for_website.py

# Step 8: Generate playoff comparison data
python 08_playoff_comparison.py
```

**Note**: Step 2 expects `data/processed/*_labeled.parquet` - ensure intermediate data exists.
**Note**: Steps 7-8 generate JSON files in `website/data/` for the static website.

## Data Flow

```
pybaseball API → data/raw/*.parquet
    → *_labeled.parquet
    → 02_cluster_pitch_types.py → *_clustered.parquet
    → 03_feature_engineering.py → *_features.parquet (73 columns)
    → 04_build_model_datasets.py → statcast_model_base.parquet
                                 → statcast_step1_swing_take.parquet
                                 → statcast_step2_whiff_contact.parquet
    → 05_train_and_validate.py  → artifacts/models/*.json
                                 → artifacts/models/model2_calibrator.joblib
                                 → artifacts/training/training_metrics.json
    → 06_player_skill_metrics.py → artifacts/skill/batter_contact_skill.parquet
                                 → artifacts/skill/pitcher_whiff_skill.parquet
                                 → artifacts/skill/skill_summary.json
    → 07_aggregate_for_website.py → website/data/pitchers_overall.json (592 pitchers)
                                  → website/data/pitchers_by_pitch_type.json
                                  → website/data/pitchers_by_zone.json
                                  → website/data/metadata.json
                                  → website/data/name_cache.json
    → 08_playoff_comparison.py    → website/data/pitchers_playoffs.json
                                  → website/data/playoff_comparison.json
```

## Architecture

### Data Fetching
- `statcast_fetcher.py` - Direct Baseball Savant data fetcher (replaces pybaseball dependency)
  - Fetches from `https://baseballsavant.mlb.com/statcast_search/csv`
  - Automatic chunking (5 days per request due to ~30k row limit)
  - Retry logic with exponential backoff
  - Parquet caching in `data/cache/`
  - Verified to produce identical output to pybaseball (see `verify_fetcher.py`)

### Pipeline Scripts
- `01_clean_targets.py` - Pulls 2023-2024 Statcast data (Apr 2023 – Nov 2024), incremental fetch with dedup, caches to parquet
- `02_cluster_pitch_types.py` - GMM clustering per pitcher (k=2-6, BIC selection), validates against pitch_type
- `03_feature_engineering.py` - Creates batter-relative, zone, game state, cluster physical, and trajectory features
- `04_build_model_datasets.py` - Imputes optional features, builds train-ready datasets
- `05_train_and_validate.py` - Trains XGBoost models with calibration, evaluates cascade performance, zone diagnostics
- `06_player_skill_metrics.py` - Evaluates batter contact skill and pitcher whiff skill using model predictions
- `07_aggregate_for_website.py` - Generates RS leaderboard JSON (overall, by pitch type, by zone)
- `08_playoff_comparison.py` - Generates playoff leaderboard and riser/faller comparison JSON

### Website (Static, GitHub Pages)
Vanilla JS frontend with no build process, designed for GitHub Pages deployment.

**Pages:**
- `website/index.html` + `app.js` - Sortable/filterable pitcher leaderboard table (RS)
- `website/pitcher.html` + `pitcher.js` - Individual pitcher detail with pitch type bars + zone heatmap
- `website/playoffs.html` + `playoffs.js` - Playoff leaderboard and RS→PO riser/faller chart
- `website/methodology.html` - Static methodology page (no JS); covers model architecture, calibration, zone grid, two-problem discovery

**Nav:** All four pages share a consistent nav with links to Leaderboard, Playoffs, Methodology.

**Season labels:** All JS files fetch `metadata.json` and populate season span elements (`#season-label`, `#data-season`, `#po-season`, `#po-season-footer`) so Phase B only needs to update `metadata.json`, not HTML files.

**Assets:**
- `website/styles.css` - Dark theme styling
- `website/data/*.json` - Pre-computed data files (total ~530KB)

**Dependencies:** Chart.js via CDN (bar charts on pitcher detail and playoff pages), no npm/bundler.
**Target weight:** <500KB total JSON data (achieved via compact array format for zone data, no-indent serialization).

### Deployment
- **GitHub Actions workflow**: `.github/workflows/pages.yml`
- **Trigger**: push to `main` branch or manual `workflow_dispatch`
- **Deploys**: `Baseball/website/` as site root
- **Pages setting**: `build_type: workflow` (NOT legacy/branch deploy)
- **Live URL**: `https://eeberman.github.io/pybaseball-ee/`
- No CNAME / custom domain
- **README.md** at repo root (`pybaseball-ee/README.md`) — rendered on GitHub repo homepage

### Website Data Pipeline

**`07_aggregate_for_website.py` (Regular Season):**
1. Loads step1 (all pitches) + step2 (swings only) filtered to `split == "test"` (2024 data)
2. Runs Model 2 inference with isotonic calibration on step2 swings
3. Extracts team info from raw parquet: `pitcher` ↔ `inning_topbot` ↔ `home_team`/`away_team` (top inning = home team pitching)
4. Name lookup chain: raw `player_name` ("Last, First" → "First Last") → MLB API fallback → `name_cache.json`
5. Aggregates at 3 levels: overall (per pitcher), by pitch type (per pitcher × pitch_type_mode), by zone (per pitcher × 3×3 grid)
6. Zone assignment uses `plate_x_batter` (batter-relative) with fixed boundaries (see Zone Grid Definition)

**`08_playoff_comparison.py` (Playoffs):**
1. Loads 2024 playoff pitches from raw parquet (`game_type` in F/D/L/W), fetches additional data from Baseball Savant API (graceful fallback on API failure)
2. Replicates full feature engineering inline: targets, batter-relative coords, zone features, game state, trajectory/deception features
3. Uses raw `pitch_type` as substitute for `pitch_type_mode` (no GMM clustering available for PO data)
4. Imputes optional features with per-column medians from the PO data itself
5. Runs Model 2 inference with calibration, aggregates per-pitcher skill
6. Merges with RS skill from `pitchers_overall.json` to compute `skill_delta = po_skill - rs_skill`
7. Assigns confidence levels: low (<100 pitches), medium (100-199), high (≥200)
8. Categorizes: big_riser (>+0.05), riser (+0.02 to +0.05), neutral (±0.02), faller (-0.02 to -0.05), big_faller (<-0.05)

### Key Output Datasets
- `statcast_step1_swing_take.parquet` - All pitches with `y_swing` target (1,442,579 rows)
- `statcast_step2_whiff_contact.parquet` - Swings only with `y_whiff` target (690,535 rows)

### Model Artifacts
- `artifacts/models/model1_swing_take.json` - XGBoost model for swing/take
- `artifacts/models/model2_whiff_contact.json` - XGBoost model for whiff/contact
- `artifacts/models/model2_calibrator.joblib` - Isotonic calibrator for Model 2 probabilities
- `artifacts/models/model_config.json` - Feature columns and thresholds for inference
- `artifacts/training/training_metrics.json` - All metrics including zone diagnostics

### Skill Evaluation Artifacts
- `artifacts/skill/batter_contact_skill.parquet` - Per-batter contact skill metrics (580 batters)
- `artifacts/skill/pitcher_whiff_skill.parquet` - Per-pitcher whiff skill metrics (626 pitchers)
- `artifacts/skill/skill_scatter.png` - Expected vs actual whiff rate scatter plots
- `artifacts/skill/skill_leaderboards.png` - Top 20 batters/pitchers leaderboards
- `artifacts/skill/skill_summary.json` - Summary statistics and top/bottom players
- `artifacts/training/feature_importance_model2.json` - Feature importance scores

---

## Qualification Thresholds

| Context | Threshold | Value |
|---------|-----------|-------|
| RS leaderboard | Min total pitches | 250 |
| Pitch type breakdown | Min pitches per type | 20 |
| Zone heatmap | Min pitches per zone | 10 |
| PO leaderboard | Min total pitches | 50 |
| RS/PO comparison chart | RS min + PO min | 250 + 50 |

Thresholds are defined as constants at the top of `07_aggregate_for_website.py` and `08_playoff_comparison.py`.

---

## Zone Grid Definition

3×3 grid using **batter-relative** coordinates (`plate_x_batter` for horizontal, `plate_z` for vertical).

**Horizontal boundaries** (from batter's perspective):
- Inside: x < −0.27 ft
- Middle: −0.27 ≤ x < 0.27 ft
- Away: x ≥ 0.27 ft

**Vertical boundaries** (absolute height):
- Low: z < 2.0 ft
- Middle: 2.0 ≤ z < 3.0 ft
- High: z ≥ 3.0 ft

**Full boundary table:**

| Zone ID | x_min | x_max | z_min | z_max |
|---------|-------|-------|-------|-------|
| high_inside | −0.83 | −0.27 | 3.0 | 4.0 |
| high_middle | −0.27 | 0.27 | 3.0 | 4.0 |
| high_away | 0.27 | 0.83 | 3.0 | 4.0 |
| middle_inside | −0.83 | −0.27 | 2.0 | 3.0 |
| middle_middle | −0.27 | 0.27 | 2.0 | 3.0 |
| middle_away | 0.27 | 0.83 | 2.0 | 3.0 |
| low_inside | −0.83 | −0.27 | 1.5 | 2.0 |
| low_middle | −0.27 | 0.27 | 1.5 | 2.0 |
| low_away | 0.27 | 0.83 | 1.5 | 2.0 |

Zone order in JSON: `["high_inside", "high_middle", "high_away", "middle_inside", "middle_middle", "middle_away", "low_inside", "low_middle", "low_away"]`

**Note:** "Inside" = toward the batter (negative x_batter), "Away" = away from the batter (positive x_batter). This is consistent regardless of L/R batter handedness because `plate_x_batter` already accounts for it.

---

## Website JSON Data Formats

### `pitchers_overall.json`
```json
{
  "last_updated": "2024-02-16",
  "season": "2024",
  "min_pitches": 250,
  "pitchers": [
    {
      "pitcher_id": 543037,
      "name": "First Last",
      "team": "NYY",
      "n_pitches": 3200,
      "n_swings": 1500,
      "actual_whiff_rate": 0.2800,
      "expected_whiff_rate": 0.2200,
      "whiff_skill": 0.0600,
      "percentile": 95
    }
  ]
}
```

### `pitchers_by_pitch_type.json`
```json
{
  "543037": {
    "FF": {"n_pitches": 1200, "n_swings": 550, "whiff_rate": 0.2100, "expected": 0.1900, "skill": 0.0200},
    "SL": {"n_pitches": 800, "n_swings": 400, "whiff_rate": 0.3500, "expected": 0.2800, "skill": 0.0700}
  }
}
```
Keyed by pitcher_id (string). Only includes pitch types with ≥20 pitches for qualified pitchers.

### `pitchers_by_zone.json`
```json
{
  "zone_order": ["high_inside", "high_middle", ...],
  "zone_bounds": {"high_inside": {"x_min": -0.83, "x_max": -0.27, "z_min": 3.0, "z_max": 4.0}, ...},
  "pitchers": {
    "543037": [
      [n_pitches, n_swings, whiff_rate, expected, skill],
      ...
    ]
  }
}
```
Compact array format: 9 entries per pitcher in fixed zone order. `null` values for whiff_rate/expected/skill when zone has <10 pitches. Zone boundaries stored once (not per pitcher) to minimize file size.

### `pitchers_playoffs.json`
Same structure as `pitchers_overall.json` with additional fields per pitcher:
- `confidence`: "low" (<100 pitches), "medium" (100-199), "high" (≥200)
- `rs_pitches`: Regular season pitch count
- `rs_qualified`: Boolean, whether pitcher met RS 250-pitch threshold

### `playoff_comparison.json`
```json
{
  "last_updated": "2024-02-16",
  "season": "2024",
  "methodology": "skill_delta = playoff_skill - regular_season_skill (using RS model for both)",
  "min_rs_pitches": 250,
  "min_po_pitches": 50,
  "pitchers": [
    {
      "pitcher_id": 543037,
      "name": "First Last",
      "team": "NYY",
      "rs_pitches": 3200,
      "rs_skill": 0.0600,
      "po_pitches": 120,
      "po_skill": 0.0900,
      "skill_delta": 0.0300,
      "category": "riser",
      "confidence": "medium",
      "rank": 1
    }
  ]
}
```
Sorted by absolute `skill_delta` descending (biggest movers first).

### `metadata.json`
Contains thresholds, totals, model version, methodology string, and timestamps. Updated by both 07 and 08 scripts.

### `name_cache.json`
```json
{"543037": {"name": "First Last"}, ...}
```
Persists MLB API name lookups between runs. Keyed by pitcher_id (string).

---

## Feature Engineering

### Batter-Relative Coordinates
Accounts for L/R batter/pitcher matchups:
- `plate_x_batter` - Horizontal plate location from batter's perspective
- `pfx_x_norm` - Horizontal movement normalized by pitcher handedness
- `release_pos_x_batter` - Release point from batter's perspective
- `same_side` - 1 if pitcher/batter same handedness, 0 otherwise

### Zone Analysis
Normalized by individual batter's strike zone:
- `in_zone` - Binary: pitch in strike zone (plate_x within 0.83 ft, plate_z between sz_bot and sz_top)
- `x_out_mag` - Horizontal distance outside zone (0 if in zone)
- `x_out_signed_batter` - Signed horizontal distance from batter perspective
- `z_out_signed` - Signed vertical distance (positive=high, negative=low)
- `zone_out_dist` - Euclidean distance outside zone
- `zone_height` - Batter's strike zone height (sz_top - sz_bot)
- `plate_z_rel` - Vertical position as fraction of zone height

### Game State
- `count_state` - Categorical: "0-0", "1-2", etc.
- `two_strikes` - Binary: 1 if 2 strikes
- `runner_on_1b`, `runner_on_2b`, `runner_on_3b` - Binary flags
- `any_runner_on`, `risp`, `bases_loaded` - Derived runner flags
- `batting_score_diff` - Score differential from batting team's perspective
- `score_bucket` - Categorical: "trail_5plus", "trail_4-3", "trail_2-1", "tied", "lead_1-2", "lead_3-4", "lead_5plus"

### Categorical Features (One-Hot Encoded in Training)
- `p_throws` - Pitcher handedness (L/R)
- `stand` - Batter handedness (L/R)
- `count_state` - Ball-strike count
- `score_bucket` - Score differential bucket
- `pitch_type_mode` - Mode pitch type for the cluster (FF, SL, CH, CU, etc.)

### Cluster Physical Statistics (EXCLUDED FROM MODEL)
Computed from training data but **excluded from model training** to prevent pitcher fingerprinting.
These features leak pitcher identity and would bias skill evaluations.

**Features excluded (9 total):**
- `cluster_mean_velocity` - Average release_speed per cluster
- `cluster_mean_eff_velocity` - Average effective_speed per cluster
- `cluster_mean_spin` - Average release_spin_rate per cluster
- `cluster_mean_pfx_x` - Average horizontal movement per cluster
- `cluster_mean_pfx_z` - Average vertical movement per cluster
- `cluster_mean_extension` - Average release extension per cluster
- `velocity_vs_cluster` - Pitch velocity deviation from cluster mean
- `spin_vs_cluster` - Spin rate deviation from cluster mean
- `pfx_z_vs_cluster` - Vertical movement deviation from cluster mean

**Rationale**: For skill evaluation, "expected" should represent what an average batter does against this exact pitch physics, not influenced by who threw it.

### Trajectory / Deception Features (NEW)
Features capturing pitch deception and late movement:

**Time-based features:**
- `time_to_plate` - Flight time from release to plate (seconds): `release_pos_y / abs(vy0)`
  - Typical range: 0.38-0.45 seconds
  - Fastballs ~0.38s, changeups ~0.43s

**Late break intensity** (movement per second - higher = more deceptive):
- `late_break_z` - Vertical movement rate: `pfx_z / time_to_plate`
- `late_break_x` - Horizontal movement rate: `pfx_x_norm / time_to_plate`

**Approach angles** (trajectory slope as seen by batter):
- `approach_angle_z` - Vertical approach: `(plate_z - release_pos_z) / release_pos_y`
  - More negative = steeper downward angle
- `approach_angle_x` - Horizontal approach: `(plate_x_batter - release_pos_x_batter) / release_pos_y`

**Acceleration magnitude:**
- `accel_magnitude` - Total Magnus force intensity: `sqrt(ax^2 + ay^2 + az^2)`

### Cluster Deviation Features (EXCLUDED FROM MODEL)
How unusual is this pitch compared to its cluster average. **Excluded** because they fingerprint individual pitchers:
- `velocity_vs_cluster` - `effective_speed - cluster_mean_velocity`
- `spin_vs_cluster` - `release_spin_rate - cluster_mean_spin`
- `pfx_z_vs_cluster` - `pfx_z - cluster_mean_pfx_z`

---

## Clustering Details

Pitcher-specific GMM clustering in `02_cluster_pitch_types.py`:
- **Features used**: release_speed, spin_rate, pfx_x/z, spin_axis, release position, velocities, accelerations
- **k selection**: BIC minimization over k=2-6
- **Minimum**: 200 pitches per pitcher
- **Validation**: ARI and NMI vs. labeled pitch_type
- **Output**: 3,053 unique clusters across all pitchers

---

## Model Training Details

### Model 1 (Swing/Take)
- XGBoost binary classifier
- Balanced classes (~52% takes, ~48% swings)
- Hyperparameters: `max_depth=6`, `eta=0.1`, `subsample=0.8`, `colsample_bytree=0.8`
- 500 rounds with early stopping (patience=50)

### Model 2 (Whiff/Contact)
- XGBoost binary classifier with class imbalance handling
- Imbalanced classes (~76.5% contact, ~23.5% whiff)
- `scale_pos_weight=3.26` to correct for class imbalance
- Regularization: `max_depth=4`, `min_child_weight=5`, `gamma=0.1`, `subsample=0.7`
- Isotonic calibration for probability outputs (improves Brier score ~25%)

### Data Splits
- **Train**: 2023 Apr-Aug (579,603 step1 / 276,007 step2)
- **Val**: 2023 Sep-Oct (137,674 step1 / 65,835 step2)
- **Test**: 2024 Apr-Sep (703,526 step1 / 338,433 step2) — RS-only; playoff data is in raw parquet but excluded from model datasets

---

## Current Model Performance (as of 2026-02-12)

**Note**: Model retrained with cluster-derived features removed (pitch-only model for clean skill evaluation).

### Model 1 (Swing/Take)
| Split | ROC-AUC | PR-AUC | Brier | Accuracy |
|-------|---------|--------|-------|----------|
| Train | 0.8667 | 0.8508 | 0.1491 | 77.9% |
| Val | 0.8440 | 0.8287 | 0.1611 | 75.8% |
| Test | 0.8426 | 0.8292 | 0.1619 | 75.8% |

### Model 2 (Whiff/Contact) - Overall (Pitch-Only, No Cluster Features)
| Split | ROC-AUC | PR-AUC | Brier | Accuracy |
|-------|---------|--------|-------|----------|
| Train | 0.7919 | 0.6020 | 0.1815 | 74.1% |
| Val | 0.7681 | 0.5733 | 0.1901 | 72.4% |
| Test | 0.7682 | 0.5675 | 0.1899 | 72.5% |

### Model 2 Zone Diagnostics (Test Set)
| Zone | ROC-AUC | PR-AUC | Whiff Rate | N |
|------|---------|--------|------------|---|
| In-zone | 0.6797 | 0.2681 | 14.4% | 228,629 (65.6%) |
| Out-of-zone | 0.7578 | 0.7049 | 40.1% | 120,064 (34.4%) |
| OOZ Breaking Balls | 0.7744 | 0.8059 | 52.6% | 42,487 (12.2%) |
| OOZ Non-Breaking | 0.7230 | 0.5881 | 33.3% | 77,577 |

### Cascade Performance (Test)
- Cascade accuracy: 66.5%
- Model 1 precision: 75.3%, recall: 73.9%
- Reachable swings coverage: 73.9%

### Calibration (Model 2)
- Brier score improvement with isotonic calibration: ~25%
- Test uncalibrated: 0.1899 → calibrated: 0.1420

---

## EDA Findings: The Two-Problem Discovery

Deep EDA revealed the whiff prediction problem is actually **two different problems**:

### Problem 1: In-Zone Pitches (65% of swings)
- Whiff rate: 14.6% (rare event)
- Model struggles here: PR-AUC only 0.27
- This is a **rare event detection** problem
- Need high recall for the minority class

### Problem 2: Out-of-Zone Pitches (35% of swings)
- Overall whiff rate: 40.1%
- **Breaking balls out-of-zone**: 52.6% whiff rate (essentially coin flips)
  - Sliders: 54.8%
  - Knuckle curves: 54.3%
  - Curveballs: 49.8%
  - Splitters: 47.1%
- Model performs better here (PR-AUC 0.70-0.80) but the underlying data is chaotic

### Implications
The model's overall PR-AUC (~0.57) is dragged down by the in-zone rare event problem, while out-of-zone breaking balls are inherently unpredictable regardless of features.

---

## OOZ Breaking Ball Probability Analysis (2026-02-12)

### Hypothesis Tested
> The model may be correctly predicting ~50% for OOZ breaking balls (52.6% actual whiff rate), and binary metrics unfairly penalize this correct uncertainty.

### Verdict: HYPOTHESIS REJECTED

The model is **NOT** just predicting ~50%. It actively attempts to distinguish between whiffs and contact, with predictions spanning the full probability range.

### Probability Distribution by Group

| Group | N | Actual Whiff% | Mean Pred | IQR | % in [0.40-0.60] |
|-------|-----|---------------|-----------|-----|------------------|
| In-Zone | 228,629 | 14.4% | 14.9% | 0.102 | 1.4% |
| OOZ Non-Breaking | 77,577 | 33.3% | 33.2% | 0.270 | 19.3% |
| OOZ Breaking | 42,487 | 52.6% | 51.8% | 0.347 | 26.5% |

### Key Discovery: Model is Well-Calibrated Within OOZ Breaking Balls

| Prediction Bin | N | Actual Whiff% | Calibration Error |
|----------------|------|---------------|-------------------|
| < 0.30 | 8,604 (20.3%) | 22.7% | -1.2% |
| 0.40-0.60 | 11,246 (26.5%) | 50.6% | +0.3% |
| >= 0.80 | 6,690 (15.7%) | 91.8% | -1.1% |

When the model predicts ~90% whiff probability, the actual rate is 91.8%. When it predicts ~22%, the actual rate is 22.7%. **The model IS finding real signal.**

### Interpretation

1. **Signal exists in OOZ breaking balls** - 15.7% get >80% whiff predictions (with 91.8% actual whiff rate)
2. **The "coin flip" characterization was incorrect** - only 26.5% of predictions fall in [0.40-0.60]
3. **Model is remarkably well-calibrated** - predictions align closely with outcomes across all bins
4. **PR-AUC limitations are real but not due to inability to distinguish** - high 52.6% base rate inherently limits class separability

### Design Decision: Keep All Pitches

We keep ALL pitches including OOZ breaking balls because:
- The model is well-calibrated across all pitch types (predictions match actual rates)
- For skill evaluation, we need to compare expected vs actual on the full pitch mix
- "Tough" pitches (OOZ breaking balls) are where skill differences emerge most clearly
- Filtering would bias skill metrics toward easy-to-predict pitches

### Artifacts
- `artifacts/training/ooz_probability_distributions.png` - Histograms by group
- `artifacts/training/ooz_calibration_curves.png` - Calibration curves by group
- `artifacts/training/ooz_probability_analysis.json` - Full analysis results

---

## Feature Importance (Model 2 - Top 20, Pitch-Only)

Model trained without cluster-derived features to enable clean skill evaluation.

| Rank | Feature | Gain |
|------|---------|------|
| 1 | in_zone | 1364.15 |
| 2 | plate_z_rel | 229.97 |
| 3 | pitch_type_mode_SI | 208.29 |
| 4 | pitch_type_mode_FF | 115.95 |
| 5 | two_strikes | 111.73 |
| 6 | pitch_type_mode_CH | 81.31 |
| 7 | count_state_0-0 | 63.01 |
| 8 | approach_angle_z | 63.00 |
| 9 | pitch_type_mode_SL | 57.46 |
| 10 | az | 52.53 |
| 11 | approach_angle_x | 49.05 |
| 12 | pitch_type_mode_NA | 39.88 |
| 13 | vy0 | 35.50 |
| 14 | count_state_1-0 | 35.25 |
| 15 | same_side | 32.08 |
| 16 | late_break_z | 30.30 |
| 17 | vz0 | 28.64 |
| 18 | stand_L | 28.29 |
| 19 | late_break_x | 28.04 |
| 20 | time_to_plate | 27.28 |

**Key trajectory features**: approach_angle_z (#8), az (#10), approach_angle_x (#11), late_break_z (#16), late_break_x (#19), time_to_plate (#20)

---

## Next Steps / Future Work

### Priority: In-Zone Prediction Improvement
Based on OOZ analysis, the model handles OOZ breaking balls well (calibrated, finding signal). **The real opportunity is in-zone predictions** where:
- Whiff rate is 14.4% (rare event problem)
- PR-AUC is only 0.27 (significant room for improvement)
- Brier score is 0.117 (best of all groups, but still improvable)

### Recommended Actions
1. **Focus on in-zone feature engineering** - What differentiates in-zone whiffs from contact?
2. **Rare event techniques for in-zone model** - SMOTE, focal loss, or threshold optimization
3. **Use calibrated probabilities** - Isotonic calibration working well across all zones

### Feature Ideas (Not Yet Implemented)
1. **Batter historical features** (requires leakage-safe computation):
   - `batter_whiff_rate` - Career/rolling whiff rate
   - `batter_chase_rate` - Rate of swinging at out-of-zone pitches
   - `batter_zone_whiff_rate` - In-zone whiff rate specifically (HIGH PRIORITY)

2. **Pitcher-batter interaction features**:
   - Historical matchup stats
   - Pitch sequence patterns

3. **Pitch sequencing features** (partially implemented):
   - ~~Previous pitch type~~ — **IMPLEMENTED** in `03_feature_engineering.py` (mirrored in `08_playoff_comparison.py`): `prev_pitch_type_mode` (shifted `pitch_type_mode`, same `[game_pk, at_bat_number, pitcher]` grouping as tunnel distance; NaN on first pitch per pitcher per AB, handled by `dummy_na` one-hot in `05` — registered in its `CATEGORICAL_COLS`) plus `same_pitch_type_as_prev` (int8 flag, 0 when no prior pitch)
   - Previous pitch location (not yet implemented)
   - ~~Velocity differential from previous pitch~~ — **IMPLEMENTED** in `03_feature_engineering.py` (mirrored in `08_playoff_comparison.py`): `velo_diff_from_prev` = `release_speed` minus the previous pitch's (same grouping as tunnel distance; negative = slower than prior pitch). Structural NaN on first pitch per pitcher per AB, not imputed in `04`/`08`.
   - ~~Tunnel distance from previous pitch~~ — **IMPLEMENTED** in `03_feature_engineering.py` (mirrored in `08_playoff_comparison.py`):
     - `tunnel_distance`: Euclidean (x, z) distance between this pitch and the previous pitch by the *same pitcher in the same at-bat* at the tunnel point, 23.8 ft from the plate (~batter's swing-decision point). Time to tunnel solved via full quadratic kinematics (`0.5*ay*t^2 + vy0*t + dy = 0`, linear fallback on invalid discriminant); (x, z) via full kinematics from `release_pos_*`, `v*0`, `a*`.
     - Grouped by `[game_pk, at_bat_number, pitcher]` sorted by `pitch_number` (pitcher in the key guards against mid-AB pitching changes).
     - NaN on the first pitch by a pitcher in an AB (~25% of rows) — structural, deliberately **not** median-imputed in `04`/`08` (XGBoost native missing handling is preferred over a fabricated median); companion flag `is_first_pitch_for_pitcher_in_ab` (int8) marks these rows.

### Architecture Considerations
- **In-zone model**: Optimize for recall (rare event), consider separate model
- **OOZ model**: Current approach is working; calibration is excellent
- **Two-model architecture**: May still be beneficial, but for different reasons than originally thought (in-zone rare event vs OOZ balanced classes)

---

## Player Skill Evaluation (`06_player_skill_metrics.py`)

### Core Concept
```
Skill = Actual - Expected
```
- **Batter contact skill** = expected_whiff - actual_whiff (positive = good contact)
- **Pitcher whiff skill** = actual_whiff - expected_whiff (positive = good at generating whiffs)

### Results (2024 Test Set)
- **348,693 swings** evaluated
- **580 batters** with >= 50 swings
- **626 pitchers** with >= 100 swings

**Key Findings:**
- Mean skill ~0 (by construction) - validates methodology
- Batter skill std: 0.062 (range: -0.23 to +0.15)
- Pitcher skill std: 0.033 (range: -0.12 to +0.16)
- Expected vs actual correlation: 0.35 (batters), 0.74 (pitchers)

### Output Schema

**Batter Contact Skill (`batter_contact_skill.parquet`):**
| Column | Description |
|--------|-------------|
| batter | MLB player ID |
| n_swings | Sample size |
| actual_whiff_rate | Observed whiff rate |
| expected_whiff_rate | Model predicted (calibrated) |
| contact_skill | expected - actual (positive = good) |
| std_error | Standard error of estimate |
| percentile | Rank 1-100 (100 = best contact) |

**Pitcher Whiff Skill (`pitcher_whiff_skill.parquet`):** Same structure with whiff_skill = actual - expected

### Future Extensions
- ~~Zone breakdowns~~ → Implemented in 07 (3×3 zone grid with per-zone skill metrics)
- ~~Pitch type breakdowns~~ → Implemented in 07 (per-pitch-type aggregation with min 20 pitches)
- ~~Player name lookups via MLB API~~ → Implemented in 07 (with `name_cache.json` persistence)
- Year-over-year comparison (remaining)

---

## Dependencies

- `pybaseball` - Statcast data source (optional, can use `statcast_fetcher.py` instead)
- `requests` - HTTP requests for direct Baseball Savant access and MLB API name lookups
- `xgboost` - Model training and inference
- `pandas`, `numpy` - Data processing
- `pyarrow` - Parquet schema inspection (used in 07/08 for column discovery without full data load)
- `scikit-learn` - GMM clustering, preprocessing, metrics, calibration
- `matplotlib` - Visualization
- `joblib` - Model serialization

---

## File Reference

### 03_feature_engineering.py Key Sections
- Lines 223-277: Trajectory/deception feature computation
- Lines 351-370: OPTIONAL_FINAL list (includes all new features)

### 04_build_model_datasets.py Key Sections
- Lines 100-124: OPTIONAL_IMPUTE list (26 columns including new trajectory features)

### 05_train_and_validate.py Key Sections
- Lines 86-100: CLUSTER_FEATURES_TO_REMOVE list (9 features excluded for pitch-only model)
- Lines 115-128: Feature filtering to remove cluster-derived features
- Lines 340-406: Zone-aware diagnostics section
- Line 670: zone_metrics saved to training_metrics.json

### 07_aggregate_for_website.py Key Sections
- Lines 43-66: Configuration constants (thresholds, input/output paths)
- Lines 235-277: Zone assignment function (batter-relative 3×3 grid logic)
- Lines 398-415: ZONE_IDS list and ZONE_BOUNDS dictionary definitions
- Lines 471-506: `fetch_pitcher_name()` - MLB API name lookup with raw-data-first fallback

### 08_playoff_comparison.py Key Sections
- Lines 50-54: Configuration (thresholds + PLAYOFF_GAME_TYPES list)
- Lines 219-338: Inline feature engineering (replicates 03_feature_engineering.py for PO data)
- Lines 485-491: `assign_confidence()` - Confidence level assignment by pitch count
- Lines 619-628: Category definitions (big_riser/riser/neutral/faller/big_faller by skill_delta)

### 01_clean_targets.py Key Sections
- Lines 112-151: Incremental raw data fetch (reuses older parquet, fetches only new date range, dedup by game_pk/at_bat_number/pitch_number)

### .github/workflows/pages.yml
- Lines 1-38: GitHub Actions deploy workflow (uploads Baseball/website/ to GitHub Pages)

### Interactive-1.ipynb Key Sections
- OOZ probability distribution analysis (cells after clustering code)
- Calibration analysis by prediction bin
- Hypothesis verification for OOZ breaking balls
