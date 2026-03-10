# NBA Synergy Engine

[![CI](https://github.com/praveenpuviindran/nba-synergy-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/praveenpuviindran/nba-synergy-engine/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Live app: https://nba-synergy-engine.streamlit.app/

This project builds a **Generative GM**: a system that takes any 4-player lineup core and recommends the best possible 5th player based on historical evidence.

The main goal is not “who is the best player overall,” but:

**"Who is the best fit for these specific 4 teammates?"**

---

## What This README Explains

This guide is written for readers with no basketball, data science, or machine learning background.

It explains:
1. What problem the Generative GM solves.
2. Every step used to build it.
3. Why it can produce different answers for different 4-player cores.
4. Which files in this repo implement each piece.

---

## Plain-English Problem Statement

In basketball, teams put 5 players on the court at once.

A common mistake is to rank players only by individual talent. In practice, team performance depends heavily on **fit**:
- Some players overlap too much (same strengths, same weaknesses).
- Some players complement each other (one creates shots, another finishes, another defends, etc.).

The Generative GM solves this:
- Input: 4 existing players.
- Output: ranked list of possible 5th players, from best fit to worst fit.

---

## Mini Glossary (No Prior Knowledge Needed)

- **Lineup**: 5 players on the court together.
- **Core**: the fixed 4 players you choose.
- **Candidate**: the possible 5th player being tested.
- **Feature**: a numeric description of player style (speed, touches, shot profile, etc.).
- **Archetype**: a broad player style category (for example, shooter, rim runner, creator).
- **Model**: a program that learns patterns from past data and predicts outcomes.
- **Synergy score**: this project’s final fit score; higher means better expected lineup fit.

---

## End-to-End Build: How Generative GM Was Created

### Step 1) Gather historical lineup outcomes
Source file: `v3_neural_synergy/data/lineups_2014_2025.csv`

For each historical 5-player lineup, we have performance outcomes including:
- Minutes played together (`MIN`)
- Score margin (`PLUS_MINUS`)
- Season label

This gives the system real examples of what lineup combinations succeeded or failed.

### Step 2) Gather player playstyle data
Source files:
- `data/processed_tracking_metrics.csv`
- `data/gmm_archetypes.csv`

Each player-season gets behavior/style measurements (movement speed, touch patterns, shot type mix, etc.) and an archetype label.

### Step 3) Standardize and clean player features
Implementation: `v3_neural_synergy/synergy_utils.py` (`build_player_feature_table`)

The pipeline:
- Converts values to numeric format.
- Handles missing values safely.
- Scales features so different units are comparable.
- Adds archetype one-hot columns.
- Adds archetype confidence.

Why this matters: the model can only learn well if features are consistent and comparable.

### Step 4) Convert each 5-player lineup into training examples
Implementation: `v3_neural_synergy/02_train_deepset.py`

From each real lineup of 5 players, the script creates **5 separate examples**:
- Keep 4 players as context (core).
- Treat the remaining player as the candidate.

This teaches the model exactly the decision we care about at inference time: **“given 4, how good is this 5th?”**

### Step 5) Engineer context-aware interaction features
Implementation: `v3_neural_synergy/synergy_utils.py` (`featurize_core_candidate`)

For each (core, candidate) pair, features include:
- Candidate vector itself.
- Core summary statistics (mean/std/min/max).
- Candidate minus core differences.
- Absolute differences.
- Element-wise interactions.
- Similarity metrics (cosine and distance stats).

Why this matters: this is the main mechanism that makes predictions change with the selected 4-player core.

### Step 6) Use a better target than raw total plus-minus
Implementation: `v3_neural_synergy/02_train_deepset.py`

Raw `PLUS_MINUS` totals are biased by minutes and context. The training target is improved in two ways:

1. **Rate normalization**:
- Convert to per-48-minute impact: `PLUS_MINUS / MIN * 48`

2. **Core-baseline demeaning**:
- Estimate each 4-player core’s baseline strength from train data.
- Train model on marginal lift above/below that baseline.

Why this matters: it shifts learning from “best players overall” to “best incremental fit for this core.”

### Step 7) Weight samples by lineup stability
Implementation: `v3_neural_synergy/02_train_deepset.py`

Lineups with more shared minutes are usually more reliable. The trainer uses minute-based weights (square-root scaled) so noisy tiny samples matter less.

### Step 8) Train a neural interaction model
Implementation: `v3_neural_synergy/synergy_utils.py` (`ContextAwareSynergyNet`) + `v3_neural_synergy/02_train_deepset.py`

The neural network learns nonlinear interactions between:
- Candidate profile,
- Core profile,
- Compatibility gaps.

Training details include:
- Train/validation/test split.
- Robust loss (SmoothL1).
- Regularization (dropout + weight decay).
- Early stopping.

### Step 9) Train a nearest-neighbor model (KNN)
Implementation: `v3_neural_synergy/02_train_deepset.py`

A KNN regressor is also trained on the same engineered feature space.

Why include this:
- It provides a local, example-based statistical check.
- If a new query looks like known historical patterns, KNN can reinforce stability.

### Step 10) Add a player quality prior and calibrate final score
Implementation: `v3_neural_synergy/02_train_deepset.py`

A season-specific player quality prior is derived from historical lineup impact and z-scored.

Final score is calibrated as:
- weighted model prediction
- plus weighted quality prior
- plus bias term

Why this matters:
- Fit-only systems can over-reward obscure role players.
- This correction balances **fit** and **quality**.

### Step 11) Save all artifacts required for consistent inference
Outputs:
- `v3_neural_synergy/synergy_model.pth` (neural weights)
- `v3_neural_synergy/synergy_artifacts.pkl` (scalers, KNN, calibration coefficients, metadata)

Why this matters: training and inference use identical preprocessing, avoiding train/inference mismatch.

### Step 12) Run inference for a user-selected 4-player core
Implementations:
- Streamlit UI: `app.py`
- CLI script: `v3_neural_synergy/03_generative_gm.py`
- SQL-powered variant: `v4_data_engineering/02_sql_generative_gm.py`

Inference flow:
1. Load model + artifacts.
2. Build vectors for all eligible candidates.
3. Compute context-aware features for each candidate with your chosen 4 players.
4. Score all candidates in one batch.
5. Rank candidates and show best/worst fits.

### Step 13) Confidence estimation
Implementation: `app.py`, `v3_neural_synergy/03_generative_gm.py`

A relative confidence score is computed from KNN neighbor distance.
- Closer to known historical patterns => higher confidence.
- Farther from known patterns => lower confidence.

Important: this is a **relative reliability indicator**, not a guaranteed probability.

---

## Why This System Produces Different Results for Different Cores

The key factors are:
1. Core-conditioned feature engineering (candidate is always evaluated against the chosen 4 players).
2. Marginal target (learns improvement over core baseline, not raw absolute team strength).
3. Interaction-aware neural network (captures nonlinear fit effects).
4. Quality calibration (prevents unrealistic fit-only rankings).
5. Shared training/inference pipeline (same transforms at both stages).

Together, these address the common failure mode where every core gets the same generic top players.

---

## File-by-File Map (Generative GM)

- `v3_neural_synergy/synergy_utils.py`
  - Shared model class, preprocessing, feature engineering.
- `v3_neural_synergy/02_train_deepset.py`
  - Full training pipeline and artifact export.
- `v3_neural_synergy/03_generative_gm.py`
  - Command-line inference for a selected core.
- `app.py`
  - Streamlit app interface for interactive lineup optimization.
- `v4_data_engineering/02_sql_generative_gm.py`
  - SQL-backed inference path.
- `v3_neural_synergy/synergy_model.pth`
  - Trained neural network weights.
- `v3_neural_synergy/synergy_artifacts.pkl`
  - Non-neural artifacts required to reproduce inference behavior.

---

## How To Run

### 1) Train / retrain the Generative GM
```bash
.venv/bin/python v3_neural_synergy/02_train_deepset.py
```

### 2) Run command-line recommendations
```bash
.venv/bin/python v3_neural_synergy/03_generative_gm.py
```

### 3) Run SQL-powered recommendation path
```bash
.venv/bin/python v4_data_engineering/02_sql_generative_gm.py
```

### 4) Launch web app
```bash
.venv/bin/streamlit run app.py
```

---

## Current Scope and Limitations

- Uses historical lineup and tracking data from 2014–2025.
- Learns from available lineup combinations; truly novel combinations can have lower confidence.
- Synergy score is a model-based estimate, not a guarantee of real-game results.
- Results depend on data quality and season coverage.

---

## Project Structure

```text
├── app.py
├── nba_sql.db
├── requirements.txt
├── v1_boxscore_project/
├── v2_tracking_project/
├── v3_neural_synergy/
│   ├── data/
│   ├── 01_fetch_lineups.py
│   ├── 02_train_deepset.py
│   ├── 03_generative_gm.py
│   ├── synergy_utils.py
│   ├── synergy_model.pth
│   └── synergy_artifacts.pkl
├── api/
│   ├── main.py
│   ├── database.py
│   ├── run.py
│   └── models/schemas.py
├── tests/
│   ├── conftest.py
│   ├── test_feature_pipeline.py
│   ├── test_model.py
│   ├── test_simulation.py
│   └── test_api.py
├── .github/workflows/ci.yml
├── Dockerfile
├── MODEL_CARD.md
├── requirements-dev.txt
└── v4_data_engineering/
    ├── 01_build_sql_db.py
    └── 02_sql_generative_gm.py
```

---

## REST API

A FastAPI layer exposes the synergy engine as a REST API.

### Start the server
```bash
pip install -r requirements-dev.txt
python -m api.run
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/lineup/optimize` | Score all candidates against a 4-player core |
| `GET` | `/archetypes` | List GMM archetype labels for a given season |
| `GET` | `/sql/lineup-stats` | Advanced SQL analytics — top lineups by plus-minus with per-minute efficiency and rank |

### Example: optimize a lineup
```bash
curl -X POST http://localhost:8000/lineup/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "core_players": ["Shai Gilgeous-Alexander", "Jalen Williams", "Chet Holmgren", "Luguentz Dort"],
    "top_n": 5
  }'
```

### Database
Set `DATABASE_URL` to a PostgreSQL connection string for production.  The API
falls back to the bundled `nba_sql.db` (SQLite) when `DATABASE_URL` is not set.

---

## Uncertainty Quantification

The Generative GM uses **Monte Carlo Dropout** to report a confidence interval
alongside every synergy score.

### How it works
1. At inference time, Dropout layers are kept active (training mode).
2. 50 independent forward passes are run per candidate.
3. The **mean** of the 50 passes is the displayed score; the **standard deviation (σ)** is the uncertainty.

### Confidence tiers

| σ | Badge | Interpretation |
|---|---|---|
| < 0.02 | **High** | Well within training distribution |
| 0.02 – 0.05 | **Med** | Moderate uncertainty |
| ≥ 0.05 | **Low** | Potentially out-of-distribution |

The Streamlit app shows `Score (±σ)` for every candidate and includes an
expandable **Uncertainty Calibration Plot** (score vs σ, coloured by tier).

See [MODEL_CARD.md](MODEL_CARD.md) for full architecture and evaluation details.
