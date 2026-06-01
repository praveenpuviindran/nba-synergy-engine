# NBA Synergy Engine

[![CI](https://github.com/praveenpuviindran/nba-synergy-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/praveenpuviindran/nba-synergy-engine/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Live app: https://nba-synergy-engine.streamlit.app/

---

## Project Overview

The NBA Synergy Engine is a lineup chemistry scoring system that answers one question:

> **Given 4 players already on the court, who is the optimal 5th teammate?**

Rather than ranking players by individual talent, this system models **fit**: how well a candidate player complements the specific playstyle profile of a 4-player core. The core artifact is a neural regression model trained on 10 years (2014–2025) of historical NBA lineup data, enriched with player tracking metrics and GMM-derived archetypes.

The system is exposed as:
- An **interactive Streamlit app** for lineup exploration
- A **FastAPI REST layer** for programmatic access
- A **CLI script** for batch inference
- A **SQL-backed variant** using a local SQLite warehouse

---

## Methodology

### Why Permutation Invariance Matters for Lineup Data

An NBA lineup is an *unordered set* of 5 players: the lineup (A, B, C, D, E) is identical to (C, A, E, B, D). Any model that represents a lineup as an ordered vector violates this — player ordering changes the input even though the underlying lineup does not, introducing spurious variation and making the model harder to train and less sample-efficient.

This project uses a **DeepSet-inspired approach**: the core-4 is reduced to order-invariant summary statistics (mean, std, min, max across the feature dimension) before being concatenated with the candidate vector and fed to the neural network. This guarantees that any permutation of the 4 core players produces identical predictions.

### Feature Engineering: `featurize_core_candidate`

For each (core-4, candidate-1) pair, the feature vector contains:

1. **Candidate vector** — the candidate's own playstyle embedding (tracking metrics + archetype one-hot + archetype confidence)
2. **Core aggregates** — mean, std, min, max over the 4 core players (permutation-invariant)
3. **Difference features** — candidate minus core mean, and absolute differences
4. **Interaction features** — element-wise product of candidate vector and core mean
5. **Similarity scalars** — per-player cosine similarity, L2 distances, and mean internal core diversity

This makes every prediction conditioned on the **specific 4 players chosen**, not just the candidate in isolation.

### Training Target: Marginal Fit, Not Raw Plus-Minus

Raw `PLUS_MINUS` totals are biased by context (strong cores inflate every candidate's apparent value). The model instead learns **marginal candidate lift**:

1. **Rate normalization** — convert `PLUS_MINUS / MIN × 48` to a per-possession-equivalent scale.
2. **Core-baseline demeaning** (Empirical-Bayes shrinkage) — subtract each core's estimated strength so the model learns what the *candidate alone* contributes above or below expectation for that core.

### Neural Architecture (`ContextAwareSynergyNet`)

A 4-layer MLP with LayerNorm, SiLU activations, and Dropout:

```
Input → Linear(256) → LayerNorm → SiLU → Dropout(0.20)
      → Linear(128) → SiLU → Dropout(0.15)
      → Linear(64)  → SiLU
      → Linear(1)   [marginal net pts / 48]
```

Training uses AdamW + SmoothL1 loss (robust to outlier lineups), with minute-weighted samples (√MIN), early stopping (patience=28), and a stratified 70/15/15 train/val/test split.

### Ensemble: Neural + KNN + Quality Calibration

A `KNeighborsRegressor` (k=35, distance-weighted) is trained in parallel. The two models are blended with a grid-searched α coefficient. A final linear calibration step incorporates a **season-specific player quality prior** (z-scored historical impact per player-season) to prevent fit-only rankings from rewarding obscure role players at the expense of high-quality stars.

### Uncertainty Quantification (Monte Carlo Dropout)

At inference time, Dropout layers remain active across 50 stochastic forward passes. The mean is the displayed synergy score; the standard deviation σ measures model uncertainty:

| σ | Confidence | Interpretation |
|---|---|---|
| < 0.02 | High | Well within training distribution |
| 0.02–0.05 | Med | Moderate uncertainty |
| ≥ 0.05 | Low | Potentially out-of-distribution |

See [MODEL_CARD.md](MODEL_CARD.md) for full architecture details and quantitative evaluation results.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Neural model | PyTorch (`ContextAwareSynergyNet`) |
| Ensemble | scikit-learn `KNeighborsRegressor` |
| Feature preprocessing | scikit-learn `StandardScaler` |
| NBA data | `nba_api` (lineup, tracking, player metadata) |
| Data wrangling | pandas, numpy |
| Web app | Streamlit |
| REST API | FastAPI + Uvicorn |
| SQL analytics | SQLite (`nba_sql.db`) + SQLAlchemy |
| Visualization | Plotly |
| CI | GitHub Actions |
| Containerization | Docker |

---

## Setup

### Prerequisites

- Python 3.11+
- (macOS) `brew install libomp` if you encounter `libomp.dylib` errors

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train the model

Requires `v3_neural_synergy/data/lineups_2014_2025.csv` and `data/gmm_archetypes.csv`:

```bash
python v3_neural_synergy/02_train_deepset.py
```

Artifacts are saved to:
- `v3_neural_synergy/synergy_model.pth` (neural weights)
- `v3_neural_synergy/synergy_artifacts.pkl` (scalers, KNN, calibration coefficients)

### Launch the Streamlit app

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

### Run CLI recommendations

```bash
python v3_neural_synergy/03_generative_gm.py
```

### Run the SQL-backed recommendation path

```bash
python v4_data_engineering/02_sql_generative_gm.py
```

### Start the REST API

```bash
pip install -r requirements-dev.txt
python -m api.run
```

API available at `http://localhost:8000`.

### Run tests

```bash
pytest tests/
```

---

## REST API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/lineup/optimize` | Score all candidates against a 4-player core |
| `GET` | `/archetypes` | List GMM archetype labels for a given season |
| `GET` | `/sql/lineup-stats` | Top lineups by plus-minus with per-minute efficiency |

**Example request:**

```bash
curl -X POST http://localhost:8000/lineup/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "core_players": ["Shai Gilgeous-Alexander", "Jalen Williams", "Chet Holmgren", "Luguentz Dort"],
    "top_n": 5
  }'
```

Set `DATABASE_URL` to a PostgreSQL connection string for production. Falls back to the bundled `nba_sql.db` (SQLite) when unset.

---

## Project Structure

```text
├── app.py                          # Streamlit interactive UI
├── api/                            # FastAPI REST layer
│   ├── main.py
│   ├── database.py
│   ├── run.py
│   └── models/schemas.py
├── v3_neural_synergy/              # DeepSet model + training pipeline
│   ├── 01_fetch_lineups.py
│   ├── 02_train_deepset.py
│   ├── 03_generative_gm.py
│   ├── synergy_utils.py
│   ├── synergy_model.pth
│   └── synergy_artifacts.pkl
├── v4_data_engineering/            # SQL-backed inference path
│   ├── 01_build_sql_db.py
│   └── 02_sql_generative_gm.py
├── v1_boxscore_project/            # v1: boxscore-based clustering
├── v2_tracking_project/            # v2: GMM tracking archetypes
├── tests/                          # pytest suite
├── Dockerfile
├── MODEL_CARD.md
├── requirements.txt
└── requirements-dev.txt
```

---

## Limitations

- Coverage: 2014–2025 seasons. Novel player combinations that never appeared historically together have lower confidence (higher σ).
- The synergy score is a model estimate, not a guarantee of real-game outcomes.
- Tracking data quality varies by season; earlier seasons have sparser coverage.
