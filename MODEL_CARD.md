# Model Card — ContextAwareSynergyNet

## Model Overview

| Field | Details |
|---|---|
| **Name** | ContextAwareSynergyNet |
| **Version** | v3 |
| **Type** | Deep neural regressor (DeepSet-inspired) |
| **Task** | Lineup chemistry scoring — predict 5th-player fit for a given 4-player core |
| **Input dim** | 175 features |
| **Output** | Scalar synergy score (calibrated plus-minus proxy) |
| **Framework** | PyTorch |
| **Artifact** | `v3_neural_synergy/synergy_model.pth` |

---

## Architecture

```
Linear(175, 256) → LayerNorm(256) → SiLU → Dropout(0.20)
Linear(256, 128)               → SiLU → Dropout(0.15)
Linear(128, 64)                → SiLU
Linear(64, 1)
```

Dropout layers are repurposed at inference time for **Monte Carlo Dropout**
uncertainty estimation (see _Uncertainty Quantification_ below).

---

## Training Data

| Source | Coverage | Rows |
|---|---|---|
| NBA tracking data (SpeedDistance, TouchTime, ShotType) | 2014-15 → 2024-25 | ~540 player-seasons |
| Historical 5-player lineup outcomes (plus-minus, GP, MIN) | 2014-15 → 2024-25 | ~20 000 lineup records |

### Input Features (175-dim)

Each row represents one *(core, candidate)* pair:

- **candidate_vector** (21-dim): 9 tracking stats (z-scored) + 10 archetype one-hot + archetype confidence + quality prior
- **core aggregations** (21 × 4 = 84-dim): mean, std, min, max over the 4-player core
- **interaction terms** (21 × 2 = 42-dim): element-wise difference and absolute difference between candidate and core mean
- **product terms** (21-dim): element-wise product of candidate and core mean
- **scalar context features** (7-dim): core quality stats (mean/std/min/max), candidate quality, and core archetype diversity

### Labels

Normalised per-minute plus-minus aggregated over all lineups where the given
5-player combination appeared together.  Lineups with fewer than 2 GP were
excluded to reduce noise.

---

## Evaluation

| Metric | Value |
|---|---|
| MAE (hold-out, 2023-24) | ≈ 0.31 per-minute PM |
| Spearman ρ (rank correlation) | ≈ 0.68 |
| Coverage (candidate pool ≥ 10) | 100% of 2024-25 rostered players |

---

## Uncertainty Quantification

The model uses **Monte Carlo Dropout** (Gal & Ghahramani, 2016):

1. At inference time, Dropout modules are kept in `train()` mode.
2. 50 independent forward passes are run per candidate batch.
3. The **mean** of the 50 samples becomes the point estimate; the **standard deviation** (σ) quantifies epistemic uncertainty.

**Confidence tiers** (displayed in the Streamlit app):

| σ | Tier | Interpretation |
|---|---|---|
| < 0.02 | **High** | Model is confident; candidate well-represented in training distribution |
| 0.02 – 0.05 | **Med** | Moderate uncertainty; treat score directionally |
| ≥ 0.05 | **Low** | High uncertainty; player may be out-of-distribution |

---

## Intended Use

- **Primary**: Exploratory roster construction tool for analysts and fans.
- **Secondary**: Demonstration of DeepSet architectures for set-valued sports inputs.

## Out-of-Scope Use

- Live game-time decision support without additional validation.
- Financial or contractual player evaluation without domain expert review.

---

## Limitations & Biases

- Training data covers NBA players only; no college or international data.
- Lineup sample size bias: star-heavy lineups accumulate more minutes, producing
  lower-variance labels for high-profile combinations.
- The quality prior (QUALITY_PRIOR_Z) partially encodes historical reputation,
  which can dampen score dispersion for established veterans.
- Draft picks and rookies with no tracking history will have imputed features
  and higher uncertainty (σ ≥ 0.05 expected).

---

## Citation

```bibtex
@misc{nba_synergy_engine,
  author = {Puviindran, Praveen},
  title  = {NBA Synergy Engine: Context-Aware Lineup Chemistry Scoring},
  year   = {2025},
  url    = {https://github.com/praveenpuviindran/nba-synergy-engine}
}
```
