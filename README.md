# NBA Synergy Engine

A data-science project that models NBA lineup fit using player tracking data. The goal is simple: given a core of four players, identify which fifth player maximizes lineup chemistry.

## Summary
- **Problem:** Player evaluation is usually individual; lineup success depends on interaction effects.
- **Approach:** Learn player archetypes, then predict five‑man lineup performance with a permutation‑invariant neural network.
- **Outcome:** A “Generative GM” module ranks the best and worst fifth‑player fits for any core lineup.

## What This Project Does
1. Groups players into playstyle archetypes using tracking metrics.
2. Learns how combinations of playstyles translate to lineup Net Rating.
3. Tests every possible fifth player to recommend the best fit for a chosen core.

## Data
- **Seasons:** 2014–2025
- **Sources:** NBA tracking data and lineup performance data
- **Scale:** 170,000+ possessions and thousands of qualified lineups
- **Storage:** Normalized SQLite database for fast queries

## Methods
### 1) Archetype Discovery and Market Trends
- **Model:** Gaussian Mixture Model (GMM) with PCA
- **Goal:** Discover playstyle clusters (e.g., heliocentric creators, movement shooters, rim runners)
- **Goal:** Analyze the market trends to identify future archetype potentials


### 2) Generative GM
- **Input:** Any four‑player core
- **Process:** Batch inference across the player pool to simulate all possible fifth‑player combinations
- **Output:** Ranked best and worst fits for that core

## Results
- **Model objective:** Predict lineup Net Rating from tracking features
- **Reported performance:** ~39.94 RMSE on held‑out lineups (as measured during training)

## How To Run
### Streamlit App
Link: https://nba-synergy-engine.streamlit.app/

## Project Structure
```text
├── app.py                     # Streamlit web app
├── nba_sql.db                 # SQLite database
├── requirements.txt           # Python dependencies
├── v1_boxscore_project/       # Legacy analysis
├── v2_tracking_project/       # GMM clustering scripts
├── v3_neural_synergy/         # Deep learning + Generative GM
│   ├── synergy_model.pth      # Trained model weights
│   └── 02_train_deepset.py    # DeepSet training script
└── v4_data_engineering/       # ETL and database build
```

## Notes
- The Generative GM uses the same variance‑aware DeepSet architecture as training, so the recommended fifth player changes meaningfully with the selected core.
