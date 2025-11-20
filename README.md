# NBA Synergy Engine (Moneyball 2.0)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nba-synergy-engine.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![SQL](https://img.shields.io/badge/Database-SQLite-orange)

**A Permutation-Invariant Neural Network (DeepSet) that predicts lineup chemistry and optimizes roster construction using 11 years of player tracking data.**

---

## Project Overview
Traditional NBA analytics evaluate players in isolation (e.g., PER, Win Shares). However, basketball is a chemical reaction—a high-usage scorer might pair poorly with another ball-dominant guard but perfectly with a low-usage rim runner.

**The NBA Synergy Engine** solves the "Fit Problem" by:
1.  **Classifying Playstyles:** Using Unsupervised Learning (GMM) on tracking data to identify 10 modern archetypes.
2.  **Predicting Synergy:** Using a **DeepSet Neural Network** to predict the Net Rating of any 5-man unit based on the vector sum of their playstyles.
3.  **Generative Optimization:** An AI "General Manager" that mathematically solves for the optimal 5th starter to maximize a specific team's chemistry.

## Technical Architecture

### 1. Data Engineering (ETL & SQL)
* **Source:** NBA API (Player Tracking Data + Lineup Performance).
* **Pipeline:** Ingested **170,000+ possessions** across 11 seasons (2014–2025).
* **Storage:** Migrated raw CSV data into a relational **SQLite Database**, designing normalized schemas for Players, Lineups, and Archetypes to enable O(1) query performance during inference.

### 2. Unsupervised Learning (The "Meta")
* **Model:** Gaussian Mixture Models (GMM) with PCA dimensionality reduction.
* **Input:** 12 tracking metrics (Speed, Micro-Touches, Dribbles per Touch).
* **Outcome:** Identified 10 Latent Archetypes, mathematically validating concepts like the "Heliocentric Creator" (Cluster 6) and predicting the extinction of the "Traditional Facilitator" (Cluster 1).

### 3. Deep Learning (The "Brain")
* **Architecture:** Custom **Permutation-Invariant DeepSet** (built in PyTorch).
* **Problem Solved:** Traditional Neural Networks treat inputs sequentially (Player 1 != Player 2). A DeepSet architecture uses a shared Encoder and a Sum-Pooling layer to ensure that {Luka, Kyrie} is treated identically to {Kyrie, Luka}.
* **Performance:** Trained on 3,500+ qualified lineups to predict Net Rating with **39.94 RMSE**.

## The "Generative GM" Module
A vectorized optimization engine that:
1.  Takes a 4-man Core (e.g., SGA, Jalen Williams, Chet Holmgren, Dort).
2.  Scans the entire NBA roster (450+ players).
3.  Performs **Batch Inference** via PyTorch to simulate 450 hypothetical lineups instantly.
4.  Returns the mathematically optimal 5th player.
    * *Real-World Validation:* The model correctly identified that the OKC Thunder (a ball-dominant core) maximize their Net Rating by adding a **Cluster 2 Rim Runner** (e.g., Clint Capela), rejecting other high-usage stars.

## Project Structure
```text
├── app.py                     # Streamlit Web Application (Frontend)
├── nba_sql.db                 # SQLite Database (Local Storage)
├── requirements.txt           # Cloud Dependencies
├── v1_boxscore_project/       # Legacy: Static analysis & Evolution Charts
├── v2_tracking_project/       # Machine Learning: GMM Clustering Scripts
├── v3_neural_synergy/         # Deep Learning: PyTorch Training & Generative Logic
│   ├── synergy_model.pth      # Trained Neural Network Weights
│   └── 02_train_deepset.py    # DeepSet Architecture Definition
└── v4_data_engineering/       # Engineering: SQL Migration & ETL Pipelines
