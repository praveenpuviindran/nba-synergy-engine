# Moneyball 2.0: Quantifying the Evolution of NBA Playstyles

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://nba-meta-evolution-v1-11-25.streamlit.app/])

**An unsupervised machine learning project that uses Player Tracking Data (Speed, Distance, Micro-Touches) to identify the 10 "True Archetypes" of the modern NBA and forecast their market value through 2030.**

---

## The Problem
Traditional NBA positions (PG, SG, SF, PF, C) are obsolete. Calling LeBron James a "Small Forward" or Nikola Jokić a "Center" fails to capture their actual role on the court.

While box scores tell us *what* happened (Points, Rebounds), they don't tell us *how* it happened. This project abandons box scores in favor of **Player Tracking Data**—analyzing how players move, how long they hold the ball, and where they operate—to scientifically classify the league's talent.

## Key Findings
Using **Gaussian Mixture Models (GMM)** on data from 2014–2025, this project identified 10 distinct playstyles.

* **The "King" Archetype:** The **Heliocentric Creator** (e.g., Luka Dončić, Shai Gilgeous-Alexander) is the most valuable and fastest-growing asset in the NBA, with a projected value growth of **54%** by 2030.
* **The Decline of the "Point God":** The **Traditional Facilitator** (e.g., Chris Paul, Tyus Jones)—players with low offensive speed and high touch time but low scoring—is statistically flatlining in value.
* **The LeBron Effect:** The model successfully detected LeBron James's tactical shift from "Heliocentric Star" (2018) to "Traditional Facilitator" (2020 Lakers PG year) to "Versatile Big" (2024).

## Technical Methodology

### 1. Data Pipeline (ETL)
* **Source:** NBA API (`nba_api`)
* **Scope:** 11 Seasons (2014–2025), 4,700+ Player-Seasons.
* **Features:** Engineered advanced metrics from raw tracking data:
    * *Offensive Motor:* Average Movement Speed (MPH)
    * *Ball Dominance:* Average Seconds Per Touch & Dribbles Per Touch
    * *Shot Diet:* Pull-Up vs. Catch & Shoot Ratios

### 2. Machine Learning Model
* **Dimensionality Reduction:** Applied **Principal Component Analysis (PCA)** to reduce 12 tracking metrics into core components (explaining 95% of variance).
* **Clustering:** Utilized **Gaussian Mixture Models (GMM)** rather than K-Means.
    * *Why GMM?* NBA players are hybrids. GMM provides "soft clustering" (probabilities), allowing us to see that a player might be **60% Rim Runner** and **40% Versatile Big**.

### 3. Forecasting
* **Metric:** Defined a synthetic "Impact Score" based on efficiency per touch and offensive motor.
* **Regression:** Trained Linear Regression models on the historical performance of each cluster to forecast the "Meta" of the NBA through 2030.

## The 10 Identified Archetypes

| ID | Label | Description | Examples |
| :--- | :--- | :--- | :--- |
| **6** | **Heliocentric Stars** | High Dribbles, High Time/Touch, Elite Scoring | *Luka Dončić, SGA* |
| **0** | **Movement Snipers** | High Off-Speed, High Catch & Shoot % | *Klay Thompson, Isaiah Joe* |
| **5** | **Versatile Bigs** | Paint Touches + Playmaking/DHO Hubs | *Nikola Jokić, Draymond Green* |
| **8** | **Dominant Bigs** | Pure Paint Scoring, Low Speed | *Giannis, Embiid* |
| **2** | **Rim Runners** | High Def-Speed, Zero Dribbles | *Clint Capela, Jarrett Allen* |
| **1** | **Traditional Facilitators** | High Touch Time, Low Scoring Efficiency | *Chris Paul, Tre Jones* |

*(See App for full list)*
