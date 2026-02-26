import os
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from v3_neural_synergy.synergy_utils import (
    ContextAwareSynergyNet,
    attach_player_quality_column,
    build_player_feature_table,
    featurize_core_candidate,
)

# CONFIG
DB_PATH = 'nba_sql.db'
MODEL_PATH = 'v3_neural_synergy/synergy_model.pth'
ARTIFACT_PATH = 'v3_neural_synergy/synergy_artifacts.pkl'
CORE_LINEUP = ["Shai Gilgeous-Alexander", "Jalen Williams", "Chet Holmgren", "Luguentz Dort"]
TARGET_SEASON = '2024-25'

print(f"--- SQL-POWERED GM: Optimizing for {CORE_LINEUP} ---")

if not os.path.exists(MODEL_PATH) or not os.path.exists(ARTIFACT_PATH):
    print("Error: missing model artifacts. Run `python3 v3_neural_synergy/02_train_deepset.py` first.")
    raise SystemExit(1)

# 1) Load model + artifacts
with open(ARTIFACT_PATH, 'rb') as f:
    artifacts = pickle.load(f)
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model = ContextAwareSynergyNet(input_dim=int(checkpoint['input_dim']))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 2) Query SQL for tracking + archetype context
conn = sqlite3.connect(DB_PATH)
query = """
SELECT p.PLAYER_ID, p.PLAYER_NAME, p.SEASON_LABEL,
       p.OFF_SPEED, p.DEF_SPEED, p.TIME_PER_TOUCH, p.DRIBBLES_PER_TOUCH,
       p.PTS_PER_TOUCH, p.DRIVE_PCT, p.CATCH_SHOOT_PCT, p.PULL_UP_PCT, p.PAINT_TOUCH_PCT,
       a.ARCHETYPE_ID, a.ARCHETYPE_CONFIDENCE
FROM players p
LEFT JOIN archetypes a
    ON p.PLAYER_ID = a.PLAYER_ID
   AND p.SEASON_LABEL = a.SEASON_LABEL
WHERE p.SEASON_LABEL = ?
"""
season_df = pd.read_sql(query, conn, params=[TARGET_SEASON])
conn.close()

print(f"Loaded {len(season_df)} player rows for season {TARGET_SEASON} from SQL")

if season_df.empty:
    print('Error: no player rows returned from SQL query.')
    raise SystemExit(1)

# 3) Build vectors with training-time scalers
quality_col = artifacts.get('quality_col', 'QUALITY_PRIOR_Z')
season_df = attach_player_quality_column(
    season_df,
    artifacts.get('player_quality_z', {}),
    col_name=quality_col,
)
vector_df, _, _, player_feature_cols = build_player_feature_table(
    season_df,
    tracking_scaler=artifacts['tracking_scaler'],
    archetype_ids=artifacts['archetype_ids'],
    extra_numeric_cols=[quality_col],
)

name_to_vec = {
    row['PLAYER_NAME']: row[player_feature_cols].values.astype(np.float32)
    for _, row in vector_df.iterrows()
}
name_to_quality = {row['PLAYER_NAME']: float(row[quality_col]) for _, row in vector_df.iterrows()}

missing_core = [p for p in CORE_LINEUP if p not in name_to_vec]
if missing_core:
    print(f"Error: missing core players in SQL season {TARGET_SEASON}: {missing_core}")
    raise SystemExit(1)

core_matrix = np.stack([name_to_vec[p] for p in CORE_LINEUP], axis=0)

# 4) Score all candidates
candidate_names = [name for name in sorted(name_to_vec) if name not in CORE_LINEUP]
X_raw = np.array([featurize_core_candidate(core_matrix, name_to_vec[name]) for name in candidate_names], dtype=np.float32)
X_scaled = artifacts['input_scaler'].transform(X_raw).astype(np.float32)

with torch.no_grad():
    nn_scores = model(torch.from_numpy(X_scaled)).squeeze(1).numpy()
knn_scores = artifacts['knn_model'].predict(X_scaled)
alpha = float(artifacts['blend_alpha'])
base_scores = alpha * nn_scores + (1.0 - alpha) * knn_scores
quality_arr = np.array([name_to_quality.get(name, 0.0) for name in candidate_names], dtype=np.float32)
coef = np.array(artifacts.get('score_calibration_coef', [1.0, 0.0, 0.0]), dtype=np.float32)
scores = (coef[0] * base_scores) + (coef[1] * quality_arr) + coef[2]

predictions = sorted(zip(candidate_names, scores), key=lambda x: x[1], reverse=True)

print("\n--- TOP SQL RECOMMENDATIONS ---")
for i, (name, rating) in enumerate(predictions[:10], 1):
    print(f"{i:>2}. {name:<25} score={rating:+.2f}")
