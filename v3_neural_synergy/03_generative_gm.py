import os
import pickle
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
MODEL_PATH = 'v3_neural_synergy/synergy_model.pth'
ARTIFACT_PATH = 'v3_neural_synergy/synergy_artifacts.pkl'
PLAYER_DATA = 'data/gmm_archetypes.csv'
CORE_LINEUP = ["Shai Gilgeous-Alexander", "Jalen Williams", "Chet Holmgren", "Luguentz Dort"]
TARGET_SEASON = '2024-25'

print(f"--- GENERATIVE GM: Finding the best contextual 5th for {CORE_LINEUP} ---")

if not os.path.exists(MODEL_PATH) or not os.path.exists(ARTIFACT_PATH):
    print("Error: missing model artifacts. Run `python3 v3_neural_synergy/02_train_deepset.py` first.")
    raise SystemExit(1)

# 1) Load Model + Artifacts
with open(ARTIFACT_PATH, 'rb') as f:
    artifacts = pickle.load(f)
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

model = ContextAwareSynergyNet(input_dim=int(checkpoint['input_dim']))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 2) Load and vectorize player-season features
raw_df = pd.read_csv(PLAYER_DATA)
quality_col = artifacts.get('quality_col', 'QUALITY_PRIOR_Z')
raw_df = attach_player_quality_column(raw_df, artifacts.get('player_quality_z', {}), col_name=quality_col)
player_table, _, _, player_feature_cols = build_player_feature_table(
    raw_df,
    tracking_scaler=artifacts['tracking_scaler'],
    archetype_ids=artifacts['archetype_ids'],
    extra_numeric_cols=[quality_col],
)

season_df = player_table[player_table['SEASON_LABEL'] == TARGET_SEASON].copy()
name_to_vec = {
    row['PLAYER_NAME']: row[player_feature_cols].values.astype(np.float32)
    for _, row in season_df.iterrows()
}
name_to_quality = {row['PLAYER_NAME']: float(row[quality_col]) for _, row in season_df.iterrows()}

# 3) Validate core
missing = [name for name in CORE_LINEUP if name not in name_to_vec]
if missing:
    print(f"Error: missing core players in {TARGET_SEASON} data: {missing}")
    raise SystemExit(1)

core_matrix = np.stack([name_to_vec[p] for p in CORE_LINEUP], axis=0)

# 4) Score all candidates in one batch
candidate_names = [p for p in sorted(name_to_vec) if p not in CORE_LINEUP]
X_raw = np.array([featurize_core_candidate(core_matrix, name_to_vec[c]) for c in candidate_names], dtype=np.float32)
X_scaled = artifacts['input_scaler'].transform(X_raw).astype(np.float32)

with torch.no_grad():
    nn_scores = model(torch.from_numpy(X_scaled)).squeeze(1).numpy()
knn_scores = artifacts['knn_model'].predict(X_scaled)
alpha = float(artifacts['blend_alpha'])
base_scores = alpha * nn_scores + (1.0 - alpha) * knn_scores
quality_arr = np.array([name_to_quality.get(c, 0.0) for c in candidate_names], dtype=np.float32)
coef = np.array(artifacts.get('score_calibration_coef', [1.0, 0.0, 0.0]), dtype=np.float32)
final_scores = (coef[0] * base_scores) + (coef[1] * quality_arr) + coef[2]

k_for_conf = min(10, int(getattr(artifacts['knn_model'], 'n_neighbors', 10)))
dists, _ = artifacts['knn_model'].kneighbors(X_scaled, n_neighbors=k_for_conf)
confidence = 1.0 / (1.0 + dists.mean(axis=1))

predictions = list(zip(candidate_names, final_scores, confidence))
predictions.sort(key=lambda x: x[1], reverse=True)

# 5) Report
print("\n--- TOP 10 FITS (Synergy Score) ---")
for i, (name, score, conf) in enumerate(predictions[:10], 1):
    print(f"{i:>2}. {name:<26} score={score:+.2f}  conf={conf:.3f}")

print("\n--- WORST 10 FITS (Synergy Score) ---")
for i, (name, score, conf) in enumerate(predictions[-10:], 1):
    print(f"{i:>2}. {name:<26} score={score:+.2f}  conf={conf:.3f}")
