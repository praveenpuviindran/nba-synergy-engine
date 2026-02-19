import sqlite3
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

# CONFIG
DB_PATH = 'nba_sql.db'
MODEL_PATH = 'v3_neural_synergy/synergy_model.pth'
CORE_LINEUP = ["Shai Gilgeous-Alexander", "Jalen Williams", "Chet Holmgren", "Luguentz Dort"]
TARGET_SEASON = '2024-25'
# Querying one season keeps inference memory bounded in SQL mode.

print(f"--- SQL-POWERED GM: Optimizing for {CORE_LINEUP} ---")

# 1. Model Definition (Standard)
class NBADeepSet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        return self.decoder(torch.sum(self.encoder(x), dim=1))

# 2. SQL QUERY (The New Part)
# Instead of loading all history, we request ONLY the season we need.
# This is O(1) memory usage vs O(N) loading huge CSVs.
conn = sqlite3.connect(DB_PATH)

query = """
SELECT * FROM players
WHERE SEASON_LABEL = ?
"""
df = pd.read_sql(query, conn, params=[TARGET_SEASON])
conn.close()

print(f"Loaded {len(df)} players from SQL Database for season {TARGET_SEASON}")

# 3. Prepare Data (Same Logic)
feature_cols = ['OFF_SPEED', 'DEF_SPEED', 'TIME_PER_TOUCH', 'DRIBBLES_PER_TOUCH', 
                'PTS_PER_TOUCH', 'DRIVE_PCT', 'CATCH_SHOOT_PCT', 'PULL_UP_PCT', 'PAINT_TOUCH_PCT']

scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

name_to_vec = {row['PLAYER_NAME']: row[feature_cols].values.astype('float32') for _, row in df.iterrows()}

# 4. Inference
model = NBADeepSet(input_dim=9)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
except:
    print("Error: Run V3 training first.")
    exit()
model.eval()

core_vectors = []
for name in CORE_LINEUP:
    if name in name_to_vec:
        core_vectors.append(name_to_vec[name])

if len(core_vectors) != 4:
    print("Error: Core players not found in database.")
    exit()

# 5. Run Scan
predictions = []
with torch.no_grad():
    for candidate, vec in name_to_vec.items():
        if candidate in CORE_LINEUP: continue
        lineup = core_vectors + [vec]
        tensor_input = torch.FloatTensor(np.array([lineup]))
        rating = model(tensor_input).item()
        predictions.append((candidate, rating))

predictions.sort(key=lambda x: x[1], reverse=True)

print("\n--- 🏆 TOP SQL RECOMMENDATIONS ---")
for i, (name, rating) in enumerate(predictions[:5]):
    print(f"{i+1}. {name:<25} (+{rating:.2f})")
