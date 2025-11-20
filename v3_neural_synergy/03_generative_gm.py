import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# CONFIG
MODEL_PATH = 'v3_neural_synergy/synergy_model.pth'
PLAYER_DATA = 'data/processed_tracking_metrics.csv'

# Define the Core 4 (OKC Thunder Example)
# We want to find the perfect 5th man to play with this core.
CORE_LINEUP = ["Shai Gilgeous-Alexander", "Jalen Williams", "Chet Holmgren", "Luguentz Dort"]

print(f"--- GENERATIVE GM: Finding the perfect 5th for {CORE_LINEUP} ---")

# 1. Re-Define Model Structure (Must match training exactly)
class NBADeepSet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        player_embeddings = self.encoder(x)
        team_embedding = torch.sum(player_embeddings, dim=1)
        net_rating = self.decoder(team_embedding)
        return net_rating

# 2. Load Data & Model
df = pd.read_csv(PLAYER_DATA)
feature_cols = ['OFF_SPEED', 'DEF_SPEED', 'TIME_PER_TOUCH', 'DRIBBLES_PER_TOUCH', 
                'PTS_PER_TOUCH', 'DRIVE_PCT', 'CATCH_SHOOT_PCT', 'PULL_UP_PCT', 'PAINT_TOUCH_PCT']

# Scale Features (Consistency is key)
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Build Vector Dictionary & Name Lookup
name_to_vec = {}

for _, row in df.iterrows():
    # Use most recent season available for the player
    if row['SEASON_LABEL'] == '2024-25':
        vec = row[feature_cols].values.astype('float32')
        name_to_vec[row['PLAYER_NAME']] = vec

# Load Neural Net
model = NBADeepSet(input_dim=9)
try:
    model.load_state_dict(torch.load(MODEL_PATH))
except RuntimeError:
    # If saved on GPU but running on CPU, or vice versa
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# 3. Prepare the Core 4 Vectors
core_vectors = []
for name in CORE_LINEUP:
    if name in name_to_vec:
        core_vectors.append(name_to_vec[name])
    else:
        print(f"WARNING: Could not find {name} in 2024-25 data.")

if len(core_vectors) != 4:
    print("Error: Need exactly 4 valid players to run.")
    exit()

# 4. The "Generative" Loop
# Try every single player in the database as the 5th man
predictions = []

print(f"Scanning {len(name_to_vec)} candidates...")

with torch.no_grad():
    for candidate_name, candidate_vec in name_to_vec.items():
        # Skip if candidate is already in the lineup
        if candidate_name in CORE_LINEUP:
            continue
            
        # Construct the 5-man unit
        # Shape: (1, 5, 9) -> Batch size 1, 5 players, 9 features
        lineup = core_vectors + [candidate_vec]
        tensor_input = torch.FloatTensor(np.array([lineup]))
        
        # Predict
        pred_rating = model(tensor_input).item()
        predictions.append((candidate_name, pred_rating))

# 5. Rank and Reveal
predictions.sort(key=lambda x: x[1], reverse=True)

print("\n--- 🏆 TOP 5 FITS (Highest Predicted Net Rating) ---")
for i, (name, rating) in enumerate(predictions[:5]):
    print(f"{i+1}. {name:<25} (+{rating:.2f})")

print("\n--- 📉 WORST 5 FITS (Chemistry Killers) ---")
for i, (name, rating) in enumerate(predictions[-5:]):
    print(f"{i+1}. {name:<25} ({rating:.2f})")