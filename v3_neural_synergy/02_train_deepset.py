import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# CONFIGURATION
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

print("--- INITIALIZING NEURAL SYNERGY ENGINE ---")

# 1. Load Data
# We assume you run this from the ROOT folder
lineups_path = 'v3_neural_synergy/data/lineups_2014_2025.csv'
player_stats_path = 'data/processed_tracking_metrics.csv'

print(f"Loading Lineups from: {lineups_path}")
print(f"Loading Player Stats from: {player_stats_path}")

lineups = pd.read_csv(lineups_path)
players = pd.read_csv(player_stats_path)

# 2. Create the "Player Feature Lookup"
# We need to turn a Player ID + Season into a vector of numbers
feature_cols = ['OFF_SPEED', 'DEF_SPEED', 'TIME_PER_TOUCH', 'DRIBBLES_PER_TOUCH', 
                'PTS_PER_TOUCH', 'DRIVE_PCT', 'CATCH_SHOOT_PCT', 'PULL_UP_PCT', 'PAINT_TOUCH_PCT']

# Scale features (Crucial for Neural Networks)
scaler = StandardScaler()
players[feature_cols] = scaler.fit_transform(players[feature_cols])

# Create a fast dictionary for lookups: key=(ID, Season), value=Vector
player_dict = {}
for _, row in players.iterrows():
    key = (int(row['PLAYER_ID']), row['SEASON_LABEL'])
    player_dict[key] = row[feature_cols].values.astype('float32')

print("Player dictionary built.")

# 3. Construct Training Tensors
# We map every lineup to 5 player vectors
X_list = []
y_list = []
skipped_count = 0

for _, row in lineups.iterrows():
    # Clean IDs (API format is sometimes "-123-456-")
    raw_ids = str(row['GROUP_ID']).replace('-', ' ').split()
    season = row['SEASON_LABEL']
    
    vectors = []
    for pid in raw_ids:
        try:
            vec = player_dict[(int(pid), season)]
            vectors.append(vec)
        except KeyError:
            # Player missing from tracking data (rare, but happens)
            pass
            
    # Only accept lineups where we found all 5 players
    if len(vectors) == 5:
        X_list.append(vectors)
        y_list.append(row['PLUS_MINUS']) # Target Variable: Net Rating
    else:
        skipped_count += 1

X = np.array(X_list) # Shape: (N_samples, 5_players, 9_features)
y = np.array(y_list)

print(f"Training Data Ready: {len(X)} samples (Skipped {skipped_count} incomplete lineups)")

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# 4. Define the DeepSet Model
class NBADeepSet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Encoder (Phi): Analyzes individual player capability
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Decoder (Rho): Analyzes the "Team Sum" to predict winning
        self.decoder = nn.Sequential(
            nn.Linear(16, 16), # Input is size 16 because we summed the 5 encoded vectors
            nn.ReLU(),
            nn.Linear(16, 1)   # Output is Net Rating
        )
        
    def forward(self, x):
        # x shape: (Batch, 5_players, 9_features)
        
        # 1. Encode Each Player
        player_embeddings = self.encoder(x) # Shape: (Batch, 5, 16)
        
        # 2. Sum Pool (Permutation Invariance)
        # This creates the "Chemistry" vector. Order doesn't matter.
        team_embedding = torch.sum(player_embeddings, dim=1) # Shape: (Batch, 16)
        
        # 3. Decode to Win Prediction
        net_rating = self.decoder(team_embedding)
        return net_rating

# 5. Training Loop
model = NBADeepSet(input_dim=9)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n--- TRAINING NEURAL NETWORK ---")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    preds = model(X_train_tensor)
    loss = criterion(preds, y_train_tensor)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

# 6. Evaluation
model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_loss = criterion(test_preds, y_test_tensor)
    rmse = np.sqrt(test_loss.item())
    
    print(f"\nFINAL RESULTS:")
    print(f"Test RMSE: {rmse:.2f} (Avg Error in Net Rating points)")
    
    # Save Model
    save_path = 'v3_neural_synergy/synergy_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")