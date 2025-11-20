import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# CONFIG
BATCH_SIZE = 64
EPOCHS = 80  # Increased Epochs for complex logic
LEARNING_RATE = 0.0005 # Slower learning rate for stability

print("--- INITIALIZING VARIANCE-AWARE SYNERGY ENGINE ---")

# 1. Load Data
lineups_path = 'v3_neural_synergy/data/lineups_2014_2025.csv'
player_stats_path = 'data/processed_tracking_metrics.csv'

lineups = pd.read_csv(lineups_path)
players = pd.read_csv(player_stats_path)

# 2. Create Features
feature_cols = ['OFF_SPEED', 'DEF_SPEED', 'TIME_PER_TOUCH', 'DRIBBLES_PER_TOUCH', 
                'PTS_PER_TOUCH', 'DRIVE_PCT', 'CATCH_SHOOT_PCT', 'PULL_UP_PCT', 'PAINT_TOUCH_PCT']

scaler = StandardScaler()
players[feature_cols] = scaler.fit_transform(players[feature_cols])

player_dict = {}
for _, row in players.iterrows():
    key = (int(row['PLAYER_ID']), row['SEASON_LABEL'])
    player_dict[key] = row[feature_cols].values.astype('float32')

# 3. Construct Tensors
X_list = []
y_list = []

for _, row in lineups.iterrows():
    raw_ids = str(row['GROUP_ID']).replace('-', ' ').split()
    season = row['SEASON_LABEL']
    vectors = []
    for pid in raw_ids:
        try:
            vec = player_dict[(int(pid), season)]
            vectors.append(vec)
        except KeyError:
            pass
    if len(vectors) == 5:
        X_list.append(vectors)
        y_list.append(row['PLUS_MINUS'])

X = np.array(X_list)
y = np.array(y_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# 4. Define the VARIANCE-AWARE DeepSet
class NBADeepSet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Encoder: Embeds playstyle
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Decoder: Reads (Sum + Std_Dev) -> Net Rating
        # Input is 64 because (32 from Sum + 32 from Std)
        self.decoder = nn.Sequential(
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Dropout(0.2), # Prevents overfitting to "Bigs are good"
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x shape: (Batch, 5, 9)
        player_embeddings = self.encoder(x) # (Batch, 5, 32)
        
        # POOLING: The Magic Trick
        # We capture Total Talent (Sum) AND Skill Diversity (Std)
        # If everyone is a Center, Std will be low (Bad). 
        # If you have a Guard and a Center, Std will be high (Good).
        team_sum = torch.sum(player_embeddings, dim=1) # (Batch, 32)
        team_std = torch.std(player_embeddings, dim=1) # (Batch, 32)
        
        # Combine them
        team_vector = torch.cat([team_sum, team_std], dim=1) # (Batch, 64)
        
        return self.decoder(team_vector)

# 5. Training
model = NBADeepSet(input_dim=9)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Training on {len(X_train)} samples with Diversity-Aware Pooling...")

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train_tensor)
    loss = criterion(preds, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

# 6. Save
save_path = 'v3_neural_synergy/synergy_model.pth'
torch.save(model.state_dict(), save_path)
print(f"Retrained Model saved to: {save_path}")