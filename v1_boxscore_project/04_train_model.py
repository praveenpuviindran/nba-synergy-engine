import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = 'data/player_season_stats.csv'
OUTPUT_FILE = 'data/clustered_nba_data.csv'

print("--- STARTING MACHINE LEARNING MODEL ---")

# 1. Load Data
df = pd.read_csv(INPUT_FILE)

# 2. Select Features for Clustering
# We exclude names/IDs/Years because we want to cluster based on *playing style* only.
features = [
    'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT', 
    'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 
    'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 
    '3P_RATE', 'FT_RATE', 'AST_TOV'
]

# Drop rows with any remaining NaNs (just in case)
X = df[features].dropna()

# 3. Scale the Data (Crucial for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train K-Means Model
# We choose k=8 to capture distinct roles (Starters, Bench, Bigs, Guards, Wings, etc.)
k = 8
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 5. Assign Clusters Back to Original Data
# We align the indices to ensure the labels match the correct players
df.loc[X.index, 'ARCHETYPE_ID'] = kmeans.labels_

# 6. Analyze the Clusters (Give them human-readable names roughly)
print("\n--- CLUSTER SIZES ---")
print(df['ARCHETYPE_ID'].value_counts().sort_index())

# 7. Save the Result
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nModel trained. Data with Archetypes saved to: {OUTPUT_FILE}")

# --- OPTIONAL: SAVE A VISUALIZATION ---
# We use a simple chart to show where the clusters differ on Points vs 3P Rate
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PTS', y='3P_RATE', hue='ARCHETYPE_ID', palette='viridis', alpha=0.6)
plt.title('NBA Archetypes: Scoring Volume vs. 3-Point Frequency')
plt.savefig('data/cluster_visualization.png')
print("Visualization saved to data/cluster_visualization.png")