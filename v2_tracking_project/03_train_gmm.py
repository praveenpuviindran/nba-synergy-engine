import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = 'data/processed_tracking_metrics.csv'
OUTPUT_FILE = 'data/gmm_archetypes.csv'

print("--- STARTING GMM CLUSTERING ---")

# 1. Load Data
df = pd.read_csv(INPUT_FILE)

# 2. Prepare Features
# We exclude ID/Name/Season from the math
feature_cols = [
    'OFF_SPEED', 'DEF_SPEED', 'TIME_PER_TOUCH', 'DRIBBLES_PER_TOUCH',
    'PTS_PER_TOUCH', 'DRIVE_PCT', 'CATCH_SHOOT_PCT', 'PULL_UP_PCT', 'PAINT_TOUCH_PCT'
]
X = df[feature_cols]

# 3. Scale Data (StandardScaler)
# GMM is very sensitive to scale. We must normalize.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Dimensionality Reduction (PCA)
# We want to keep enough components to explain 95% of the variance.
# This removes "noise" while keeping the "signal".
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA Reduced Features from {len(feature_cols)} to {X_pca.shape[1]} Components")

# 5. Train Gaussian Mixture Model (GMM)
# We select 10 Clusters to capture nuance (Ball Handlers, Wings, Bigs, Hybrids)
n_clusters = 10
gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
gmm.fit(X_pca)

# 6. Get Probabilities (The "Soft" Labels)
# This gives us the probability of the player belonging to EACH of the 10 clusters
probs = gmm.predict_proba(X_pca)

# We assign the "Primary" Archetype (highest probability)
primary_labels = gmm.predict(X_pca)
df['ARCHETYPE_ID'] = primary_labels

# We also save the "Max Probability" to see how well they fit
# (e.g., 0.99 means "Pure Archetype", 0.40 means "Hybrid")
df['ARCHETYPE_CONFIDENCE'] = np.max(probs, axis=1)

# 7. Analyze Cluster Sizes
print("\n--- CLUSTER DISTRIBUTION ---")
print(df['ARCHETYPE_ID'].value_counts().sort_index())

# 8. Save
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved Clustered Data to: {OUTPUT_FILE}")

# --- VISUALIZATION: PCA PROJECTION ---
# We plot the first 2 PCA components to see if the clusters look distinct
plt.figure(figsize=(10, 8))
df['PCA_1'] = X_pca[:, 0]
df['PCA_2'] = X_pca[:, 1]
sns.scatterplot(data=df, x='PCA_1', y='PCA_2', hue='ARCHETYPE_ID', palette='tab10', alpha=0.6)
plt.title('NBA Archetypes: GMM Clusters on Tracking Data')
plt.savefig('data/gmm_visualization.png')
print("Visualization saved to data/gmm_visualization.png")