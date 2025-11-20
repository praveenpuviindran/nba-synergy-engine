import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# We need to load the ORIGINAL logs to get "PLUS_MINUS" or "W_PCT" for value
# Since we filtered data, we need to rejoin with performance metrics.
# For this demo, we will assume "PTS_PER_TOUCH" and "EFFICIENCY" proxies from our processed data
# are the "Value" we are tracking. 

INPUT_FILE = 'data/gmm_archetypes.csv'
OUTPUT_IMG = 'data/archetype_value_forecast.png'

print("--- STARTING VALUE FORECAST ---")

df = pd.read_csv(INPUT_FILE)

# 1. Define "Value"
# In a real team environment, we would use RAPM or EPM.
# Here, we define Value = (PTS_PER_TOUCH * 10) + (OFF_SPEED * 2) - (DEF_SPEED * 2)
# (This is a simplified "Impact Score" for demonstration)
df['IMPACT_SCORE'] = (df['PTS_PER_TOUCH'] * 100) + (df['OFF_SPEED'] * 10)

# 2. Calculate Average Impact per Archetype per Year
trends = df.groupby(['SEASON_LABEL', 'ARCHETYPE_ID'])['IMPACT_SCORE'].mean().reset_index()

# 3. Forecast (Linear Regression per Cluster)
unique_clusters = df['ARCHETYPE_ID'].unique()
forecasts = []

# Map seasons to numbers (0, 1, 2...) for regression
seasons = sorted(df['SEASON_LABEL'].unique())
season_map = {season: i for i, season in enumerate(seasons)}
trends['SEASON_NUM'] = trends['SEASON_LABEL'].map(season_map)

plt.figure(figsize=(14, 8))

for cluster in unique_clusters:
    cluster_data = trends[trends['ARCHETYPE_ID'] == cluster]
    
    # Train Regression
    X = cluster_data[['SEASON_NUM']]
    y = cluster_data['IMPACT_SCORE']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next 5 years (2025-2030)
    future_X = np.array(range(len(seasons), len(seasons) + 5)).reshape(-1, 1)
    future_y = model.predict(future_X)
    
    # Calculate Slope (Growth Rate)
    slope = model.coef_[0]
    label = f"Cluster {cluster} (Growth: {slope:.2f})"
    
    # Plot History
    sns.lineplot(data=cluster_data, x='SEASON_NUM', y='IMPACT_SCORE', label=label, alpha=0.7)
    
    # Plot Future (Dotted Line)
    # We just plot the line from last data point to end of forecast
    last_val = cluster_data['IMPACT_SCORE'].iloc[-1]
    last_season = cluster_data['SEASON_NUM'].iloc[-1]
    
    plt.plot([last_season, future_X[-1][0]], [last_val, future_y[-1]], linestyle='--', linewidth=2)

plt.title('Projected Value of NBA Archetypes (2025-2030)', fontsize=16)
plt.xlabel('Season Index (0=2014, 15=2030)', fontsize=12)
plt.ylabel('Impact Score (Efficiency + Motor)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUTPUT_IMG)

print(f"Forecast saved to {OUTPUT_IMG}")
print("--- PROJECT COMPLETE ---")