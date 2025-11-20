import pandas as pd
import numpy as np  # We need numpy to detect Infinity

INPUT_FILE = 'data/cleaned_nba_logs.csv'
OUTPUT_FILE = 'data/player_season_stats.csv'

print("--- STARTING FEATURE ENGINEERING ---")

# 1. Load Cleaned Data
df = pd.read_csv(INPUT_FILE)

# 2. Group by Player and Season
aggregations = {
    'GAME_ID': 'count',      
    'MIN': 'mean',           
    'PTS': 'mean',
    'REB': 'mean',
    'AST': 'mean',
    'STL': 'mean',
    'BLK': 'mean',
    'TOV': 'mean',
    'PF': 'mean',            
    'FGM': 'sum',
    'FGA': 'sum',
    'FG3M': 'sum',
    'FG3A': 'sum',
    'FTM': 'sum',
    'FTA': 'sum',
    'OREB': 'sum',
    'DREB': 'sum'
}

season_stats = df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'SEASON_LABEL']).agg(aggregations).reset_index()
season_stats.rename(columns={'GAME_ID': 'GP'}, inplace=True)

# 3. Filter: Remove "Cup of Coffee" Players (< 10 games)
season_stats = season_stats[season_stats['GP'] >= 10]

# 4. Calculate Derived Metrics
season_stats['FG_PCT'] = season_stats['FGM'] / season_stats['FGA']
season_stats['FG3_PCT'] = season_stats['FG3M'] / season_stats['FG3A']
season_stats['FT_PCT'] = season_stats['FTM'] / season_stats['FTA']

# Style Metrics
season_stats['3P_RATE'] = season_stats['FG3A'] / season_stats['FGA']
season_stats['FT_RATE'] = season_stats['FTA'] / season_stats['FGA']
season_stats['TS_PCT'] = season_stats['PTS'] / (2 * (season_stats['FGA'] + 0.44 * season_stats['FTA']))
season_stats['AST_TOV'] = season_stats['AST'] / season_stats['TOV']

# --- THE FIX: CLEAN UP INFINITY AND NAN ---
# Replace Infinity (from dividing by zero) with 0
season_stats.replace([np.inf, -np.inf], 0, inplace=True)
# Replace NaN (missing data) with 0
season_stats = season_stats.fillna(0)

# 5. Save
season_stats.to_csv(OUTPUT_FILE, index=False)
print("--- ENGINEERING COMPLETE ---")
print(f"Season Aggregates Saved to: {OUTPUT_FILE}")