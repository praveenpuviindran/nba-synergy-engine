import pandas as pd
import os

# 1. Setup Paths
INPUT_FILE = 'data/nba_game_logs_2010_2025.csv'
OUTPUT_FILE = 'data/cleaned_nba_logs.csv'

print("--- STARTING DATA CLEANING ---")

# 2. Load Data
df = pd.read_csv(INPUT_FILE)
print(f"Raw Data Loaded: {len(df)} rows")

# 3. Filter: Regular Season Only
# In NBA API, Game IDs starting with '2' are Regular Season. 
df['GAME_ID'] = df['GAME_ID'].astype(str)
df = df[df['GAME_ID'].str.startswith('2')]
print(f"Rows after filtering for Regular Season: {len(df)}")

# 4. Fix Data Types
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# Convert 'MIN' (Minutes) to numeric (Handle "24:15" vs 24.25)
def clean_minutes(x):
    if isinstance(x, str):
        if ':' in x:
            minutes, seconds = x.split(':')
            return float(minutes) + float(seconds)/60
        return float(x)
    return float(x)

df['MIN'] = df['MIN'].apply(clean_minutes)

# 5. Feature Selection 
cols_to_keep = [
    'SEASON_LABEL', 'GAME_ID', 'GAME_DATE', 'PLAYER_ID', 'PLAYER_NAME', 
    'TEAM_ABBREVIATION', 'MATCHUP', 'WL', 'MIN', 'PTS', 'FGM', 'FGA', 
    'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 
    'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS'
]
df = df[cols_to_keep]

# 6. Save
df.to_csv(OUTPUT_FILE, index=False)
print("--- CLEANING COMPLETE ---")
print(f"Cleaned data saved to: {OUTPUT_FILE}")
print(f"Final Row Count: {len(df)}")