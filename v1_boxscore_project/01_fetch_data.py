import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
import time
import os

# --- UPDATE: NOW PULLING 2010-2025 ---
# Function to generate season strings (e.g., "2010-11")
def get_season_list(start_year, end_year):
    seasons = []
    for year in range(start_year, end_year + 1):
        next_year_suffix = str(year + 1)[-2:] # Get last two digits (e.g., '11')
        seasons.append(f"{year}-{next_year_suffix}")
    return seasons

# Generate 2010-11 to 2024-25
SEASONS = get_season_list(2010, 2024) 

OUTPUT_FOLDER = 'data'
OUTPUT_FILE = 'nba_game_logs_2010_2025.csv' # Renamed file to reflect new range

print("--- STARTING 15-YEAR DATA INGESTION ---")
print(f"Target Seasons: {SEASONS}")

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

all_season_logs = []

for season in SEASONS:
    print(f"Fetching data for {season}...")
    try:
        log = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='P').get_data_frames()[0]
        log['SEASON_LABEL'] = season
        all_season_logs.append(log)
        print(f"   > Success: {len(log)} rows.")
        time.sleep(1) 
    except Exception as e:
        print(f"   > ERROR fetching {season}: {e}")

if all_season_logs:
    final_df = pd.concat(all_season_logs, ignore_index=True)
    full_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)
    final_df.to_csv(full_path, index=False)
    print("\n--- INGESTION COMPLETE ---")
    print(f"Total Records: {len(final_df)}")
else:
    print("FAILURE: No data collected.")