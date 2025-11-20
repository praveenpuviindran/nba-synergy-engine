import pandas as pd
from nba_api.stats.endpoints import leaguedashptstats
import time
import os

# CONFIGURATION
SEASONS = [
    '2013-14', '2014-15', '2015-16', '2016-17', '2017-18', 
    '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', 
    '2023-24', '2024-25'
]

OUTPUT_FOLDER = 'data'
OUTPUT_FILE = 'tracking_data_2014_2025.csv'

MEASURE_TYPES = ['Possessions', 'SpeedDistance', 'Defense', 'Efficiency']

print("--- STARTING TRACKING DATA INGESTION (FIXED) ---")

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

all_seasons_data = []

for season in SEASONS:
    print(f"Processing Season: {season}...")
    
    season_merged = None
    
    try:
        for measure in MEASURE_TYPES:
            print(f"   > Fetching {measure}...")
            
            api_call = leaguedashptstats.LeagueDashPtStats(
                season=season,
                player_or_team='Player',
                pt_measure_type=measure
            )
            df = api_call.get_data_frames()[0]
            
            # --- THE FIX IS HERE ---
            if measure == 'Possessions':
                # Corrected 'AVG_DRIB_PER_TOUCH'
                cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GP', 'MIN', 
                        'AVG_SEC_PER_TOUCH', 'AVG_DRIB_PER_TOUCH', 'TOUCHES', 
                        'ELBOW_TOUCHES', 'POST_TOUCHES', 'PAINT_TOUCHES']
            elif measure == 'SpeedDistance':
                cols = ['PLAYER_ID', 'AVG_SPEED', 'DIST_MILES', 'AVG_SPEED_OFF', 'AVG_SPEED_DEF']
            elif measure == 'Defense':
                cols = ['PLAYER_ID', 'STL', 'BLK'] 
            elif measure == 'Efficiency':
                cols = ['PLAYER_ID', 'DRIVE_PTS', 'CATCH_SHOOT_PTS', 'PULL_UP_PTS', 'PAINT_TOUCH_PTS']

            # Filter
            available_cols = [c for c in cols if c in df.columns]
            df_subset = df[available_cols]

            # Merge
            if season_merged is None:
                season_merged = df_subset
            else:
                season_merged = pd.merge(season_merged, df_subset, on='PLAYER_ID', how='left')
            
            time.sleep(0.6) 
        
        if season_merged is not None:
            season_merged['SEASON_LABEL'] = season
            all_seasons_data.append(season_merged)
            print(f"   > Completed {season}: {len(season_merged)} players.")
    
    except Exception as e:
        print(f"   > ERROR in {season}: {e}")

# Save
if all_seasons_data:
    final_df = pd.concat(all_seasons_data, ignore_index=True)
    final_df.fillna(0, inplace=True)
    
    path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)
    final_df.to_csv(path, index=False)
    print("\n--- INGESTION COMPLETE ---")
    print(f"Saved {len(final_df)} rows to {path}")
else:
    print("FAILURE: No data collected.")