import pandas as pd
from nba_api.stats.endpoints import leaguedashlineups
import time
import os

# CONFIGURATION
SEASONS = [
    '2014-15', '2015-16', '2016-17', '2017-18', '2018-19', 
    '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25'
]

# We save this inside the V3 folder to keep things organized
OUTPUT_FOLDER = 'v3_neural_synergy/data'
OUTPUT_FILE = 'lineups_2014_2025.csv'

print("--- STARTING LINEUP INGESTION (The Training Data) ---")

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

all_lineups = []

for season in SEASONS:
    print(f"Processing {season}...")
    try:
        # Fetch 5-man lineups
        # measure_type_detailed_defense='Base' gives us Net Rating
        api = leaguedashlineups.LeagueDashLineups(
            season=season, 
            measure_type_detailed_defense='Base', 
            group_quantity=5
        )
        df = api.get_data_frames()[0]
        
        # FILTER: Garbage Time Removal
        # We only learn from lineups that played at least 48 minutes together (approx 1 full game total)
        # This removes "fluke" lineups that played 2 minutes and got lucky.
        df = df[df['MIN'] > 48]
        
        df['SEASON_LABEL'] = season
        
        # Important: The 'GROUP_ID' column contains the 5 Player IDs separated by dashes
        # We will need this later to link to our Deep Learning model.
        
        all_lineups.append(df)
        print(f"   > Found {len(df)} qualified lineups.")
        
        # Sleep to respect API limits
        time.sleep(1)
        
    except Exception as e:
        print(f"   > Error fetching {season}: {e}")

# Save to CSV
if all_lineups:
    final_df = pd.concat(all_lineups, ignore_index=True)
    
    path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)
    final_df.to_csv(path, index=False)
    
    print("\n--- INGESTION COMPLETE ---")
    print(f"Total Training Samples: {len(final_df)}")
    print(f"Saved to: {path}")
else:
    print("FAILURE: No data collected.")