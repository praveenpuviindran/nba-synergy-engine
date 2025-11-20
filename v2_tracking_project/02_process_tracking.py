import pandas as pd
import numpy as np

INPUT_FILE = 'data/tracking_data_2014_2025.csv'
OUTPUT_FILE = 'data/processed_tracking_metrics.csv'

print("--- STARTING FEATURE ENGINEERING ---")

# 1. Load Data
df = pd.read_csv(INPUT_FILE)

# 2. Filter Garbage Time
df['MPG'] = df['MIN'] / df['GP']
df = df[df['GP'] >= 15]
df = df[df['MPG'] >= 10]

print(f"Filtered Roster Size: {len(df)} (Removed low-minute players)")

# 3. Create Playstyle Ratios
# Total Tracking Points
df['TRACKING_PTS_TOTAL'] = df['DRIVE_PTS'] + df['CATCH_SHOOT_PTS'] + df['PULL_UP_PTS'] + df['PAINT_TOUCH_PTS']
df = df[df['TRACKING_PTS_TOTAL'] > 0]

# Ratios
df['DRIVE_PCT'] = df['DRIVE_PTS'] / df['TRACKING_PTS_TOTAL']
df['CATCH_SHOOT_PCT'] = df['CATCH_SHOOT_PTS'] / df['TRACKING_PTS_TOTAL']
df['PULL_UP_PCT'] = df['PULL_UP_PTS'] / df['TRACKING_PTS_TOTAL']
df['PAINT_TOUCH_PCT'] = df['PAINT_TOUCH_PTS'] / df['TRACKING_PTS_TOTAL']

# Efficiency
df['PTS_PER_TOUCH'] = df['TRACKING_PTS_TOTAL'] / df['TOUCHES']

# --- THE FIX IS HERE ---
# Renaming the column correctly
df.rename(columns={
    'AVG_SPEED_OFF': 'OFF_SPEED',
    'AVG_SPEED_DEF': 'DEF_SPEED',
    'AVG_SEC_PER_TOUCH': 'TIME_PER_TOUCH',
    'AVG_DRIB_PER_TOUCH': 'DRIBBLES_PER_TOUCH'  # MATCHES NEW API NAME
}, inplace=True)

# 4. Select Final Features
features = [
    'PLAYER_ID', 'PLAYER_NAME', 'SEASON_LABEL',
    'OFF_SPEED', 'DEF_SPEED',
    'TIME_PER_TOUCH', 'DRIBBLES_PER_TOUCH',
    'PTS_PER_TOUCH',
    'DRIVE_PCT', 'CATCH_SHOOT_PCT', 'PULL_UP_PCT', 'PAINT_TOUCH_PCT'
]

final_df = df[features].copy()
final_df.fillna(0, inplace=True)

# 5. Save
final_df.to_csv(OUTPUT_FILE, index=False)
print("--- PROCESSING COMPLETE ---")
print(f"Saved Feature Matrix to: {OUTPUT_FILE}")
print(f"Final Row Count: {len(final_df)}")