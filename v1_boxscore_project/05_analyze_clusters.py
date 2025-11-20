import pandas as pd

INPUT_FILE = 'data/clustered_nba_data.csv'

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("--- ANALYZING CLUSTER IDENTITIES ---")

# 1. Load Data
df = pd.read_csv(INPUT_FILE)

# 2. Group by Cluster and Calculate Averages
# We only look at the stats that define a "role"
cols_to_analyze = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3P_RATE', 'FG_PCT', 'MIN', 'GP']
cluster_profiles = df.groupby('ARCHETYPE_ID')[cols_to_analyze].mean().round(2)

# 3. Add a "Count" column so we know how big the group is
cluster_profiles['COUNT'] = df['ARCHETYPE_ID'].value_counts()

# 4. Sort by Points (PTS) to see the "Stars" at the top
cluster_profiles = cluster_profiles.sort_values(by='PTS', ascending=False)

print(cluster_profiles)

# 5. Sample Players
# Print 3 random names from each cluster so we can sanity check
print("\n--- SAMPLE PLAYERS PER CLUSTER ---")
for cluster_id in cluster_profiles.index:
    sample_players = df[df['ARCHETYPE_ID'] == cluster_id]['PLAYER_NAME'].sample(3).values
    print(f"Cluster {cluster_id}: {sample_players}")