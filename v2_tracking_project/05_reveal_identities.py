import pandas as pd

INPUT_FILE = 'data/gmm_archetypes.csv'

print("--- ARCHETYPE REVEAL ---")

df = pd.read_csv(INPUT_FILE)

# Get the list of clusters sorted by their "Value" (Impact Score) from high to low
# We approximate value using the same formula as the plot
df['IMPACT_SCORE'] = (df['PTS_PER_TOUCH'] * 100) + (df['OFF_SPEED'] * 10)
cluster_order = df.groupby('ARCHETYPE_ID')['IMPACT_SCORE'].mean().sort_values(ascending=False).index

print(f"{'ID':<5} {'AVG IMPACT':<12} {'SAMPLE PLAYERS'}")
print("-" * 60)

for cluster_id in cluster_order:
    # Get players in this cluster from the most recent season (2024-25)
    recent_season = df[df['SEASON_LABEL'] == '2024-25']
    cluster_players = recent_season[recent_season['ARCHETYPE_ID'] == cluster_id]
    
    # If no players in 2025 (rare), check 2024
    if cluster_players.empty:
        cluster_players = df[(df['SEASON_LABEL'] == '2023-24') & (df['ARCHETYPE_ID'] == cluster_id)]
    
    # Get the top 3 "most confident" matches for this archetype
    # (The players who define the style perfectly)
    top_examples = cluster_players.sort_values(by='ARCHETYPE_CONFIDENCE', ascending=False).head(3)['PLAYER_NAME'].values
    
    avg_score = df[df['ARCHETYPE_ID'] == cluster_id]['IMPACT_SCORE'].mean()
    
    print(f"{cluster_id:<5} {avg_score:<12.1f} {top_examples}")