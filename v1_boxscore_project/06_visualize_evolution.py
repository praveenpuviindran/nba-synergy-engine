import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = 'data/clustered_nba_data.csv'
OUTPUT_IMAGE = 'data/archetype_evolution.png'

print("--- GENERATING EVOLUTION CHART ---")

# 1. Load Data
df = pd.read_csv(INPUT_FILE)

# 2. Map IDs to Human Labels (Based on your specific cluster output)
# NOTE: If you re-run K-Means, these IDs might swap. 
# Based on your provided text: 7=Stars, 3=Paint Beasts, 0=Shooters
archetype_map = {
    7: 'Elite Scorers',
    3: 'Paint Beasts',
    0: 'High-Volume Shooters',
    4: 'Secondary Creators',
    1: 'Role Bigs',
    2: 'Rotation Wings',
    5: 'Deep Bench Bigs',
    6: 'Deep Bench Guards'
}

df['Archetype'] = df['ARCHETYPE_ID'].map(archetype_map)

# 3. Calculate Proportion per Season
# We want to know: "In 2012, what % of the league were Paint Beasts?"
evolution = df.groupby(['SEASON_LABEL', 'Archetype']).size().reset_index(name='Count')

# Get total players per season to calculate percentage
season_totals = df.groupby('SEASON_LABEL').size().reset_index(name='Total')
evolution = evolution.merge(season_totals, on='SEASON_LABEL')
evolution['Percentage'] = (evolution['Count'] / evolution['Total']) * 100

# 4. Plotting
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# We filter out "Deep Bench" because they clutter the chart and aren't interesting
main_archetypes = [
    'Elite Scorers', 'Paint Beasts', 'High-Volume Shooters', 
    'Secondary Creators', 'Role Bigs', 'Rotation Wings'
]
evolution_filtered = evolution[evolution['Archetype'].isin(main_archetypes)]

sns.lineplot(
    data=evolution_filtered, 
    x='SEASON_LABEL', 
    y='Percentage', 
    hue='Archetype', 
    linewidth=2.5,
    palette='tab10'
)

plt.title('The Evolution of NBA Player Archetypes (2010-2025)', fontsize=16, weight='bold')
plt.ylabel('Share of League (%)', fontsize=12)
plt.xlabel('Season', fontsize=12)
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# 5. Save
plt.savefig(OUTPUT_IMAGE)
print(f"Evolution chart saved to: {OUTPUT_IMAGE}")