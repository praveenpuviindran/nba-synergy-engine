import sqlite3
import pandas as pd
import os

# CONFIG
DB_NAME = 'nba_sql.db'
TRACKING_CSV = 'data/processed_tracking_metrics.csv'
LINEUPS_CSV = 'v3_neural_synergy/data/lineups_2014_2025.csv'
ARCHETYPES_CSV = 'data/gmm_archetypes.csv'

print("--- STARTING SQL MIGRATION ---")

# 1. Connect to Database (Creates it if it doesn't exist)
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# 2. Define Schemas (Showing off SQL DDL skills)
# We drop tables if they exist to allow re-running the script cleanly
cursor.execute("DROP TABLE IF EXISTS players")
cursor.execute("DROP TABLE IF EXISTS lineups")
cursor.execute("DROP TABLE IF EXISTS archetypes")

# Create Players Table
create_players_table = """
CREATE TABLE players (
    PLAYER_ID INTEGER,
    PLAYER_NAME TEXT,
    SEASON_LABEL TEXT,
    OFF_SPEED REAL,
    DEF_SPEED REAL,
    TIME_PER_TOUCH REAL,
    DRIBBLES_PER_TOUCH REAL,
    PTS_PER_TOUCH REAL,
    DRIVE_PCT REAL,
    CATCH_SHOOT_PCT REAL,
    PULL_UP_PCT REAL,
    PAINT_TOUCH_PCT REAL,
    PRIMARY KEY (PLAYER_ID, SEASON_LABEL)
);
"""

# Create Lineups Table
create_lineups_table = """
CREATE TABLE lineups (
    GROUP_ID TEXT,
    GROUP_NAME TEXT,
    TEAM_ABBREVIATION TEXT,
    GP INTEGER,
    W INTEGER,
    L INTEGER,
    MIN REAL,
    PLUS_MINUS REAL,
    SEASON_LABEL TEXT
);
"""

# Create Archetypes Table (The Meta Data)
create_archetypes_table = """
CREATE TABLE archetypes (
    PLAYER_ID INTEGER,
    PLAYER_NAME TEXT,
    SEASON_LABEL TEXT,
    ARCHETYPE_ID INTEGER,
    ARCHETYPE_CONFIDENCE REAL,
    PRIMARY KEY (PLAYER_ID, SEASON_LABEL)
);
"""

print("Creating Table Schemas...")
cursor.execute(create_players_table)
cursor.execute(create_lineups_table)
cursor.execute(create_archetypes_table)

# 3. Ingest Data (ETL)
print("Ingesting CSVs into SQL...")

# Load CSVs
df_players = pd.read_csv(TRACKING_CSV)
df_lineups = pd.read_csv(LINEUPS_CSV)
df_archetypes = pd.read_csv(ARCHETYPES_CSV)

# Write to SQL
# if_exists='append' because we already created the table structure above
df_players.to_sql('players', conn, if_exists='append', index=False)
df_lineups.to_sql('lineups', conn, if_exists='append', index=False)
df_archetypes.to_sql('archetypes', conn, if_exists='append', index=False)

# 4. Verification Query
print("--- VERIFICATION CHECK ---")
test_query = """
SELECT PLAYER_NAME, OFF_SPEED 
FROM players 
WHERE SEASON_LABEL = '2024-25' 
ORDER BY OFF_SPEED DESC 
LIMIT 3;
"""
results = pd.read_sql(test_query, conn)
print("Top 3 Fastest Offensive Players (2025):")
print(results)

conn.close()
print(f"\nDatabase built successfully: {DB_NAME}")