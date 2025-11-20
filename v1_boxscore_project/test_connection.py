from nba_api.stats.static import players
import pandas as pd

#1 Test Basic Python
print("--- SYSTEM CHECK INITIATED ---")

#2 Test NBA API Connection
print("Querying NBA API for player data...")

try:
    # Fetch all players
    nba_players = players.get_players()

    # Search for Lebron James to verify data integrity
    lebron = [p for p in nba_players if p['full_name'] == 'LeBron James'][0]

    print(f"SUCCESS: Found player data.")
    print(f"Player Name: {lebron['full_name']}")
    print(f"Player ID: {lebron['id']}")
    print("--- SYSTEM CHECK COMPLETE ---")

except Exception as e:
    print("f:FAILURE: {e}") 