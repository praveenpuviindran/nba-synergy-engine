import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="NBA Meta Evolution", layout="wide")
st.title("NBA Synergy Engine (Moneyball 2.0)")
st.markdown("### Quantifying the shift from 'Positions' to 'Playstyles' (2014-2025)")

# --- MODEL DEFINITION (Variance-Aware) ---
# This matches the architecture you just retrained
class NBADeepSet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Encoder: Embeds playstyle
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Decoder: Reads (Sum + Std_Dev) -> Net Rating
        self.decoder = nn.Sequential(
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x shape: (Batch, 5, 9)
        player_embeddings = self.encoder(x)
        
        # POOLING: Sum (Talent) + Std (Diversity)
        team_sum = torch.sum(player_embeddings, dim=1)
        team_std = torch.std(player_embeddings, dim=1)
        
        # Combine them
        team_vector = torch.cat([team_sum, team_std], dim=1)
        
        return self.decoder(team_vector)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/gmm_archetypes.csv')
    tracking = pd.read_csv('data/processed_tracking_metrics.csv')
    return df, tracking

@st.cache_resource
def load_model():
    model = NBADeepSet(input_dim=9)
    try:
        # Load weights (Handle CPU mapping for Cloud)
        model.load_state_dict(torch.load('v3_neural_synergy/synergy_model.pth', map_location=torch.device('cpu')))
    except:
        st.error("Model weights not found. Please ensure 'synergy_model.pth' is in the 'v3_neural_synergy' folder.")
    model.eval()
    return model

df, tracking_df = load_data()
model = load_model()

# --- PREPARE VECTORS FOR INFERENCE ---
feature_cols = ['OFF_SPEED', 'DEF_SPEED', 'TIME_PER_TOUCH', 'DRIBBLES_PER_TOUCH', 
                'PTS_PER_TOUCH', 'DRIVE_PCT', 'CATCH_SHOOT_PCT', 'PULL_UP_PCT', 'PAINT_TOUCH_PCT']

# Scale features exactly as trained
scaler = StandardScaler()
tracking_df[feature_cols] = scaler.fit_transform(tracking_df[feature_cols])

# Create Vector Lookup (Name -> Tensor) for 2024-25 season
name_to_vec = {}
for _, row in tracking_df[tracking_df['SEASON_LABEL'] == '2024-25'].iterrows():
    name_to_vec[row['PLAYER_NAME']] = row[feature_cols].values.astype('float32')

# --- ARCHETYPE MAPPING ---
archetype_map = {
    6: "Heliocentric Stars", 0: "Movement Snipers", 7: "Scoring Wings",
    5: "Versatile Bigs", 8: "Dominant Bigs", 3: "Defensive Wings",
    2: "Rim Runners", 4: "Raw Athletic Bigs", 9: "Veteran Guards",
    1: "Traditional Facilitators"
}
df['Archetype Label'] = df['ARCHETYPE_ID'].map(archetype_map)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Analysis Controls")
# The app intentionally exposes three layers: V1 (history), V2 (macro trends), V3 (optimization).
view_mode = st.sidebar.radio("Choose Module:", ["Player Evolution (V1)", "Market Trends (V2)", "Generative GM (V3)"])

# ==========================================
# VIEW 1: PLAYER EVOLUTION
# ==========================================
if view_mode == "Player Evolution (V1)":
    st.subheader("Player Career Arc")
    
    all_players = sorted(df['PLAYER_NAME'].unique())
    selected_player = st.selectbox("Select a Player:", all_players, index=all_players.index("LeBron James") if "LeBron James" in all_players else 0)
    
    # Filter Data & Force Sort
    player_data = df[df['PLAYER_NAME'] == selected_player].sort_values('SEASON_LABEL')
    all_seasons_sorted = sorted(df['SEASON_LABEL'].unique())

    # Plot
    fig = px.scatter(
        player_data, 
        x='SEASON_LABEL', 
        y='ARCHETYPE_ID',
        color='Archetype Label',
        size='ARCHETYPE_CONFIDENCE',
        hover_data=['OFF_SPEED', 'PTS_PER_TOUCH'],
        title=f"The Evolution of {selected_player}",
        height=500
    )
    
    # Force X-Axis Order (Fixes the timeline bug)
    fig.update_xaxes(categoryorder='array', categoryarray=all_seasons_sorted)
    fig.update_yaxes(tickvals=list(archetype_map.keys()), ticktext=list(archetype_map.values()))
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("### Underlying Metrics")
    st.dataframe(player_data[['SEASON_LABEL', 'Archetype Label', 'PTS_PER_TOUCH', 'OFF_SPEED', 'DRIBBLES_PER_TOUCH']])

# ==========================================
# VIEW 2: MARKET TRENDS
# ==========================================
elif view_mode == "Market Trends (V2)":
    st.subheader("The Stock Market of Playstyles")
    st.write("Which archetypes are becoming more valuable to winning?")
    
    # Calculate "Value" (Impact Score)
    df['IMPACT_SCORE'] = (df['PTS_PER_TOUCH'] * 100) + (df['OFF_SPEED'] * 10)
    
    # Group by Year and Archetype
    trends = df.groupby(['SEASON_LABEL', 'Archetype Label'])['IMPACT_SCORE'].mean().reset_index()
    
    # Plot Line Chart
    fig = px.line(
        trends, 
        x='SEASON_LABEL', 
        y='IMPACT_SCORE', 
        color='Archetype Label',
        markers=True,
        title="Projected Value of NBA Playstyles (2014-2025)",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("Observation: Note how 'Heliocentric Stars' and 'Movement Snipers' are trending up, while 'Traditional Facilitators' are flatlining.")

# ==========================================
# VIEW 3: GENERATIVE GM (AI OPTIMIZER)
# ==========================================
elif view_mode == "Generative GM (V3)":
    st.subheader("AI Roster Construction")
    st.markdown("Select 4 players. The Neural Network will scan the entire NBA to find the **mathematically perfect 5th player** to maximize Net Rating.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        available_players = sorted(list(name_to_vec.keys()))
        core = st.multiselect(
            "Select 4 Starters:", 
            available_players, 
            default=["Shai Gilgeous-Alexander", "Jalen Williams", "Chet Holmgren", "Luguentz Dort"], 
            max_selections=4
        )
    
    with col2:
        st.write("**Filter Candidates:**")
        filter_type = st.radio("Looking for:", ["Any", "Ball Handler", "Wing/Shooter", "Big Man"])

    # --- FILTER LOGIC ---
    candidate_pool = [p for p in available_players if p not in core]
    
    # Get 2025 Archetypes for filtering
    current_archetypes = df[df['SEASON_LABEL'] == '2024-25'].set_index('PLAYER_NAME')['ARCHETYPE_ID'].to_dict()
    
    if filter_type == "Ball Handler":
        # Clusters: 6 (Heliocentric), 1 (Traditional Facilitator), 9 (Veteran Guard)
        candidate_pool = [p for p in candidate_pool if current_archetypes.get(p) in [6, 1, 9]]
    elif filter_type == "Wing/Shooter":
        # Clusters: 0 (Sniper), 7 (Scorer), 3 (Defensive Wing)
        candidate_pool = [p for p in candidate_pool if current_archetypes.get(p) in [0, 7, 3]]
    elif filter_type == "Big Man":
        # Clusters: 8 (Dominant), 5 (Versatile), 2 (Rim Runner), 4 (Raw)
        candidate_pool = [p for p in candidate_pool if current_archetypes.get(p) in [8, 5, 2, 4]]

    # --- INFERENCE ENGINE ---
    if len(core) < 4:
        st.warning("Please select exactly 4 players to run the simulation.")
    else:
        if st.button(f"Run Synergy Engine ({len(candidate_pool)} Candidates)"):
            
            # 1. Prepare Core Vectors
            core_vectors = [name_to_vec[p] for p in core]
            
            # 2. Batch Input Creation
            batch_input = []
            valid_candidates = []
            
            for cand in candidate_pool:
                if cand in name_to_vec:
                    lineup = core_vectors + [name_to_vec[cand]]
                    batch_input.append(lineup)
                    valid_candidates.append(cand)
            
            # 3. Neural Inference (One Batch)
            tensor_input = torch.FloatTensor(np.array(batch_input))
            
            with torch.no_grad():
                ratings = model(tensor_input).flatten().numpy()
            
            # 4. Display Results
            results_df = pd.DataFrame({
                "Player": valid_candidates, 
                "Predicted Net Rating": ratings
            }).sort_values(by="Predicted Net Rating", ascending=False)
            
            # Add Archetype context
            results_df['Archetype ID'] = results_df['Player'].map(current_archetypes)
            results_df['Archetype Label'] = results_df['Archetype ID'].map(archetype_map)
            
            st.divider()
            
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"Best Fits ({filter_type})")
                st.dataframe(results_df.head(5)[['Player', 'Predicted Net Rating', 'Archetype Label']].style.format({"Predicted Net Rating": "{:.2f}"}))
                
            with c2:
                st.error("Chemistry Killers")
                st.dataframe(results_df.tail(5)[['Player', 'Predicted Net Rating', 'Archetype Label']].style.format({"Predicted Net Rating": "{:.2f}"}))
            
            st.info("Note: The Variance-Aware model penalizes lineups with low skill diversity (e.g., 5 Centers), prioritizing complementary roles.")
