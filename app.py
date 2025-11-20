import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="NBA Meta Evolution", layout="wide")
st.title("Moneyball 2.0: The Neural Synergy Engine")
st.markdown("### From Archetypes (V2) to Generative Rosters (V3)")

# --- MODEL DEFINITION ---
class NBADeepSet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
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
        
        # Updated Pooling Logic
        team_sum = torch.sum(player_embeddings, dim=1)
        team_std = torch.std(player_embeddings, dim=1)
        team_vector = torch.cat([team_sum, team_std], dim=1)
        
        net_rating = self.decoder(team_vector)
        return net_rating

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
        # Load with map_location to ensure it works on Cloud CPUs
        model.load_state_dict(torch.load('v3_neural_synergy/synergy_model.pth', map_location=torch.device('cpu')))
    except:
        st.error("Model weights not found.")
    model.eval()
    return model

df, tracking_df = load_data()
model = load_model()

# --- PREPARE VECTORS ---
feature_cols = ['OFF_SPEED', 'DEF_SPEED', 'TIME_PER_TOUCH', 'DRIBBLES_PER_TOUCH', 
                'PTS_PER_TOUCH', 'DRIVE_PCT', 'CATCH_SHOOT_PCT', 'PULL_UP_PCT', 'PAINT_TOUCH_PCT']

scaler = StandardScaler()
tracking_df[feature_cols] = scaler.fit_transform(tracking_df[feature_cols])

name_to_vec = {}
for _, row in tracking_df[tracking_df['SEASON_LABEL'] == '2024-25'].iterrows():
    name_to_vec[row['PLAYER_NAME']] = row[feature_cols].values.astype('float32')

# --- SIDEBAR ---
st.sidebar.header("Analysis Controls")
view_mode = st.sidebar.radio("Choose Module:", ["Player Evolution (V1)", "Market Trends (V2)", "Generative GM (V3)"])

# --- VIEW 1: PLAYER EVOLUTION ---
if view_mode == "Player Evolution (V1)":
    st.subheader("Player Career Arc")
    archetype_map = {
        6: "Heliocentric Stars", 0: "Movement Snipers", 7: "Scoring Wings",
        5: "Versatile Bigs", 8: "Dominant Bigs", 3: "Defensive Wings",
        2: "Rim Runners", 4: "Raw Athletic Bigs", 9: "Veteran Guards",
        1: "Traditional Facilitators"
    }
    df['Archetype Label'] = df['ARCHETYPE_ID'].map(archetype_map)
    all_players = sorted(df['PLAYER_NAME'].unique())
    selected_player = st.selectbox("Select a Player:", all_players, index=all_players.index("LeBron James") if "LeBron James" in all_players else 0)
    
    player_data = df[df['PLAYER_NAME'] == selected_player].sort_values('SEASON_LABEL')
    all_seasons_sorted = sorted(df['SEASON_LABEL'].unique())

    fig = px.scatter(player_data, x='SEASON_LABEL', y='ARCHETYPE_ID', color='Archetype Label', size='ARCHETYPE_CONFIDENCE', title=f"The Evolution of {selected_player}", height=500)
    fig.update_xaxes(categoryorder='array', categoryarray=all_seasons_sorted)
    fig.update_yaxes(tickvals=list(archetype_map.keys()), ticktext=list(archetype_map.values()))
    st.plotly_chart(fig, use_container_width=True)

# --- VIEW 2: MARKET TRENDS ---
elif view_mode == "Market Trends (V2)":
    st.subheader("The Stock Market of Playstyles")
    df['IMPACT_SCORE'] = (df['PTS_PER_TOUCH'] * 100) + (df['OFF_SPEED'] * 10)
    archetype_map = {
        6: "Heliocentric Stars", 0: "Movement Snipers", 7: "Scoring Wings",
        5: "Versatile Bigs", 8: "Dominant Bigs", 3: "Defensive Wings",
        2: "Rim Runners", 4: "Raw Athletic Bigs", 9: "Veteran Guards",
        1: "Traditional Facilitators"
    }
    df['Archetype Label'] = df['ARCHETYPE_ID'].map(archetype_map)
    trends = df.groupby(['SEASON_LABEL', 'Archetype Label'])['IMPACT_SCORE'].mean().reset_index()
    fig = px.line(trends, x='SEASON_LABEL', y='IMPACT_SCORE', color='Archetype Label', markers=True, height=600)
    st.plotly_chart(fig, use_container_width=True)

# --- VIEW 3: GENERATIVE GM (OPTIMIZED) ---
elif view_mode == "Generative GM (V3)":
    st.subheader("AI Roster Construction")
    st.markdown("Select 4 players. The Neural Network will scan the entire NBA in **milliseconds** to find the perfect 5th.")
    
    available_players = sorted(list(name_to_vec.keys()))
    core = st.multiselect("Select 4 Starters:", available_players, default=["Shai Gilgeous-Alexander", "Jalen Williams", "Chet Holmgren", "Luguentz Dort"], max_selections=4)
    
    if len(core) < 4:
        st.warning("Please select exactly 4 players.")
    else:
        if st.button("Run Neural Synergy Engine"):
            core_vectors = [name_to_vec[p] for p in core]
            candidates = [p for p in available_players if p not in core]
            
            # --- OPTIMIZATION: BATCH INFERENCE ---
            # Instead of a loop, we create ONE massive tensor containing all lineups
            batch_input = []
            for cand in candidates:
                lineup = core_vectors + [name_to_vec[cand]]
                batch_input.append(lineup)
            
            # Convert to Tensor (Batch_Size, 5, 9)
            tensor_input = torch.FloatTensor(np.array(batch_input))
            
            # Run Model ONCE
            with torch.no_grad():
                ratings = model(tensor_input).flatten().numpy()
            
            # Combine names with results
            results_df = pd.DataFrame({
                "Player": candidates,
                "Predicted Net Rating": ratings
            }).sort_values(by="Predicted Net Rating", ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.success("Top 5 Fits")
                st.dataframe(results_df.head(5).style.format({"Predicted Net Rating": "{:.2f}"}))
            with col2:
                st.error("Bottom 5 Fits")
                st.dataframe(results_df.tail(5).style.format({"Predicted Net Rating": "{:.2f}"}))