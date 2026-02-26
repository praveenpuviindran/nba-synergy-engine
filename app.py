import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from v3_neural_synergy.synergy_utils import (
    ContextAwareSynergyNet,
    attach_player_quality_column,
    build_player_feature_table,
    featurize_core_candidate,
)

# --- CONFIGURATION ---
st.set_page_config(page_title="NBA Meta Evolution", layout="wide")
st.title("NBA Synergy Engine (Moneyball 2.0)")
st.markdown("### Quantifying the shift from 'Positions' to 'Playstyles' (2014-2025)")

MODEL_PATH = 'v3_neural_synergy/synergy_model.pth'
ARTIFACT_PATH = 'v3_neural_synergy/synergy_artifacts.pkl'


@st.cache_data
def load_archetype_data():
    return pd.read_csv('data/gmm_archetypes.csv')


@st.cache_resource
def load_synergy_engine():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ARTIFACT_PATH):
        return None, None

    with open(ARTIFACT_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

    input_dim = int(checkpoint.get('input_dim'))
    model = ContextAwareSynergyNet(input_dim=input_dim)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model, artifacts


df = load_archetype_data()
model, artifacts = load_synergy_engine()


def infer_archetype_map(archetype_df: pd.DataFrame, season='2024-25'):
    """
    Cluster IDs from GMM are arbitrary; infer human-readable labels from centroid behavior.
    This prevents stale hardcoded ID->name mappings (e.g., guards labeled as bigs).
    """
    feature_cols = [
        'OFF_SPEED',
        'DEF_SPEED',
        'TIME_PER_TOUCH',
        'DRIBBLES_PER_TOUCH',
        'PTS_PER_TOUCH',
        'DRIVE_PCT',
        'CATCH_SHOOT_PCT',
        'PULL_UP_PCT',
        'PAINT_TOUCH_PCT',
    ]

    source = archetype_df[archetype_df['SEASON_LABEL'] == season].copy()
    if source.empty:
        source = archetype_df.copy()

    centroids = source.groupby('ARCHETYPE_ID')[feature_cols].mean()
    raw_map = {}

    for archetype_id, row in centroids.iterrows():
        if row['PAINT_TOUCH_PCT'] >= 0.75:
            label = 'Dominant Bigs'
        elif row['PAINT_TOUCH_PCT'] >= 0.52:
            label = 'Interior Bigs'
        elif row['TIME_PER_TOUCH'] >= 4.2 and row['PULL_UP_PCT'] >= 0.30:
            label = 'Heliocentric Creators'
        elif row['TIME_PER_TOUCH'] >= 3.8 and row['DRIBBLES_PER_TOUCH'] >= 3.3:
            label = 'Lead Guards'
        elif row['CATCH_SHOOT_PCT'] >= 0.52 and row['DRIBBLES_PER_TOUCH'] <= 1.6:
            label = 'Movement Shooters'
        elif row['DRIVE_PCT'] >= 0.28 and row['DRIBBLES_PER_TOUCH'] >= 1.8:
            label = 'Slashing Guards/Wings'
        elif row['CATCH_SHOOT_PCT'] >= 0.42 and row['DEF_SPEED'] >= 4.05:
            label = '3-and-D Wings'
        elif row['PAINT_TOUCH_PCT'] >= 0.28:
            label = 'Rim Pressure Bigs'
        else:
            label = 'Two-Way Forwards'

        raw_map[int(archetype_id)] = label

    # Keep labels unique for chart legends if multiple clusters share the same style bucket.
    label_counts = {}
    archetype_map = {}
    for archetype_id, label in sorted(raw_map.items()):
        label_counts[label] = label_counts.get(label, 0) + 1
        count = label_counts[label]
        archetype_map[archetype_id] = label if count == 1 else f'{label} ({count})'

    return archetype_map


archetype_map = infer_archetype_map(df, season='2024-25')
df['Archetype Label'] = df['ARCHETYPE_ID'].map(archetype_map).fillna('Unlabeled Archetype')

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Analysis Controls")
view_mode = st.sidebar.radio("Choose Module:", ["Player Evolution (V1)", "Market Trends (V2)", "Generative GM (V3)"])

# ==========================================
# VIEW 1: PLAYER EVOLUTION
# ==========================================
if view_mode == "Player Evolution (V1)":
    st.subheader("Player Career Arc")

    all_players = sorted(df['PLAYER_NAME'].unique())
    selected_player = st.selectbox(
        "Select a Player:",
        all_players,
        index=all_players.index("LeBron James") if "LeBron James" in all_players else 0,
    )

    player_data = df[df['PLAYER_NAME'] == selected_player].sort_values('SEASON_LABEL')
    all_seasons_sorted = sorted(df['SEASON_LABEL'].unique())

    fig = px.scatter(
        player_data,
        x='SEASON_LABEL',
        y='ARCHETYPE_ID',
        color='Archetype Label',
        size='ARCHETYPE_CONFIDENCE',
        hover_data=['OFF_SPEED', 'PTS_PER_TOUCH'],
        title=f"The Evolution of {selected_player}",
        height=500,
    )

    fig.update_xaxes(categoryorder='array', categoryarray=all_seasons_sorted)
    fig.update_yaxes(tickvals=list(archetype_map.keys()), ticktext=list(archetype_map.values()))

    st.plotly_chart(fig, use_container_width=True)

    st.write("### Underlying Metrics")
    st.dataframe(
        player_data[
            ['SEASON_LABEL', 'Archetype Label', 'PTS_PER_TOUCH', 'OFF_SPEED', 'DRIBBLES_PER_TOUCH']
        ]
    )

# ==========================================
# VIEW 2: MARKET TRENDS
# ==========================================
elif view_mode == "Market Trends (V2)":
    st.subheader("The Stock Market of Playstyles")
    st.write("Which archetypes are becoming more valuable to winning?")

    df['IMPACT_SCORE'] = (df['PTS_PER_TOUCH'] * 100) + (df['OFF_SPEED'] * 10)
    trends = df.groupby(['SEASON_LABEL', 'Archetype Label'])['IMPACT_SCORE'].mean().reset_index()

    fig = px.line(
        trends,
        x='SEASON_LABEL',
        y='IMPACT_SCORE',
        color='Archetype Label',
        markers=True,
        title="Projected Value of NBA Playstyles (2014-2025)",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Observation: Note how 'Heliocentric Stars' and 'Movement Snipers' are trending up, while "
        "'Traditional Facilitators' are flatlining."
    )

# ==========================================
# VIEW 3: GENERATIVE GM (AI OPTIMIZER)
# ==========================================
elif view_mode == "Generative GM (V3)":
    st.subheader("AI Roster Construction")
    st.markdown(
        "Select 4 players. The engine evaluates every possible 5th player and ranks the best contextual fits."
    )

    if model is None or artifacts is None:
        st.error(
            "Model artifacts not found. Run `python3 v3_neural_synergy/02_train_deepset.py` "
            "to train and export `synergy_model.pth` + `synergy_artifacts.pkl`."
        )
        st.stop()

    quality_col = artifacts.get('quality_col', 'QUALITY_PRIOR_Z')
    enriched_df = attach_player_quality_column(df, artifacts.get('player_quality_z', {}), col_name=quality_col)

    player_table, _, _, player_feature_cols = build_player_feature_table(
        enriched_df,
        tracking_scaler=artifacts['tracking_scaler'],
        archetype_ids=artifacts['archetype_ids'],
        extra_numeric_cols=[quality_col],
    )

    current = player_table[player_table['SEASON_LABEL'] == '2024-25'].copy()
    if current.empty:
        st.error("No 2024-25 player feature rows found in `data/gmm_archetypes.csv`.")
        st.stop()

    name_to_vec = {
        row['PLAYER_NAME']: row[player_feature_cols].values.astype(np.float32)
        for _, row in current.iterrows()
    }
    name_to_quality = {row['PLAYER_NAME']: float(row[quality_col]) for _, row in current.iterrows()}

    col1, col2 = st.columns([2, 1])

    with col1:
        available_players = sorted(list(name_to_vec.keys()))
        core = st.multiselect(
            "Select 4 Starters:",
            available_players,
            default=["Shai Gilgeous-Alexander", "Jalen Williams", "Chet Holmgren", "Luguentz Dort"],
            max_selections=4,
        )

    with col2:
        st.write("**Filter Candidates:**")
        filter_type = st.radio("Looking for:", ["Any", "Ball Handler", "Wing/Shooter", "Big Man"])

    candidate_pool = [p for p in available_players if p not in core]

    current_archetypes = current.set_index('PLAYER_NAME')['ARCHETYPE_ID'].to_dict()
    current_labels = {name: archetype_map.get(arch_id, 'Unlabeled Archetype') for name, arch_id in current_archetypes.items()}

    if filter_type == "Ball Handler":
        keywords = ('Creator', 'Guard', 'Facilitator')
        candidate_pool = [p for p in candidate_pool if any(k in current_labels.get(p, '') for k in keywords)]
    elif filter_type == "Wing/Shooter":
        keywords = ('Wing', 'Shooter', 'Forward')
        candidate_pool = [p for p in candidate_pool if any(k in current_labels.get(p, '') for k in keywords)]
    elif filter_type == "Big Man":
        keywords = ('Big', 'Rim')
        candidate_pool = [p for p in candidate_pool if any(k in current_labels.get(p, '') for k in keywords)]

    if len(core) < 4:
        st.warning("Please select exactly 4 players to run the simulation.")
    else:
        if st.button(f"Run Synergy Engine ({len(candidate_pool)} Candidates)"):
            if len(candidate_pool) == 0:
                st.warning("No candidates matched the selected filter for this core.")
                st.stop()

            core_matrix = np.stack([name_to_vec[p] for p in core], axis=0)

            batch_input = []
            valid_candidates = []
            for cand in candidate_pool:
                cand_vec = name_to_vec.get(cand)
                if cand_vec is None:
                    continue
                batch_input.append(featurize_core_candidate(core_matrix, cand_vec))
                valid_candidates.append(cand)

            if not batch_input:
                st.warning("No valid candidate feature vectors were available.")
                st.stop()

            X = np.array(batch_input, dtype=np.float32)
            X_scaled = artifacts['input_scaler'].transform(X).astype(np.float32)

            with torch.no_grad():
                nn_scores = model(torch.from_numpy(X_scaled)).squeeze(1).numpy()

            knn_scores = artifacts['knn_model'].predict(X_scaled)
            alpha = float(artifacts['blend_alpha'])
            base_scores = alpha * nn_scores + (1.0 - alpha) * knn_scores
            quality_arr = np.array([name_to_quality.get(c, 0.0) for c in valid_candidates], dtype=np.float32)
            coef = np.array(artifacts.get('score_calibration_coef', [1.0, 0.0, 0.0]), dtype=np.float32)
            final_scores = (coef[0] * base_scores) + (coef[1] * quality_arr) + coef[2]

            k_for_conf = min(10, int(getattr(artifacts['knn_model'], 'n_neighbors', 10)))
            dists, _ = artifacts['knn_model'].kneighbors(X_scaled, n_neighbors=k_for_conf)
            confidence = 1.0 / (1.0 + dists.mean(axis=1))

            results_df = pd.DataFrame(
                {
                    "Player": valid_candidates,
                    "Synergy Score": final_scores,
                    "Model Confidence": confidence,
                }
            ).sort_values(by="Synergy Score", ascending=False)

            results_df['Archetype ID'] = results_df['Player'].map(current_archetypes)
            results_df['Archetype Label'] = results_df['Archetype ID'].map(archetype_map)

            st.divider()

            c1, c2 = st.columns(2)
            with c1:
                st.success(f"Best Fits ({filter_type})")
                st.dataframe(
                    results_df.head(5)[
                        ['Player', 'Synergy Score', 'Model Confidence', 'Archetype Label']
                    ].style.format(
                        {
                            "Synergy Score": "{:.2f}",
                            "Model Confidence": "{:.3f}",
                        }
                    )
                )

            with c2:
                st.error("Chemistry Killers")
                st.dataframe(
                    results_df.tail(5)[
                        ['Player', 'Synergy Score', 'Model Confidence', 'Archetype Label']
                    ].style.format(
                        {
                            "Synergy Score": "{:.2f}",
                            "Model Confidence": "{:.3f}",
                        }
                    )
                )

            st.info(
                "Synergy Score is a context-conditioned marginal fit metric (higher is better). "
                "It blends a neural interaction model with KNN similarity correction so results change materially "
                "with the selected 4-player core."
            )
