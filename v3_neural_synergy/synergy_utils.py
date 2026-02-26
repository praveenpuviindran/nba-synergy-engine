import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

TRACKING_FEATURES = [
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
ARCHETYPE_COL = 'ARCHETYPE_ID'
ARCHETYPE_CONF_COL = 'ARCHETYPE_CONFIDENCE'


class ContextAwareSynergyNet(nn.Module):
    """Neural regressor over engineered core+candidate interaction features."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def parse_group_ids(group_id: str):
    return [int(pid) for pid in str(group_id).replace('-', ' ').split() if pid.strip()]


def season_to_start_year(season_label: str):
    try:
        return int(str(season_label).split('-')[0])
    except (ValueError, IndexError):
        return -1


def build_player_feature_table(
    archetypes_df: pd.DataFrame,
    tracking_scaler=None,
    archetype_ids=None,
    extra_numeric_cols=None,
):
    """
    Returns a copy of archetypes_df with standardized tracking features + archetype one-hot.
    Also returns fitted scalers/metadata if not supplied.
    """
    df = archetypes_df.copy()

    for idx, col in enumerate(TRACKING_FEATURES):
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if tracking_scaler is not None and hasattr(tracking_scaler, 'mean_'):
            fill_value = float(tracking_scaler.mean_[idx])
        else:
            fill_value = float(df[col].mean())
        if np.isnan(fill_value):
            fill_value = 0.0
        df[col] = df[col].fillna(fill_value)

    if tracking_scaler is None:
        tracking_scaler = StandardScaler()
        df[TRACKING_FEATURES] = tracking_scaler.fit_transform(df[TRACKING_FEATURES])
    else:
        df[TRACKING_FEATURES] = tracking_scaler.transform(df[TRACKING_FEATURES])

    if archetype_ids is None:
        archetype_ids = sorted(int(v) for v in df[ARCHETYPE_COL].dropna().astype(int).unique())

    archetype_cols = []
    for archetype_id in archetype_ids:
        col = f'ARCH_{archetype_id}'
        archetype_cols.append(col)
        df[col] = (df[ARCHETYPE_COL].astype(float).fillna(-1).astype(int) == int(archetype_id)).astype(float)

    extra_numeric_cols = extra_numeric_cols or []
    df[ARCHETYPE_CONF_COL] = pd.to_numeric(df[ARCHETYPE_CONF_COL], errors='coerce').fillna(0.0)
    for col in extra_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    player_feature_cols = TRACKING_FEATURES + archetype_cols + [ARCHETYPE_CONF_COL] + list(extra_numeric_cols)
    return df, tracking_scaler, archetype_ids, player_feature_cols


def attach_player_quality_column(player_df: pd.DataFrame, quality_map: dict, col_name='QUALITY_PRIOR_Z'):
    df = player_df.copy()
    df[col_name] = [
        float(quality_map.get((int(pid), season), 0.0))
        for pid, season in zip(df['PLAYER_ID'].values, df['SEASON_LABEL'].values)
    ]
    return df


def featurize_core_candidate(core_vectors: np.ndarray, candidate_vector: np.ndarray):
    """Permutation-invariant engineered context vector for (core-4, candidate-1)."""
    core_mean = core_vectors.mean(axis=0)
    core_std = core_vectors.std(axis=0)
    core_min = core_vectors.min(axis=0)
    core_max = core_vectors.max(axis=0)

    diff = candidate_vector - core_mean
    abs_diff = np.abs(diff)
    prod = candidate_vector * core_mean

    cand_norm = np.linalg.norm(candidate_vector) + 1e-8
    core_norms = np.linalg.norm(core_vectors, axis=1) + 1e-8
    cos_per_player = (core_vectors @ candidate_vector) / (core_norms * cand_norm)

    l2_per_player = np.linalg.norm(core_vectors - candidate_vector, axis=1)

    scalar_features = np.array(
        [
            float(cos_per_player.mean()),
            float(cos_per_player.min()),
            float(cos_per_player.max()),
            float(l2_per_player.mean()),
            float(l2_per_player.min()),
            float(l2_per_player.max()),
            float(core_std.mean()),
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            candidate_vector,
            core_mean,
            core_std,
            core_min,
            core_max,
            diff,
            abs_diff,
            prod,
            scalar_features,
        ]
    ).astype(np.float32)


def build_player_vector_dict(player_df: pd.DataFrame, player_feature_cols):
    vectors = {}
    for _, row in player_df.iterrows():
        key = (int(row['PLAYER_ID']), row['SEASON_LABEL'])
        vectors[key] = row[player_feature_cols].values.astype(np.float32)
    return vectors


def latest_name_to_vector(player_df: pd.DataFrame, player_feature_cols):
    """Map player name -> latest season vector available in player_df."""
    name_to_vec = {}
    name_to_year = {}

    for _, row in player_df.iterrows():
        name = row['PLAYER_NAME']
        year = season_to_start_year(row['SEASON_LABEL'])
        prev = name_to_year.get(name)
        if prev is None or year > prev:
            name_to_year[name] = year
            name_to_vec[name] = row[player_feature_cols].values.astype(np.float32)

    return name_to_vec, name_to_year
