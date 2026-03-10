"""Shared pytest fixtures for the NBA Synergy Engine test suite."""

import os
import numpy as np
import pandas as pd
import pytest
import torch

from v3_neural_synergy.synergy_utils import (
    ContextAwareSynergyNet,
    TRACKING_FEATURES,
)

# Canonical input dimensionality for the trained model
_DEFAULT_INPUT_DIM = 175

# Seed for reproducible synthetic data
_RNG = np.random.default_rng(42)


@pytest.fixture(scope="session")
def sample_possession_df() -> pd.DataFrame:
    """50-row synthetic DataFrame matching the gmm_archetypes.csv schema."""
    n = 50
    player_ids = list(range(1000, 1050))
    archetype_ids = _RNG.integers(0, 10, size=n)

    tracking_data = {col: _RNG.uniform(0.0, 1.0, size=n) for col in TRACKING_FEATURES}

    arch_cols = {f"ARCH_{i}": (archetype_ids == i).astype(float) for i in range(10)}

    df = pd.DataFrame(
        {
            "PLAYER_ID": player_ids,
            "PLAYER_NAME": [f"Player_{i}" for i in player_ids],
            "SEASON_LABEL": _RNG.choice(["2022-23", "2023-24", "2024-25"], size=n),
            "ARCHETYPE_ID": archetype_ids,
            "ARCHETYPE_CONFIDENCE": _RNG.uniform(0.5, 1.0, size=n),
            "QUALITY_PRIOR_Z": _RNG.standard_normal(size=n),
            **tracking_data,
            **arch_cols,
        }
    )
    return df


@pytest.fixture(scope="session")
def loaded_model() -> ContextAwareSynergyNet:
    """Load model from checkpoint if available; fall back to default weights."""
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "v3_neural_synergy", "synergy_model.pth"
    )
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        input_dim = int(checkpoint.get("input_dim", _DEFAULT_INPUT_DIM))
        net = ContextAwareSynergyNet(input_dim=input_dim)
        net.load_state_dict(checkpoint["state_dict"])
    else:
        net = ContextAwareSynergyNet(input_dim=_DEFAULT_INPUT_DIM)
    net.eval()
    return net


@pytest.fixture(scope="session")
def sample_lineup(sample_possession_df: pd.DataFrame) -> list[int]:
    """5 player IDs drawn from the synthetic possession DataFrame."""
    return list(sample_possession_df["PLAYER_ID"].head(5))
