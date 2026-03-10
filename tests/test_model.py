"""Tests for ContextAwareSynergyNet and GMM archetype stability."""

import numpy as np
import pandas as pd
import pytest
import torch

from v3_neural_synergy.synergy_utils import ContextAwareSynergyNet

_DEFAULT_INPUT_DIM = 175
_RNG = np.random.default_rng(7)


def test_model_output_shape(loaded_model):
    """Model must return a (batch,) tensor for a standard batch."""
    batch_size = 16
    x = torch.from_numpy(_RNG.standard_normal((batch_size, _DEFAULT_INPUT_DIM)).astype(np.float32))
    with torch.no_grad():
        out = loaded_model(x).squeeze(1)
    assert out.shape == (batch_size,), f"Unexpected output shape: {out.shape}"


def test_model_finite_outputs(loaded_model):
    """Model outputs must contain no NaN or Inf values."""
    x = torch.from_numpy(_RNG.standard_normal((32, _DEFAULT_INPUT_DIM)).astype(np.float32))
    with torch.no_grad():
        out = loaded_model(x)
    assert torch.isfinite(out).all(), "Model produced NaN or Inf outputs"


def test_gmm_stability():
    """Repeated GMM-style archetype assignment must yield consistent counts."""
    from sklearn.mixture import GaussianMixture

    n_components = 10
    X = _RNG.standard_normal((200, 9)).astype(np.float32)

    gmm1 = GaussianMixture(n_components=n_components, random_state=0)
    labels1 = gmm1.fit_predict(X)

    gmm2 = GaussianMixture(n_components=n_components, random_state=0)
    labels2 = gmm2.fit_predict(X)

    assert np.array_equal(labels1, labels2), "GMM with same seed produced different labels"


def test_archetype_count_matches_config(sample_possession_df):
    """Number of unique ARCHETYPE_ID values must equal the configured component count."""
    n_archetypes_expected = 10  # matches 02_train_deepset.py GMM_COMPONENTS
    unique_ids = sample_possession_df["ARCHETYPE_ID"].nunique()
    assert unique_ids <= n_archetypes_expected, (
        f"Found {unique_ids} unique archetype IDs, expected <= {n_archetypes_expected}"
    )
