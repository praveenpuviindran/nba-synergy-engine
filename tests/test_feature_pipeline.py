"""Tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from v3_neural_synergy.synergy_utils import (
    TRACKING_FEATURES,
    build_player_feature_table,
    featurize_core_candidate,
)


def test_feature_table_expected_columns(sample_possession_df):
    """build_player_feature_table must produce all tracking feature columns."""
    result, _, _, feature_cols = build_player_feature_table(sample_possession_df)
    for col in TRACKING_FEATURES:
        assert col in result.columns, f"Missing column: {col}"
    assert len(feature_cols) > 0, "feature_cols must be non-empty"


def test_feature_table_no_nulls(sample_possession_df):
    """After pipeline, no NaN values should remain in feature columns."""
    result, _, _, feature_cols = build_player_feature_table(sample_possession_df)
    null_counts = result[feature_cols].isnull().sum()
    assert null_counts.sum() == 0, f"NaN values found:\n{null_counts[null_counts > 0]}"


def test_feature_table_row_count_preserved(sample_possession_df):
    """Row count must be identical before and after the pipeline."""
    result, _, _, _ = build_player_feature_table(sample_possession_df)
    assert len(result) == len(sample_possession_df)


def test_featurize_permutation_invariance():
    """featurize_core_candidate output must be identical regardless of core player order."""
    rng = np.random.default_rng(0)
    feature_dim = 21
    core = rng.standard_normal((4, feature_dim)).astype(np.float32)
    candidate = rng.standard_normal(feature_dim).astype(np.float32)

    feat_original = featurize_core_candidate(core, candidate)

    shuffled_idx = [2, 0, 3, 1]
    feat_shuffled = featurize_core_candidate(core[shuffled_idx], candidate)

    np.testing.assert_allclose(
        feat_original,
        feat_shuffled,
        atol=1e-5,
        err_msg="Feature vector changed when core player order was shuffled",
    )
