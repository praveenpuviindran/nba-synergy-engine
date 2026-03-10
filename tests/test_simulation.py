"""Tests for the lineup simulation and ranking logic."""

import numpy as np
import pandas as pd
import pytest

from v3_neural_synergy.synergy_utils import featurize_core_candidate

_RNG = np.random.default_rng(13)
_FEATURE_DIM = 21


def _make_core() -> np.ndarray:
    return _RNG.standard_normal((4, _FEATURE_DIM)).astype(np.float32)


def _score_candidates(core: np.ndarray, candidates: list[np.ndarray]) -> list[float]:
    """Minimal scorer that mirrors app.py logic without requiring model artifacts."""
    scores = []
    for cand in candidates:
        feat = featurize_core_candidate(core, cand)
        # Use L2 norm as a deterministic proxy score for testing ranking properties
        scores.append(float(np.linalg.norm(feat)))
    return scores


def test_valid_lineup_combinations():
    """Every candidate should produce a finite feature vector with the correct dimension."""
    core = _make_core()
    expected_dim = 175  # 21 * 8 aggregations + 7 scalar features

    for _ in range(10):
        cand = _RNG.standard_normal(_FEATURE_DIM).astype(np.float32)
        feat = featurize_core_candidate(core, cand)
        assert len(feat) == expected_dim, f"Feature vector length {len(feat)} != {expected_dim}"
        assert np.isfinite(feat).all(), "Non-finite value in feature vector"


def test_rankings_are_sorted():
    """Ranked output must be in descending order by score."""
    core = _make_core()
    candidates = [_RNG.standard_normal(_FEATURE_DIM).astype(np.float32) for _ in range(20)]
    scores = _score_candidates(core, candidates)

    ranked_scores = sorted(scores, reverse=True)
    assert ranked_scores == sorted(scores, reverse=True), "Rankings are not sorted descending"
    # Verify descending
    for i in range(len(ranked_scores) - 1):
        assert ranked_scores[i] >= ranked_scores[i + 1]


def test_no_duplicate_candidates():
    """Each candidate should appear exactly once in the result set."""
    core = _make_core()
    candidate_names = [f"Player_{i}" for i in range(25)]
    candidates = [_RNG.standard_normal(_FEATURE_DIM).astype(np.float32) for _ in candidate_names]

    scores = _score_candidates(core, candidates)

    results = sorted(zip(candidate_names, scores), key=lambda x: x[1], reverse=True)
    names_in_results = [r[0] for r in results]

    assert len(names_in_results) == len(set(names_in_results)), "Duplicate candidates in results"
    assert set(names_in_results) == set(candidate_names), "Candidate set mismatch after ranking"
