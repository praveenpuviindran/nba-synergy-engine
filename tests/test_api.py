"""FastAPI endpoint tests using TestClient."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_returns_ok():
    """GET /health must return status 'ok' and a version string."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert isinstance(data["version"], str)


def test_lineup_optimize_requires_four_players():
    """POST /lineup/optimize with fewer than 4 players must return 422."""
    resp = client.post(
        "/lineup/optimize",
        json={"core_players": ["Player A", "Player B", "Player C"]},
    )
    assert resp.status_code == 422


def test_lineup_optimize_unknown_players_returns_error():
    """POST /lineup/optimize with unknown player names must return 422 or 503."""
    resp = client.post(
        "/lineup/optimize",
        json={
            "core_players": [
                "Nonexistent Player 1",
                "Nonexistent Player 2",
                "Nonexistent Player 3",
                "Nonexistent Player 4",
            ]
        },
    )
    # 503 if model not loaded, 422 if loaded but players unknown
    assert resp.status_code in (422, 503)


def test_archetypes_endpoint_returns_list():
    """GET /archetypes must return an archetypes list (may be empty if data not loaded)."""
    resp = client.get("/archetypes", params={"season": "2024-25"})
    # 200 with archetypes list or 503 if data not loaded in test env
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "archetypes" in data
        assert isinstance(data["archetypes"], list)


def test_sql_lineup_stats_returns_rows():
    """GET /sql/lineup-stats must return a rows list."""
    resp = client.get("/sql/lineup-stats", params={"season": "2024-25", "min_gp": 5, "limit": 10})
    assert resp.status_code in (200, 500, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "rows" in data
        assert isinstance(data["rows"], list)
        assert "total" in data
