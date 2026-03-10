"""NBA Synergy Engine REST API.

Endpoints
---------
GET  /health                    Liveness probe
POST /lineup/optimize           Score all candidates against a 4-player core
GET  /archetypes                List GMM archetype labels for a season
GET  /sql/lineup-stats          Advanced SQL analytics on historical lineups
"""

from __future__ import annotations

import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import text

from api.database import get_engine
from api.models.schemas import (
    ArchetypeInfo,
    ArchetypesResponse,
    CandidateResult,
    HealthResponse,
    LineupOptimizeRequest,
    LineupOptimizeResponse,
    LineupStatRow,
    LineupStatsResponse,
)
from v3_neural_synergy.synergy_utils import (
    ContextAwareSynergyNet,
    attach_player_quality_column,
    build_player_feature_table,
    featurize_core_candidate,
)

_API_VERSION = "1.0.0"
_REPO_ROOT = Path(__file__).parent.parent
_MODEL_PATH = _REPO_ROOT / "v3_neural_synergy" / "synergy_model.pth"
_ARTIFACT_PATH = _REPO_ROOT / "v3_neural_synergy" / "synergy_artifacts.pkl"
_ARCHETYPE_CSV = _REPO_ROOT / "data" / "gmm_archetypes.csv"

@asynccontextmanager
async def _lifespan(application: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="NBA Synergy Engine API",
    description="Lineup chemistry scoring and analytics endpoints.",
    version=_API_VERSION,
    lifespan=_lifespan,
)

# --------------------------------------------------------------------------- #
# Startup: load heavy artifacts once                                           #
# --------------------------------------------------------------------------- #

_model: Optional[ContextAwareSynergyNet] = None
_artifacts: Optional[dict] = None
_archetype_df: Optional[pd.DataFrame] = None
_name_to_vec: dict[str, np.ndarray] = {}
_name_to_quality: dict[str, float] = {}
_current_archetypes: dict[str, int] = {}
_archetype_map: dict[int, str] = {}
_player_feature_cols: list[str] = []
_db_engine = None


def _infer_archetype_map(archetype_df: pd.DataFrame, season: str = "2024-25") -> dict[int, str]:
    feature_cols = [
        "OFF_SPEED", "DEF_SPEED", "TIME_PER_TOUCH", "DRIBBLES_PER_TOUCH",
        "PTS_PER_TOUCH", "DRIVE_PCT", "CATCH_SHOOT_PCT", "PULL_UP_PCT", "PAINT_TOUCH_PCT",
    ]
    source = archetype_df[archetype_df["SEASON_LABEL"] == season].copy()
    if source.empty:
        source = archetype_df.copy()
    centroids = source.groupby("ARCHETYPE_ID")[feature_cols].mean()
    raw_map: dict[int, str] = {}
    for archetype_id, row in centroids.iterrows():
        if row["PAINT_TOUCH_PCT"] >= 0.75:
            label = "Dominant Bigs"
        elif row["PAINT_TOUCH_PCT"] >= 0.52:
            label = "Interior Bigs"
        elif row["TIME_PER_TOUCH"] >= 4.2 and row["PULL_UP_PCT"] >= 0.30:
            label = "Heliocentric Creators"
        elif row["TIME_PER_TOUCH"] >= 3.8 and row["DRIBBLES_PER_TOUCH"] >= 3.3:
            label = "Lead Guards"
        elif row["CATCH_SHOOT_PCT"] >= 0.52 and row["DRIBBLES_PER_TOUCH"] <= 1.6:
            label = "Movement Shooters"
        elif row["DRIVE_PCT"] >= 0.28 and row["DRIBBLES_PER_TOUCH"] >= 1.8:
            label = "Slashing Guards/Wings"
        elif row["CATCH_SHOOT_PCT"] >= 0.42 and row["DEF_SPEED"] >= 4.05:
            label = "3-and-D Wings"
        elif row["PAINT_TOUCH_PCT"] >= 0.28:
            label = "Rim Pressure Bigs"
        else:
            label = "Two-Way Forwards"
        raw_map[int(archetype_id)] = label
    label_counts: dict[str, int] = {}
    result: dict[int, str] = {}
    for aid, lbl in sorted(raw_map.items()):
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        count = label_counts[lbl]
        result[aid] = lbl if count == 1 else f"{lbl} ({count})"
    return result


def _load_artifacts() -> None:
    global _model, _artifacts, _archetype_df, _name_to_vec, _name_to_quality
    global _current_archetypes, _archetype_map, _player_feature_cols, _db_engine

    _db_engine = get_engine()

    if not _ARCHETYPE_CSV.exists():
        return

    _archetype_df = pd.read_csv(_ARCHETYPE_CSV)
    _archetype_map = _infer_archetype_map(_archetype_df, season="2024-25")
    _archetype_df["Archetype Label"] = _archetype_df["ARCHETYPE_ID"].map(_archetype_map).fillna("Unlabeled")

    if not _MODEL_PATH.exists() or not _ARTIFACT_PATH.exists():
        return

    with open(_ARTIFACT_PATH, "rb") as f:
        _artifacts = pickle.load(f)

    checkpoint = torch.load(str(_MODEL_PATH), map_location="cpu", weights_only=True)
    input_dim = int(checkpoint.get("input_dim", 175))
    _model = ContextAwareSynergyNet(input_dim=input_dim)
    _model.load_state_dict(checkpoint["state_dict"])
    _model.eval()

    quality_col = _artifacts.get("quality_col", "QUALITY_PRIOR_Z")
    enriched = attach_player_quality_column(_archetype_df, _artifacts.get("player_quality_z", {}), col_name=quality_col)
    player_table, _, _, feature_cols = build_player_feature_table(
        enriched,
        tracking_scaler=_artifacts["tracking_scaler"],
        archetype_ids=_artifacts["archetype_ids"],
        extra_numeric_cols=[quality_col],
    )
    _player_feature_cols = feature_cols

    current = player_table[player_table["SEASON_LABEL"] == "2024-25"]
    for _, row in current.iterrows():
        name = row["PLAYER_NAME"]
        _name_to_vec[name] = row[feature_cols].values.astype(np.float32)
        _name_to_quality[name] = float(row[quality_col])
        _current_archetypes[name] = int(row["ARCHETYPE_ID"])


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=_API_VERSION)


@app.post("/lineup/optimize", response_model=LineupOptimizeResponse, tags=["Synergy"])
def lineup_optimize(request: LineupOptimizeRequest) -> LineupOptimizeResponse:
    if _model is None or _artifacts is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")

    missing = [p for p in request.core_players if p not in _name_to_vec]
    if missing:
        raise HTTPException(status_code=422, detail=f"Unknown players: {missing}")

    candidate_pool = [p for p in _name_to_vec if p not in request.core_players]

    if request.filter_type == "Ball Handler":
        keywords = ("Creator", "Guard", "Facilitator")
        candidate_pool = [p for p in candidate_pool if any(k in _archetype_map.get(_current_archetypes.get(p, -1), "") for k in keywords)]
    elif request.filter_type == "Wing/Shooter":
        keywords = ("Wing", "Shooter", "Forward")
        candidate_pool = [p for p in candidate_pool if any(k in _archetype_map.get(_current_archetypes.get(p, -1), "") for k in keywords)]
    elif request.filter_type == "Big Man":
        keywords = ("Big", "Rim")
        candidate_pool = [p for p in candidate_pool if any(k in _archetype_map.get(_current_archetypes.get(p, -1), "") for k in keywords)]

    if not candidate_pool:
        return LineupOptimizeResponse(core=request.core_players, candidates=[], total_evaluated=0)

    core_matrix = np.stack([_name_to_vec[p] for p in request.core_players], axis=0)
    batch_input, valid_candidates = [], []
    for cand in candidate_pool:
        vec = _name_to_vec.get(cand)
        if vec is not None:
            batch_input.append(featurize_core_candidate(core_matrix, vec))
            valid_candidates.append(cand)

    X = np.array(batch_input, dtype=np.float32)
    X_scaled = _artifacts["input_scaler"].transform(X).astype(np.float32)

    with torch.no_grad():
        nn_scores = _model(torch.from_numpy(X_scaled)).squeeze(1).numpy()

    knn_scores = _artifacts["knn_model"].predict(X_scaled)
    alpha = float(_artifacts["blend_alpha"])
    base_scores = alpha * nn_scores + (1.0 - alpha) * knn_scores
    quality_arr = np.array([_name_to_quality.get(c, 0.0) for c in valid_candidates], dtype=np.float32)
    coef = np.array(_artifacts.get("score_calibration_coef", [1.0, 0.0, 0.0]), dtype=np.float32)
    final_scores = (coef[0] * base_scores) + (coef[1] * quality_arr) + coef[2]

    k = min(10, int(getattr(_artifacts["knn_model"], "n_neighbors", 10)))
    dists, _ = _artifacts["knn_model"].kneighbors(X_scaled, n_neighbors=k)
    confidence = 1.0 / (1.0 + dists.mean(axis=1))

    ranked = sorted(
        zip(valid_candidates, final_scores.tolist(), confidence.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    candidates_out = [
        CandidateResult(
            player=p,
            synergy_score=round(s, 4),
            model_confidence=round(c, 4),
            archetype_label=_archetype_map.get(_current_archetypes.get(p, -1)),
        )
        for p, s, c in ranked[: request.top_n]
    ]
    return LineupOptimizeResponse(core=request.core_players, candidates=candidates_out, total_evaluated=len(valid_candidates))


@app.get("/archetypes", response_model=ArchetypesResponse, tags=["Analytics"])
def get_archetypes(season: str = Query(default="2024-25")) -> ArchetypesResponse:
    if _archetype_df is None:
        raise HTTPException(status_code=503, detail="Archetype data not loaded.")
    amap = _infer_archetype_map(_archetype_df, season=season)
    season_df = _archetype_df[_archetype_df["SEASON_LABEL"] == season]
    if season_df.empty:
        season_df = _archetype_df
    counts = season_df.groupby("ARCHETYPE_ID").size().to_dict()
    archetypes = [
        ArchetypeInfo(archetype_id=aid, label=lbl, player_count=int(counts.get(aid, 0)))
        for aid, lbl in sorted(amap.items())
    ]
    return ArchetypesResponse(season=season, archetypes=archetypes)


@app.get("/sql/lineup-stats", response_model=LineupStatsResponse, tags=["Analytics"])
def sql_lineup_stats(
    season: str = Query(default="2024-25"),
    min_gp: int = Query(default=5, ge=1),
    limit: int = Query(default=50, ge=1, le=500),
) -> LineupStatsResponse:
    if _db_engine is None:
        raise HTTPException(status_code=503, detail="Database not available.")

    query = text("""
        SELECT
            GROUP_NAME,
            TEAM_ABBREVIATION,
            GP,
            MIN                                                         AS minutes,
            PLUS_MINUS,
            CASE WHEN MIN > 0 THEN ROUND(PLUS_MINUS / MIN, 4) ELSE NULL END AS pm_per_minute,
            SEASON_LABEL,
            RANK() OVER (
                PARTITION BY SEASON_LABEL
                ORDER BY PLUS_MINUS DESC
            )                                                           AS pm_rank
        FROM lineups
        WHERE SEASON_LABEL = :season
          AND GP >= :min_gp
        ORDER BY PLUS_MINUS DESC
        LIMIT :limit
    """)

    try:
        with _db_engine.connect() as conn:
            rows = conn.execute(query, {"season": season, "min_gp": min_gp, "limit": limit}).fetchall()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    result_rows = [
        LineupStatRow(
            group_name=row[0],
            team_abbreviation=row[1],
            gp=int(row[2]),
            minutes=float(row[3]),
            plus_minus=float(row[4]),
            pm_per_minute=float(row[5]) if row[5] is not None else None,
            season_label=row[6],
            pm_rank=int(row[7]) if row[7] is not None else None,
        )
        for row in rows
    ]
    return LineupStatsResponse(rows=result_rows, total=len(result_rows))
