"""Pydantic request/response schemas for the NBA Synergy Engine REST API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str


class LineupOptimizeRequest(BaseModel):
    core_players: list[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Exactly 4 player names forming the starting core.",
        examples=[["Shai Gilgeous-Alexander", "Jalen Williams", "Chet Holmgren", "Luguentz Dort"]],
    )
    top_n: int = Field(default=10, ge=1, le=50, description="Number of candidates to return.")
    filter_type: Optional[str] = Field(
        default=None,
        description="Optional archetype filter: 'Ball Handler', 'Wing/Shooter', or 'Big Man'.",
    )


class CandidateResult(BaseModel):
    player: str
    synergy_score: float
    model_confidence: float
    synergy_score_std: Optional[float] = None
    confidence_tier: Optional[str] = None
    archetype_label: Optional[str] = None


class LineupOptimizeResponse(BaseModel):
    core: list[str]
    candidates: list[CandidateResult]
    total_evaluated: int


class ArchetypeInfo(BaseModel):
    archetype_id: int
    label: str
    player_count: int


class ArchetypesResponse(BaseModel):
    season: str
    archetypes: list[ArchetypeInfo]


class LineupStatRow(BaseModel):
    group_name: str
    team_abbreviation: Optional[str]
    gp: int
    minutes: float
    plus_minus: float
    pm_per_minute: Optional[float]
    season_label: str
    pm_rank: Optional[int]


class LineupStatsResponse(BaseModel):
    rows: list[LineupStatRow]
    total: int
