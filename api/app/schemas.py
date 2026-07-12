"""Modelos Pydantic del contrato de la API (§6)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Event(BaseModel):
    """Un evento de navegación. 'transaction' no es una entrada válida (§4.3)."""

    type: Literal["view", "addtocart"]
    item_category: str = "<oov>"
    seconds_since_prev: float = Field(ge=0.0)


class ScoreRequest(BaseModel):
    events: list[Event] = Field(min_length=1)
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)


class ScoreResponse(BaseModel):
    conversion_probability: float
    label: str
    recommended_action: str
    segment: str
    model_breakdown: dict[str, float]
    threshold: float


class BatchRow(BaseModel):
    lead_id: str
    probability: float
    label: str
    segment: str
    recommended_action: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str
