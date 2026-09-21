"""Typed domain objects shared by adapters, metrics, and persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ClassificationExample:
    """A single labeled classification item."""

    identifier: str
    text: str
    expected_label: str


@dataclass(frozen=True, slots=True)
class Prediction:
    """One adapter decision and its observable execution metadata."""

    predicted_label: str
    probabilities: dict[str, float] | None
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    raw_response: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """Persisted, line-oriented benchmark record."""

    example: ClassificationExample
    prediction: Prediction
    model_name: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Small, JSON-serializable summary emitted when a run is complete."""

    run_id: str
    model_name: str
    dataset_name: str
    split: str
    sample_count: int
    metrics: dict[str, float | None] = field(default_factory=dict)
    estimated_cost_usd: float | None = None
