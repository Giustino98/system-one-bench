"""Typed domain objects shared by datasets, adapters, metrics, and persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ChoiceExample:
    """A single-choice benchmark item with example-specific alternatives."""

    identifier: str
    task: str
    text: str
    choices: Mapping[str, str]
    expected_choice: str
    instruction: str
    answer_prefix: str | None = None

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("A choice example must contain at least one alternative.")
        if self.expected_choice not in self.choices:
            raise ValueError("The expected choice must be present in choices.")

    @property
    def choice_keys(self) -> tuple[str, ...]:
        return tuple(self.choices)


@dataclass(frozen=True, slots=True)
class Prediction:
    """One adapter decision and its observable execution metadata."""

    predicted_choice: str
    probabilities: dict[str, float] | None
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    reasoning: str | None = None
    finish_reason: str | None = None
    error: str | None = None
    raw_response: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """Persisted, line-oriented benchmark record."""

    example: ChoiceExample
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
