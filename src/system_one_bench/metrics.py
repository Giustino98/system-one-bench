"""Metrics with explicit availability rules for optional confidence information."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from system_one_bench.domain import PredictionRecord


def compute_metrics(
    records: Sequence[PredictionRecord], labels: Sequence[str]
) -> dict[str, float | None]:
    """Compute quality, latency, and calibrated-confidence metrics when possible."""
    if not records:
        raise ValueError("Cannot compute metrics for an empty run.")
    expected = [record.example.expected_label for record in records]
    predicted = [record.prediction.predicted_label for record in records]
    latencies = np.asarray([record.prediction.latency_ms for record in records], dtype=float)
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(expected, predicted, labels=list(labels), average="macro", zero_division=0)
        ),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "brier_score": None,
        "ece": None,
    }
    probabilities = [record.prediction.probabilities for record in records]
    if all(probability is not None for probability in probabilities):
        vectors = [probability for probability in probabilities if probability is not None]
        metrics["brier_score"] = _multiclass_brier(expected, vectors, labels)
        metrics["ece"] = _expected_calibration_error(expected, vectors, labels)
    return metrics


def estimate_cost_usd(
    records: Sequence[PredictionRecord], input_rate: float, output_rate: float
) -> float | None:
    """Prefer provider-billed cost, otherwise calculate from complete token usage."""
    billed_costs = [record.prediction.cost_usd for record in records]
    if all(cost is not None for cost in billed_costs):
        return sum(cost for cost in billed_costs if cost is not None)
    token_pairs = [(r.prediction.input_tokens, r.prediction.output_tokens) for r in records]
    if any(
        input_tokens is None or output_tokens is None for input_tokens, output_tokens in token_pairs
    ):
        return None
    input_total = sum(input_tokens for input_tokens, _ in token_pairs if input_tokens is not None)
    output_total = sum(
        output_tokens for _, output_tokens in token_pairs if output_tokens is not None
    )
    return (input_total * input_rate + output_total * output_rate) / 1_000_000


def _multiclass_brier(
    expected: Sequence[str], probabilities: Sequence[dict[str, float]], labels: Sequence[str]
) -> float:
    scores = []
    for truth, probability in zip(expected, probabilities, strict=True):
        scores.append(
            sum((probability.get(label, 0.0) - float(label == truth)) ** 2 for label in labels)
        )
    return float(np.mean(scores))


def _expected_calibration_error(
    expected: Sequence[str],
    probabilities: Sequence[dict[str, float]],
    labels: Sequence[str],
    bins: int = 10,
) -> float:
    confidences = np.asarray(
        [max(item.get(label, 0.0) for label in labels) for item in probabilities]
    )
    guesses = [max(labels, key=lambda label: item.get(label, 0.0)) for item in probabilities]
    correct = np.asarray(
        [guess == truth for guess, truth in zip(guesses, expected, strict=True)], dtype=float
    )
    ece = 0.0
    for lower, upper in zip(
        np.linspace(0, 1, bins, endpoint=False), np.linspace(1 / bins, 1, bins), strict=True
    ):
        mask = (confidences >= lower) & (confidences < upper if upper < 1 else confidences <= upper)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidences[mask].mean()))
    return ece
