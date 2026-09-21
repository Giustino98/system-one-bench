"""Metrics with explicit availability rules for optional confidence information."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from system_one_bench.domain import PredictionRecord


def compute_metrics(
    records: Sequence[PredictionRecord], choices: Sequence[str]
) -> dict[str, float | None]:
    """Compute global/task quality, latency, and calibration when possible."""
    if not records:
        raise ValueError("Cannot compute metrics for an empty run.")
    expected = [record.example.expected_choice for record in records]
    predicted = [record.prediction.predicted_choice for record in records]
    latencies = np.asarray([record.prediction.latency_ms for record in records], dtype=float)
    invalid_count = sum(
        record.prediction.predicted_choice not in record.example.choices for record in records
    )
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(expected, predicted, labels=list(choices), average="macro", zero_division=0)
        ),
        "invalid_output_rate": invalid_count / len(records),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "brier_score": None,
        "ece": None,
    }
    by_task: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        by_task[record.example.task].append(record)
    for task, task_records in sorted(by_task.items()):
        task_expected = [record.example.expected_choice for record in task_records]
        task_predicted = [record.prediction.predicted_choice for record in task_records]
        metrics[f"accuracy_{task}"] = float(accuracy_score(task_expected, task_predicted))

    probabilities = [record.prediction.probabilities for record in records]
    if all(probability is not None for probability in probabilities):
        vectors = [probability for probability in probabilities if probability is not None]
        metrics["brier_score"] = _multiclass_brier(records, vectors)
        metrics["ece"] = _expected_calibration_error(records, vectors)
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
    records: Sequence[PredictionRecord], probabilities: Sequence[dict[str, float]]
) -> float:
    scores = []
    for record, probability in zip(records, probabilities, strict=True):
        scores.append(
            sum(
                (probability.get(choice, 0.0) - float(choice == record.example.expected_choice))
                ** 2
                for choice in record.example.choice_keys
            )
        )
    return float(np.mean(scores))


def _expected_calibration_error(
    records: Sequence[PredictionRecord],
    probabilities: Sequence[dict[str, float]],
    bins: int = 10,
) -> float:
    confidences = np.asarray(
        [
            max(item.get(choice, 0.0) for choice in record.example.choice_keys)
            for record, item in zip(records, probabilities, strict=True)
        ]
    )
    guesses = [
        max(record.example.choice_keys, key=lambda choice: item.get(choice, 0.0))
        for record, item in zip(records, probabilities, strict=True)
    ]
    correct = np.asarray(
        [
            guess == record.example.expected_choice
            for guess, record in zip(guesses, records, strict=True)
        ],
        dtype=float,
    )
    ece = 0.0
    for lower, upper in zip(
        np.linspace(0, 1, bins, endpoint=False), np.linspace(1 / bins, 1, bins), strict=True
    ):
        mask = (confidences >= lower) & (confidences < upper if upper < 1 else confidences <= upper)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidences[mask].mean()))
    return ece
