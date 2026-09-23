from system_one_bench.domain import ChoiceExample, Prediction, PredictionRecord
from system_one_bench.metrics import compute_metrics, estimate_cost_usd


def _record(
    expected: str,
    predicted: str,
    probability: dict[str, float] | None,
    task: str = "three_objects",
    error: str | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        example=ChoiceExample(
            "id",
            task,
            "problem",
            {"A": "first", "B": "second"},
            expected,
            "Choose one.",
        ),
        prediction=Prediction(
            predicted,
            probability,
            latency_ms=10,
            input_tokens=5,
            output_tokens=2,
            error=error,
        ),
        model_name="test-model",
    )


def test_metrics_include_global_and_per_task_accuracy() -> None:
    records = [
        _record("A", "A", {"A": 0.9, "B": 0.1}, "three_objects"),
        _record("B", "A", {"A": 0.6, "B": 0.4}, "five_objects"),
    ]

    metrics = compute_metrics(records, ["A", "B"])

    assert metrics["accuracy"] == 0.5
    assert metrics["accuracy_three_objects"] == 1.0
    assert metrics["accuracy_five_objects"] == 0.0
    assert metrics["brier_score"] is not None
    assert metrics["ece"] is not None


def test_metrics_hide_calibration_without_probabilities_and_count_invalids() -> None:
    metrics = compute_metrics([_record("A", "__invalid__:almost", None)], ["A", "B"])

    assert metrics["invalid_output_rate"] == 1.0
    assert metrics["error_rate"] == 0.0
    assert metrics["brier_score"] is None
    assert metrics["ece"] is None


def test_metrics_count_recorded_execution_errors() -> None:
    metrics = compute_metrics(
        [_record("A", "__error__:TimeoutError", None, error="TimeoutError: timed out")],
        ["A", "B"],
    )

    assert metrics["accuracy"] == 0.0
    assert metrics["invalid_output_rate"] == 1.0
    assert metrics["error_rate"] == 1.0


def test_cost_requires_complete_usage_and_applies_rates() -> None:
    assert estimate_cost_usd([_record("A", "A", None)], 1.0, 2.0) == 9 / 1_000_000
