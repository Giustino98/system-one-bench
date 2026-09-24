from system_one_bench.domain import ChoiceExample, Prediction, PredictionRecord
from system_one_bench.metrics import compute_metrics, estimate_cost_usd


def record(
    expected: str,
    predicted: str | None,
    probability: dict[str, float] | None,
    task: str,
) -> PredictionRecord:
    return PredictionRecord(
        example=ChoiceExample(
            "id", task, "problem", {"A": "first", "B": "second"}, expected, "Choose one."
        ),
        prediction=Prediction(
            predicted,
            probability,
            latency_ms=10,
            input_tokens=5,
            output_tokens=2,
        ),
        model_name="test-model",
    )


def test_metrics_include_per_task_accuracy_and_calibration() -> None:
    records = [
        record("A", "A", {"A": 0.9, "B": 0.1}, "three_objects"),
        record("B", "A", {"A": 0.6, "B": 0.4}, "five_objects"),
    ]

    metrics = compute_metrics(records, ["A", "B"])

    assert metrics["accuracy"] == 0.5
    assert metrics["accuracy_three_objects"] == 1.0
    assert metrics["accuracy_five_objects"] == 0.0
    assert metrics["brier_score"] is not None
    assert metrics["ece"] is not None


def test_none_is_counted_as_an_invalid_model_output() -> None:
    metrics = compute_metrics([record("A", None, None, "three_objects")], ["A", "B"])

    assert metrics["accuracy"] == 0.0
    assert metrics["invalid_output_rate"] == 1.0
    assert metrics["brier_score"] is None


def test_cost_uses_complete_token_usage() -> None:
    assert estimate_cost_usd([record("A", "A", None, "three_objects")], 1.0, 2.0) == 9 / 1_000_000
