from system_one_bench.domain import ClassificationExample, Prediction, PredictionRecord
from system_one_bench.metrics import compute_metrics, estimate_cost_usd


def _record(
    expected: str, predicted: str, probability: dict[str, float] | None
) -> PredictionRecord:
    return PredictionRecord(
        example=ClassificationExample("id", "message", expected),
        prediction=Prediction(
            predicted, probability, latency_ms=10, input_tokens=5, output_tokens=2
        ),
        model_name="test-model",
    )


def test_metrics_include_calibration_when_every_prediction_has_probabilities() -> None:
    records = [_record("a", "a", {"a": 0.9, "b": 0.1}), _record("b", "a", {"a": 0.6, "b": 0.4})]

    metrics = compute_metrics(records, ["a", "b"])

    assert metrics["accuracy"] == 0.5
    assert metrics["brier_score"] is not None
    assert metrics["ece"] is not None


def test_metrics_hide_calibration_for_models_without_probabilities() -> None:
    metrics = compute_metrics([_record("a", "a", None)], ["a", "b"])

    assert metrics["brier_score"] is None
    assert metrics["ece"] is None


def test_metrics_count_invalid_outputs() -> None:
    metrics = compute_metrics([_record("a", "__invalid__:almost_a", None)], ["a", "b"])

    assert metrics["invalid_output_rate"] == 1.0


def test_cost_requires_complete_usage_and_applies_rates() -> None:
    record = _record("a", "a", None)

    assert estimate_cost_usd([record], input_rate=1.0, output_rate=2.0) == 9 / 1_000_000
