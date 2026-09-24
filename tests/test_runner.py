import json
from pathlib import Path

from pytest import MonkeyPatch

from system_one_bench.config import BenchmarkConfig
from system_one_bench.domain import ChoiceExample, Prediction, PredictionRecord
from system_one_bench.persistence import RunWriter
from system_one_bench.runner import run_benchmark


class _FlakyAdapter:
    model_name = "flaky-model"

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def classify(self, example: ChoiceExample) -> Prediction:
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("generation timed out")
        return Prediction(example.expected_choice, None, latency_ms=10)

    def close(self) -> None:
        self.closed = True


class _RecordingAdapter:
    model_name = "recording-model"

    def __init__(self) -> None:
        self.called_identifiers: list[str] = []
        self.closed = False

    def classify(self, example: ChoiceExample) -> Prediction:
        self.called_identifiers.append(example.identifier)
        return Prediction(example.expected_choice, None, latency_ms=10)

    def close(self) -> None:
        self.closed = True


def _example(identifier: str) -> ChoiceExample:
    return ChoiceExample(
        identifier=identifier,
        task="logical_deduction_three_objects",
        text="problem",
        choices={"A": "first", "B": "second"},
        expected_choice="A",
        instruction="Choose one.",
    )


def test_runner_records_failed_sample_and_continues(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _FlakyAdapter()
    config = BenchmarkConfig.model_validate(
        {
            "dataset": {
                "kind": "bbh_logical_deduction",
                "name": "test-dataset",
                "tasks": ["logical_deduction_three_objects"],
            },
            "model": {"kind": "lmstudio_qwen", "name": "test-model"},
            "run": {
                "output_dir": tmp_path,
                "dry_run": False,
                "persist_raw_responses": True,
                "continue_on_error": True,
            },
        }
    )
    examples = [_example("first"), _example("second")]

    monkeypatch.setattr(
        "system_one_bench.runner.load_benchmark_dataset",
        lambda dataset, seed: (examples, ["A", "B"]),
    )
    monkeypatch.setattr("system_one_bench.runner.create_adapter", lambda benchmark: adapter)

    directory = run_benchmark(config)

    records = [
        json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()
    ]
    summary = json.loads((directory / "summary.json").read_text())
    assert len(records) == 2
    assert records[0]["prediction"]["error"].startswith("TimeoutError:")
    assert records[1]["prediction"]["predicted_choice"] == "A"
    assert summary["sample_count"] == 2
    assert summary["metrics"]["error_rate"] == 0.5
    assert adapter.closed is True


def test_runner_resumes_only_missing_examples(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    config = BenchmarkConfig.model_validate(
        {
            "dataset": {
                "kind": "bbh_logical_deduction",
                "name": "test-dataset",
                "tasks": ["logical_deduction_three_objects"],
            },
            "model": {"kind": "lmstudio_qwen", "name": "test-model"},
            "run": {"output_dir": tmp_path, "dry_run": False, "persist_raw_responses": True},
        }
    )
    examples = [_example("first"), _example("second")]
    directory = tmp_path / "interrupted-run"
    writer = RunWriter(directory, persist_raw_responses=True)
    writer.write_metadata(
        {
            "run_id": "interrupted-run",
            "started_at": "2026-01-01T00:00:00+00:00",
            "config": config.model_dump(mode="json"),
            "choice_key_order": ["A", "B"],
        }
    )
    writer.append_record(
        PredictionRecord(examples[0], Prediction("A", None, latency_ms=10), "recording-model")
    )
    adapter = _RecordingAdapter()
    monkeypatch.setattr(
        "system_one_bench.runner.load_benchmark_dataset",
        lambda dataset, seed: (examples, ["A", "B"]),
    )
    monkeypatch.setattr("system_one_bench.runner.create_adapter", lambda benchmark: adapter)

    output = run_benchmark(config, resume_directory=directory)

    records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    summary = json.loads((output / "summary.json").read_text())
    assert output == directory
    assert adapter.called_identifiers == ["second"]
    assert adapter.closed is True
    assert [record["example"]["identifier"] for record in records] == ["first", "second"]
    assert summary["sample_count"] == 2
