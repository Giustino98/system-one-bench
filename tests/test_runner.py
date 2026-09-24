import json
from pathlib import Path

from pytest import MonkeyPatch

from system_one_bench.config import BenchmarkConfig
from system_one_bench.domain import ChoiceExample, Prediction, PredictionRecord
from system_one_bench.persistence import RunWriter
from system_one_bench.runner import run_benchmark


class RecordingAdapter:
    model_name = "recording-model"

    def __init__(self) -> None:
        self.called_identifiers: list[str] = []
        self.closed = False

    def predict(self, example: ChoiceExample) -> Prediction:
        self.called_identifiers.append(example.identifier)
        return Prediction(example.expected_choice, None, latency_ms=10)

    def close(self) -> None:
        self.closed = True


def example(identifier: str) -> ChoiceExample:
    return ChoiceExample(
        identifier=identifier,
        task="logical_deduction_three_objects",
        text="problem",
        choices={"A": "first", "B": "second"},
        expected_choice="A",
        instruction="Choose one.",
    )


def config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "dataset": {"tasks": ["logical_deduction_three_objects"]},
            "model": {"kind": "mlx_qwen", "name": "test-model"},
            "run": {"output_dir": tmp_path, "dry_run": False},
        }
    )


def test_runner_writes_records_and_summary(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    adapter = RecordingAdapter()
    examples = [example("first"), example("second")]
    monkeypatch.setattr(
        "system_one_bench.runner.load_bbh_dataset",
        lambda dataset, seed: (examples, ["A", "B"]),
    )
    monkeypatch.setattr("system_one_bench.runner.create_adapter", lambda benchmark: adapter)

    directory = run_benchmark(config(tmp_path))

    records = [
        json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()
    ]
    summary = json.loads((directory / "summary.json").read_text())
    assert [record["example"]["identifier"] for record in records] == ["first", "second"]
    assert summary["sample_count"] == 2
    assert adapter.closed is True


def test_runner_resumes_only_missing_examples(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    benchmark_config = config(tmp_path)
    examples = [example("first"), example("second")]
    directory = tmp_path / "interrupted-run"
    writer = RunWriter(directory)
    writer.write_metadata(
        {
            "run_id": "interrupted-run",
            "started_at": "2026-01-01T00:00:00+00:00",
            "config": benchmark_config.model_dump(mode="json"),
            "choice_key_order": ["A", "B"],
        }
    )
    writer.append_record(
        PredictionRecord(examples[0], Prediction("A", None, latency_ms=10), "recording-model")
    )
    adapter = RecordingAdapter()
    monkeypatch.setattr(
        "system_one_bench.runner.load_bbh_dataset",
        lambda dataset, seed: (examples, ["A", "B"]),
    )
    monkeypatch.setattr("system_one_bench.runner.create_adapter", lambda benchmark: adapter)

    output = run_benchmark(benchmark_config, resume_directory=directory)

    records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert adapter.called_identifiers == ["second"]
    assert [record["example"]["identifier"] for record in records] == ["first", "second"]
