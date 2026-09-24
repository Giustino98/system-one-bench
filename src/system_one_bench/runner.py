"""Reproducible benchmark orchestration; no live call is made in dry-run mode."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from system_one_bench.adapters import MlxQwenAdapter, OpenRouterJevAdapter
from system_one_bench.adapters.base import ModelAdapter
from system_one_bench.config import BenchmarkConfig, JevConfig, QwenConfig
from system_one_bench.dataset import load_bbh_dataset
from system_one_bench.domain import PredictionRecord, RunSummary
from system_one_bench.metrics import compute_metrics, estimate_cost_usd
from system_one_bench.persistence import RunWriter

logger = logging.getLogger(__name__)


def dry_run_plan(config: BenchmarkConfig) -> dict[str, object]:
    """Return a safe, no-network plan suitable for review before spending money."""
    return {
        "mode": "dry-run",
        "dataset": f"{config.dataset.name}:{config.dataset.split}",
        "tasks": list(config.dataset.tasks),
        "limit_per_task": config.dataset.limit,
        "model": {"kind": config.model.kind, "name": config.model.name},
        "will_call_model": False,
        "output_dir": str(config.run.output_dir),
    }


def run_benchmark(config: BenchmarkConfig, *, resume_directory: Path | None = None) -> Path:
    """Run one configured model and persist independently auditable artifacts."""
    if config.run.dry_run:
        raise RuntimeError(
            "Refusing to run because run.dry_run is true. Set it to false explicitly."
        )
    examples, choices = load_bbh_dataset(config.dataset, seed=config.run.seed)
    examples_by_identifier = {example.identifier: example for example in examples}
    if len(examples_by_identifier) != len(examples):
        raise ValueError("Dataset contains duplicate example identifiers; cannot support resume.")
    if resume_directory is None:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{config.model.kind}"
        writer = RunWriter(config.run.output_dir / run_id)
        writer.write_metadata(
            {
                "run_id": run_id,
                "started_at": datetime.now(UTC).isoformat(),
                "config": config.model_dump(mode="json"),
                "choice_key_order": choices,
            }
        )
        records: list[PredictionRecord] = []
    else:
        writer = RunWriter.resume(resume_directory)
        metadata = writer.read_metadata()
        _validate_resume_metadata(metadata, config, choices)
        run_id = _required_string(metadata, "run_id")
        records = writer.read_records(examples_by_identifier)
        logger.info(
            "Resuming %s: %s/%s items already persisted",
            run_id,
            len(records),
            len(examples),
        )

    completed_identifiers = {record.example.identifier for record in records}
    pending_examples = [
        example for example in examples if example.identifier not in completed_identifiers
    ]
    adapter: ModelAdapter | None = None
    try:
        if pending_examples:
            adapter = create_adapter(config)
        for position, example in enumerate(pending_examples, start=len(records) + 1):
            logger.info("Classifying item %s/%s", position, len(examples))
            assert adapter is not None
            prediction = adapter.predict(example)
            record = PredictionRecord(
                example=example, prediction=prediction, model_name=adapter.model_name
            )
            writer.append_record(record)
            records.append(record)
    finally:
        if adapter is not None:
            adapter.close()
    metrics = compute_metrics(records, choices)
    cost = estimate_cost_usd(
        records,
        config.model.pricing.input_per_million_usd,
        config.model.pricing.output_per_million_usd,
    )
    writer.write_summary(
        RunSummary(
            run_id=run_id,
            model_name=_run_model_name(records, config),
            dataset_name=config.dataset.name,
            split=config.dataset.split,
            sample_count=len(records),
            metrics=metrics,
            estimated_cost_usd=cost,
        )
    )
    return writer.directory


def _validate_resume_metadata(
    metadata: dict[str, object], config: BenchmarkConfig, choices: list[str]
) -> None:
    """Reject a resume that could silently mix different experimental conditions."""
    if metadata.get("config") != config.model_dump(mode="json"):
        raise ValueError(
            "Resume configuration differs from metadata.json; start a new run instead."
        )
    if metadata.get("choice_key_order") != choices:
        raise ValueError(
            "Resume choice key order differs from metadata.json; start a new run instead."
        )
    _required_string(metadata, "run_id")


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise TypeError(f"Resume metadata is missing a valid {key!r}.")
    return value


def _run_model_name(records: list[PredictionRecord], config: BenchmarkConfig) -> str:
    """Use persisted identity on a resume, or the adapter identity for a fresh run."""
    if records:
        return records[0].model_name
    return config.model.name


def create_adapter(config: BenchmarkConfig) -> ModelAdapter:
    """Construct one of the two supported model adapters."""
    match config.model:
        case JevConfig():
            return OpenRouterJevAdapter(config.model)
        case QwenConfig():
            return MlxQwenAdapter(config.model)
