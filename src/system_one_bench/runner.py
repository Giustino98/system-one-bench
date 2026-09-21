"""Reproducible benchmark orchestration; no live call is made in dry-run mode."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from system_one_bench.adapters import JevAdapter, MlxQwenAdapter, OpenRouterJevAdapter
from system_one_bench.adapters.base import ClassifierAdapter
from system_one_bench.config import BenchmarkConfig
from system_one_bench.dataset import load_classification_dataset
from system_one_bench.domain import PredictionRecord, RunSummary
from system_one_bench.metrics import compute_metrics, estimate_cost_usd
from system_one_bench.persistence import RunWriter

logger = logging.getLogger(__name__)


def dry_run_plan(config: BenchmarkConfig) -> dict[str, object]:
    """Return a safe, no-network plan suitable for review before spending money."""
    return {
        "mode": "dry-run",
        "dataset": f"{config.dataset.name}:{config.dataset.split}",
        "limit": config.dataset.limit,
        "model": {"kind": config.model.kind, "name": config.model.name},
        "will_call_model": False,
        "output_dir": str(config.run.output_dir),
    }


def run_benchmark(config: BenchmarkConfig) -> Path:
    """Run one configured model and persist independently auditable artifacts."""
    if config.run.dry_run:
        raise RuntimeError(
            "Refusing to run because run.dry_run is true. Set it to false explicitly."
        )
    examples, labels = load_classification_dataset(config.dataset)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{config.model.kind}"
    writer = RunWriter(config.run.output_dir / run_id, config.run.persist_raw_responses)
    writer.write_metadata(
        {
            "run_id": run_id,
            "started_at": datetime.now(UTC).isoformat(),
            "config": config.model_dump(mode="json"),
            "label_order": labels,
        }
    )
    adapter = create_adapter(config)
    records: list[PredictionRecord] = []
    try:
        for position, example in enumerate(examples, start=1):
            logger.info("Classifying item %s/%s", position, len(examples))
            prediction = adapter.classify(example, labels)
            record = PredictionRecord(
                example=example, prediction=prediction, model_name=adapter.model_name
            )
            writer.append_record(record)
            records.append(record)
    finally:
        adapter.close()
    metrics = compute_metrics(records, labels)
    cost = estimate_cost_usd(
        records,
        config.model.pricing.input_per_million_usd,
        config.model.pricing.output_per_million_usd,
    )
    writer.write_summary(
        RunSummary(
            run_id=run_id,
            model_name=adapter.model_name,
            dataset_name=config.dataset.name,
            split=config.dataset.split,
            sample_count=len(records),
            metrics=metrics,
            estimated_cost_usd=cost,
        )
    )
    return writer.directory


def create_adapter(config: BenchmarkConfig) -> ClassifierAdapter:
    """Create the chosen adapter at the last responsible moment."""
    match config.model.kind:
        case "jev":
            return JevAdapter(config.model)
        case "jev_openrouter":
            return OpenRouterJevAdapter(config.model)
        case "mlx_qwen":
            return MlxQwenAdapter(config.model)
