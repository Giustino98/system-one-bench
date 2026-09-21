"""Append-only run artifacts that can be inspected without a database."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from system_one_bench.domain import PredictionRecord, RunSummary


class RunWriter:
    """Owns a single run directory and writes JSON/JSONL artifacts."""

    def __init__(self, directory: Path, persist_raw_responses: bool) -> None:
        self.directory = directory
        self._persist_raw_responses = persist_raw_responses
        directory.mkdir(parents=True, exist_ok=False)
        self._records_path = directory / "predictions.jsonl"

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        self._write_json("metadata.json", metadata)

    def append_record(self, record: PredictionRecord) -> None:
        safe_record = record
        if not self._persist_raw_responses:
            safe_record = replace(record, prediction=replace(record.prediction, raw_response=None))
        with self._records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe_record.as_dict(), ensure_ascii=False) + "\n")

    def write_summary(self, summary: RunSummary) -> None:
        self._write_json("summary.json", asdict(summary))

    def _write_json(self, filename: str, payload: dict[str, Any]) -> None:
        with (self.directory / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
