"""Append-only run artifacts that can be inspected without a database."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from system_one_bench.domain import (
    ChoiceExample,
    Prediction,
    PredictionRecord,
    RunSummary,
)


class RunWriter:
    """Owns a single run directory and writes JSON/JSONL artifacts."""

    def __init__(self, directory: Path, persist_raw_responses: bool) -> None:
        self.directory = directory
        self._persist_raw_responses = persist_raw_responses
        directory.mkdir(parents=True, exist_ok=False)
        self._records_path = directory / "predictions.jsonl"

    @classmethod
    def resume(cls, directory: Path, persist_raw_responses: bool) -> RunWriter:
        """Re-open an interrupted run without replacing its append-only records."""
        if not directory.is_dir():
            raise ValueError(f"Resume directory does not exist: {directory}")
        if not (directory / "metadata.json").is_file():
            raise ValueError(f"Resume directory has no metadata.json: {directory}")
        writer = cls.__new__(cls)
        writer.directory = directory
        writer._persist_raw_responses = persist_raw_responses
        writer._records_path = directory / "predictions.jsonl"
        return writer

    def read_metadata(self) -> dict[str, Any]:
        """Read the immutable run metadata used to validate a resume request."""
        with (self.directory / "metadata.json").open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError("metadata.json must contain a JSON object.")
        return payload

    def read_records(self, examples: dict[str, ChoiceExample]) -> list[PredictionRecord]:
        """Load and validate append-only records against the reproducible dataset."""
        if not self._records_path.exists():
            return []
        records: list[PredictionRecord] = []
        seen_identifiers: set[str] = set()
        with self._records_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    stored_example = payload["example"]
                    identifier = stored_example["identifier"]
                    example = examples[identifier]
                    prediction = Prediction(**payload["prediction"])
                    model_name = payload["model_name"]
                except (KeyError, TypeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"Invalid prediction record at line {line_number} in {self._records_path}."
                    ) from error
                if identifier in seen_identifiers:
                    raise ValueError(f"Duplicate completed example in resume data: {identifier}")
                if stored_example != asdict(example):
                    raise ValueError(
                        f"Resume record does not match current dataset example: {identifier}"
                    )
                if not isinstance(model_name, str):
                    raise TypeError(f"Invalid model_name in resume record: {identifier}")
                seen_identifiers.add(identifier)
                records.append(PredictionRecord(example, prediction, model_name))
        return records

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
