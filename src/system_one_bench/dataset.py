"""Pinned BBH Logical Deduction dataset loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from datasets import load_dataset

from system_one_bench.config import DatasetConfig, LogicalDeductionTask
from system_one_bench.domain import ChoiceExample


def load_bbh_dataset(config: DatasetConfig, *, seed: int) -> tuple[list[ChoiceExample], list[str]]:
    """Load configured tasks in order; ``limit`` applies independently to each task."""
    examples: list[ChoiceExample] = []
    choice_keys: set[str] = set()
    for task_index, task in enumerate(config.tasks):
        dataset = load_dataset(
            config.name,
            task,
            split=config.split,
            revision=config.revision,
        )
        if config.limit is not None:
            dataset = dataset.shuffle(seed=seed + task_index)
            dataset = dataset.select(range(min(config.limit, len(dataset))))
        for row_index, row in enumerate(dataset):
            example = parse_bbh_row(row, task, row_index, config.instruction)
            examples.append(example)
            choice_keys.update(example.choice_keys)
    return examples, sorted(choice_keys)


def parse_bbh_row(
    row: Mapping[str, Any],
    task: LogicalDeductionTask,
    row_index: int,
    instruction: str,
) -> ChoiceExample:
    """Parse one trusted dataset row into the benchmark domain model."""
    text = row["input"]
    alternatives = row["choices"]
    target_index = row["target_idx"]
    if not isinstance(text, str):
        raise TypeError("BBH input must be a string.")
    if not isinstance(alternatives, Sequence) or isinstance(alternatives, str):
        raise TypeError("BBH choices must be a sequence.")
    if not isinstance(target_index, int):
        raise TypeError("BBH target_idx must be an integer.")

    choices = {chr(65 + index): str(value) for index, value in enumerate(alternatives)}
    expected_choice = chr(65 + target_index)
    source_id = row.get("id", row_index)
    return ChoiceExample(
        identifier=f"{task}-{source_id}",
        task=task,
        text=text.strip(),
        choices=choices,
        expected_choice=expected_choice,
        instruction=instruction,
    )
