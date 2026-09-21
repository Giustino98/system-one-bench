"""Dataset access isolated from benchmark orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from datasets import ClassLabel, load_dataset

from system_one_bench.config import DatasetConfig
from system_one_bench.domain import ChoiceExample

BBH_LOGICAL_DEDUCTION_TASKS = (
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
)


def humanize_label(label: str) -> str:
    """Turn a machine label into a neutral readable option."""
    return label.replace("_", " ")


def load_benchmark_dataset(
    config: DatasetConfig, *, seed: int = 42
) -> tuple[list[ChoiceExample], list[str]]:
    """Dispatch dataset loading without coupling the runner to a benchmark."""
    if config.kind == "bbh_logical_deduction":
        return load_bbh_logical_deduction_dataset(config, seed=seed)
    return load_classification_dataset(config, seed=seed)


def load_classification_dataset(
    config: DatasetConfig, *, seed: int = 42
) -> tuple[list[ChoiceExample], list[str]]:
    """Load a Hugging Face classification split and preserve its label order."""
    load_kwargs: dict[str, Any] = {"split": config.split}
    if config.revision is not None:
        load_kwargs["revision"] = config.revision
    if config.data_files is None:
        dataset = load_dataset(config.name, **load_kwargs)
    else:
        dataset = load_dataset("parquet", data_files=config.data_files, **load_kwargs)
    label_feature = dataset.features[config.label_field]
    if not isinstance(label_feature, ClassLabel):
        raise TypeError(f"{config.label_field!r} must be a ClassLabel feature.")
    labels = list(label_feature.names)
    if config.limit is not None:
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(min(config.limit, len(dataset))))
    choices = {label: humanize_label(label) for label in labels}
    instruction = config.instruction or "Choose the single best category for the input."
    records: list[ChoiceExample] = []
    for index, row in enumerate(dataset):
        label_index = row[config.label_field]
        if not isinstance(label_index, int):
            raise TypeError("Expected an integer class label.")
        records.append(
            ChoiceExample(
                identifier=f"{config.split}-{index}",
                task=config.name,
                text=str(row[config.text_field]),
                choices=choices,
                expected_choice=labels[label_index],
                instruction=instruction,
            )
        )
    return records, labels


def load_bbh_logical_deduction_dataset(
    config: DatasetConfig, *, seed: int = 42
) -> tuple[list[ChoiceExample], list[str]]:
    """Load the pinned BBH logical-deduction subsets with a stable task order.

    ``limit`` is applied per task so every smoke run represents 3, 5, and 7 objects.
    """
    invalid_tasks = set(config.tasks) - set(BBH_LOGICAL_DEDUCTION_TASKS)
    if invalid_tasks:
        raise ValueError(f"Unsupported BBH logical deduction tasks: {sorted(invalid_tasks)!r}")
    if config.revision is None:
        raise ValueError("BBH must be pinned with dataset.revision for reproducibility.")

    instruction = config.instruction or "Choose the option that follows from the problem."
    examples: list[ChoiceExample] = []
    all_keys: set[str] = set()
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
            example = _parse_bbh_row(row, task, row_index, instruction)
            examples.append(example)
            all_keys.update(example.choice_keys)
    return examples, sorted(all_keys)


def _parse_bbh_row(
    row: Mapping[str, Any], task: str, row_index: int, instruction: str
) -> ChoiceExample:
    raw_input = row.get("input")
    raw_choices = row.get("choices")
    target_index = row.get("target_idx")
    if not isinstance(raw_input, str):
        raise TypeError("BBH rows must contain a string input field.")
    if not isinstance(raw_choices, Sequence) or isinstance(raw_choices, str):
        raise TypeError("BBH rows must contain a sequence of choices.")
    if not isinstance(target_index, int):
        raise TypeError("BBH rows must contain an integer target_idx field.")
    if len(raw_choices) > 26:
        raise ValueError("BBH example has too many choices for canonical A-Z keys.")
    choices = {chr(65 + index): str(text) for index, text in enumerate(raw_choices)}
    if target_index < 0 or target_index >= len(choices):
        raise ValueError(f"Invalid BBH target index: {target_index!r}")
    source_identifier = row.get("id")
    identifier = source_identifier if isinstance(source_identifier, str) else str(row_index)
    return ChoiceExample(
        identifier=f"{task}-{identifier}",
        task=task,
        text=raw_input.strip(),
        choices=choices,
        expected_choice=chr(65 + target_index),
        instruction=instruction,
        answer_prefix="ANSWER:",
    )
