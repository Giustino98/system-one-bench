"""Dataset access isolated from benchmark orchestration."""

from __future__ import annotations

from datasets import ClassLabel, load_dataset

from system_one_bench.config import DatasetConfig
from system_one_bench.domain import ClassificationExample


def humanize_label(label: str) -> str:
    """Turn BANKING77's machine labels into a neutral readable option."""
    return label.replace("_", " ")


def load_classification_dataset(
    config: DatasetConfig, *, seed: int = 42
) -> tuple[list[ClassificationExample], list[str]]:
    """Load a Hugging Face classification split and preserve its label order."""
    if config.data_files is None:
        dataset = load_dataset(config.name, split=config.split)
    else:
        dataset = load_dataset("parquet", data_files=config.data_files, split=config.split)
    label_feature = dataset.features[config.label_field]
    if not isinstance(label_feature, ClassLabel):
        raise TypeError(f"{config.label_field!r} must be a ClassLabel feature.")
    labels = list(label_feature.names)
    if config.limit is not None:
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(min(config.limit, len(dataset))))
    records: list[ClassificationExample] = []
    for index, row in enumerate(dataset):
        label_index = row[config.label_field]
        if not isinstance(label_index, int):
            raise TypeError("Expected an integer class label.")
        records.append(
            ClassificationExample(
                identifier=f"{config.split}-{index}",
                text=str(row[config.text_field]),
                expected_label=labels[label_index],
            )
        )
    return records, labels
