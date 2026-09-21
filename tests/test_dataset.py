from datasets import ClassLabel, Dataset, Features, Value

import system_one_bench.dataset as dataset_module
from system_one_bench.config import DatasetConfig
from system_one_bench.dataset import load_classification_dataset


def test_limited_dataset_is_shuffled_deterministically(monkeypatch) -> None:
    dataset = Dataset.from_dict(
        {
            "text": [f"message-{index}" for index in range(20)],
            "label": [index % 2 for index in range(20)],
        },
        features=Features(
            {
                "text": Value("string"),
                "label": ClassLabel(names=["a", "b"]),
            }
        ),
    )
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: dataset)
    config = DatasetConfig(name="test", limit=5)

    first, labels = load_classification_dataset(config, seed=7)
    repeated, _ = load_classification_dataset(config, seed=7)
    different, _ = load_classification_dataset(config, seed=8)

    first_texts = [item.text for item in first]
    assert labels == ["a", "b"]
    assert len(first) == 5
    assert first_texts == [item.text for item in repeated]
    assert first_texts != [item.text for item in different]
    assert first_texts != [f"message-{index}" for index in range(5)]
