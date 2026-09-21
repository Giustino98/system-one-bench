from datasets import ClassLabel, Dataset, Features, Value

import system_one_bench.dataset as dataset_module
from system_one_bench.config import DatasetConfig
from system_one_bench.dataset import (
    load_bbh_logical_deduction_dataset,
    load_classification_dataset,
)


def test_limited_classification_dataset_is_shuffled_deterministically(monkeypatch) -> None:
    dataset = Dataset.from_dict(
        {
            "text": [f"message-{index}" for index in range(20)],
            "label": [index % 2 for index in range(20)],
        },
        features=Features({"text": Value("string"), "label": ClassLabel(names=["a", "b"])}),
    )
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: dataset)

    first, labels = load_classification_dataset(DatasetConfig(name="test", limit=5), seed=7)
    repeated, _ = load_classification_dataset(DatasetConfig(name="test", limit=5), seed=7)

    assert labels == ["a", "b"]
    assert [item.text for item in first] == [item.text for item in repeated]
    assert first[0].choices == {"a": "a", "b": "b"}


def test_bbh_loader_maps_problem_options_and_target_and_pins_revision(monkeypatch) -> None:
    dataset = Dataset.from_dict(
        {
            "id": ["sample-1"],
            "input": ["Red is left of blue."],
            "choices": [["Red is first", "Blue is first"]],
            "target_idx": [0],
        }
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_load_dataset(*args: object, **kwargs: object) -> Dataset:
        calls.append((args, kwargs))
        return dataset

    monkeypatch.setattr(dataset_module, "load_dataset", fake_load_dataset)
    config = DatasetConfig(
        kind="bbh_logical_deduction",
        name="lighteval/bbh",
        revision="pinned-sha",
        tasks=("logical_deduction_three_objects",),
    )

    examples, keys = load_bbh_logical_deduction_dataset(config)

    assert calls[0][0] == ("lighteval/bbh", "logical_deduction_three_objects")
    assert calls[0][1]["revision"] == "pinned-sha"
    assert keys == ["A", "B"]
    assert examples[0].text == "Red is left of blue."
    assert examples[0].choices == {"A": "Red is first", "B": "Blue is first"}
    assert examples[0].expected_choice == "A"
    assert examples[0].answer_prefix == "ANSWER:"
