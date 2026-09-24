from datasets import Dataset

import system_one_bench.dataset as dataset_module
from system_one_bench.config import DatasetConfig
from system_one_bench.dataset import load_bbh_dataset


def test_loader_maps_choices_and_uses_the_pinned_revision(monkeypatch) -> None:
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
    config = DatasetConfig(tasks=("logical_deduction_three_objects",), limit=1)

    examples, keys = load_bbh_dataset(config, seed=42)

    assert calls[0][0] == ("lighteval/bbh", "logical_deduction_three_objects")
    assert calls[0][1]["revision"] == config.revision
    assert keys == ["A", "B"]
    assert examples[0].choices == {"A": "Red is first", "B": "Blue is first"}
    assert examples[0].expected_choice == "A"
