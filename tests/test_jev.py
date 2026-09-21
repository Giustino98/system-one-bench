import pytest

from system_one_bench.adapters.jev import _build_payload, _parse_choice
from system_one_bench.domain import ChoiceExample


def _example() -> ChoiceExample:
    return ChoiceExample(
        identifier="id",
        task="logical_deduction_three_objects",
        text="Which object is first?",
        choices={"A": "The red object", "B": "The blue object"},
        expected_choice="A",
        instruction="Choose the option that follows from the problem.",
        answer_prefix="ANSWER:",
    )


def test_build_payload_uses_choice_keys_and_alternative_text_as_criteria() -> None:
    payload = _build_payload(_example(), "typesafe/jev-1.13")

    assert payload["state"] == {"problem": "Which object is first?"}
    question = payload["questions"]["answer"]
    assert question["criteria"] == {"A": "The red object", "B": "The blue object"}
    assert question["instructions"] == "Choose the option that follows from the problem."
    assert "customer_message" not in payload["state"]


def test_parse_choice_normalizes_probabilities() -> None:
    selected, probabilities = _parse_choice(
        {
            "answers": {
                "answer": {
                    "choice": "B",
                    "probabilities": {"A": 1, "B": 2},
                }
            }
        },
        ["A", "B"],
    )

    assert selected == "B"
    assert probabilities == {"A": pytest.approx(1 / 3), "B": pytest.approx(2 / 3)}


def test_parse_choice_rejects_a_key_outside_the_example() -> None:
    with pytest.raises(ValueError, match="invalid benchmark choice"):
        _parse_choice({"answers": {"answer": "C"}}, ["A", "B"])
