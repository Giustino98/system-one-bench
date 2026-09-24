import pytest

from system_one_bench.adapters.jev import build_payload, parse_choice
from system_one_bench.domain import ChoiceExample


def example() -> ChoiceExample:
    return ChoiceExample(
        identifier="id",
        task="logical_deduction_three_objects",
        text="Which object is first?",
        choices={"A": "The red object", "B": "The blue object"},
        expected_choice="A",
        instruction="Choose the option that follows from the problem.",
    )


def test_payload_preserves_problem_and_alternative_text() -> None:
    payload = build_payload(example(), "typesafe/jev-1.13")

    assert payload["state"] == {"problem": "Which object is first?"}
    assert payload["questions"]["answer"]["criteria"] == {
        "A": "The red object",
        "B": "The blue object",
    }


def test_choice_response_is_parsed_with_normalized_probabilities() -> None:
    selected, probabilities = parse_choice(
        {"answers": {"answer": {"choice": "B", "probabilities": {"A": 1, "B": 2}}}},
        example(),
    )

    assert selected == "B"
    assert probabilities == {"A": pytest.approx(1 / 3), "B": pytest.approx(2 / 3)}


def test_protocol_drift_fails_immediately() -> None:
    with pytest.raises(KeyError):
        parse_choice({"choices": {"answer": "A"}}, example())
