import pytest

from system_one_bench.adapters.jev import _build_payload, _parse_choice
from system_one_bench.domain import ClassificationExample


def test_build_payload_uses_official_choice_criteria_shape() -> None:
    payload = _build_payload(
        ClassificationExample("id", "Where is my card?", "card_arrival"),
        ["card_arrival", "cash_withdrawal"],
        "typesafe/jev-1.13",
    )

    question = payload["questions"]["intent"]
    assert "options" not in question
    assert question["criteria"] == {"card_arrival": None, "cash_withdrawal": None}


def test_parse_choice_normalizes_probabilities() -> None:
    selected, probabilities = _parse_choice(
        {
            "answers": {
                "intent": {
                    "choice": "cash_withdrawal",
                    "probabilities": {"cash_withdrawal": 2, "card_payment": 1},
                }
            }
        },
        ["cash_withdrawal", "card_payment"],
    )

    assert selected == "cash_withdrawal"
    assert probabilities == {
        "cash_withdrawal": pytest.approx(2 / 3),
        "card_payment": pytest.approx(1 / 3),
    }


def test_parse_choice_rejects_a_label_outside_the_benchmark() -> None:
    with pytest.raises(ValueError, match="invalid benchmark label"):
        _parse_choice({"answers": {"intent": "invented_label"}}, ["cash_withdrawal"])
