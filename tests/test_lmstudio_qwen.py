import pytest

from system_one_bench.adapters.lmstudio_qwen import _parse_label


def test_parse_label_reads_openai_compatible_response() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "card_arrival",
                }
            }
        ]
    }

    assert _parse_label(body, ["card_arrival", "cash_withdrawal"]) == "card_arrival"


def test_parse_label_accepts_markdown_escaped_underscores() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": r"receive\_money",
                }
            }
        ]
    }

    assert _parse_label(body, ["receive_money", "cash_withdrawal"]) == "receive_money"


def test_parse_label_keeps_near_match_invalid() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": r"receive\_money",
                }
            }
        ]
    }

    assert _parse_label(body, ["receiving_money"]) == r"__invalid__:receive\_money"


def test_parse_label_rejects_missing_choices() -> None:
    with pytest.raises(ValueError, match="completion choice"):
        _parse_label({}, ["card_arrival"])


def test_parse_label_marks_out_of_vocabulary_output() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "invented_label",
                }
            }
        ]
    }

    assert _parse_label(body, ["card_arrival"]) == "__invalid__:invented_label"
