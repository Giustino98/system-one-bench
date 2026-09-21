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


def test_parse_label_rejects_missing_choices() -> None:
    with pytest.raises(ValueError, match="completion choice"):
        _parse_label({}, ["card_arrival"])


def test_parse_label_rejects_out_of_vocabulary_output() -> None:
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
    with pytest.raises(ValueError, match="invalid label"):
        _parse_label(body, ["card_arrival"])
