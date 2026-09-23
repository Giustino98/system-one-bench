import pytest

from system_one_bench.adapters.lmstudio_qwen import (
    _messages,
    _parse_choice,
    _parse_native_output,
    _parse_openai_reasoning,
    _parse_structured_choice,
    _structured_response_format,
)
from system_one_bench.domain import ChoiceExample


def test_banking_parser_preserves_escaped_underscore_compatibility() -> None:
    example = ChoiceExample(
        identifier="id",
        task="banking77",
        text="How do I receive money?",
        choices={"receive_money": "receive money", "cash_withdrawal": "cash withdrawal"},
        expected_choice="receive_money",
        instruction="Choose the category.",
    )
    body = {"choices": [{"message": {"content": r"receive\_money"}}]}

    assert _parse_choice(body, example) == "receive_money"


def test_native_response_keeps_reasoning_separate_from_final_answer() -> None:
    body = {
        "output": [
            {"type": "reasoning", "content": "Ordering the objects."},
            {"type": "message", "content": "Short conclusion.\nANSWER: B"},
        ]
    }

    final_answer, reasoning = _parse_native_output(body)

    assert final_answer == "Short conclusion.\nANSWER: B"
    assert reasoning == "Ordering the objects."


def test_structured_response_uses_example_specific_choice_enum() -> None:
    example = ChoiceExample(
        identifier="id",
        task="logical_deduction_three_objects",
        text="problem",
        choices={"A": "first", "B": "second", "C": "third"},
        expected_choice="B",
        instruction="Choose one.",
    )

    response_format = _structured_response_format(example)
    enum = response_format["json_schema"]["schema"]["properties"]["choice"]["enum"]

    assert enum == ["A", "B", "C"]
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False


def test_structured_parser_and_reasoning_are_separate() -> None:
    example = ChoiceExample(
        identifier="id",
        task="logical_deduction_three_objects",
        text="problem",
        choices={"A": "first", "B": "second"},
        expected_choice="B",
        instruction="Choose one.",
    )
    body = {
        "choices": [
            {
                "message": {
                    "content": '{"choice":"B"}',
                    "reasoning_content": "Ordered the objects.",
                },
                "finish_reason": "stop",
            }
        ]
    }

    assert _parse_structured_choice(body, example) == "B"
    assert _parse_openai_reasoning(body) == "Ordered the objects."


@pytest.mark.parametrize(
    "content",
    ['{"choice":"C"}', '{"choice":"B","extra":true}', "not-json"],
)
def test_structured_parser_rejects_invalid_payloads(content: str) -> None:
    example = ChoiceExample(
        identifier="id",
        task="logical_deduction_three_objects",
        text="problem",
        choices={"A": "first", "B": "second"},
        expected_choice="B",
        instruction="Choose one.",
    )

    with pytest.raises(ValueError):
        _parse_structured_choice({"choices": [{"message": {"content": content}}]}, example)


def test_messages_use_qwen_native_thinking_switches() -> None:
    example = ChoiceExample(
        identifier="id",
        task="logical_deduction_three_objects",
        text="problem",
        choices={"A": "first", "B": "second"},
        expected_choice="B",
        instruction="Choose one.",
    )

    thinking = _messages(example, thinking=True, structured=True)
    no_thinking = _messages(example, thinking=False, structured=True)

    assert thinking[0]["content"].endswith("/think")
    assert no_thinking[0]["content"].endswith("/no_think")
    assert "ANSWER:" not in thinking[1]["content"]
