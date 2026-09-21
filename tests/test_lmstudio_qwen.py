from system_one_bench.adapters.lmstudio_qwen import _parse_choice, _parse_native_output
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
