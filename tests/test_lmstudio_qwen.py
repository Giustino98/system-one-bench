from system_one_bench.adapters.lmstudio_qwen import _parse_choice
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
