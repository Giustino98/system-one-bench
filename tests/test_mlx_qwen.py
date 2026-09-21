from typing import Any

from system_one_bench.adapters.mlx_qwen import (
    _extract_choice,
    _prompt,
    _split_thinking_output,
)
from system_one_bench.domain import ChoiceExample


def _bbh_example() -> ChoiceExample:
    return ChoiceExample(
        identifier="id",
        task="logical_deduction_three_objects",
        text="Which object is first?",
        choices={"A": "red", "B": "blue", "C": "green"},
        expected_choice="B",
        instruction="Choose the option that follows from the problem.",
        answer_prefix="ANSWER:",
    )


def test_native_thinking_is_enabled_through_the_chat_template() -> None:
    class Tokenizer:
        options: dict[str, Any]

        def apply_chat_template(self, messages: object, **kwargs: Any) -> str:
            self.options = kwargs
            return "rendered"

    tokenizer = Tokenizer()

    assert _prompt(tokenizer, _bbh_example(), thinking=True) == "rendered"
    assert tokenizer.options["enable_thinking"] is True


def test_reasoning_is_separated_and_only_exact_final_answer_is_scored() -> None:
    reasoning, final = _split_thinking_output(
        "<think>Ordering the objects gives blue first.</think>\nANSWER: B"
    )

    assert reasoning == "Ordering the objects gives blue first."
    assert final == "ANSWER: B"
    assert _extract_choice(final, _bbh_example()) == "B"


def test_bbh_parser_has_no_fuzzy_or_embedded_answer_matching() -> None:
    example = _bbh_example()

    assert _extract_choice("B", example).startswith("__invalid__:")
    assert _extract_choice("The answer is ANSWER: B", example).startswith("__invalid__:")
    assert _extract_choice("ANSWER: blue", example).startswith("__invalid__:")
