from typing import Any

from system_one_bench.adapters.mlx_qwen import (
    _allowed_post_thinking_tokens,
    _eos_token_ids,
    build_prompt,
    parse_final_choice,
    split_thinking_output,
)
from system_one_bench.domain import ChoiceExample


def example() -> ChoiceExample:
    return ChoiceExample(
        identifier="id",
        task="logical_deduction_three_objects",
        text="Which object is first?",
        choices={"A": "red", "B": "blue", "C": "green"},
        expected_choice="B",
        instruction="Choose the option that follows from the problem.",
    )


def test_prompt_enables_native_thinking() -> None:
    class Tokenizer:
        options: dict[str, Any]

        def apply_chat_template(self, messages: object, **kwargs: Any) -> str:
            self.options = kwargs
            return "rendered"

    tokenizer = Tokenizer()

    assert build_prompt(tokenizer, example()) == "rendered"
    assert tokenizer.options["enable_thinking"] is True


def test_reasoning_and_final_json_are_separate() -> None:
    reasoning, final = split_thinking_output('<think>Blue must be first.</think>\n{"choice":"B"}')

    assert reasoning == "Blue must be first."
    assert parse_final_choice(final, example()) == "B"


def test_invalid_or_unfinished_final_output_returns_none() -> None:
    assert parse_final_choice("", example()) is None
    assert parse_final_choice('{"choice":"Z"}', example()) is None
    assert parse_final_choice('{"choice":"B","extra":true}', example()) is None


def test_post_thinking_constraint_activates_after_marker() -> None:
    marker = (90, 91)
    suffixes = ((10, 20, 30), (10, 21, 30))
    eos = (99,)

    assert _allowed_post_thinking_tokens((1, 2), marker, suffixes, eos) is None
    assert _allowed_post_thinking_tokens((1, 90, 91), marker, suffixes, eos) == (10,)
    assert _allowed_post_thinking_tokens((90, 91, 10), marker, suffixes, eos) == (20, 21)
    assert _allowed_post_thinking_tokens((90, 91, 10, 20, 30), marker, suffixes, eos) == eos


def test_eos_token_uses_tokenizer_fallback() -> None:
    class Tokenizer:
        eos_token_ids = {None}
        eos_token_id = None
        eos_token = "<|im_end|>"

        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == self.eos_token
            return 151645

    assert _eos_token_ids(Tokenizer()) == (151645,)
