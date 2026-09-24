from typing import Any

from system_one_bench.adapters.mlx_qwen import (
    _allowed_post_thinking_tokens,
    _eos_token_ids,
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


def test_thinking_flag_is_always_passed_to_the_chat_template() -> None:
    class Tokenizer:
        options: dict[str, Any]

        def apply_chat_template(self, messages: object, **kwargs: Any) -> str:
            self.options = kwargs
            return "rendered"

    thinking_tokenizer = Tokenizer()
    no_thinking_tokenizer = Tokenizer()

    assert _prompt(thinking_tokenizer, _bbh_example(), thinking=True) == "rendered"
    assert _prompt(no_thinking_tokenizer, _bbh_example(), thinking=False) == "rendered"
    assert thinking_tokenizer.options["enable_thinking"] is True
    assert no_thinking_tokenizer.options["enable_thinking"] is False


def test_reasoning_is_separated_and_only_exact_final_answer_is_scored() -> None:
    reasoning, final = _split_thinking_output(
        "<think>Ordering the objects gives blue first.</think>\nANSWER: B"
    )

    assert reasoning == "Ordering the objects gives blue first."
    assert final == "ANSWER: B"
    assert _extract_choice(final, _bbh_example()) == "B"


def test_structured_final_answer_is_parsed_without_touching_reasoning() -> None:
    reasoning, final = _split_thinking_output(
        '<think>Ordering the objects gives blue first.</think>\n{"choice":"B"}'
    )

    assert reasoning == "Ordering the objects gives blue first."
    assert _extract_choice(final, _bbh_example(), structured=True) == "B"
    assert _extract_choice('{"choice":"Z"}', _bbh_example(), structured=True).startswith(
        "__invalid__:"
    )


def test_bbh_parser_has_no_fuzzy_or_embedded_answer_matching() -> None:
    example = _bbh_example()

    assert _extract_choice("B", example).startswith("__invalid__:")
    assert _extract_choice("The answer is ANSWER: B", example).startswith("__invalid__:")
    assert _extract_choice("ANSWER: blue", example).startswith("__invalid__:")


def test_bbh_parser_scores_only_an_exact_answer_on_the_last_line() -> None:
    example = _bbh_example()

    assert _extract_choice("Concise conclusion.\nANSWER: B", example) == "B"
    assert _extract_choice("Concise conclusion.\nANSWER: (B)", example) == "B"
    assert _extract_choice("ANSWER: B\nTrailing text", example).startswith("__invalid__:")


def test_post_thinking_constraint_activates_only_after_marker() -> None:
    marker = (90, 91)
    suffixes = ((10, 20, 30), (10, 21, 30))
    eos = (99,)

    assert _allowed_post_thinking_tokens((1, 2), marker, suffixes, eos) is None
    assert _allowed_post_thinking_tokens((1, 90, 91), marker, suffixes, eos) == (10,)
    assert _allowed_post_thinking_tokens((90, 91, 10), marker, suffixes, eos) == (
        20,
        21,
    )
    assert _allowed_post_thinking_tokens((90, 91, 10, 20, 30), marker, suffixes, eos) == eos
    assert _allowed_post_thinking_tokens((90, 91, 10, 20, 30, 99), marker, suffixes, eos) == eos
    assert _allowed_post_thinking_tokens((90, 91, 42), marker, suffixes, eos) == ()


def test_eos_token_falls_back_from_missing_wrapper_id() -> None:
    class Tokenizer:
        eos_token_ids = {None}
        eos_token_id = None
        eos_token = "<|im_end|>"

        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == self.eos_token
            return 151645

    assert _eos_token_ids(Tokenizer()) == (151645,)
