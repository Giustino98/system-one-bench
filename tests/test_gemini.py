from system_one_bench.adapters.gemini import _build_payload, _parse_choice, _usage_tokens
from system_one_bench.config import ModelConfig
from system_one_bench.domain import ChoiceExample


def _example() -> ChoiceExample:
    return ChoiceExample(
        identifier="id",
        task="logical_deduction_three_objects",
        text="The red object is first.",
        choices={"A": "The red object is first.", "B": "The blue object is first."},
        expected_choice="A",
        instruction="Choose the option that follows from the problem.",
        answer_prefix="ANSWER:",
    )


def _config() -> ModelConfig:
    return ModelConfig(
        kind="gemini",
        name="gemini-3.8-flash",
        api_key_env="GEMINI_API_KEY",
        max_tokens=1024,
        temperature=0.0,
        thinking=True,
        thinking_level="medium",
    )


def test_payload_keeps_the_shared_problem_and_enables_gemini_thinking() -> None:
    payload = _build_payload(_example(), _config())

    assert payload["systemInstruction"] == {"parts": [{"text": "Solve the single-choice problem."}]}
    user_text = payload["contents"][0]["parts"][0]["text"]
    assert "The red object is first." in user_text
    assert "(A) The red object is first." in user_text
    assert 'Return the selected choice in the JSON field "choice".' in user_text
    assert payload["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "medium",
        "includeThoughts": True,
    }
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert payload["generationConfig"]["responseJsonSchema"] == {
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
        "required": ["choice"],
        "additionalProperties": False,
        "propertyOrdering": ["choice"],
    }


def test_parser_separates_thought_summary_and_scores_only_final_text() -> None:
    body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "The red object is given as first."},
                        {"text": '{"choice":"A"}'},
                    ]
                }
            }
        ]
    }

    choice, reasoning = _parse_choice(body, _example())

    assert choice == "A"
    assert reasoning == "The red object is given as first."


def test_parser_marks_malformed_or_out_of_vocab_structured_output_invalid() -> None:
    malformed = {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}
    out_of_vocab = {"candidates": [{"content": {"parts": [{"text": '{"choice":"C"}'}]}}]}

    malformed_choice, _ = _parse_choice(malformed, _example())
    out_of_vocab_choice, _ = _parse_choice(out_of_vocab, _example())

    assert malformed_choice.startswith("__invalid__:")
    assert out_of_vocab_choice.startswith("__invalid__:")


def test_usage_counts_thought_tokens_as_output() -> None:
    input_tokens, output_tokens = _usage_tokens(
        {"usageMetadata": {"promptTokenCount": 12, "totalTokenCount": 37}}
    )

    assert input_tokens == 12
    assert output_tokens == 25
