"""Google Gemini REST adapter for the shared single-choice benchmark contract."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from json import JSONDecodeError, loads
from typing import Any

import httpx

from system_one_bench.adapters.prompts import build_choice_prompt
from system_one_bench.config import ModelConfig
from system_one_bench.domain import ChoiceExample, Prediction

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
SYSTEM_INSTRUCTION = "Solve the single-choice problem."
STRUCTURED_ANSWER_INSTRUCTION = 'Return the selected choice in the JSON field "choice".'


class GeminiAdapter:
    """Call Gemini's GenerateContent REST endpoint without an SDK dependency."""

    def __init__(self, config: ModelConfig) -> None:
        if config.kind != "gemini":
            raise ValueError("GeminiAdapter requires a 'gemini' model configuration.")
        self._config = config
        self._client = httpx.Client(timeout=config.timeout_seconds)

    @property
    def model_name(self) -> str:
        return self._config.name

    def classify(self, example: ChoiceExample) -> Prediction:
        api_key = os.getenv(self._config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing {self._config.api_key_env}. Copy .env.example and add a key "
                "before a live run."
            )
        start = time.perf_counter()
        response = self._post_with_retry(api_key, _build_payload(example, self._config))
        body: dict[str, Any] = response.json()
        latency_ms = (time.perf_counter() - start) * 1_000
        selected, reasoning = _parse_choice(body, example)
        input_tokens, output_tokens = _usage_tokens(body)
        return Prediction(
            predicted_choice=selected,
            probabilities=None,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning=reasoning,
            raw_response=body,
        )

    def close(self) -> None:
        self._client.close()

    def _post_with_retry(self, api_key: str, payload: dict[str, Any]) -> httpx.Response:
        base_url = str(self._config.base_url or DEFAULT_GEMINI_BASE_URL).rstrip("/")
        endpoint = f"{base_url}/models/{self._config.name}:generateContent"
        retry_statuses = {429, 500, 502, 503, 504}
        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._client.post(
                    endpoint,
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json=payload,
                )
            except httpx.TransportError:
                if attempt >= self._config.max_retries:
                    raise
                time.sleep(min(2**attempt, 8))
                continue
            if response.status_code not in retry_statuses or attempt >= self._config.max_retries:
                response.raise_for_status()
                return response
            time.sleep(min(2**attempt, 8))
        raise RuntimeError("Unreachable retry state.")


def _build_payload(example: ChoiceExample, config: ModelConfig) -> dict[str, Any]:
    """Use the identical problem/options prompt sent to the Qwen BBH baseline."""
    generation_config: dict[str, Any] = {
        "candidateCount": 1,
        "maxOutputTokens": config.max_tokens,
        "temperature": config.temperature,
    }
    if config.top_p is not None:
        generation_config["topP"] = config.top_p
    if config.top_k is not None:
        generation_config["topK"] = config.top_k
    if config.thinking:
        generation_config["thinkingConfig"] = {
            "thinkingLevel": config.thinking_level or "medium",
            "includeThoughts": True,
        }
    generation_config["responseMimeType"] = "application/json"
    generation_config["responseJsonSchema"] = _choice_schema(example)
    return {
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": build_choice_prompt(
                            example, answer_instruction=STRUCTURED_ANSWER_INSTRUCTION
                        )
                    }
                ],
            }
        ],
        "generationConfig": generation_config,
    }


def _choice_schema(example: ChoiceExample) -> dict[str, Any]:
    """Constrain output to one canonical answer key for this exact example."""
    return {
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": list(example.choice_keys)}},
        "required": ["choice"],
        "additionalProperties": False,
        "propertyOrdering": ["choice"],
    }


def _parse_choice(body: Mapping[str, Any], example: ChoiceExample) -> tuple[str, str | None]:
    """Separate thought summaries and parse only the schema-constrained final JSON."""
    candidates = body.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, str) or not candidates:
        raise ValueError("Gemini response did not include a candidate.")
    candidate = candidates[0]
    if not isinstance(candidate, Mapping):
        raise ValueError("Gemini response candidate has an invalid shape.")
    content = candidate.get("content")
    if not isinstance(content, Mapping):
        raise ValueError("Gemini response did not include candidate content.")
    parts = content.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, str):
        raise ValueError("Gemini response did not include candidate content parts.")
    answers: list[str] = []
    thoughts: list[str] = []
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        text = part.get("text")
        if not isinstance(text, str):
            continue
        if part.get("thought") is True:
            thoughts.append(text)
        else:
            answers.append(text)
    if not answers:
        raise ValueError("Gemini response did not include final assistant text.")
    reasoning = "\n".join(thoughts).strip() or None
    return _parse_structured_choice("\n".join(answers).strip(), example), reasoning


def _parse_structured_choice(text: str, example: ChoiceExample) -> str:
    """Defensively validate Gemini JSON even though the API schema constrains it."""
    try:
        value = loads(text)
    except JSONDecodeError:
        return f"__invalid__:{text}"
    if not isinstance(value, Mapping) or set(value) != {"choice"}:
        return f"__invalid__:{text}"
    choice = value.get("choice")
    if not isinstance(choice, str) or choice not in example.choices:
        return f"__invalid__:{text}"
    return choice


def _usage_tokens(body: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """Count thought tokens as output so cost estimates cover Gemini thinking."""
    usage = body.get("usageMetadata")
    if not isinstance(usage, Mapping):
        return None, None
    input_tokens = _integer_or_none(usage.get("promptTokenCount"))
    total_tokens = _integer_or_none(usage.get("totalTokenCount"))
    if input_tokens is None or total_tokens is None or total_tokens < input_tokens:
        return input_tokens, _integer_or_none(usage.get("candidatesTokenCount"))
    return input_tokens, total_tokens - input_tokens


def _integer_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None
