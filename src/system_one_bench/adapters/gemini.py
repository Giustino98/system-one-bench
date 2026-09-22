"""Google Gemini REST adapter for the shared single-choice benchmark contract."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from system_one_bench.adapters.lmstudio_qwen import _user_prompt
from system_one_bench.adapters.mlx_qwen import _extract_choice
from system_one_bench.config import ModelConfig
from system_one_bench.domain import ChoiceExample, Prediction

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
SYSTEM_INSTRUCTION = "Solve the single-choice problem."


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
    return {
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": _user_prompt(example)}]}],
        "generationConfig": generation_config,
    }


def _parse_choice(body: Mapping[str, Any], example: ChoiceExample) -> tuple[str, str | None]:
    """Separate Gemini thought summaries and score only the final assistant content."""
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
    return _extract_choice("\n".join(answers).strip(), example), reasoning


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
