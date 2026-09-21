"""Local Qwen adapter using LM Studio's OpenAI-compatible server."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import httpx

from system_one_bench.adapters.mlx_qwen import _extract_choice
from system_one_bench.config import ModelConfig
from system_one_bench.domain import ChoiceExample, Prediction


class LmStudioQwenAdapter:
    """Classify with a Qwen checkpoint already loaded in LM Studio."""

    def __init__(self, config: ModelConfig) -> None:
        if config.kind != "lmstudio_qwen":
            raise ValueError("LmStudioQwenAdapter requires an 'lmstudio_qwen' model configuration.")
        self._config = config
        self._client = httpx.Client(timeout=config.timeout_seconds)

    @property
    def model_name(self) -> str:
        return self._config.api_model or self._config.name

    def classify(self, example: ChoiceExample) -> Prediction:
        if self._config.lmstudio_api == "native":
            return self._classify_native(example)
        payload: dict[str, Any] = {
            "model": self._config.api_model or self._config.name,
            "messages": _messages(example),
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "stream": False,
        }
        if self._config.top_p is not None:
            payload["top_p"] = self._config.top_p
        if self._config.top_k is not None:
            payload["top_k"] = self._config.top_k
        base_url = str(self._config.base_url or "http://127.0.0.1:1234/v1").rstrip("/")
        start = time.perf_counter()
        response = self._client.post(f"{base_url}/chat/completions", json=payload)
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        latency_ms = (time.perf_counter() - start) * 1_000
        selected = _parse_choice(body, example)
        usage = body.get("usage")
        if not isinstance(usage, Mapping):
            usage = {}
        return Prediction(
            predicted_choice=selected,
            probabilities=None,
            latency_ms=latency_ms,
            input_tokens=_integer_or_none(usage.get("prompt_tokens")),
            output_tokens=_integer_or_none(usage.get("completion_tokens")),
            raw_response=body,
        )

    def _classify_native(self, example: ChoiceExample) -> Prediction:
        payload: dict[str, Any] = {
            "model": self._config.api_model or self._config.name,
            "input": _user_prompt(example),
            "system_prompt": "Solve the single-choice problem.",
            "max_output_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "reasoning": "on" if self._config.thinking else "off",
            "stream": False,
            "store": False,
        }
        if self._config.top_p is not None:
            payload["top_p"] = self._config.top_p
        if self._config.top_k is not None:
            payload["top_k"] = self._config.top_k
        base_url = str(self._config.base_url or "http://127.0.0.1:1234").rstrip("/")
        start = time.perf_counter()
        response = self._client.post(f"{base_url}/api/v1/chat", json=payload)
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        latency_ms = (time.perf_counter() - start) * 1_000
        final_answer, reasoning = _parse_native_output(body)
        stats = body.get("stats")
        if not isinstance(stats, Mapping):
            stats = {}
        return Prediction(
            predicted_choice=_extract_choice(final_answer, example),
            probabilities=None,
            latency_ms=latency_ms,
            input_tokens=_integer_or_none(stats.get("input_tokens")),
            output_tokens=_integer_or_none(stats.get("total_output_tokens")),
            reasoning=reasoning,
            raw_response=body,
        )

    def close(self) -> None:
        self._client.close()


def _messages(example: ChoiceExample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Solve the single-choice problem."},
        {"role": "user", "content": _user_prompt(example)},
    ]


def _user_prompt(example: ChoiceExample) -> str:
    options = "\n".join(f"({key}) {text}" for key, text in example.choices.items())
    suffix = (
        f"Return the final line exactly as {example.answer_prefix} <choice>."
        if example.answer_prefix
        else "Reply with exactly one allowed choice key."
    )
    return f"{example.instruction}\n\nProblem:\n{example.text}\n\nChoices:\n{options}\n\n{suffix}"


def _parse_native_output(body: Mapping[str, Any]) -> tuple[str, str | None]:
    output = body.get("output")
    if not isinstance(output, list):
        raise ValueError("LM Studio native response did not include an output list.")
    messages: list[str] = []
    reasoning: list[str] = []
    for item in output:
        if not isinstance(item, Mapping) or not isinstance(item.get("content"), str):
            continue
        if item.get("type") == "message":
            messages.append(item["content"])
        elif item.get("type") == "reasoning":
            reasoning.append(item["content"])
    if not messages:
        raise ValueError("LM Studio native response did not include a message output.")
    reasoning_text = "\n".join(reasoning).strip() or None
    return "\n".join(messages).strip(), reasoning_text


def _parse_choice(body: Mapping[str, Any], example: ChoiceExample) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LM Studio response did not include a completion choice.")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("LM Studio completion choice has an invalid shape.")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("LM Studio response did not include an assistant message.")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("LM Studio assistant message did not contain text.")
    return _extract_choice(content, example)


def _integer_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None
