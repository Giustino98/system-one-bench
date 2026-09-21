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
        return self._config.name

    def classify(self, example: ChoiceExample) -> Prediction:
        payload: dict[str, Any] = {
            "model": self._config.name,
            "messages": _messages(example),
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "stream": False,
        }
        if self._config.top_p is not None:
            payload["top_p"] = self._config.top_p
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

    def close(self) -> None:
        self._client.close()


def _messages(example: ChoiceExample) -> list[dict[str, str]]:
    options = "\n".join(f"({key}) {text}" for key, text in example.choices.items())
    suffix = (
        f"End with exactly {example.answer_prefix} <choice>."
        if example.answer_prefix
        else "Reply with exactly one allowed choice key."
    )
    return [
        {"role": "system", "content": "Solve the single-choice problem."},
        {
            "role": "user",
            "content": (
                f"{example.instruction}\n\nProblem:\n{example.text}\n\nChoices:\n{options}\n\n{suffix}"
            ),
        },
    ]


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
