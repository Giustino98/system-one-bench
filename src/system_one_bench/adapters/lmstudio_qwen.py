"""Local Qwen adapter using LM Studio's OpenAI-compatible server."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

import httpx

from system_one_bench.adapters.mlx_qwen import _extract_choice
from system_one_bench.adapters.prompts import build_choice_prompt
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
            "messages": _messages(
                example,
                thinking=self._config.thinking,
                structured=self._config.structured_output,
            ),
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "stream": False,
        }
        if self._config.structured_output:
            payload["response_format"] = _structured_response_format(example)
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
        selected = (
            _parse_structured_choice(body, example)
            if self._config.structured_output
            else _parse_choice(body, example)
        )
        usage = body.get("usage")
        if not isinstance(usage, Mapping):
            usage = {}
        return Prediction(
            predicted_choice=selected,
            probabilities=None,
            latency_ms=latency_ms,
            input_tokens=_integer_or_none(usage.get("prompt_tokens")),
            output_tokens=_integer_or_none(usage.get("completion_tokens")),
            reasoning=_parse_openai_reasoning(body),
            finish_reason=_parse_finish_reason(body),
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
            finish_reason=_string_or_none(stats.get("stop_reason")),
            raw_response=body,
        )

    def close(self) -> None:
        self._client.close()


def _messages(example: ChoiceExample, *, thinking: bool, structured: bool) -> list[dict[str, str]]:
    thinking_switch = "/think" if thinking else "/no_think"
    return [
        {
            "role": "system",
            "content": f"Solve the single-choice problem. {thinking_switch}",
        },
        {"role": "user", "content": _user_prompt(example, structured=structured)},
    ]


def _user_prompt(example: ChoiceExample, *, structured: bool = False) -> str:
    answer_instruction = (
        "Return only the choice using the required response schema." if structured else None
    )
    return build_choice_prompt(example, answer_instruction=answer_instruction)


def _structured_response_format(example: ChoiceExample) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "benchmark_choice",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"choice": {"type": "string", "enum": list(example.choice_keys)}},
                "required": ["choice"],
                "additionalProperties": False,
            },
        },
    }


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
    message = _completion_message(body)
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("LM Studio assistant message did not contain text.")
    return _extract_choice(content, example)


def _parse_structured_choice(body: Mapping[str, Any], example: ChoiceExample) -> str:
    message = _completion_message(body)
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("LM Studio structured response did not contain text.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("LM Studio structured response was not valid JSON.") from error
    if not isinstance(parsed, dict) or set(parsed) != {"choice"}:
        raise ValueError("LM Studio structured response must contain only 'choice'.")
    selected = parsed["choice"]
    if not isinstance(selected, str) or selected not in example.choices:
        raise ValueError("LM Studio structured response contained an invalid choice.")
    return selected


def _completion_message(body: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LM Studio response did not include a completion choice.")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("LM Studio completion choice has an invalid shape.")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("LM Studio response did not include an assistant message.")
    return message


def _parse_openai_reasoning(body: Mapping[str, Any]) -> str | None:
    message = _completion_message(body)
    for key in ("reasoning", "reasoning_content"):
        reasoning = message.get(key)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()
    return None


def _parse_finish_reason(body: Mapping[str, Any]) -> str | None:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    return _string_or_none(choices[0].get("finish_reason"))


def _integer_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None
