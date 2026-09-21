"""HTTP adapters for TypeSafe AI's System One / Jev Choice primitive."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from system_one_bench.config import ModelConfig
from system_one_bench.domain import ChoiceExample, Prediction


class _BaseJevAdapter:
    """Shared Decisions transport for direct TypeSafe and OpenRouter."""

    expected_kind: str
    endpoint_path: str

    def __init__(self, config: ModelConfig) -> None:
        if config.kind != self.expected_kind:
            raise ValueError(
                f"{type(self).__name__} requires a {self.expected_kind!r} model configuration."
            )
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
        payload = _build_payload(example, self._config.name)
        start = time.perf_counter()
        response = self._post_with_retry(api_key, payload)
        body: dict[str, Any] = response.json()
        latency_ms = (time.perf_counter() - start) * 1_000
        selected, probabilities = _parse_choice(body, example.choice_keys)
        usage = body.get("usage", {})
        if not isinstance(usage, Mapping):
            usage = {}
        return Prediction(
            predicted_choice=selected,
            probabilities=probabilities,
            latency_ms=latency_ms,
            input_tokens=_integer_or_none(usage.get("input_tokens")),
            output_tokens=_integer_or_none(usage.get("output_tokens")),
            cost_usd=_float_or_none(usage.get("cost")),
            raw_response=body,
        )

    def close(self) -> None:
        self._client.close()

    def _post_with_retry(self, api_key: str, payload: dict[str, Any]) -> httpx.Response:
        endpoint = f"{str(self._config.base_url).rstrip('/')}/{self.endpoint_path}"
        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            except httpx.TransportError:
                if attempt >= self._config.max_retries:
                    raise
                time.sleep(min(2**attempt, 8))
                continue
            if response.status_code not in {429, 529} or attempt >= self._config.max_retries:
                response.raise_for_status()
                return response
            retry_after = response.headers.get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
            time.sleep(min(delay, 10))
        raise RuntimeError("Unreachable retry state.")


class JevAdapter(_BaseJevAdapter):
    """Call Jev through TypeSafe's direct System One endpoint."""

    expected_kind = "jev"
    endpoint_path = "v1/systemone"


class OpenRouterJevAdapter(_BaseJevAdapter):
    """Call Jev through OpenRouter's alpha Decisions endpoint."""

    expected_kind = "jev_openrouter"
    endpoint_path = "decisions"


def _build_payload(example: ChoiceExample, model_name: str) -> dict[str, Any]:
    """Map arbitrary single-choice examples to Jev's typed Choice primitive."""
    return {
        "model": model_name,
        "state": {"problem": example.text},
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": example.instruction,
                "criteria": dict(example.choices),
            }
        },
    }


def _parse_choice(
    body: Mapping[str, Any], choices: Sequence[str]
) -> tuple[str, dict[str, float] | None]:
    """Extract an answer without silently accepting keys outside the benchmark."""
    answer: Any = body.get("answers", {}).get("answer")
    if answer is None:
        answer = body.get("choices", {}).get("answer")
    if isinstance(answer, str):
        selected = answer
        probabilities = None
    elif isinstance(answer, Mapping):
        selected = answer.get("choice", answer.get("value", answer.get("selected")))
        probabilities = _normalise_probabilities(answer.get("probabilities"), choices)
    else:
        raise ValueError("Jev response did not include an 'answer' Choice response.")
    if not isinstance(selected, str) or selected not in choices:
        raise ValueError(f"Jev returned invalid benchmark choice: {selected!r}")
    return selected, probabilities


def _normalise_probabilities(raw: Any, choices: Sequence[str]) -> dict[str, float] | None:
    if not isinstance(raw, Mapping):
        return None
    values = {choice: float(raw[choice]) for choice in choices if choice in raw}
    total = sum(values.values())
    if total <= 0:
        return None
    return {choice: value / total for choice, value in values.items()}


def _integer_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None
