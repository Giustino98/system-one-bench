"""Jev adapter for OpenRouter's Decisions API."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

import httpx

from system_one_bench.config import JevConfig
from system_one_bench.domain import ChoiceExample, Prediction


class OpenRouterJevAdapter:
    def __init__(self, config: JevConfig) -> None:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing required environment variable: {config.api_key_env}")
        self._config = config
        self._api_key = api_key
        self._client = httpx.Client(timeout=config.timeout_seconds)

    @property
    def model_name(self) -> str:
        return self._config.name

    def predict(self, example: ChoiceExample) -> Prediction:
        started_at = time.perf_counter()
        response = self._post_with_retry(build_payload(example, self._config.name))
        body = response.json()
        if not isinstance(body, dict):
            raise TypeError("OpenRouter returned a non-object response.")
        selected, probabilities = parse_choice(body, example)
        input_tokens, output_tokens, cost_usd = parse_usage(body.get("usage"))
        return Prediction(
            predicted_choice=selected,
            probabilities=probabilities,
            latency_ms=(time.perf_counter() - started_at) * 1_000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            raw_response=body,
        )

    def close(self) -> None:
        self._client.close()

    def _post_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        endpoint = f"{str(self._config.base_url).rstrip('/')}/decisions"
        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            except httpx.TransportError:
                if attempt == self._config.max_retries:
                    raise
                time.sleep(min(2**attempt, 8))
                continue
            if response.status_code not in {429, 529}:
                response.raise_for_status()
                return response
            if attempt == self._config.max_retries:
                response.raise_for_status()
            time.sleep(retry_delay(response, attempt))
        raise AssertionError("retry loop exhausted")


def build_payload(example: ChoiceExample, model_name: str) -> dict[str, Any]:
    """Represent the same problem and alternatives as a Jev Choice question."""
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


def parse_choice(
    body: Mapping[str, Any], example: ChoiceExample
) -> tuple[str, dict[str, float] | None]:
    """Parse the documented Choice response shape and fail on protocol drift."""
    answers = body["answers"]
    if not isinstance(answers, Mapping):
        raise TypeError("Jev answers must be an object.")
    answer = answers["answer"]
    if isinstance(answer, str):
        selected = answer
        probabilities = None
    elif isinstance(answer, Mapping):
        selected = answer["choice"]
        probabilities = parse_probabilities(answer.get("probabilities"), example)
    else:
        raise TypeError("Jev answer must be a choice string or object.")
    if not isinstance(selected, str) or selected not in example.choices:
        raise ValueError(f"Jev returned an unknown choice: {selected!r}")
    return selected, probabilities


def parse_probabilities(value: object, example: ChoiceExample) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("Jev probabilities must be an object.")
    probabilities = {choice: float(value[choice]) for choice in example.choice_keys}
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError("Jev probabilities must have a positive sum.")
    return {choice: probability / total for choice, probability in probabilities.items()}


def parse_usage(value: object) -> tuple[int | None, int | None, float | None]:
    if value is None:
        return None, None, None
    if not isinstance(value, Mapping):
        raise TypeError("OpenRouter usage must be an object.")
    input_tokens = value.get("input_tokens")
    output_tokens = value.get("output_tokens")
    cost = value.get("cost")
    return (
        input_tokens if isinstance(input_tokens, int) else None,
        output_tokens if isinstance(output_tokens, int) else None,
        float(cost) if isinstance(cost, int | float) else None,
    )


def retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after and retry_after.isdigit():
        return min(float(retry_after), 10)
    return float(min(2**attempt, 10))
