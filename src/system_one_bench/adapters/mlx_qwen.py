"""Local Apple-Silicon baseline using MLX-LM and a Qwen instruct checkpoint."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from system_one_bench.config import ModelConfig
from system_one_bench.domain import ClassificationExample, Prediction


class MlxQwenAdapter:
    """Generate one exact label locally. Import MLX lazily so setup remains lightweight."""

    def __init__(self, config: ModelConfig) -> None:
        if config.kind != "mlx_qwen":
            raise ValueError("MlxQwenAdapter requires an 'mlx_qwen' model configuration.")
        self._config = config
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    @property
    def model_name(self) -> str:
        return self._config.name

    def classify(self, example: ClassificationExample, labels: Sequence[str]) -> Prediction:
        model, tokenizer = self._load()
        prompt = _prompt(tokenizer, example.text, labels)
        from mlx_lm import generate

        generation_options: dict[str, Any] = {}
        if self._config.temperature > 0:
            from mlx_lm.sample_utils import make_sampler

            generation_options["sampler"] = make_sampler(temp=self._config.temperature)

        start = time.perf_counter()
        answer = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=self._config.max_tokens,
            verbose=False,
            **generation_options,
        )
        latency_ms = (time.perf_counter() - start) * 1_000
        selected = _extract_label(answer, labels)
        return Prediction(predicted_label=selected, probabilities=None, latency_ms=latency_ms)

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            try:
                from mlx_lm import load
            except ImportError as error:
                raise RuntimeError(
                    "Install the local extra first: uv sync --extra local"
                ) from error
            loaded = load(self._config.name)
            self._model = loaded[0]
            self._tokenizer = loaded[1]
        return self._model, self._tokenizer

    def close(self) -> None:
        """MLX owns no external connection; the method satisfies the adapter protocol."""


def _prompt(tokenizer: Any, text: str, labels: Sequence[str]) -> str:
    options = "\n".join(f"- {label}" for label in labels)
    messages = [
        {
            "role": "system",
            "content": "Classify banking messages. Reply with exactly one allowed label.",
        },
        {
            "role": "user",
            "content": f"Message: {text}\n\nAllowed labels:\n{options}",
        },
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError("The tokenizer chat template did not return text.")
    return rendered


def _extract_label(answer: str, labels: Sequence[str]) -> str:
    normalised = answer.strip().lower().replace(" ", "_").replace("-", "_")
    for label in labels:
        if normalised == label or normalised.startswith(f"{label}\n"):
            return label
    raise ValueError(f"Local model returned an invalid label: {answer!r}")
