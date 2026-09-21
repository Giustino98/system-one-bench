"""Local Apple-Silicon adapter using MLX-LM and Qwen chat templates."""

from __future__ import annotations

import re
import time
from typing import Any

from system_one_bench.config import ModelConfig
from system_one_bench.domain import ChoiceExample, Prediction


class MlxQwenAdapter:
    """Generate one canonical choice locally; load MLX lazily for lightweight tests."""

    def __init__(self, config: ModelConfig) -> None:
        if config.kind != "mlx_qwen":
            raise ValueError("MlxQwenAdapter requires an 'mlx_qwen' model configuration.")
        self._config = config
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._seeded = False

    @property
    def model_name(self) -> str:
        return self._config.name

    def classify(self, example: ChoiceExample) -> Prediction:
        model, tokenizer = self._load()
        prompt = _prompt(tokenizer, example, thinking=self._config.thinking)
        from mlx_lm import generate

        if not self._seeded and self._config.seed is not None:
            import mlx.core as mx

            mx.random.seed(self._config.seed)
            self._seeded = True

        generation_options: dict[str, Any] = {}
        if (
            self._config.temperature > 0
            or self._config.top_p is not None
            or self._config.top_k is not None
        ):
            from mlx_lm.sample_utils import make_sampler

            sampler_options: dict[str, Any] = {"temp": self._config.temperature}
            if self._config.top_p is not None:
                sampler_options["top_p"] = self._config.top_p
            if self._config.top_k is not None:
                sampler_options["top_k"] = self._config.top_k
            generation_options["sampler"] = make_sampler(**sampler_options)

        start = time.perf_counter()
        output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=self._config.max_tokens,
            verbose=False,
            **generation_options,
        )
        latency_ms = (time.perf_counter() - start) * 1_000
        reasoning, final_answer = _split_thinking_output(output)
        selected = _extract_choice(final_answer, example)
        return Prediction(
            predicted_choice=selected,
            probabilities=None,
            latency_ms=latency_ms,
            reasoning=reasoning,
            raw_response={"output": output, "final_answer": final_answer},
        )

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


def _prompt(tokenizer: Any, example: ChoiceExample, *, thinking: bool) -> str:
    options = "\n".join(f"({key}) {text}" for key, text in example.choices.items())
    format_instruction = (
        f"\n\nReturn the final line exactly as {example.answer_prefix} <choice>."
        if example.answer_prefix
        else "\n\nReply with exactly one allowed choice key."
    )
    messages = [
        {"role": "system", "content": "Solve the single-choice problem."},
        {
            "role": "user",
            "content": (
                f"{example.instruction}\n\nProblem:\n{example.text}\n\n"
                f"Choices:\n{options}{format_instruction}"
            ),
        },
    ]
    template_options: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": thinking,
    }
    rendered = tokenizer.apply_chat_template(messages, **template_options)
    if not isinstance(rendered, str):
        raise TypeError("The tokenizer chat template did not return text.")
    return rendered


def _split_thinking_output(output: str) -> tuple[str | None, str]:
    """Separate Qwen3's native think block from the answer used for scoring."""
    if "</think>" not in output:
        return None, output.strip()
    reasoning, final_answer = output.split("</think>", maxsplit=1)
    reasoning = reasoning.removeprefix("<think>").strip()
    return reasoning or None, final_answer.strip()


def _extract_choice(answer: str, example: ChoiceExample) -> str:
    """Parse one exact canonical key; BBH deliberately has no fuzzy fallback."""
    if example.answer_prefix:
        alternatives = "|".join(map(re.escape, example.choice_keys))
        pattern = re.compile(rf"^{re.escape(example.answer_prefix)}\s*({alternatives})$")
        last_line = answer.rstrip().splitlines()[-1] if answer.strip() else ""
        match = pattern.fullmatch(last_line)
        if match is not None:
            return match.group(1)
    else:
        normalized = answer.strip().lower().replace("\\_", "_").replace(" ", "_")
        for choice in example.choice_keys:
            if normalized == choice.lower():
                return choice
    return f"__invalid__:{answer.strip()}"
