"""Local Apple-Silicon adapter using MLX-LM and Qwen chat templates."""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable, Iterable
from typing import Any

from system_one_bench.config import QwenConfig
from system_one_bench.domain import ChoiceExample, Prediction


class MlxQwenAdapter:
    """Generate one canonical choice locally; load MLX lazily for lightweight tests."""

    def __init__(self, config: QwenConfig) -> None:
        self._config = config
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._seeded = False

    @property
    def model_name(self) -> str:
        return self._config.name

    def predict(self, example: ChoiceExample) -> Prediction:
        model, tokenizer = self._load()
        prompt = build_prompt(tokenizer, example)
        from mlx_lm import stream_generate

        if not self._seeded:
            import mlx.core as mx

            mx.random.seed(self._config.seed)
            self._seeded = True

        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(
            temp=self._config.temperature,
            top_p=self._config.top_p,
            top_k=self._config.top_k,
        )
        logits_processors = [_post_thinking_json_processor(tokenizer, example)]

        start = time.perf_counter()
        output_parts: list[str] = []
        last_response: Any | None = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=self._config.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            output_parts.append(response.text)
            last_response = response
        latency_ms = (time.perf_counter() - start) * 1_000
        if last_response is None:
            raise RuntimeError("MLX generation returned no response.")
        output = "".join(output_parts)
        reasoning, final_answer = split_thinking_output(output)
        selected = parse_final_choice(final_answer, example)
        return Prediction(
            predicted_choice=selected,
            probabilities=None,
            latency_ms=latency_ms,
            input_tokens=int(last_response.prompt_tokens),
            output_tokens=int(last_response.generation_tokens),
            reasoning=reasoning,
            finish_reason=last_response.finish_reason,
            raw_response={
                "output": output,
                "final_answer": final_answer,
                "generation_tokens": int(last_response.generation_tokens),
                "generation_tokens_per_second": float(last_response.generation_tps),
                "peak_memory_gb": float(last_response.peak_memory),
            },
        )

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            try:
                from mlx_lm import load
            except ImportError as error:
                raise RuntimeError(
                    "Install the local extra first: uv sync --extra local"
                ) from error
            model_reference = self._config.name
            if self._config.model_path is not None:
                local_path = self._config.model_path.expanduser()
                if not local_path.is_dir():
                    raise FileNotFoundError(f"MLX model directory not found: {local_path}")
                model_reference = str(local_path)
            loaded = load(model_reference)
            self._model = loaded[0]
            self._tokenizer = loaded[1]
            self._tokenizer.eos_token_ids = set(_eos_token_ids(self._tokenizer))
        return self._model, self._tokenizer

    def close(self) -> None:
        """Release references and cached Metal buffers after the benchmark."""
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx
        except ImportError:
            return
        mx.clear_cache()


def build_prompt(tokenizer: Any, example: ChoiceExample) -> str:
    """Render the shared problem with Qwen's native thinking mode enabled."""
    options = "\n".join(f"({key}) {text}" for key, text in example.choices.items())
    messages = [
        {"role": "system", "content": "Solve the single-choice problem."},
        {
            "role": "user",
            "content": (
                f"{example.instruction}\n\nProblem:\n{example.text}\n\n"
                f'Choices:\n{options}\n\nAfter thinking, return only {{"choice":"<choice>"}}.'
            ),
        },
    ]
    template_options: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }
    rendered = tokenizer.apply_chat_template(messages, **template_options)
    if not isinstance(rendered, str):
        raise TypeError("The tokenizer chat template did not return text.")
    return rendered


def split_thinking_output(output: str) -> tuple[str | None, str]:
    """Separate Qwen3's native think block from the answer used for scoring."""
    if "</think>" not in output:
        return None, output.strip()
    reasoning, final_answer = output.split("</think>", maxsplit=1)
    reasoning = reasoning.removeprefix("<think>").strip()
    return reasoning or None, final_answer.strip()


def parse_final_choice(answer: str, example: ChoiceExample) -> str | None:
    """Parse the constrained final JSON; an unfinished generation is an invalid output."""
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"choice"}:
        return None
    choice = payload["choice"]
    return choice if isinstance(choice, str) and choice in example.choices else None


def _post_thinking_json_processor(
    tokenizer: Any,
    example: ChoiceExample,
) -> Callable[[Any, Any], Any]:
    """Constrain only the post-`</think>` suffix to one canonical JSON object."""
    think_end = tuple(tokenizer.encode("</think>", add_special_tokens=False))
    if not think_end:
        raise ValueError("Tokenizer could not encode the Qwen thinking terminator.")
    suffixes = tuple(
        _encode_suffix_after_marker(
            tokenizer,
            think_end,
            f'\n{{"choice":"{choice}"}}',
        )
        for choice in example.choice_keys
    )
    eos_tokens = _eos_token_ids(tokenizer)

    def processor(tokens: Any, logits: Any) -> Any:
        allowed = _allowed_post_thinking_tokens(
            tuple(int(token) for token in tokens.tolist()),
            think_end,
            suffixes,
            eos_tokens,
        )
        if allowed is None:
            return logits
        if not allowed:
            raise ValueError("Generated post-thinking output left the structured choice grammar.")

        import mlx.core as mx

        vocabulary = mx.arange(logits.shape[-1])
        mask = vocabulary == allowed[0]
        for token_id in allowed[1:]:
            mask = mx.logical_or(mask, vocabulary == token_id)
        return mx.where(mask, logits, mx.full_like(logits, -float("inf")))

    return processor


def _encode_suffix_after_marker(
    tokenizer: Any,
    marker: tuple[int, ...],
    suffix: str,
) -> tuple[int, ...]:
    combined = tuple(tokenizer.encode(f"</think>{suffix}", add_special_tokens=False))
    if combined[: len(marker)] == marker:
        return combined[len(marker) :]
    return tuple(tokenizer.encode(suffix, add_special_tokens=False))


def _eos_token_ids(tokenizer: Any) -> tuple[int, ...]:
    raw_ids = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(raw_ids, int):
        return (raw_ids,)
    if isinstance(raw_ids, Iterable) and not isinstance(raw_ids, (str, bytes)):
        ids = tuple(int(token_id) for token_id in raw_ids if token_id is not None)
        if ids:
            return ids
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        return (eos_token_id,)
    eos_token = getattr(tokenizer, "eos_token", None)
    if isinstance(eos_token, str):
        converted = tokenizer.convert_tokens_to_ids(eos_token)
        if isinstance(converted, int) and converted >= 0:
            return (converted,)
    raise ValueError("Tokenizer does not expose an EOS token id.")


def _allowed_post_thinking_tokens(
    tokens: tuple[int, ...],
    marker: tuple[int, ...],
    suffixes: tuple[tuple[int, ...], ...],
    eos_tokens: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Return allowed next tokens, or None while the model is still reasoning."""
    marker_start = _last_subsequence_start(tokens, marker)
    if marker_start is None:
        return None
    generated_suffix = tokens[marker_start + len(marker) :]
    for suffix in suffixes:
        trailing = generated_suffix[len(suffix) :]
        if (
            generated_suffix[: len(suffix)] == suffix
            and trailing
            and all(token in eos_tokens for token in trailing)
        ):
            return eos_tokens
    candidates = tuple(
        suffix for suffix in suffixes if suffix[: len(generated_suffix)] == generated_suffix
    )
    if not candidates:
        return ()
    if any(len(candidate) == len(generated_suffix) for candidate in candidates):
        return eos_tokens
    return tuple(dict.fromkeys(candidate[len(generated_suffix)] for candidate in candidates))


def _last_subsequence_start(
    values: tuple[int, ...],
    needle: tuple[int, ...],
) -> int | None:
    for start in range(len(values) - len(needle), -1, -1):
        if values[start : start + len(needle)] == needle:
            return start
    return None
