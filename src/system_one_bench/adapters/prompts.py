"""Model-neutral rendering of a shared single-choice benchmark problem."""

from __future__ import annotations

from system_one_bench.domain import ChoiceExample


def build_choice_prompt(example: ChoiceExample, *, answer_instruction: str | None = None) -> str:
    """Render the task, problem, and alternatives with a model-specific answer contract."""
    options = "\n".join(f"({key}) {text}" for key, text in example.choices.items())
    default_instruction = (
        f"Return the final line exactly as {example.answer_prefix} <choice>."
        if example.answer_prefix
        else "Reply with exactly one allowed choice key."
    )
    suffix = answer_instruction or default_instruction
    return f"{example.instruction}\n\nProblem:\n{example.text}\n\nChoices:\n{options}\n\n{suffix}"
