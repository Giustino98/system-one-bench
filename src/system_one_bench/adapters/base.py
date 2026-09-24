"""Shared model adapter contract."""

from typing import Protocol

from system_one_bench.domain import ChoiceExample, Prediction


class ModelAdapter(Protocol):
    @property
    def model_name(self) -> str: ...

    def predict(self, example: ChoiceExample) -> Prediction: ...

    def close(self) -> None: ...
