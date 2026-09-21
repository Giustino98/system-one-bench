"""Common adapter contract: every model receives exactly the same task."""

from __future__ import annotations

from typing import Protocol

from system_one_bench.domain import ChoiceExample, Prediction


class ClassifierAdapter(Protocol):
    """Synchronous model interface with choices carried by each example."""

    @property
    def model_name(self) -> str: ...

    def classify(self, example: ChoiceExample) -> Prediction: ...

    def close(self) -> None: ...
