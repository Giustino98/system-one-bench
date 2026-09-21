"""Common adapter contract: every model receives exactly the same task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from system_one_bench.domain import ClassificationExample, Prediction


class ClassifierAdapter(Protocol):
    """Synchronous model interface, deliberately small for faithful comparisons."""

    @property
    def model_name(self) -> str: ...

    def classify(self, example: ClassificationExample, labels: Sequence[str]) -> Prediction: ...

    def close(self) -> None: ...
