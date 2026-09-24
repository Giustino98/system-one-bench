"""Typed configuration parsed once at the CLI boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

LogicalDeductionTask = Literal[
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(StrictModel):
    name: str = "lighteval/bbh"
    revision: str = "1b61f099fcbf9e55691ef8cc6b4b8fb431dae097"
    split: str = "train"
    tasks: tuple[LogicalDeductionTask, ...] = Field(
        default=(
            "logical_deduction_three_objects",
            "logical_deduction_five_objects",
            "logical_deduction_seven_objects",
        ),
        min_length=1,
    )
    instruction: str = "Choose the option that follows from the problem."
    limit: int | None = Field(default=None, gt=0)


class PricingConfig(StrictModel):
    input_per_million_usd: float = Field(default=0.0, ge=0)
    output_per_million_usd: float = Field(default=0.0, ge=0)


class JevConfig(StrictModel):
    kind: Literal["jev_openrouter"]
    name: str = "typesafe/jev-1.13"
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: HttpUrl = HttpUrl("https://openrouter.ai/api/alpha")
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    pricing: PricingConfig = Field(default_factory=PricingConfig)


class QwenConfig(StrictModel):
    kind: Literal["mlx_qwen"]
    name: str = "lmstudio-community/Qwen3-14B-MLX-4bit"
    revision: str = "b5d17e319ff9734f059b42b8b1f0834932bbb12c"
    model_path: Path | None = None
    max_tokens: int = Field(default=8192, gt=0)
    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=0.95, gt=0, le=1)
    top_k: int = Field(default=20, ge=0)
    seed: int = 42
    pricing: PricingConfig = Field(default_factory=PricingConfig)


ModelConfig = Annotated[JevConfig | QwenConfig, Field(discriminator="kind")]


class RunConfig(StrictModel):
    output_dir: Path = Path("results")
    seed: int = 42
    dry_run: bool = True


class BenchmarkConfig(StrictModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig
    run: RunConfig = Field(default_factory=RunConfig)


def load_config(path: Path) -> BenchmarkConfig:
    """Parse YAML into one of the two supported benchmark configurations."""
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return BenchmarkConfig.model_validate(payload)
