"""Configuration loading. YAML is parsed then validated before a run starts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "PolyAI/banking77"
    split: str = "test"
    text_field: str = "text"
    label_field: str = "label"
    limit: int | None = Field(default=None, gt=0)
    data_files: dict[str, str] | None = None


class PricingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_million_usd: float = Field(default=0.0, ge=0)
    output_per_million_usd: float = Field(default=0.0, ge=0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["jev", "jev_openrouter", "mlx_qwen", "lmstudio_qwen"]
    name: str
    api_key_env: str = "TYPESAFE_API_KEY"
    base_url: HttpUrl | None = None
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_tokens: int = Field(default=12, gt=0)
    temperature: float = Field(default=0.0, ge=0)
    pricing: PricingConfig = Field(default_factory=PricingConfig)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Path("results")
    seed: int = 42
    dry_run: bool = True
    persist_raw_responses: bool = False


class BenchmarkConfig(BaseModel):
    """The entire declarative contract of a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig
    run: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode="after")
    def require_url_for_jev(self) -> BenchmarkConfig:
        if self.model.kind == "jev" and self.model.base_url is None:
            # The adapter also falls back to the environment; this makes configs explicit.
            self.model.base_url = "https://api.typesafe.ai"  # type: ignore[assignment]
        if self.model.kind == "jev_openrouter" and self.model.base_url is None:
            self.model.base_url = "https://openrouter.ai/api/alpha"  # type: ignore[assignment]
        return self


def load_config(path: Path) -> BenchmarkConfig:
    """Parse and validate one YAML config, with useful errors from Pydantic."""
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("The YAML root must be a mapping.")
    return BenchmarkConfig.model_validate(data)
