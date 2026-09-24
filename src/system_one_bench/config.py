"""Configuration loading. YAML is parsed then validated before a run starts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["classification", "bbh_logical_deduction"] = "classification"
    name: str = "PolyAI/banking77"
    revision: str | None = None
    split: str = "test"
    tasks: tuple[str, ...] = ()
    text_field: str = "text"
    label_field: str = "label"
    instruction: str | None = None
    limit: int | None = Field(default=None, gt=0)
    data_files: dict[str, str] | None = None


class PricingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_million_usd: float = Field(default=0.0, ge=0)
    output_per_million_usd: float = Field(default=0.0, ge=0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["gemini", "jev", "jev_openrouter", "mlx_qwen", "lmstudio_qwen"]
    name: str
    model_path: Path | None = None
    api_model: str | None = None
    lmstudio_api: Literal["openai", "native"] = "openai"
    structured_output: bool = False
    api_key_env: str = "TYPESAFE_API_KEY"
    base_url: HttpUrl | None = None
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_tokens: int = Field(default=12, gt=0)
    temperature: float = Field(default=0.0, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    thinking: bool = False
    thinking_level: Literal["low", "medium", "high"] | None = None
    seed: int | None = None
    pricing: PricingConfig = Field(default_factory=PricingConfig)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Path("results")
    seed: int = 42
    dry_run: bool = True
    persist_raw_responses: bool = False
    continue_on_error: bool = False


class BenchmarkConfig(BaseModel):
    """The entire declarative contract of a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig
    run: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode="after")
    def validate_benchmark(self) -> BenchmarkConfig:
        if self.model.kind == "jev" and self.model.base_url is None:
            self.model.base_url = "https://api.typesafe.ai"  # type: ignore[assignment]
        if self.model.kind == "jev_openrouter" and self.model.base_url is None:
            self.model.base_url = "https://openrouter.ai/api/alpha"  # type: ignore[assignment]
        if (
            self.model.kind == "lmstudio_qwen"
            and self.model.structured_output
            and self.model.lmstudio_api != "openai"
        ):
            raise ValueError("LM Studio structured output requires the OpenAI-compatible API.")
        if self.dataset.kind == "bbh_logical_deduction" and not self.dataset.tasks:
            raise ValueError("BBH logical deduction requires at least one dataset task.")
        return self


def load_config(path: Path) -> BenchmarkConfig:
    """Parse and validate one YAML config, with useful errors from Pydantic."""
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("The YAML root must be a mapping.")
    return BenchmarkConfig.model_validate(data)
