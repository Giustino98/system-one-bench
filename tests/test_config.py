from pathlib import Path

import pytest
from pydantic import ValidationError

from system_one_bench.config import JevConfig, QwenConfig, load_config

ROOT = Path(__file__).parents[1]


def test_configs_share_the_exact_dataset_protocol() -> None:
    jev = load_config(ROOT / "configs" / "jev-1.13-openrouter.yaml")
    qwen = load_config(ROOT / "configs" / "qwen3-14b-mlx.yaml")

    assert isinstance(jev.model, JevConfig)
    assert isinstance(qwen.model, QwenConfig)
    assert jev.dataset == qwen.dataset
    assert jev.run.dry_run is True
    assert qwen.run.dry_run is True
    assert qwen.model.max_tokens == 8192


def test_unknown_configuration_fields_fail_during_parsing(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("model:\n  kind: mlx_qwen\n  unsupported: true\n")

    with pytest.raises(ValidationError):
        load_config(config)
