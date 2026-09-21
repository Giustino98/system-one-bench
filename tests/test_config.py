from pathlib import Path

from system_one_bench.config import load_config
from system_one_bench.runner import dry_run_plan


def test_example_config_is_valid_and_safe() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "banking77-jev.yaml")

    assert config.run.dry_run is True
    assert dry_run_plan(config)["will_call_model"] is False


def test_openrouter_config_is_valid_and_safe() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "banking77-jev-openrouter.yaml")

    assert config.model.kind == "jev_openrouter"
    assert config.model.api_key_env == "OPENROUTER_API_KEY"
    assert config.run.dry_run is True
    assert config.dataset.data_files is not None
    assert config.dataset.data_files["test"].endswith(".parquet")
