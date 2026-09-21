from pathlib import Path

from system_one_bench.config import load_config
from system_one_bench.runner import dry_run_plan


def test_banking_config_remains_valid_and_safe() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "banking77-jev.yaml")

    assert config.dataset.kind == "classification"
    assert config.run.dry_run is True
    assert dry_run_plan(config)["will_call_model"] is False


def test_bbh_configs_are_safe_and_enable_native_qwen_thinking() -> None:
    root = Path(__file__).parents[1]
    jev = load_config(root / "configs" / "bbh-logical-deduction-jev-openrouter.yaml")
    qwen = load_config(root / "configs" / "bbh-logical-deduction-qwen3-14b.yaml")

    assert jev.model.name == "typesafe/jev-1.13"
    assert qwen.model.name == "mlx-community/Qwen3-14B-4bit"
    assert qwen.model.thinking is True
    assert jev.run.dry_run is qwen.run.dry_run is True
    assert jev.dataset.tasks == qwen.dataset.tasks
    assert jev.dataset.instruction == qwen.dataset.instruction
