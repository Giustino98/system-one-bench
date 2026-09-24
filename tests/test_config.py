from pathlib import Path

from system_one_bench.config import load_config
from system_one_bench.runner import dry_run_plan


def test_banking_config_remains_valid_and_safe() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "banking77-jev.yaml")

    assert config.dataset.kind == "classification"
    assert config.run.dry_run is True
    assert dry_run_plan(config)["will_call_model"] is False


def test_bbh_configs_enable_structured_qwen_thinking() -> None:
    root = Path(__file__).parents[1]
    jev = load_config(root / "configs" / "bbh-logical-deduction-jev-openrouter.yaml")
    qwen = load_config(root / "configs" / "bbh-logical-deduction-qwen3-14b.yaml")

    assert jev.model.name == "typesafe/jev-1.13"
    assert qwen.model.name == "lmstudio-community/Qwen3-14B-MLX-4bit"
    assert qwen.model.kind == "mlx_qwen"
    assert qwen.model.model_path is not None
    assert str(qwen.model.model_path).endswith("Qwen3-14B-MLX-4bit")
    assert qwen.model.structured_output is True
    assert qwen.model.thinking is True
    assert qwen.model.max_tokens == 8192
    assert qwen.run.continue_on_error is True
    assert qwen.run.dry_run is False
    assert jev.dataset.tasks == qwen.dataset.tasks
    assert jev.dataset.instruction == qwen.dataset.instruction


def test_gemini_bbh_config_matches_the_shared_protocol() -> None:
    root = Path(__file__).parents[1]
    gemini = load_config(root / "configs" / "bbh-logical-deduction-gemini-3.8-flash.yaml")
    qwen = load_config(root / "configs" / "bbh-logical-deduction-qwen3-14b.yaml")

    assert gemini.model.kind == "gemini"
    assert gemini.model.name == "gemini-3.8-flash"
    assert gemini.model.api_key_env == "GEMINI_API_KEY"
    assert gemini.model.thinking is True
    assert gemini.model.thinking_level == "medium"
    assert gemini.dataset.tasks == qwen.dataset.tasks
    assert gemini.dataset.instruction == qwen.dataset.instruction
