UV ?= uv
LOCAL_CONFIG ?= configs/banking77-mlx-qwen.yaml
JEV_CONFIG ?= configs/banking77-jev-openrouter.yaml

.DEFAULT_GOAL := help

.PHONY: help setup test lint format-check typecheck check validate-local plan-local \
	run-local validate-jev plan-jev run-jev

help: ## Show the available commands.
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create/sync the uv environment with development and MLX dependencies.
	$(UV) sync --extra dev --extra local

test: ## Run the test suite.
	$(UV) run pytest

lint: ## Run Ruff lint checks.
	$(UV) run ruff check .

format-check: ## Check formatting without changing files.
	$(UV) run ruff format --check .

typecheck: ## Run strict static type checks.
	$(UV) run mypy src

check: lint format-check typecheck test ## Run all code-quality checks.

validate-local: ## Validate the local MLX experiment configuration.
	$(UV) run system-one-bench validate-config $(LOCAL_CONFIG)

plan-local: validate-local ## Print the local MLX experiment plan without running it.
	$(UV) run system-one-bench plan $(LOCAL_CONFIG)

run-local: ## Run the local MLX experiment (YAML must have dry_run: false).
	$(UV) run system-one-bench run $(LOCAL_CONFIG)

validate-jev: ## Validate the Jev/OpenRouter experiment configuration.
	$(UV) run system-one-bench validate-config $(JEV_CONFIG)

plan-jev: validate-jev ## Print the Jev experiment plan without making paid calls.
	$(UV) run system-one-bench plan $(JEV_CONFIG)

run-jev: ## Run the Jev experiment (requires a key and dry_run: false).
	$(UV) run system-one-bench run $(JEV_CONFIG)
