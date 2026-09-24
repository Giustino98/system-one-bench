UV ?= uv
LOCAL_CONFIG ?= configs/banking77-mlx-qwen.yaml
JEV_CONFIG ?= configs/banking77-jev-openrouter.yaml
BBH_QWEN_CONFIG ?= configs/bbh-logical-deduction-qwen3-14b.yaml
BBH_JEV_CONFIG ?= configs/bbh-logical-deduction-jev-openrouter.yaml
BBH_GEMINI_CONFIG ?= configs/bbh-logical-deduction-gemini-3.8-flash.yaml
ENV_FILE ?= .env
RESUME_DIR ?=

.DEFAULT_GOAL := help

.PHONY: help setup test lint format-check typecheck check validate-local plan-local \
	run-local validate-jev plan-jev run-jev plan-bbh-qwen run-bbh-qwen plan-bbh-jev run-bbh-jev \
	plan-bbh-gemini run-bbh-gemini resume-bbh-qwen

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
	@test -f $(ENV_FILE) || { echo "Missing $(ENV_FILE). Copy .env.example and add the required API key."; exit 1; }
	$(UV) run --env-file $(ENV_FILE) system-one-bench run $(JEV_CONFIG)

plan-bbh-qwen: ## Validate and print the safe Qwen3 BBH smoke-test plan.
	$(UV) run system-one-bench validate-config $(BBH_QWEN_CONFIG)
	$(UV) run system-one-bench plan $(BBH_QWEN_CONFIG)

run-bbh-qwen: ## Run Qwen3 BBH locally (YAML must have dry_run: false).
	$(UV) run system-one-bench run $(BBH_QWEN_CONFIG)

resume-bbh-qwen: ## Resume Qwen BBH; pass RESUME_DIR=results/<run-id>.
	@test -n "$(RESUME_DIR)" || { echo "Set RESUME_DIR=results/<run-id>."; exit 1; }
	$(UV) run system-one-bench run $(BBH_QWEN_CONFIG) --resume $(RESUME_DIR)

plan-bbh-jev: ## Validate and print the safe Jev BBH smoke-test plan.
	$(UV) run system-one-bench validate-config $(BBH_JEV_CONFIG)
	$(UV) run system-one-bench plan $(BBH_JEV_CONFIG)

run-bbh-jev: ## Run Jev BBH (requires a key and dry_run: false).
	@test -f $(ENV_FILE) || { echo "Missing $(ENV_FILE). Copy .env.example and add the required API key."; exit 1; }
	$(UV) run --env-file $(ENV_FILE) system-one-bench run $(BBH_JEV_CONFIG)

plan-bbh-gemini: ## Validate and print the safe Gemini 3.8 Flash BBH smoke-test plan.
	$(UV) run system-one-bench validate-config $(BBH_GEMINI_CONFIG)
	$(UV) run system-one-bench plan $(BBH_GEMINI_CONFIG)

run-bbh-gemini: ## Run Gemini 3.8 Flash BBH (requires a key and dry_run: false).
	@test -f $(ENV_FILE) || { echo "Missing $(ENV_FILE). Copy .env.example and add the required API key."; exit 1; }
	$(UV) run --env-file $(ENV_FILE) system-one-bench run $(BBH_GEMINI_CONFIG)
