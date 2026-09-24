UV ?= uv
QWEN_CONFIG ?= configs/qwen3-14b-mlx.yaml
JEV_CONFIG ?= configs/jev-1.13-openrouter.yaml
ENV_FILE ?= .env
RESUME_DIR ?=

.DEFAULT_GOAL := help

.PHONY: help setup test lint format-check typecheck check plan-qwen run-qwen resume-qwen \
	plan-jev run-jev

help: ## Show available commands.
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install development and MLX dependencies with uv.
	$(UV) sync --extra dev --extra local

test: ## Run unit tests.
	$(UV) run pytest

lint: ## Run Ruff lint checks.
	$(UV) run ruff check .

format-check: ## Check formatting.
	$(UV) run ruff format --check .

typecheck: ## Run strict mypy checks.
	$(UV) run mypy src

check: lint format-check typecheck test ## Run every local quality check.

plan-qwen: ## Validate and print the Qwen plan without inference.
	$(UV) run system-one-bench plan $(QWEN_CONFIG)

run-qwen: ## Run Qwen after setting dry_run: false.
	$(UV) run system-one-bench run $(QWEN_CONFIG)

resume-qwen: ## Resume Qwen with RESUME_DIR=results/<run-id>.
	@test -n "$(RESUME_DIR)" || { echo "Set RESUME_DIR=results/<run-id>."; exit 1; }
	$(UV) run system-one-bench run $(QWEN_CONFIG) --resume $(RESUME_DIR)

plan-jev: ## Validate and print the Jev plan without API calls.
	$(UV) run system-one-bench plan $(JEV_CONFIG)

run-jev: ## Run Jev after setting dry_run: false and configuring .env.
	@test -f $(ENV_FILE) || { echo "Missing $(ENV_FILE). Copy .env.example first."; exit 1; }
	$(UV) run --env-file $(ENV_FILE) system-one-bench run $(JEV_CONFIG)
