# Standard dev commands (noirdoc engineering standard). Run `make help`.
.DEFAULT_GOAL := help

.PHONY: help install lint fmt fmt-check typecheck test test-slow check models

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Set up the dev environment
	uv sync --extra dev

lint: ## Lint (ruff)
	uv run ruff check .

fmt: ## Auto-format (ruff)
	uv run ruff format .

fmt-check: ## Check formatting (ruff)
	uv run ruff format --check .

typecheck: ## Type-check (mypy, strict)
	uv run mypy .

test: ## Run fast tests (excludes slow ML-model tests)
	uv run python -m pytest -m "not slow"

test-slow: ## Run slow tests (loads ML models)
	uv run python -m pytest -m slow

check: fmt-check lint typecheck test ## Run all gates (mirrors CI/pre-commit)

models: ## Download ML model weights
	uv run noirdoc models pull
