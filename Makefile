SHELL := /bin/bash
.DEFAULT_GOAL := help

SRC := src tests tools
UV := uv run --locked --extra local

ifeq ($(MAKELEVEL),0)
$(shell python3 tools/banner.py >&2)
endif

TASKS := unit integration

.PHONY: help install format lint typecheck dead test check build clean

help: ## List every target
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv from uv.lock with the CPU engine
	uv sync --locked --extra local

format: ## Rewrite sources with the formatter and autofixes
	$(UV) ruff format $(SRC)
	$(UV) ruff check --fix $(SRC)

lint: ## Formatter check and lint, no rewrites
	$(UV) ruff format --check $(SRC)
	$(UV) ruff check $(SRC)

typecheck: ## mypy strict over every source directory
	$(UV) mypy

dead: ## Dead-code sweep
	$(UV) vulture

test: ## Tests: TASK=unit (default), or TASK=integration with MIMIR_RELEASE_DIR and MIMIR_FIXTURES_DIR set
	$(if $(filter $(or $(TASK),unit),$(TASKS)),,$(error TASK must be one of: $(TASKS)))
	$(UV) pytest $(if $(filter integration,$(TASK)),-m integration)

check: lint typecheck dead test ## The gate: lint, typecheck, dead code, unit tests

build: ## Build the sdist and wheel into dist/
	uv build

clean: ## Delete every cache and build output
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache -o -name htmlcov -o -name build -o -name dist \) -not -path './.venv/*' -prune -exec rm -rf {} +
	find . \( -name '*.pyc' -o -name .coverage \) -not -path './.venv/*' -delete
