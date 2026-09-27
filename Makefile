SHELL := /bin/bash
.DEFAULT_GOAL := help

SRC := src tests tools
EXTRAS := --extra local --extra server --extra mcp
UV := uv run --locked $(EXTRAS)

ifeq ($(MAKELEVEL),0)
$(shell python3 tools/banner.py >&2)
endif

TASKS := unit integration
VARIANTS := cpu cuda
VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)

.PHONY: help install format lint typecheck dead test check openapi serve inspect load image build clean

help: ## List every target
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv from uv.lock with the CPU engine, server and MCP server
	uv sync --locked $(EXTRAS)

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

openapi: ## Write the server's OpenAPI document to openapi.json
	$(UV) python tools/openapi.py openapi.json

serve: ## Serve an unsigned local release on 127.0.0.1:8000: RELEASE=<release dir>
	$(if $(RELEASE),,$(error RELEASE is required))
	$(UV) mimir serve --model $(RELEASE) --allow-unsigned --device cpu

inspect: ## Check MCP tool schemas with the MCP Inspector (operator, npx downloads it): RELEASE=<release dir>
	$(if $(RELEASE),,$(error RELEASE is required))
	npx --yes @modelcontextprotocol/inspector@2.8.0 --cli $(CURDIR)/.venv/bin/mimir mcp --generic-tools --model $(RELEASE) --allow-unsigned --device cpu --method tools/list --strict

load: ## Load-test a running server: URL=http://host:port RELEASE=<release dir> [OUT=file.json]
	$(if $(URL),,$(error URL is required))
	$(if $(RELEASE),,$(error RELEASE is required))
	$(UV) python tools/load.py --url $(URL) --release $(RELEASE) $(if $(OUT),--out $(OUT))

image: ## Build a runtime image (operator, needs Docker): VARIANT=cpu|cuda
	$(if $(filter $(VARIANT),$(VARIANTS)),,$(error VARIANT must be one of: $(VARIANTS)))
	docker build -f docker/$(VARIANT).Dockerfile -t ghcr.io/vathosai/mimir:$(VERSION)-$(VARIANT) .

build: ## Build the sdist and wheel into dist/
	uv build

clean: ## Delete every cache and build output
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache -o -name htmlcov -o -name build -o -name dist \) -not -path './.venv/*' -prune -exec rm -rf {} +
	find . \( -name '*.pyc' -o -name .coverage \) -not -path './.venv/*' -delete
