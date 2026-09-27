SHELL := /bin/bash
.DEFAULT_GOAL := help

SRC := src tests tools examples
FRAMEWORKS := openai-agents langchain pydantic-ai llamaindex adk agent-framework smolagents
EXTRAS := --extra local --extra server --extra mcp $(foreach framework,$(FRAMEWORKS),--extra $(framework)) --group docs
UV := uv run --locked $(EXTRAS)
# CrewAI pins the MCP SDK 1, and the MCP clients of Agent Framework and smolagents still need
# it, so they have an environment of their own. Its MCP tests start the main environment's
# server. These files are checked there and only there.
MCP1_ENVIRONMENT := .venv-mcp1
MCP1_EXTRAS := --extra local --extra crewai --extra agent-framework --extra smolagents --no-group examples --group examples-mcp1
MCP1 := UV_PROJECT_ENVIRONMENT=$(MCP1_ENVIRONMENT) uv run --locked $(MCP1_EXTRAS)
MCP1_TESTS := tests/unit/mimir/integrations/test_crewai.py tests/unit/examples/test_crewai.py \
	tests/integration/examples/test_crewai.py tests/integration/examples/test_agent_framework.py \
	tests/integration/examples/test_smolagents.py
MCP1_FILES := $(MCP1_TESTS) src/mimir/integrations/crewai.py examples/crewai \
	examples/agent_framework/mcp_agent.py examples/smolagents/mcp_agent.py
MCP1_MATCH := $(MCP1_FILES) $(addsuffix /%,$(MCP1_FILES))
MAIN_PYTEST := $(foreach file,$(MCP1_TESTS),--ignore=$(file))
MAIN_MYPY := $(foreach file,$(MCP1_FILES),--exclude '^$(file)')
# Tier 1 frameworks as extra:test module, tested at their extras' lower bounds.
TIER_ONE := openai-agents:openai_agents langchain:langchain pydantic-ai:pydantic_ai crewai:crewai

ifeq ($(MAKELEVEL),0)
$(shell python3 tools/banner.py >&2)
endif

TASKS := unit integration minimum
VARIANTS := cpu cuda
VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)

.PHONY: help install format lint typecheck dead test docs check openapi serve mcp example inspect load image publish build clean

help: ## List every target
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Create both virtualenvs from uv.lock, and the TypeScript example's node_modules
	uv sync --locked $(EXTRAS)
	UV_PROJECT_ENVIRONMENT=$(MCP1_ENVIRONMENT) uv sync --locked $(MCP1_EXTRAS)
	npm ci --prefix examples/vercel_ai --no-audit --no-fund

format: ## Rewrite sources with the formatter and autofixes
	$(UV) ruff format $(SRC)
	$(UV) ruff check --fix $(SRC)

lint: ## Formatter check and lint, no rewrites
	$(UV) ruff format --check $(SRC)
	$(UV) ruff check $(SRC)

typecheck: ## mypy strict over every source directory, each file in its environment
	$(UV) mypy $(MAIN_MYPY)
	$(MCP1) mypy $(MCP1_FILES)
	npm --prefix examples/vercel_ai run --silent typecheck

dead: ## Dead-code sweep
	$(UV) vulture

test: ## Tests: TASK=unit (default), integration (MIMIR_RELEASE_DIR, MIMIR_FIXTURES_DIR), minimum (Tier 1 at lower bounds)
	$(if $(filter $(or $(TASK),unit),$(TASKS)),,$(error TASK must be one of: $(TASKS)))
ifeq ($(or $(TASK),unit),unit)
	$(UV) pytest $(MAIN_PYTEST)
	$(MCP1) pytest $(MCP1_TESTS)
else ifeq ($(TASK),integration)
	$(UV) pytest -m integration $(MAIN_PYTEST)
	MIMIR_SERVER=$(CURDIR)/.venv/bin/mimir $(MCP1) pytest -m integration $(MCP1_TESTS)
else
	@set -e; for pair in $(TIER_ONE); do \
		uv run --isolated --resolution lowest-direct --no-group examples --extra local \
			--extra $${pair%%:*} pytest -p no:cacheprovider tests/unit/mimir/integrations/test_$${pair##*:}.py; \
	done
endif

# Strict mode fails on links and references but not on griffe's docstring warnings.
docs: ## Build the docs site into site/, failing on any warning
	set -o pipefail; $(UV) zensical build --strict --clean 2>&1 | awk '{ print } /^griffe:/ { bad = 1 } END { exit bad }'

check: lint typecheck dead docs test ## The gate: lint, typecheck, dead code, docs build, unit tests

openapi: ## Write the server's OpenAPI document to openapi.json
	$(UV) python tools/openapi.py openapi.json

serve: ## Serve an unsigned local release on 127.0.0.1:8000: RELEASE=<release dir>
	$(if $(RELEASE),,$(error RELEASE is required))
	$(UV) mimir serve --model $(RELEASE) --allow-unsigned --device cpu

mcp: ## Serve MCP tools over Streamable HTTP on 127.0.0.1:8000: TOOLS=<file> [RELEASE=<unsigned release dir>]
	$(if $(TOOLS),,$(error TOOLS is required))
	$(UV) mimir mcp --http --tools $(TOOLS) $(if $(RELEASE),--model $(RELEASE) --allow-unsigned --device cpu)

example: ## Run an example on the Hub model (operator, needs its provider's key): NAME=<framework>/<script>
	$(if $(wildcard examples/$(NAME).py examples/$(NAME).ts),,$(error NAME must name a script under examples/, e.g. openai_agents/agent))
ifeq ($(NAME),vercel_ai/agent)
	npm --prefix examples/vercel_ai start
else
	$(if $(filter $(MCP1_MATCH),examples/$(NAME).py),$(MCP1),$(UV)) python examples/$(NAME).py
endif

inspect: ## Check MCP tool schemas with the MCP Inspector (operator, npx downloads it): RELEASE=<release dir>
	$(if $(RELEASE),,$(error RELEASE is required))
	npx --yes @modelcontextprotocol/inspector@2.8.0 --cli $(CURDIR)/.venv/bin/mimir mcp --generic-tools --model $(RELEASE) --allow-unsigned --device cpu -- --method tools/list --strict

load: ## Load-test a running server: URL=http://host:port RELEASE=<release dir> [OUT=file.json]
	$(if $(URL),,$(error URL is required))
	$(if $(RELEASE),,$(error RELEASE is required))
	$(UV) python tools/load.py --url $(URL) --release $(RELEASE) $(if $(OUT),--out $(OUT))

image: ## Build a runtime image (operator, needs Docker): VARIANT=cpu|cuda
	$(if $(filter $(VARIANT),$(VARIANTS)),,$(error VARIANT must be one of: $(VARIANTS)))
	docker build -f docker/$(VARIANT).Dockerfile -t ghcr.io/mythologic/mimir:$(VERSION)-$(VARIANT) .

publish: ## Start a publishing workflow on main (operator): ACTION=sign REVISION=<Hub commit> | ACTION=release MODEL_COMMIT=<signed Hub commit>
ifeq ($(ACTION),sign)
	$(if $(REVISION),,$(error REVISION is required))
	gh workflow run sign-model.yml --repo mythologic/mimir --ref main -f revision=$(REVISION)
else ifeq ($(ACTION),release)
	$(if $(MODEL_COMMIT),,$(error MODEL_COMMIT is required))
	gh workflow run release.yml --repo mythologic/mimir --ref main -f model_commit=$(MODEL_COMMIT)
else
	$(error ACTION must be one of: sign release)
endif

build: ## Build the sdist and wheel into dist/
	uv build

clean: ## Delete every cache and build output
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache -o -name htmlcov -o -name build -o -name dist -o -name site -o -name .cache \) -not -path './.venv*' -not -path '*/node_modules/*' -prune -exec rm -rf {} +
	find . \( -name '*.pyc' -o -name .coverage \) -not -path './.venv*' -delete
