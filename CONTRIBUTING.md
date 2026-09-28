# Contributing to mimir

Thank you for your interest in contributing. This document covers everything you need to get your environment running, the conventions the codebase follows, and how to submit a change.

## Before you open a pull request

Open an issue first for anything beyond a typo fix. This keeps work visible, avoids duplicated effort, and lets maintainers give early feedback on direction before you invest time writing code.

Security vulnerabilities must be reported privately. See [SECURITY.md](SECURITY.md).

## Setting up

You need Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/abderahmane-ai/mimir.git
cd mimir
make install
```

`make install` creates a virtual environment and installs the package in editable mode with all development and example dependencies. Run the checks to confirm the environment is healthy:

```bash
make check
make test
```

## Running checks

| Command | What it runs |
|---|---|
| `make check` | ruff (lint + format), mypy, vulture, the strict docs build, unit tests |
| `make test` | pytest unit tests |
| `make test TASK=integration` | integration tests (needs the released artifact and network) |
| `make test TASK=minimum` | tests against the minimum declared dependency versions |

All of these must pass before a pull request is merged.

## Code conventions

- **Formatting and linting**: ruff, configured in `pyproject.toml`. Run `make check` to catch issues before pushing.
- **Types**: the codebase runs under mypy strict. All public-facing functions must be fully annotated.
- **Imports**: relative imports are banned across the entire project. Use absolute imports from the `mimir` namespace.
- **Optional dependencies**: extras (`onnxruntime`, `tokenizers`, `sigstore`, framework packages) are imported only inside the functions that need them, not at module level. This keeps `import mimir` fast and the base install light.
- **Tests**: unit tests live in `tests/unit/`, integration tests in `tests/integration/`. New behaviour should come with a unit test; new integration points should come with an integration test where practical.

## Pull request checklist

- [ ] `make check` passes with no new warnings or errors.
- [ ] `make test` passes.
- [ ] New behaviour is covered by tests.
- [ ] Public API changes are reflected in the relevant docs page under `docs/`.
- [ ] The commit message is a short imperative sentence (`Add X`, `Fix Y`, not `Added X` or `Fixes #123`).

## Project layout

```
src/mimir/         core package
  core/            data models: specs, contexts, results
  runtime/         local engine (ONNX Runtime, Sigstore, hardware)
  client/          HTTP client
  integrations/    framework adapters
  cli/             Typer CLI
  compat/          Jev and Laya compatibility shims
tests/
  unit/            fast, no network, no model
  integration/     needs the release artifact or a running server
docs/              Zensical source
examples/          one native, one MCP and one checked agent per framework
tools/             release and Hub management scripts
```

## Licence

By submitting a pull request you agree that your contribution will be licensed under the [Apache 2.0 licence](LICENSE) that covers the MIMIR SDK.

The MIMIR-1 model weights are governed separately by the [MIMIR Model License](MODEL-LICENSE.md); contributions to this repository do not attach to them.
