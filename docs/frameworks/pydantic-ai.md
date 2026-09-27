# PydanticAI

```bash
pip install "mimirai[local,pydantic-ai]"
```

`mimir.integrations.pydantic_ai.as_toolset` gives a `FunctionToolset` of decision tools with their
typed results. `guard(toolset, check)` wraps any toolset: a denied call fails with the check's
reason, and an escalated call ends the run with `DeferredToolRequests` (declare it in the
agent's `output_type`). Tested from `pydantic-ai-slim` 2.16.

## Native tools

```python
--8<-- "examples/pydantic_ai/agent.py"
```

## Tool-call check

```python
--8<-- "examples/pydantic_ai/guarded_agent.py"
```

## Over MCP

```python
--8<-- "examples/pydantic_ai/mcp_agent.py"
```

