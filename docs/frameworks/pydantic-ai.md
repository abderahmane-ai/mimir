# PydanticAI

```bash
pip install "mimirai[local,pydantic-ai]"
```

`mimir.integrations.pydantic_ai.as_toolset` converts a collection of decision tools into a `FunctionToolset` with typed return values for use in a PydanticAI agent.

`guard(toolset, check)` wraps any toolset with a tool-call check: a certified denial fails the tool call with the check's reason, and an escalated call ends the run by raising `DeferredToolRequests`. Declare `DeferredToolRequests` in the agent's `output_type` to handle escalations cleanly.

Tested from `pydantic-ai-slim` 2.16.

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
