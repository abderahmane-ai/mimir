# OpenAI Agents SDK

```bash
pip install "mimir-decisions[local,openai-agents]"
```

`mimir.integrations.openai_agents.as_function_tool` converts a decision tool into a `FunctionTool` for use in an `Agent`'s tool list. The typed result is returned as structured data.

`guard(tool, check)` gates any `FunctionTool` with a tool-call check: a certified denial is rejected with the check's reason, and an escalated call pauses the run with `RunState.approve` or `RunState.reject` for human review.

Tested from `openai-agents` 0.21.

## Native tools

```python
--8<-- "examples/openai_agents/agent.py"
```

## Tool-call check

```python
--8<-- "examples/openai_agents/guarded_agent.py"
```

## Over MCP

```python
--8<-- "examples/openai_agents/mcp_agent.py"
```
