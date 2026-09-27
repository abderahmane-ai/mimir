# OpenAI Agents SDK

```bash
pip install "mimirai[local,openai-agents]"
```

`mimir.integrations.openai_agents.as_function_tool` turns a decision tool into a `FunctionTool`.
`guard(tool, check)` gates any function tool with a tool-call check: a denied call is rejected
with the check's reason, and an escalated call pauses the run for a person
(`RunState.approve` or `RunState.reject`). Tested from `openai-agents` 0.21.

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

