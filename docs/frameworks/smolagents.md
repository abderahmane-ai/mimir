# smolagents

```bash
pip install "mimirai[local,smolagents]"
```

`mimir.integrations.smolagents.as_smolagents_tool` converts a decision tool into a tool for `ToolCallingAgent` and `CodeAgent`. The typed result is returned as a JSON object, so agent code reads `result["status"]` and `result["answer"]`.

smolagents callbacks fire after each agent step rather than before a tool call, so tool-call checks cannot be wired in as pre-call middleware. Use the MCP transport or the HTTP client to integrate checks into a smolagents pipeline from outside the framework.

> **Note.** smolagents uses MCP SDK 2, which conflicts with CrewAI's pinned MCP SDK 1. Keep them in separate environments.

Tested from `smolagents` 1.26.

## Native tools

```python
--8<-- "examples/smolagents/agent.py"
```

## Over MCP

```python
--8<-- "examples/smolagents/mcp_agent.py"
```
