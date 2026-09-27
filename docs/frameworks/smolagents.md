# smolagents

```bash
pip install "mimirai[local,smolagents]"
```

`mimir.integrations.smolagents.as_smolagents_tool` gives a tool for `ToolCallingAgent` and
`CodeAgent` returning the result as a JSON object, so code reads `result["status"]` and
`result["answer"]`. smolagents callbacks run after each step, not before a tool call, so
tool-call checks are not adapted.

## Native tools

```python
--8<-- "examples/smolagents/agent.py"
```

## Over MCP

```python
--8<-- "examples/smolagents/mcp_agent.py"
```

