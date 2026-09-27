# Google ADK

```bash
pip install "mimirai[local,adk]"
```

`mimir.integrations.google_adk.as_adk_tool` gives a tool declared with the decision tool's
argument schema. `tool_call_callback(check)` gives a `before_tool_callback` for an `LlmAgent`:
a denied call is skipped with an `error` response carrying the reason, and an escalated call
asks for ADK's tool confirmation.

## Native tools

```python
--8<-- "examples/google_adk/agent.py"
```

## Tool-call check

```python
--8<-- "examples/google_adk/guarded_agent.py"
```

## Over MCP

```python
--8<-- "examples/google_adk/mcp_agent.py"
```

