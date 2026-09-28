# Google ADK

```bash
pip install "mimir-decisions[local,adk]"
```

`mimir.integrations.google_adk.as_adk_tool` converts a decision tool into an ADK tool declared with the decision tool's argument schema, for use in an `LlmAgent`'s tool list.

`tool_call_callback(check)` returns a `before_tool_callback` for an `LlmAgent`: a certified denial skips the tool call and returns an `error` response carrying the check's reason; an escalated call asks for ADK's built-in tool confirmation before allowing the call to proceed.

Tested from `google-adk` 2.10.

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
