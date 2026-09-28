# Microsoft Agent Framework

```bash
pip install "mimirai[local,agent-framework]"
```

`mimir.integrations.agent_framework.as_function_tool` converts a decision tool into a `FunctionTool` that returns the typed result as JSON.

`ToolCallCheckMiddleware` is function middleware that intercepts each tool call before it runs. Only calls that receive a certified `ALLOW` are forwarded to the tool. A certified `DENY` and an uncertified `ESCALATE` both block the call, and the model receives the check's reason instead.

Tested from `agent-framework-core` 1.19.

## Native tools

```python
--8<-- "examples/agent_framework/agent.py"
```

## Tool-call check

```python
--8<-- "examples/agent_framework/guarded_agent.py"
```

## Over MCP

```python
--8<-- "examples/agent_framework/mcp_agent.py"
```
