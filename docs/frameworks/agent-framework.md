# Microsoft Agent Framework

```bash
pip install "mimirai[local,agent-framework]"
```

`mimir.integrations.agent_framework.as_function_tool` gives a `FunctionTool` returning the
result as JSON. `ToolCallCheckMiddleware` is function middleware that runs only certified calls:
a denied or escalated call is not run, and the model reads the check's reason instead.

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

