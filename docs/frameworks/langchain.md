# LangChain and LangGraph

```bash
pip install "mimir-decisions[local,langchain]"
```

`mimir.integrations.langchain.as_structured_tool` converts a decision tool into a `StructuredTool` for use with `create_agent` and LangGraph's `ToolNode`. The typed result is the tool message's `artifact`.

`ToolCallCheckMiddleware` is a runnable middleware that intercepts each tool call before it runs: a certified denial becomes an error tool message, and an escalated call interrupts the graph with the human-in-the-loop request, pausing execution until a reviewer approves or rejects.

Tested from `langchain` 1.3.

## Native tools

```python
--8<-- "examples/langchain/agent.py"
```

## Tool-call check

```python
--8<-- "examples/langchain/guarded_agent.py"
```

## Over MCP

```python
--8<-- "examples/langchain/mcp_agent.py"
```
