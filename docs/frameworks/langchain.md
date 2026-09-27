# LangChain and LangGraph

```bash
pip install "mimirai[local,langchain]"
```

`mimir.integrations.langchain.as_structured_tool` gives a `StructuredTool` for `create_agent` and
LangGraph's `ToolNode`; the typed result is the tool message's `artifact`.
`ToolCallCheckMiddleware` decides each call before it runs: a denied call becomes an error tool
message, and an escalated call interrupts the graph with the human-in-the-loop request.
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

