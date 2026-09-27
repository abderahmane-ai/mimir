# LlamaIndex

```bash
pip install "mimirai[local,llamaindex]"
```

`mimir.integrations.llamaindex.as_llamaindex_tool` gives a tool for `FunctionAgent`,
`ReActAgent` and `AgentWorkflow`; the typed result is the tool output's `raw_output`.
LlamaIndex has no hook before a tool call runs, so tool-call checks are not adapted.

## Native tools

```python
--8<-- "examples/llamaindex/agent.py"
```

## Over MCP

```python
--8<-- "examples/llamaindex/mcp_agent.py"
```

