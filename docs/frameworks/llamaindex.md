# LlamaIndex

```bash
pip install "mimirai[local,llamaindex]"
```

`mimir.integrations.llamaindex.as_llamaindex_tool` converts a decision tool into a tool compatible with `FunctionAgent`, `ReActAgent`, and `AgentWorkflow`. The typed result is available as the tool output's `raw_output`.

LlamaIndex does not expose a hook that runs before a tool call, so tool-call checks cannot be wired in as pre-call middleware. Use the MCP transport or the HTTP client to integrate checks into a LlamaIndex pipeline from outside the framework.

Tested from `llama-index-core` 0.14.25.

## Native tools

```python
--8<-- "examples/llamaindex/agent.py"
```

## Over MCP

```python
--8<-- "examples/llamaindex/mcp_agent.py"
```
