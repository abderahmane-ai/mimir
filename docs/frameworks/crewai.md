# CrewAI

```bash
pip install "mimirai[local,crewai]"
```

`mimir.integrations.crewai.as_crewai_tool` gives a `BaseTool` whose `result_schema` is the typed
result. `tool_call_hook(check, approve=...)` gives a hook for
`register_before_tool_call_hook`: a denied call is blocked, and an escalated call goes to
`approve`, which returns whether it may run. Tested from `crewai` 1.15. CrewAI pins the MCP SDK
1, so install it apart from `mimirai[mcp]`; its MCP example connects over HTTP.

## Native tools

```python
--8<-- "examples/crewai/agent.py"
```

## Tool-call check

```python
--8<-- "examples/crewai/guarded_agent.py"
```

## Over MCP

```python
--8<-- "examples/crewai/mcp_agent.py"
```

