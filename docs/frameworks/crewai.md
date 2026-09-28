# CrewAI

```bash
pip install "mimirai[local,crewai]"
```

`mimir.integrations.crewai.as_crewai_tool` converts a decision tool into a `BaseTool` whose `result_schema` is the typed result, for use in a CrewAI `Agent`'s tool list.

`tool_call_hook(check, approve=...)` returns a hook for `register_before_tool_call_hook`: a certified denial blocks the call, and an escalated call is forwarded to your `approve` function, which receives the tool name and arguments and returns whether the call may run.

Tested from `crewai` 1.15.

> **Note.** CrewAI pins the MCP SDK to version 1, which conflicts with `mimirai[mcp]`. Install CrewAI separately from the MCP extra. The MCP example for CrewAI connects over HTTP rather than stdio for this reason.

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
