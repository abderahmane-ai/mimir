# Tool-call checks

A tool-call check decides, against rules you write, whether an agent's pending tool call may
run. The rules are read as passages and the call as fields (`tool`, `arguments.<name>`). A
certified yes allows the call, a certified no denies it, and anything else escalates it to a
person.

```python
check = model.tool_call_check(
    ["Refunds above 500 dollars need a manager's approval."], tools=["issue_refund"]
)
outcome = check("issue_refund", {"order": "4412", "amount": 900})
outcome.permission    # Permission.ALLOW, Permission.DENY or Permission.ESCALATE
outcome.reason        # one sentence for the agent or the approver
```

`tools` limits the check to those tool names, and calls to other tools run without a
decision; without it the check applies to every call.
`acall` is the async form.

Write the rules the call is judged against: without them the check has nothing to decide by.
Only a certified outcome allows or denies; treat `ESCALATE` as the normal path for a call the
policy cannot vouch for.

Each framework adapter wires a check into that framework's own approval hook; see the table in
[Frameworks](../frameworks/openai-agents.md).
