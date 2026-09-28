# Tool-call checks

A tool-call check decides, against rules you write, whether an agent's pending tool call may proceed. The rules are read as passages; the call is presented as fields (`tool`, `arguments.<name>`). A certified yes allows the call, a certified no denies it, and an uncertified result escalates it to a person.

```python
check = model.tool_call_check(
    ["Refunds above 500 dollars need a manager's approval."], tools=["issue_refund"]
)
outcome = check("issue_refund", {"order": "4412", "amount": 900})
outcome.permission    # Permission.ALLOW, Permission.DENY or Permission.ESCALATE
outcome.reason        # one sentence for the agent or the approver
```

`tools` limits the check to those tool names; calls to any other tool pass through without a decision. Without `tools`, the check applies to every call.

`acall` is the async form.

## Writing rules

Rules are the passages the check reads as evidence. Write them as plain-language policies that name the tools and the conditions:

```python
rules = [
    "Refunds above 500 dollars require a manager's approval.",
    "Only verified customer accounts may request a chargeback.",
    "Password reset emails may only be sent to the account's registered address.",
]
check = model.tool_call_check(rules, tools=["issue_refund", "send_email", "request_chargeback"])
```

Without rules, the check has no evidence to decide by and will defer every call. Write rules that are specific to your tools and your risk policy — the more precise the rule, the more confidently the model can certify its verdict.

## Handling outcomes

Treat `ESCALATE` as the normal path for any call the policy cannot confidently vouch for. Only `ALLOW` and `DENY` are certified; `ESCALATE` means the model saw the call but its confidence did not clear the threshold.

```python
match outcome.permission:
    case Permission.ALLOW:
        run_the_tool()
    case Permission.DENY:
        tell_the_agent(outcome.reason)
    case Permission.ESCALATE:
        request_human_approval(outcome.reason)
```

## Framework adapters

Each framework adapter wires a check into that framework's own approval hook. See the [Frameworks](../frameworks/openai-agents.md) section for the specific hook names and escalation behaviour per SDK.
