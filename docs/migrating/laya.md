# Migrating from Laya

`mimir.compat.laya.v1` provides Laya 0.3.20's `load(...).predict(state, questions)` interface, answering in Laya's shape. Existing Laya callers can migrate by changing what they import.

```python
from mimir.compat.laya.v1 import load

agent = load("Mythologic/MIMIR-1")
answers = agent.predict(
    "Help! My payouts have been failing for 3 days.",
    {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": ["billing", "technical", "sales"],
        }
    },
)
```

## Differences from Laya

- Questions follow Jev's format, extended with list-valued choice criteria and `noul` `labels`.
- `predict_batch(states, questions)` answers the same set of questions about each state in the batch.
- Probability values are rounded to 4 decimal places, matching Laya's rounding.
- MIMIR has no separate act head. `action.act_probability` is `1.0` when the certified decision may be acted on (`DECIDED` or `ABSTAINED`), and `0.0` when it defers — evaluated at the risk level passed to `load` (1% by default).

## Limitations

The shim makes no commitment to track later Laya versions. It provides a migration path, not a long-term compatibility guarantee.

For access to the full MIMIR result — the certificate, the deferral reason, and the relevant context slices — call [`Mimir`](../surfaces/python.md) directly rather than going through the compat shim.
