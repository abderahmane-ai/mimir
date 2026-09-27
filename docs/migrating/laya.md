# Migrating from Laya

`mimir.compat.laya.v1` offers Laya 0.3.20's `load(...).predict(state, questions)`, answering in
Laya's shape:

```python
from mimir.compat.laya.v1 import load

agent = load("vathosai/mimir-1")
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

- Questions are Laya's: Jev's format plus list-valued choice criteria and noul `labels`.
- `predict_batch(states, questions)` answers the same questions about each state.
- Values are rounded to 4 decimals, as Laya rounds them.
- MIMIR has no separate act head: `action.act_probability` is 1.0 when the certified decision
  may be acted on (`DECIDED` or `ABSTAINED`) and 0.0 when it is deferred, at the risk given to
  `load` (1% by default).

The shim makes no promise to follow later Laya versions. For the certificate, the deferral
reason and the relevant context, call [`Mimir`](../surfaces/python.md) directly.
