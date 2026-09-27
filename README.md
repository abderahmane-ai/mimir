# mimirai

Typed, calibrated and certified decisions from **MIMIR**, a non-generative decision model by
VathosAI. Give it a context, a question and the options; get back a typed answer, calibrated
probabilities, the parts of the context it relied on, and a certified signal for when to act
and when to escalate.

```bash
pip install "mimirai[local]"        # CPU engine
pip install "mimirai[local-gpu]"    # CUDA engine
pip install mimirai                 # data models and the HTTP client only
```

Python 3.11 or later.

## Quickstart

```python
from mimir import Mimir

model = Mimir.from_pretrained("vathosai/mimir-1")
result = model.choose(
    "My card was charged twice for the same order.",
    "Which team should handle this ticket?",
    options={"billing": "Billing: payments, refunds", "security": "Security: account access"},
)

result.status         # Status.DECIDED, Status.ABSTAINED or Status.DEFERRED
result.answer         # an option id, or None when no option applies
result.probabilities  # calibrated probability of each option id
result.certificate    # the certified threshold the decision was checked against
```

`answer` is the model's prediction. `status` is the policy's verdict on it:

- `DECIDED`: the answer is an option and is certified at the requested risk.
- `ABSTAINED`: the answer is "none of the options" and is certified.
- `DEFERRED`: not certified; `result.deferral.reason` is `below_threshold`,
  `out_of_distribution` or `no_certified_threshold`.

## Decisions

| Spec | Answer |
|---|---|
| `Choice(question, options)` | an option id, or None |
| `MultiChoice(question, options)` | the option ids that apply |
| `YesNo(question)` | `True` or `False` |
| `Verify(claim)` | `supported`, `contradicted` or `not_enough_information` |
| `Rank(question, candidates)` | candidate ids, best first |
| `Rate(question, levels)` | a level id, levels given lowest first |
| `Estimate(question, low, high, unit)` | a number in `[low, high]` with an interval |

```python
from mimir import Context, Field, Passage, Rate, Table

context = Context(
    passages=[Passage(title="Ticket #4412", text="The export has failed every night this week.")],
    tables=[Table.from_rows([["2026-03-02", "failed"]], header=["date", "status"])],
    fields=Field.from_json({"customer": {"plan": "enterprise", "seats": 240}}),
)
result = model.decide(context, Rate("How urgent is this?", ["low", "medium", "high"]), risk=0.01)
```

A context can also be a string, a list of strings, or a dict read as a JSON state. Numbers and
dates in tables and fields are typed. `decide_many` batches many decisions, and every method
has an async form (`adecide`, `adecide_many`, ...).

## Certification

`decide` takes a risk level certified by the loaded policy (`model.info().risk_levels`). A
decision is taken only when its calibrated confidence reaches a threshold certified on held-out
data to keep the error rate of taken decisions at or below that risk with 95% confidence, and
when the context passes the out-of-distribution gate. `decide_uncertified` returns the raw
model answer with no policy applied.

A certificate covers one exact configuration: model files, variant, ONNX Runtime version,
execution provider and options. On hardware the certificate does not list, the first load runs
the release's equivalence set and requires every decision to match. To certify thresholds on
your own labelled data:

```bash
mimir calibrate labelled.jsonl --risk 0.01 --confidence 0.95 --out policy.json
```

```python
model = Mimir.from_pretrained("vathosai/mimir-1", policy="policy.json")
```

## Remote use

```python
from mimir import MimirClient

remote = MimirClient("https://mimir.internal", api_key="...")
remote.choose("...", "Which team?", options=["billing", "security"])
```

`MimirClient` has the same interface as `Mimir`, so code and tools accept either.

## Decision tools

```python
from mimir import Choice

route_ticket = model.tool(
    "route_ticket",
    Choice("Which team should handle this ticket?", options=["billing", "security"]),
    description="Route a support ticket to the team that owns it.",
)
route_ticket("My card was charged twice")
route_ticket.input_schema, route_ticket.output_schema
```

## Command line

| Command | Description |
|---|---|
| `mimir decide` | one decision from flags, or a JSON request on stdin |
| `mimir bench FILE` | accuracy, coverage and realised risk on labelled decisions |
| `mimir calibrate FILE` | certify thresholds on labelled decisions |
| `mimir schema` | JSON Schemas of every spec, result and request |
| `mimir download` | download and verify a release for offline use |
| `mimir doctor` | report the environment; `--verify` loads the model |

## Integrity

Releases are loaded from a pinned Hugging Face revision. Before anything is read, the
manifest's Sigstore signature is verified against the VathosAI release workflow, every file is
checked against the manifest's SHA-256, and the ONNX graph is checked against its operator
allowlist and signature. No pickle is used anywhere.

## Migrating

`mimir.compat.systemone.v1` converts Jev `/v1/systemone` requests and answers, and
`mimir.compat.laya.v1` offers `load(...).predict(state, questions)` in Laya 0.3.20's shape.

## License

The `mimirai` package is licensed under Apache 2.0. The MIMIR model weights are distributed
under their own license on the Hugging Face Hub.
