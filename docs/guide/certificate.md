# The certificate

Every decision answers: `answer` always holds the model's prediction, with calibrated
`probabilities`. `status` says whether the answer cleared the operating floor:

| Status | Meaning | What to do |
|---|---|---|
| `DECIDED` | `answer` is an option, above the floor | act on it |
| `ABSTAINED` | no listed option applies, above the floor | act on "none of these" |
| `DEFERRED` | `answer` is below the floor | have a person review it; `answer` still shows the model's prediction |

`actionable` is true for `decided` and `abstained`. In `standard` mode there is no floor,
so every answer is actionable. In `threshold` mode the floor is your `min_confidence`; in
`certified` mode it is the policy's threshold at the requested risk.

## Modes

```python
ticket = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
question = "Which department should handle this request?"
teams = {
    "billing": "Billing: invoices, payments, refunds",
    "technical": "Technical: bugs, outages, system errors",
    "sales": "Sales: pricing, new contracts",
    "other": "Other: everything else",
}

model.choose(ticket, question, teams)                                     # standard
model.choose(ticket, question, teams, mode="threshold", min_confidence=0.7)
model.choose(ticket, question, teams, mode="certified", risk=0.05)        # 95% SLA (agents)
model.choose(ticket, question, teams, mode="certified", risk=0.01)        # 99% SLA (mission-critical)
```

- `standard` (the default) answers every request: the calibrated argmax, never deferred
  except by abstention.
- `threshold` defers answers below `min_confidence`, keeping the answer. Pick the floor
  your pipeline can defend; 0.5 asks for a majority, 0.9 for near-certainty.
- `certified` defers answers below the threshold the release certified at `risk` for this
  decision type (`risk` defaults to 0.05, the 95% SLA for high-throughput autonomous agents; pass `risk=0.01` for strict aerospace/financial SLAs). A type with no certified threshold at the risk behaves as `standard` for
  the status, with `certified` false and no certificate.

## What a certificate records

`result.certified` is true when the loaded policy holds a threshold for the decision type
at the requested risk and the score passes it. `result.certificate` then carries the
evidence that certified it:

| Field | Meaning |
|---|---|
| `risk`, `confidence` | the certified error rate and the statistical confidence of the certificate |
| `threshold` | the calibrated score a decision must reach to be certified |
| `records`, `taken`, `errors` | held-out decisions, how many cleared the threshold, how many of those were wrong |
| `p_value` | exact binomial test of `errors` out of `taken` at rate `risk` |
| `coverage`, `coverage_interval` | `taken / records`, with a 95% Wilson interval |
| `model`, `revision`, `variant`, `origin` | what was certified, and whether by the release workflow or by you |

Thresholds are tested from the strictest down, each with an exact binomial test, stopping at the first that fails. The certified threshold is the last that passed. The risk levels the release holds are in `model.info().risk_levels`.

## One exact configuration

A certificate is a property of the specific numbers that produced the probabilities. It records the weights, the Torch version, the device and the hardware it was measured on. Loading a policy into a configuration it was not made for raises immediately. On hardware the certificate does not list, the first load runs the release's equivalence set and requires every decision to match; `mimir doctor --verify` runs the same check on demand.

## Certifying on your own data

To certify thresholds on your own labelled decisions, write one JSON object per line:

```json
{"context": "Hi, we were billed twice for March.", "decision": {"type": "choice", "question": "Which department should handle this request?", "options": {"billing": "Billing: invoices, payments, refunds", "technical": "Technical: bugs, outages, system errors", "sales": "Sales: pricing, new contracts", "other": "Other: everything else"}, "label": "billing"}
```

| Spec | Label |
|---|---|
| `choice` | an option id, or `null` when no option applies |
| `multi_choice` | a list of option ids that apply |
| `yes_no` | `true` or `false` |
| `verify` | `supported`, `contradicted`, or `not_enough_information` |
| `rank` | the id of the best candidate, or a list of ids that are equally best |
| `rate` | a level id |
| `estimate` | a number within `[low, high]` |

```bash
mimir calibrate labelled.jsonl --risk 0.01 --confidence 0.95 --out policy.json
mimir bench held-out.jsonl --policy policy.json --risk 0.01
```

```python
model = Mimir.from_pretrained("Mythologic/MIMIR-1", policy="policy.json")
```

A custom policy keeps the release's calibration intact and replaces its thresholds with ones certified on your data at one risk level. Each decision type is tested at an equal share of `1 - confidence`. The policy is bound to the weights, runtime, and hardware it was made on — run `mimir calibrate` where you run the model. `mimir bench` reports accuracy, coverage, certified share, and realised risk per decision type, each with a 95% Wilson interval.
