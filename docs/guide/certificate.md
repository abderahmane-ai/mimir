# The certificate

A certified decision comes with a promise: among decisions taken at risk level `r`, the share
that are wrong is at most `r`, with 95% confidence. The policy keeps that promise by taking a
decision only when its calibrated confidence reaches a threshold certified on held-out data, and
deferring everything else.

## Status

| Status | Meaning | What to do |
|---|---|---|
| `DECIDED` | `answer` is an option, certified at the requested risk | act on it |
| `ABSTAINED` | no listed option applies, certified | act on "none of these" |
| `DEFERRED` | the policy does not authorise acting | escalate; `answer` still shows the model's view |

A deferral carries its reason:

- `below_threshold`: `confidence` is under the certified `threshold`;
- `out_of_distribution`: the context is unlike the data the thresholds were certified on
  (`gate_p_value` is at or below the policy's gate level);
- `no_certified_threshold`: nothing is certified for this decision type at this risk.

## What a certificate records

`result.certificate` is the threshold the decision was checked against, with its evidence:

| Field | Meaning |
|---|---|
| `risk`, `confidence` | the certified error rate and the confidence of the certificate |
| `threshold` | the calibrated confidence a decision must reach |
| `records`, `taken`, `errors` | held-out decisions, how many reached the threshold, how many of those were wrong |
| `p_value` | the exact binomial test of `errors` out of `taken` at rate `risk` |
| `coverage`, `coverage_interval` | `taken / records`, with a 95% Wilson interval |
| `model`, `revision`, `variant`, `origin` | what was certified, and whether by the release or by you |

Thresholds are tested from the strictest down, each with an exact binomial test, stopping at
the first that fails; the certified threshold is the last that passed. The certified risk levels
are in `model.info().risk_levels`.

## One exact configuration

A certificate is a property of the numbers that produced the probabilities, so it names the
model files, the variant (`fp32` on CPU, `fp16` on CUDA), the ONNX Runtime version, the
execution provider and its options, and the hardware it was measured on. Loading a policy into
any other configuration raises. On hardware the certificate does not list, the first load runs
the release's equivalence set and requires every decision to match; `mimir doctor --verify`
runs the same check.

## Your own certificate

To certify thresholds on your own labelled decisions, write one JSON object per line:

```json
{"context": "My card was charged twice", "decision": {"type": "choice", "question": "Which team?", "options": ["billing", "security"]}, "label": "billing"}
```

| Spec | Label |
|---|---|
| `choice` | an option id, or null when no option applies |
| `multi_choice` | the list of option ids that apply |
| `yes_no` | true or false |
| `verify` | `supported`, `contradicted` or `not_enough_information` |
| `rank` | the id of the best candidate, or the list of ids that are equally best |
| `rate` | a level id |
| `estimate` | a number within `[low, high]` |

```bash
mimir calibrate labelled.jsonl --risk 0.01 --confidence 0.95 --out policy.json
mimir bench held-out.jsonl --policy policy.json --risk 0.01
```

```python
model = Mimir.from_pretrained("mythologic/mimir-1", policy="policy.json")
```

A custom policy keeps the release's calibration and replaces its thresholds with ones certified
on your data at one risk level, each decision type tested at an equal share of `1 - confidence`.
It is bound to the model, runtime and hardware it was made on. `mimir bench` reports accuracy,
coverage and realised risk per decision type, each with a 95% Wilson interval.
