# The certificate

A certified decision comes with a formal promise: among all decisions taken at risk level `r`, the share that are wrong is at most `r`, with 95% confidence. The policy keeps this promise by taking a decision only when its calibrated confidence reaches a threshold certified on held-out data, and deferring everything else.

## Status

| Status | Meaning | What to do |
|---|---|---|
| `DECIDED` | `answer` is an option, certified at the requested risk | act on it |
| `ABSTAINED` | no listed option applies, certified | act on "none of these" |
| `DEFERRED` | the policy does not authorise acting | escalate; `answer` still shows the model's best guess |

A deferral carries its reason:

- `below_threshold` — `confidence` is under the certified `threshold` for this decision type at this risk level.
- `out_of_distribution` — the context is unlike the data the thresholds were certified on; `gate_p_value` is at or below the policy's gate level.
- `no_certified_threshold` — nothing is certified for this decision type at this risk level.

## What a certificate records

`result.certificate` is the threshold the decision was checked against, along with the evidence that certified it:

| Field | Meaning |
|---|---|
| `risk`, `confidence` | the certified error rate and the statistical confidence of the certificate |
| `threshold` | the calibrated confidence a decision must reach to be taken |
| `records`, `taken`, `errors` | held-out decisions, how many cleared the threshold, how many of those were wrong |
| `p_value` | exact binomial test of `errors` out of `taken` at rate `risk` |
| `coverage`, `coverage_interval` | `taken / records`, with a 95% Wilson interval |
| `model`, `revision`, `variant`, `origin` | what was certified, and whether by the release workflow or by you |

Thresholds are tested from the strictest down, each with an exact binomial test, stopping at the first that fails. The certified threshold is the last that passed. The risk levels certified by the release are in `model.info().risk_levels`.

## One exact configuration

A certificate is a property of the specific numbers that produced the probabilities. It records the model files, the variant (`fp32` on CPU, `fp16` on CUDA), the ONNX Runtime version, the execution provider and its options, and the hardware it was measured on.

Loading a policy into a configuration it was not made for raises immediately. On hardware that is not listed in the certificate, the first load runs the release's equivalence set and requires every decision to match; `mimir doctor --verify` runs the same check on demand.

## Certifying on your own data

To certify thresholds on your own labelled decisions, write one JSON object per line:

```json
{"context": "My card was charged twice", "decision": {"type": "choice", "question": "Which team?", "options": ["billing", "security"]}, "label": "billing"}
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

A custom policy keeps the release's calibration intact and replaces its thresholds with ones certified on your data at one risk level. Each decision type is tested at an equal share of `1 - confidence`. The policy is bound to the model, runtime, and hardware it was made on — run `mimir calibrate` where you run the model. `mimir bench` reports accuracy, coverage, and realised risk per decision type, each with a 95% Wilson interval.
