# Calibration and scoring

Certify thresholds on your own labelled decisions, and score results against labels.
The file format both read is one JSON object per line —
`{"context": ..., "decision": <spec>, "label": ...}` — with the labels
[The certificate](../guide/certificate.md#your-own-certificate) lists.

```python
from mimir import Mimir
from mimir.core.labels import read_labelled
from mimir.evaluation.bench import bench
from mimir.runtime.calibrate import calibrate

model = Mimir.from_pretrained("Mythologic/MIMIR-1")
labelled = read_labelled("labelled.jsonl")
calibration = calibrate(model, labelled, risk=0.01, confidence=0.95)
results = [model.decide_uncertified(item.context, item.decision) for item in labelled]
report = bench(results, [item.label for item in labelled])
```

`calibrate` replaces the release thresholds with ones certified on your data at one risk
level; `calibration.types` reports each decision type's records, taken count and
certified threshold. Certifiable types are binary, categorical, multilabel, ranking and
ordinal. `bench` reports accuracy, coverage and realised risk per spec type,
each with a 95% Wilson interval. The `mimir calibrate` and `mimir bench` commands do the
same over files; see [Command line](../surfaces/cli.md).

::: mimir.runtime.calibrate

::: mimir.evaluation.bench

::: mimir.core.labels
