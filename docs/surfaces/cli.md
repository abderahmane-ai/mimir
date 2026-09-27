# Command line

`pip install "mimirai[local]"` installs the `mimir` command. Every command takes `--help`.

| Command | Does |
|---|---|
| `mimir decide` | one decision from flags, or a `DecideRequest` JSON on stdin; the result as JSON |
| `mimir serve` | the HTTP server; `--mcp` also serves MCP at `/mcp` |
| `mimir mcp` | the MCP server, over stdio or `--http` |
| `mimir bench FILE` | accuracy, coverage and realised risk on labelled decisions |
| `mimir calibrate FILE` | certify thresholds on labelled decisions and write a custom policy |
| `mimir schema [NAME]` | JSON Schemas of every spec, result and request |
| `mimir download` | download and verify a release for offline use |
| `mimir doctor` | the environment; `--verify` loads the model and runs the equivalence check |

```bash
mimir decide --type choice --question "Which team?" --option billing --option security \
  --text "My card was charged twice"
echo '{"context": "...", "decision": {"type": "yes_no", "question": "Is this urgent?"}}' | mimir decide
mimir decide --server https://mimir.internal --type yes_no --question "Is this urgent?" --text "..."
```

`--server` sends the decision to a MIMIR HTTP server instead of loading the model. The model
options (`--model`, `--revision`, `--device`, `--variant`, `--policy`) are the same on every
command that loads one. The labelled file format of `bench` and `calibrate` is in
[The certificate](../guide/certificate.md#your-own-certificate).
