# Command line

`pip install "mimir-decisions[local]"` installs the `mimir` command. Every command accepts `--help`.

| Command | Does |
|---|---|
| `mimir decide` | one decision from flags, or a `DecideRequest` JSON on stdin; prints the result as JSON |
| `mimir serve` | the HTTP server; `--mcp` also serves MCP at `/mcp` |
| `mimir mcp` | the MCP server, over stdio or `--http` |
| `mimir bench FILE` | accuracy, coverage, certified share, and realised risk on labelled decisions |
| `mimir calibrate FILE` | certify thresholds on labelled decisions and write a custom policy |
| `mimir schema [NAME]` | JSON Schemas of every spec, result, and request type |
| `mimir download` | download and verify a release for offline use |
| `mimir doctor` | the environment; `--verify` loads the model and runs the equivalence check |

## Examples

```bash
# One decision from flags
mimir decide --type choice --question "What is the customer reporting?" \
  --option "Transaction charged twice" --option "Request a refund" \
  --option "Card not working" --option "Change PIN" \
  --text "There are two identical charges from your company on my statement."

# One decision with a confidence floor
mimir decide --type choice --question "What is the customer reporting?" \
  --option "Transaction charged twice" --option "Request a refund" \
  --option "Card not working" --option "Change PIN" \
  --text "There are two identical charges from your company on my statement." \
  --mode threshold --min-confidence 0.7

# One decision from stdin
echo '{"context": "...", "decision": {"type": "yes_no", "question": "Is this urgent?"}}' | mimir decide

# Forward to a running server instead of loading the model locally
mimir decide --server https://mimir.internal --type yes_no --question "Is this urgent?" --text "..."
```

## Model arguments

The model arguments (`--model`, `--revision`, `--device`, `--variant`, `--policy`, `--offline`) are consistent across every command that loads the model locally. Pass `--server` to any command to forward decisions to a MIMIR HTTP server instead.

The labelled file format for `mimir bench` and `mimir calibrate` is described in [The certificate](../guide/certificate.md#certifying-on-your-own-data).
