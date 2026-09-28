# MCP server

```bash
pip install "mimirai[local,mcp]"
mimir mcp --tools tools.yaml
```

Each tool in the [tools file](http.md#tools-file) becomes an MCP tool whose only argument is a context. The server owns the question and the options, so the agent cannot invent them at call time. `--generic-tools` adds `mimir_choose`, `mimir_verify`, `mimir_rank`, and `mimir_rate`, which take the question and options as arguments and decide at `--risk`.

Every tool declares its full output schema, returns structured content, and is annotated read-only, idempotent, and closed-world. Its description instructs the agent to act only on `decided` or `abstained` and to escalate `deferred`. A deferral is a normal result, not an error. Invalid arguments, a model still loading, and engine failures are tool errors that name the cause.

## Transports

```bash
uvx --from "mimirai[local,mcp]" mimirai mcp --tools tools.yaml               # stdio
MIMIR_API_KEYS=... mimir mcp --http --host 0.0.0.0 --tools tools.yaml       # Streamable HTTP at /mcp
MIMIR_API_KEY=... mimir mcp --tools tools.yaml --remote https://mimir.internal  # forward to a server
mimir serve --mcp --tools tools.yaml                                          # HTTP API and /mcp together
```

`--http` serves stateless Streamable HTTP with JSON responses, behind the HTTP server's key and loopback rules. `--remote` forwards every tool call to a MIMIR HTTP server, so a local MCP host can use a GPU server without installing the model locally.

## Clients

=== "Claude Code"

    ```bash
    claude mcp add mimir -- uvx --from "mimirai[local,mcp]" mimirai mcp --tools /path/to/tools.yaml
    claude mcp add --transport http mimir https://mimir.internal/mcp --header "Authorization: Bearer ..."
    ```

=== "Claude Desktop"

    In `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`):

    ```json
    {
      "mcpServers": {
        "mimir": {
          "command": "uvx",
          "args": ["--from", "mimirai[local,mcp]", "mimirai", "mcp", "--tools", "/path/to/tools.yaml"]
        }
      }
    }
    ```

    For a remote server, add `"--remote", "https://mimir.internal"` to `args` and `"env": {"MIMIR_API_KEY": "..."}` beside them.

=== "Cursor"

    In `~/.cursor/mcp.json`, or `.cursor/mcp.json` in a project:

    ```json
    {
      "mcpServers": {
        "mimir": {
          "command": "uvx",
          "args": ["--from", "mimirai[local,mcp]", "mimirai", "mcp", "--tools", "/path/to/tools.yaml"]
        },
        "mimir-remote": {
          "url": "https://mimir.internal/mcp",
          "headers": {"Authorization": "Bearer ${env:MIMIR_API_KEY}"}
        }
      }
    }
    ```

=== "VS Code"

    In `.vscode/mcp.json`:

    ```json
    {
      "servers": {
        "mimir": {
          "command": "uvx",
          "args": ["--from", "mimirai[local,mcp]", "mimirai", "mcp", "--tools", "${workspaceFolder}/tools.yaml"]
        },
        "mimir-remote": {
          "type": "http",
          "url": "https://mimir.internal/mcp",
          "headers": {"Authorization": "Bearer ${input:mimir-key}"}
        }
      },
      "inputs": [
        {"type": "promptString", "id": "mimir-key", "description": "MIMIR API key", "password": true}
      ]
    }
    ```

The server is registered in the MCP Registry as `io.github.Mythologic/mimir`.
