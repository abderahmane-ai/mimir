# Vercel AI SDK

Any language reaches MIMIR through its MCP server. This TypeScript example starts `mimir mcp` over stdio and hands its tools to the Vercel AI SDK.

```typescript
--8<-- "examples/vercel_ai/agent.ts"
```

For a persistent remote server rather than a subprocess, start `mimir mcp --http` and point the AI SDK's MCP client at `https://mimir.internal/mcp` with an `Authorization: Bearer ...` header.
