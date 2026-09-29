/**
 * A Vercel AI SDK agent using MIMIR's MCP server over stdio.
 *
 * The server runs `examples/tools.yaml`, so the model sees `route_ticket` and passes only the
 * ticket. Needs Node 22 or later and `uv`; set `OPENAI_API_KEY`, then run
 * `make example NAME=vercel_ai/agent`.
 */

import { fileURLToPath, pathToFileURL } from "node:url";

import { createMCPClient, type MCPClient } from "@ai-sdk/mcp";
import { Experimental_StdioMCPTransport as StdioMCPTransport } from "@ai-sdk/mcp/mcp-stdio";
import { openai } from "@ai-sdk/openai";
import { generateText, type LanguageModel, stepCountIs } from "ai";

export const TOOLS = fileURLToPath(new URL("../tools.yaml", import.meta.url));
export const SERVER = {
  command: "uvx",
  args: ["--from", "mimir-decisions[local,mcp]", "mimir-decisions", "mcp", "--tools", TOOLS],
};
export const INSTRUCTIONS =
  "Route the customer's ticket with route_ticket, then tell the customer which team will " +
  "answer. If the decision is deferred, say that a person will review the ticket.";

/** MIMIR's MCP server, started as `command` with `args`.
 *
 * Discovery is off: it times out in a second while a cold server loads its weights, and the
 * client's fallback then reuses the poisoned connection. The legacy handshake waits instead.
 */
export async function mimirClient(command: string, args: string[]): Promise<MCPClient> {
  return createMCPClient({
    transport: new StdioMCPTransport({ command, args }),
    protocolVersionDiscovery: false,
  });
}

/** The model's reply to `ticket`, after it routes the ticket with the server's tools. */
export async function routeTicket(client: MCPClient, model: LanguageModel, ticket: string) {
  const tools = await client.tools();
  return generateText({ model, system: INSTRUCTIONS, prompt: ticket, tools, stopWhen: stepCountIs(4) });
}

async function main(): Promise<void> {
  const client = await mimirClient(SERVER.command, SERVER.args);
  try {
    const result = await routeTicket(client, openai("gpt-5.5"), "I was charged twice for order 4412.");
    console.log(result.text);
  } finally {
    await client.close();
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
