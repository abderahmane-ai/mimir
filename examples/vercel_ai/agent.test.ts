/**
 * The example against a real `mimir mcp` process: `MIMIR_SERVER` is the `mimir` command and
 * `MIMIR_TEST_RELEASE` a test release, as `make test TASK=integration` sets them.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import type { LanguageModelV4GenerateResult } from "@ai-sdk/provider";
import { MockLanguageModelV4 } from "ai/test";

import { mimirClient, routeTicket, TOOLS } from "./agent.ts";

function required(name: string): string {
  const value = process.env[name];
  if (value === undefined || value === "") {
    throw new Error(`${name} is not set; run this test through make test TASK=integration`);
  }
  return value;
}

const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};

function reply(content: LanguageModelV4GenerateResult["content"], finish: "tool-calls" | "stop") {
  return { content, finishReason: { unified: finish, raw: finish }, usage, warnings: [] };
}

test("the agent routes a ticket through the server", async () => {
  const release = required("MIMIR_TEST_RELEASE");
  const args = ["mcp", "--tools", TOOLS, "--model", release, "--allow-unsigned", "--device", "cpu"];
  const client = await mimirClient(required("MIMIR_SERVER"), args);
  const steps = [
    reply(
      [
        {
          type: "tool-call",
          toolCallId: "call-1",
          toolName: "route_ticket",
          input: JSON.stringify({ context: "my card was charged twice" }),
        },
      ],
      "tool-calls",
    ),
    reply([{ type: "text", text: "done" }], "stop"),
  ];
  const model = new MockLanguageModelV4({ doGenerate: async () => steps.shift()! });
  try {
    assert.deepEqual(Object.keys(await client.tools()), ["route_ticket"]);
    const result = await routeTicket(client, model, "Route.");
    assert.equal(result.text, "done");
    const [call] = result.steps[0]!.toolResults;
    assert.equal(call!.toolName, "route_ticket");
    const output = call!.output as { structuredContent: { probabilities: Record<string, number> } };
    assert.deepEqual(Object.keys(output.structuredContent.probabilities).sort(), [
      "billing",
      "security",
      "shipping",
    ]);
  } finally {
    await client.close();
  }
});
