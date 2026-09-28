"""A Google ADK agent whose refunds MIMIR checks against the refund rules first.

A certified yes runs the refund, a certified no skips it with the reason, and anything else
asks for confirmation through ADK's own flow, answered here at the console. Install
`mimir-decisions[local,adk]`, set `GOOGLE_API_KEY`, then run `make example NAME=google_adk/guarded_agent`.
"""

import asyncio
from typing import Final

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

from mimir import Decider, Mimir
from mimir.integrations.google_adk import tool_call_callback

MODEL: Final = "gemini-3.5-flash"
RULES: Final = (
    "A refund is at most the amount the customer paid for the order.",
    "Refunds above 500 dollars need a manager's approval.",
)
INSTRUCTIONS: Final = "Issue the refunds customers ask for with issue_refund."
CONFIRMATION: Final = "adk_request_confirmation"


def issue_refund(order: str, amount: float) -> str:
    """Refund `amount` dollars on `order`."""
    return f"Refunded {amount:.2f} dollars on order {order}."


def build_agent(
    decider: Decider, refund: FunctionTool | None = None, model: str | BaseLlm = MODEL
) -> LlmAgent:
    """The refunds agent, its refund tool checked by `decider` against `RULES`."""
    tool = FunctionTool(issue_refund) if refund is None else refund
    check = decider.tool_call_check(RULES, tools=[tool.name])
    return LlmAgent(
        name="refunds",
        model=model,
        instruction=INSTRUCTIONS,
        tools=[tool],
        before_tool_callback=tool_call_callback(check),
    )


async def run_with_approvals(agent: LlmAgent, text: str) -> str:
    """The agent's final reply to `text`, asking at the console about each confirmation."""
    runner = InMemoryRunner(agent=agent, app_name="refunds")
    session = await runner.session_service.create_session(app_name="refunds", user_id="customer")
    message: types.Content | None = types.Content(role="user", parts=[types.Part(text=text)])
    reply = ""
    while message is not None:
        replies: list[types.Part] = []
        async for event in runner.run_async(
            user_id="customer", session_id=session.id, new_message=message
        ):
            for call in event.get_function_calls():
                if call.name == CONFIRMATION and call.args is not None:
                    hint = call.args["toolConfirmation"]["hint"]
                    confirmed = input(f"{hint} Approve? [y/N] ").strip().lower() == "y"
                    response = types.FunctionResponse(
                        id=call.id, name=CONFIRMATION, response={"confirmed": confirmed}
                    )
                    replies.append(types.Part(function_response=response))
            if event.is_final_response() and event.content and event.content.parts:
                reply = "".join(part.text or "" for part in event.content.parts)
        message = types.Content(role="user", parts=replies) if replies else None
    return reply


async def main() -> None:
    agent = build_agent(Mimir.from_pretrained("Mythologic/MIMIR-1"))
    print(await run_with_approvals(agent, "Refund 900 dollars on order 4412."))


if __name__ == "__main__":
    asyncio.run(main())
