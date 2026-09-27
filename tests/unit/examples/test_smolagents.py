from examples.smolagents import agent
from tests.conftest import RecordingDecider
from tests.unit.mimir.integrations.test_smolagents import ScriptedModel, _call


def test_the_support_agent_routes_the_ticket() -> None:
    decider = RecordingDecider()
    model = ScriptedModel(
        [
            _call("route_ticket", {"context": "charged twice"}),
            _call("final_answer", {"answer": "done"}),
        ]
    )
    assert agent.build_agent(decider, model).run("Help.") == "done"
    assert decider.calls[0].requests[0][1].option_ids == tuple(agent.TEAMS)
