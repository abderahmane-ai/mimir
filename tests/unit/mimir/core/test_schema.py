import json

from mimir.core.schema import MODELS, json_schemas


def test_every_model_and_union_has_a_schema() -> None:
    schemas = json_schemas()
    expected = {model.__name__ for model in MODELS} | {
        "ContextInput",
        "DecisionSpec",
        "DecisionResult",
    }
    assert set(schemas) == expected


def test_schemas_are_json_and_reference_their_definitions() -> None:
    schemas = json_schemas()
    text = json.dumps(schemas)
    assert json.loads(text) == schemas
    result = schemas["DecisionResult"]
    assert isinstance(result, dict)
    discriminator = result["discriminator"]
    assert isinstance(discriminator, dict)
    mapping = discriminator["mapping"]
    assert isinstance(mapping, dict)
    assert set(mapping) == {
        "choice",
        "multi_choice",
        "yes_no",
        "verify",
        "rank",
        "rate",
        "estimate",
    }
