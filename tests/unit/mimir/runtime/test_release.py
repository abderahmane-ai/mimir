import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mimir.runtime.release import Manifest, ReleaseConfig, is_known_type_order


def test_config_parses_and_indexes_decision_types(release: Path) -> None:
    config = ReleaseConfig.model_validate_json((release / "config.json").read_bytes())
    assert config.type_index("binary") == 0
    assert config.type_index("continuous") == 5
    assert is_known_type_order(config)
    assert config.variants["fp32"].devices == ("cpu",)


def test_config_rejects_unknown_fields_and_types(release: Path) -> None:
    raw = json.loads((release / "config.json").read_text(encoding="utf-8"))
    with pytest.raises(ValidationError, match="Extra inputs"):
        ReleaseConfig.model_validate({**raw, "training_sources": []})
    with pytest.raises(ValidationError):
        ReleaseConfig.model_validate({**raw, "decision_types": ["binary", "vibes"]})


def test_config_parses_the_private_release_shape() -> None:
    raw = {
        "format_version": 1,
        "variants": {
            "fp32": {"graph": "onnx/model.onnx", "policy": "policy/fp32", "devices": ["cpu"]}
        },
        "tokenizer": "tokenizer.json",
        "decision_types": [
            "binary",
            "categorical",
            "multilabel",
            "ranking",
            "ordinal",
            "continuous",
        ],
        "risk_levels": [0.005, 0.01, 0.02, 0.05],
        "default_alpha": 0.1,
        "limits": {"options": 150, "levels": 49, "context_tokens": 26406},
        "layout": {
            "chunk_tokens": 512,
            "question_cap": 256,
            "crossing_tokens": 4096,
            "special_tokens": {
                "cls": 50281,
                "sep": 50282,
                "pad": 50283,
                "mask": 50284,
                "newline": 187,
            },
        },
        "graph": {
            "opset": {"ai.onnx": 20},
            "operators": ["ai.onnx::Add"],
            "inputs": [{"name": "chunk_ids", "dtype": "int64", "rank": 2}],
            "outputs": [{"name": "utilities", "dtype": "float32", "rank": 2}],
        },
    }
    config = ReleaseConfig.model_validate(raw)
    assert config.limits.context_tokens == 26406


def test_manifest_parses() -> None:
    manifest = Manifest.model_validate(
        {"format_version": 1, "files": {"a": "b" * 64}, "loadable_by": {"mimirai": ">=1,<2"}}
    )
    assert manifest.files == {"a": "b" * 64}
