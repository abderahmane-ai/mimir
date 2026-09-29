import json
from pathlib import Path

import numpy as np
import pytest

from mimir.core.errors import FormatVersionError, PolicyError
from mimir.policy.document import (
    Configuration,
    Fingerprint,
    Policy,
    Scaling,
    TypePolicy,
    expected_arrays,
    parse_policy,
    read_policy,
    risk_key,
    write_policy,
)
from mimir.runtime.session import torch_version
from tests.conftest import build_policy

FINGERPRINT = Fingerprint(
    model="m",
    revision="r",
    variant="fp32",
    weights_sha256="b" * 64,
    torch=torch_version(),
    configurations=(Configuration(device="cpu", hardware="h"),),
)


def _document(policy: Policy) -> dict[str, object]:
    loaded: dict[str, object] = json.loads(policy.document.model_dump_json())
    return loaded


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    policy = build_policy(FINGERPRINT)
    write_policy(policy, tmp_path / "p.json", tmp_path / "p.npz")
    loaded = read_policy(tmp_path / "p.json", tmp_path / "p.npz")
    assert loaded.document == policy.document
    assert set(loaded.arrays) == set(policy.arrays)
    for name, value in policy.arrays.items():
        np.testing.assert_array_equal(loaded.arrays[name], value)
    assert not list(tmp_path.glob("*.part"))


def test_arrays_must_match_the_document_exactly() -> None:
    policy = build_policy(FINGERPRINT)
    arrays = dict(policy.arrays)
    arrays.pop("conformal/binary/0")
    arrays["stray"] = np.zeros(4)
    with pytest.raises(PolicyError, match=r"missing \['conformal/binary/0'\].*unexpected"):
        Policy(policy.document, arrays)


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("conformal/binary/0", np.array([3.0, 1.0]), "not sorted"),
        ("conformal/ordinal", np.array([[0.1]]), "not sorted and 1-D"),
        ("conformal/binary/0", np.array([0.0, np.nan, 0.0])[None], "non-finite"),
        ("conformal/binary/0", np.zeros(4, dtype=np.float32), "float32"),
    ],
)
def test_malformed_arrays_are_rejected(name: str, value: np.ndarray, match: str) -> None:
    policy = build_policy(FINGERPRINT)
    arrays = {**policy.arrays, name: value}
    with pytest.raises(PolicyError, match=match):
        Policy(policy.document, arrays)


def test_parse_rejects_bad_json_versions_and_fields() -> None:
    policy = build_policy(FINGERPRINT)
    document = _document(policy)
    with pytest.raises(PolicyError, match="not JSON"):
        parse_policy(b"{", policy.arrays, "x")
    with pytest.raises(FormatVersionError, match="format 2"):
        parse_policy(json.dumps({**document, "format_version": 2}).encode(), policy.arrays, "x")
    with pytest.raises(PolicyError, match="label_taken"):
        parse_policy(json.dumps({**document, "label_taken": 0.6}).encode(), policy.arrays, "x")
    with pytest.raises(PolicyError, match="Extra inputs"):
        parse_policy(json.dumps({**document, "sources": 12}).encode(), policy.arrays, "x")


def test_scaling_of_picks_the_largest_bucket_not_above() -> None:
    policy = TypePolicy(
        scaling=(Scaling(bucket=2, parameters=(1.1,)), Scaling(bucket=4, parameters=(0.9,))),
        conformal=None,
        thresholds={},
    )
    assert policy.scaling_of(2).bucket == 2
    assert policy.scaling_of(3).bucket == 2
    assert policy.scaling_of(16).bucket == 4
    assert policy.scaling_of(150).bucket == 4


def test_scaling_buckets_must_ascend() -> None:
    with pytest.raises(ValueError, match="ascending"):
        TypePolicy(
            scaling=(Scaling(bucket=3, parameters=(1.0,)), Scaling(bucket=1, parameters=(1.0,))),
            conformal=None,
            thresholds={},
        )


def test_risk_levels_and_conformal_lookup() -> None:
    policy = build_policy(FINGERPRINT)
    assert policy.risk_levels == (0.01,)
    assert policy.conformal_scores("binary", 0) is policy.arrays["conformal/binary/0"]
    assert policy.conformal_scores("ordinal", 3) is policy.arrays["conformal/ordinal"]
    assert policy.conformal_scores("ranking", 0) is None
    assert "conformal/binary/0" in expected_arrays(policy.document)


@pytest.mark.parametrize(("risk", "key"), [(0.01, "0.01"), (0.005, "0.005"), (0.05, "0.05")])
def test_risk_key(risk: float, key: str) -> None:
    assert risk_key(risk) == key
