import numpy as np
import pytest

from mimir.runtime.layout import Batch
from mimir.runtime.readout import OUTPUTS, readout_at, split_outputs


def _outputs() -> dict[str, np.ndarray]:
    return {
        "utilities": np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]], dtype=np.float32),
        "thresholds": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "abstain": np.array([0.5, 0.6], dtype=np.float32),
        "ordinal_score": np.array([0.7, 0.8], dtype=np.float32),
        "histogram_logits": np.zeros((2, 4), dtype=np.float32),
        "evidence": np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.0, 0.0]], dtype=np.float32),
        "workspace": np.ones((2, 3), dtype=np.float32),
    }


def test_readout_cuts_each_request_to_its_own_options() -> None:
    values = {name: np.asarray(value, dtype=np.float64) for name, value in _outputs().items()}
    first = readout_at(values, 0, "categorical", 3)
    second = readout_at(values, 1, "binary", 1)
    assert first.utilities.tolist() == [1.0, 2.0, 3.0]
    assert first.thresholds.tolist() == pytest.approx([0.1, 0.2])
    assert second.utilities.tolist() == [4.0]
    assert second.thresholds.tolist() == []
    assert second.abstain == pytest.approx(0.6)


def test_relevance_sums_evidence_per_state_segment_and_ignores_cls_and_padding() -> None:
    batch = Batch(
        feed={"key_mask": np.array([[True, True, True, True], [True, True, False, False]])},
        key_segments=np.array([[-1, 0, 1, 1], [-1, 0, -1, -1]]),
        counts=(3, 1),
    )
    split = split_outputs(_outputs(), batch, ["categorical", "binary"], [2, 1])
    assert split[0].relevance.tolist() == pytest.approx([0.2, 0.7])
    assert split[1].relevance.tolist() == pytest.approx([0.5])


def test_outputs_are_the_contract_outputs() -> None:
    assert set(OUTPUTS) == set(_outputs())
