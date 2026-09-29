"""Torch inference: weights, device and the forward run.

The release ships fp32 safetensors with the encoder, the head and the typed-value edges.
`load_model` builds the encoder from the release's encoder config and loads the weights
strictly; `run` executes one collated batch and returns the exit depth's outputs.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import torch
from safetensors.torch import load_file
from transformers import ModernBertConfig, ModernBertModel

from mimir.runtime.model.batch import DecisionBatch, DecisionTargets
from mimir.runtime.model.config import FULL
from mimir.runtime.model.decision import DecisionModel
from mimir.runtime.model.head import DecisionHead
from mimir.runtime.model.typed_values import TypedValues

Device = Literal["cpu", "cuda"]
# The calibration exits every record at depth 2, so the release serves that depth's readout
# with the final workspace.
EXIT_DEPTH: Final = 1
ENCODER_CONFIG_FILE: Final = "encoder_config.json"
ATTENTION: Final = "sdpa"

Array = npt.NDArray[np.generic]
Floats = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """The decision model on its device, in eval mode."""

    model: DecisionModel
    device: torch.device


def torch_version() -> str:
    version: str = torch.__version__
    return version


def resolve_device(device: str) -> Device:
    """Resolve `auto`, `cpu` or `cuda`; `auto` selects CUDA when torch sees a GPU."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        message = "device 'cuda' needs a CUDA torch build with a visible GPU"
        raise ValueError(message)
    message = f"device {device!r} is not one of 'auto', 'cpu', 'cuda'"
    raise ValueError(message)


def _tensors(feed: Mapping[str, Array], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(np.ascontiguousarray(value)).to(device)
        for name, value in feed.items()
    }


def load_model(weights: Path, encoder_config: Path, device: Device) -> LoadedModel:
    """Build the model and load `weights` strictly. Raises on any missing or extra tensor."""
    state = load_file(str(weights), device="cpu")
    encoder = ModernBertModel(ModernBertConfig.from_json_file(str(encoder_config)))
    encoder.set_attn_implementation(ATTENTION)
    head = DecisionHead(
        FULL,
        state["head.typed_values.number_edges"],
        state["head.typed_values.day_edges"],
    )
    model = DecisionModel(encoder, head)
    model.load_state_dict(state, strict=True)
    torch_device = torch.device(device)
    return LoadedModel(model.float().to(torch_device).eval(), torch_device)


def run(model: LoadedModel, feed: Mapping[str, Array], question_cap: int) -> dict[str, Floats]:
    """Run one collated batch; return the exit depth's outputs as float32 arrays."""
    device = model.device
    values = _tensors(feed, device)
    empty_long = torch.zeros(0, dtype=torch.long, device=device)
    empty_float = torch.zeros(0, dtype=torch.float32, device=device)
    empty_bool = torch.zeros(0, dtype=torch.bool, device=device)
    batch = DecisionBatch(
        chunk_ids=values["chunk_ids"].long(),
        chunk_mask=values["chunk_mask"].bool(),
        chunk_record=values["chunk_record"].long(),
        question_length=values["question_length"].long(),
        key_chunk=values["key_chunk"].long(),
        key_position=values["key_position"].long(),
        key_mask=values["key_mask"].bool(),
        candidate_ids=values["candidate_ids"].long(),
        candidate_mask=values["candidate_mask"].bool(),
        candidate_text_mask=values["candidate_text_mask"].bool(),
        candidate_index=values["candidate_index"].long(),
        candidate_present=values["candidate_present"].bool(),
        typed=TypedValues(
            kind=values["typed_kind"].long(),
            number=values["typed_number"].float(),
            year=values["typed_year"].long(),
            month=values["typed_month"].long(),
            day=values["typed_day"].long(),
            second=values["typed_second"].long(),
        ),
        typed_chunk=values["typed_chunk"].long(),
        typed_position=values["typed_position"].long(),
        targets=DecisionTargets(
            decision_type=values["decision_type"].long(),
            choice=empty_long,
            labels=empty_bool,
            grades=empty_long,
            value=empty_float,
            low=empty_float,
            high=empty_float,
        ),
    )
    with torch.no_grad():
        outputs = model.model.forward_padded(batch, question_cap)
    found = outputs[EXIT_DEPTH]
    named = {
        "utilities": found.readout.utilities,
        "thresholds": found.readout.thresholds,
        "abstain": found.readout.abstain,
        "ordinal_score": found.readout.ordinal_score,
        "histogram_logits": found.readout.histogram_logits,
        "evidence": found.evidence_scores,
        "workspace": outputs[-1].workspace,
    }
    return {name: tensor.float().cpu().numpy() for name, tensor in named.items()}
