"""Typed numbers and dates as piecewise-linear features, encoded into the encoder's hidden width.

Piecewise-linear encoding over quantile bins: Gorishniy et al., NeurIPS 2022.

Copied from the training repository; held to it by the golden fixtures. The quantile edges
are buffers of the encoder, shipped in the weights; cutting them is training-time code and
lives there.
"""

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from mimir.runtime.model.config import HeadConfig

NUMBER: Final = 0
DATE: Final = 1
CYCLES: Final = 4
SECONDS_PER_DAY: Final = 86_400
# The civil-day algorithm counts years from March, so January and February join the year before.
FEBRUARY: Final = 2
# 1970-01-01 was a Thursday; with Monday as 0 that is weekday 3.
EPOCH_WEEKDAY: Final = 3


class TypedValueError(ValueError):
    """Typed values that cannot be binned."""


@dataclass(frozen=True, slots=True)
class TypedValues:
    kind: torch.Tensor
    number: torch.Tensor
    year: torch.Tensor
    month: torch.Tensor
    day: torch.Tensor
    second: torch.Tensor


def signed_log(values: torch.Tensor) -> torch.Tensor:
    return values.sign() * values.abs().log1p()


def days_from_civil(year: torch.Tensor, month: torch.Tensor, day: torch.Tensor) -> torch.Tensor:
    """Days since 1970-01-01 in the proleptic Gregorian calendar (Hinnant's algorithm)."""
    shifted = year - (month <= FEBRUARY).long()
    era = torch.div(shifted, 400, rounding_mode="floor")
    year_of_era = shifted - era * 400
    month_index = (month + 9) % 12
    day_of_year = torch.div(153 * month_index + 2, 5, rounding_mode="floor") + day - 1
    day_of_era = (
        year_of_era * 365
        + torch.div(year_of_era, 4, rounding_mode="floor")
        - torch.div(year_of_era, 100, rounding_mode="floor")
        + day_of_year
    )
    return era * 146_097 + day_of_era - 719_468


def piecewise_linear(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """[n] values over T+1 edges -> [n, T]; the first bin is open below, the last above."""
    low, high = edges[:-1], edges[1:]
    width = high - low
    safe = torch.where(width > 0, width, torch.ones_like(width))
    fraction = (values[:, None] - low) / safe
    fraction = torch.where(width > 0, fraction, (values[:, None] >= high).to(values.dtype))
    bins = edges.shape[0] - 1
    first = torch.arange(bins, device=values.device) == 0
    last = torch.arange(bins, device=values.device) == bins - 1
    fraction = torch.where(first, fraction, fraction.clamp(min=0))
    return torch.where(last, fraction, fraction.clamp(max=1))


class TypedValueEncoder(nn.Module):
    """Number bins, day bins and 4 date cycles -> MLP -> encoder width; zero at initialisation."""

    def __init__(
        self,
        config: HeadConfig,
        number_edges: torch.Tensor,
        day_edges: torch.Tensor,
    ) -> None:
        super().__init__()
        for name, edges in (("number_edges", number_edges), ("day_edges", day_edges)):
            if edges.shape != (config.value_bins + 1,) or bool((edges.diff() < 0).any()):
                message = f"{name} of shape {edges.shape} for {config.value_bins} sorted bins"
                raise TypedValueError(message)
        self.number_edges: torch.Tensor
        self.day_edges: torch.Tensor
        self.register_buffer("number_edges", number_edges.float())
        self.register_buffer("day_edges", day_edges.float())
        features = 2 * config.value_bins + 2 * CYCLES
        self.hidden = nn.Linear(features, config.width, bias=False)
        self.output = nn.Linear(config.width, config.encoder_width, bias=False)
        nn.init.zeros_(self.output.weight)

    def features(self, values: TypedValues) -> torch.Tensor:
        is_number = values.kind == NUMBER
        is_date = values.kind == DATE
        number = torch.nan_to_num(values.number.float(), nan=0.0, posinf=0.0, neginf=0.0)
        number_part = piecewise_linear(signed_log(number), self.number_edges)
        days = days_from_civil(values.year, values.month.clamp(min=1), values.day.clamp(min=1))
        day_part = piecewise_linear(days.float(), self.day_edges)
        weekday = (days + EPOCH_WEEKDAY) % 7
        angles = (
            2
            * math.pi
            * torch.stack(
                [
                    (values.month.float() - 1) / 12,
                    (values.day.float() - 1) / 31,
                    weekday.float() / 7,
                    values.second.float() / SECONDS_PER_DAY,
                ],
                dim=-1,
            )
        )
        cycles = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return torch.cat(
            [
                number_part * is_number[:, None],
                day_part * is_date[:, None],
                cycles * is_date[:, None],
            ],
            dim=-1,
        )

    def forward(self, values: TypedValues) -> torch.Tensor:
        encoded: torch.Tensor = self.output(nn.functional.silu(self.hidden(self.features(values))))
        return encoded
