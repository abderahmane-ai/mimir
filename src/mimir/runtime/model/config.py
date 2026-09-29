"""The decision head's sizes: the full configuration of DESIGN §15 and the checks it must pass."""

from dataclasses import dataclass
from typing import Final


class ConfigError(ValueError):
    """A head configuration that cannot be built."""


@dataclass(frozen=True, slots=True)
class HeadConfig:
    encoder_width: int
    encoder_layers: int
    trained_layers: int
    width: int
    head_size: int
    feedforward_hidden: int
    latents: int
    slots: int
    iterations: int
    value_bins: int
    histogram_bins: int

    def __post_init__(self) -> None:
        if self.width % self.head_size:
            message = f"width {self.width} is not a multiple of head size {self.head_size}"
            raise ConfigError(message)
        if not 0 < self.trained_layers <= self.encoder_layers:
            message = f"{self.trained_layers} trained layers of {self.encoder_layers}"
            raise ConfigError(message)
        for name in ("latents", "slots", "iterations", "value_bins", "histogram_bins"):
            if getattr(self, name) < 1:
                message = f"{name} is {getattr(self, name)}"
                raise ConfigError(message)

    @property
    def heads(self) -> int:
        return self.width // self.head_size

    @property
    def injection_layer(self) -> int:
        return self.encoder_layers - self.trained_layers


FULL: Final = HeadConfig(
    encoder_width=1024,
    encoder_layers=28,
    trained_layers=14,
    width=512,
    head_size=64,
    feedforward_hidden=1408,
    latents=32,
    slots=4,
    iterations=4,
    value_bins=64,
    histogram_bins=64,
)
