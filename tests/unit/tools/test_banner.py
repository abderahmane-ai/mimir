import re

import pytest

from banner import ART, foil_rgb, render

ESCAPE = re.compile(r"\033\[[0-9;]*m")
PHASES = [0.0, 0.25, 0.5, 0.999]


def test_plain_render_is_the_art_exactly() -> None:
    assert render(is_colour=False, phase=0.0) == "\n".join(ART) + "\n"


@pytest.mark.parametrize("phase", PHASES)
def test_colour_render_keeps_every_character_of_the_art(phase: float) -> None:
    coloured = render(is_colour=True, phase=phase)
    assert ESCAPE.sub("", coloured) == render(is_colour=False, phase=phase)


@pytest.mark.parametrize("phase", PHASES)
def test_colour_render_colours_every_visible_character(phase: float) -> None:
    coloured = render(is_colour=True, phase=phase)
    visible = sum(character != " " for line in ART for character in line)
    assert coloured.count("\033[38;2;") == visible


@pytest.mark.parametrize("phase", PHASES)
def test_foil_rgb_stays_in_channel_range(phase: float) -> None:
    width = len(ART[0])
    channels = [
        channel
        for row in range(len(ART))
        for column in range(width)
        for channel in foil_rgb(column, row, phase, width)
    ]
    assert min(channels) >= 0
    assert max(channels) <= 255


def test_foil_colour_shifts_with_phase() -> None:
    width = len(ART[0])
    assert foil_rgb(10, 2, 0.0, width) != foil_rgb(10, 2, 0.5, width)
