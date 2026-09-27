"""Print the project banner in holographic foil colours to stderr."""

import math
import os
import sys
import time
from typing import Final

ART: Final = (
    "███╗   ███╗██╗███╗   ███╗██╗██████╗ ",
    "████╗ ████║██║████╗ ████║██║██╔══██╗",
    "██╔████╔██║██║██╔████╔██║██║██████╔╝",
    "██║╚██╔╝██║██║██║╚██╔╝██║██║██╔══██╗",
    "██║ ╚═╝ ██║██║██║ ╚═╝ ██║██║██║  ██║",
    "╚═╝     ╚═╝╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═╝",
)
FACE: Final = "█"
RESET: Final = "\033[0m"
SHIMMER_PERIOD_S: Final = 6.0


def foil_rgb(column: int, row: int, phase: float, width: int) -> tuple[int, int, int]:
    angle = 2 * math.pi * (column + 2 * row) / width + 2 * math.pi * phase
    base = [170 + 85 * math.sin(angle + offset) for offset in (0.0, 2.094, 4.189)]
    band = (column - 1.5 * row) - phase * (width + 12) + 6
    glint = 0.75 * math.exp(-((band / 3.0) ** 2))
    red, green, blue = (round(channel + (255 - channel) * glint) for channel in base)
    return red, green, blue


def render(*, is_colour: bool, phase: float) -> str:
    if not is_colour:
        return "\n".join(ART) + "\n"
    width = len(ART[0])
    lines = []
    for row, line in enumerate(ART):
        cells = []
        for column, character in enumerate(line):
            if character == " ":
                cells.append(character)
                continue
            red, green, blue = foil_rgb(column, row, phase, width)
            if character != FACE:
                red, green, blue = (round(channel * 0.55) for channel in (red, green, blue))
            cells.append(f"\033[38;2;{red};{green};{blue}m{character}")
        lines.append("".join(cells) + RESET)
    return "\n".join(lines) + "\n"


def main() -> None:
    is_colour = sys.stderr.isatty() and "NO_COLOR" not in os.environ
    phase = (time.time() % SHIMMER_PERIOD_S) / SHIMMER_PERIOD_S
    sys.stderr.write(render(is_colour=is_colour, phase=phase))


if __name__ == "__main__":
    main()
