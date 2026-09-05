"""Generic on-screen drawing helpers shared by every live view.

Kept separate from the pose-specific overlay so that the calibration screen,
the tracking screen and (later) the navigation screen all look and behave
like one instrument rather than three unrelated scripts.

Colours are BGR because that is what OpenCV wants.
"""

from __future__ import annotations

import cv2
import numpy as np

# A small, deliberately limited palette. Green/amber/red carry meaning
# (in tolerance / marginal / out of tolerance) and are not used decoratively,
# so that when the navigation HUD turns red in Phase 4 it reads instantly.
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
GREEN = (80, 220, 100)
AMBER = (60, 190, 245)
RED = (70, 70, 240)
CYAN = (230, 200, 60)
BLACK = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_text_panel(
    image: np.ndarray,
    lines: list[str] | list[tuple[str, tuple[int, int, int]]],
    origin: tuple[int, int] = (12, 12),
    *,
    scale: float = 0.5,
    line_height: int = 22,
    pad: int = 10,
    bg_alpha: float = 0.55,
    width: int | None = None,
) -> np.ndarray:
    """Draw a translucent panel of text.

    The dark backing matters: white text alone becomes unreadable the moment
    a white calibration board fills the frame, which is most of the time in
    this application.
    """
    if not lines:
        return image

    norm: list[tuple[str, tuple[int, int, int]]] = [
        (ln, WHITE) if isinstance(ln, str) else ln for ln in lines  # type: ignore[misc]
    ]

    thickness = 1
    if width is None:
        widest = max(
            cv2.getTextSize(text, FONT, scale, thickness)[0][0] for text, _ in norm
        )
        width = widest + 2 * pad

    height = 2 * pad + line_height * len(norm)
    x0, y0 = origin
    x1 = min(x0 + width, image.shape[1] - 1)
    y1 = min(y0 + height, image.shape[0] - 1)

    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), BLACK, thickness=cv2.FILLED)
    cv2.addWeighted(overlay, bg_alpha, image, 1.0 - bg_alpha, 0, dst=image)
    cv2.rectangle(image, (x0, y0), (x1, y1), (60, 60, 60), 1, cv2.LINE_AA)

    # putText takes the text BASELINE, not the top of the glyphs, so each line
    # is offset down by roughly the cap height within its slot.
    baseline_offset = int(round(line_height * 0.65))
    for i, (text, colour) in enumerate(norm):
        y = y0 + pad + line_height * i + baseline_offset
        cv2.putText(image, text, (x0 + pad, y), FONT, scale, colour, thickness, cv2.LINE_AA)

    return image


def draw_coverage_grid(
    image: np.ndarray,
    occupied: np.ndarray,
    origin: tuple[int, int],
    *,
    cell: int = 14,
) -> None:
    """Miniature map of which parts of the frame the board has visited.

    Calibration quality depends on covering the image area -- particularly
    the corners, where lens distortion is largest. A number cannot convey
    "you never went to the top-left"; a picture can, and the operator can act
    on it while still holding the board.
    """
    grid = occupied.shape[0]
    x0, y0 = origin
    for r in range(grid):
        for c in range(grid):
            x, y = x0 + c * cell, y0 + r * cell
            colour = GREEN if occupied[r, c] else (55, 55, 55)
            cv2.rectangle(image, (x, y), (x + cell - 2, y + cell - 2), colour, cv2.FILLED)
    cv2.rectangle(
        image, (x0 - 2, y0 - 2), (x0 + grid * cell, y0 + grid * cell), (110, 110, 110), 1
    )


def status_colour(ok: bool, marginal: bool = False) -> tuple[int, int, int]:
    """Green / amber / red, used consistently across every screen."""
    if ok:
        return GREEN
    return AMBER if marginal else RED
