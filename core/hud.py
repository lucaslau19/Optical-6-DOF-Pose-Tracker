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


def draw_orientation_wheel(
    image: np.ndarray,
    tilts_deg: np.ndarray,
    azimuths_deg: np.ndarray,
    origin: tuple[int, int],
    *,
    radius: int = 62,
    target_cone_deg: float = 30.0,
) -> None:
    """Polar plot of the tool axis directions captured so far.

    Radius is tilt from the mean direction, angle is azimuth about it. A good
    pivot capture -- precessing the tool right around a wide cone -- fills an
    annulus at or beyond the dashed target ring. A bad one is a blob in the
    middle (never tilted enough) or an arc on one side (only swept half the
    azimuth), and both are obvious at a glance.

    This exists because orientation variety, not frame count, is what makes
    the tip solvable, and a bare counter would let someone collect 150 useless
    frames without noticing.
    """
    cx, cy = origin

    # Scale so the target cone sits comfortably inside the disc, leaving room
    # to show that the operator went further.
    display_max_deg = max(target_cone_deg * 1.8, 20.0)

    cv2.circle(image, (cx, cy), radius, (55, 55, 55), cv2.FILLED, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), radius, (110, 110, 110), 1, cv2.LINE_AA)

    # Dashed target ring at the minimum useful cone angle.
    target_r = int(radius * target_cone_deg / display_max_deg)
    for deg in range(0, 360, 14):
        a0, a1 = np.radians(deg), np.radians(deg + 7)
        p0 = (int(cx + target_r * np.cos(a0)), int(cy + target_r * np.sin(a0)))
        p1 = (int(cx + target_r * np.cos(a1)), int(cy + target_r * np.sin(a1)))
        cv2.line(image, p0, p1, AMBER, 1, cv2.LINE_AA)

    for tilt, az in zip(np.atleast_1d(tilts_deg), np.atleast_1d(azimuths_deg)):
        r = radius * min(float(tilt) / display_max_deg, 1.0)
        a = np.radians(float(az))
        p = (int(round(cx + r * np.cos(a))), int(round(cy + r * np.sin(a))))
        colour = GREEN if tilt >= target_cone_deg else GREY
        cv2.circle(image, p, 2, colour, -1, cv2.LINE_AA)

    cv2.circle(image, (cx, cy), 2, WHITE, -1, cv2.LINE_AA)


def status_colour(ok: bool, marginal: bool = False) -> tuple[int, int, int]:
    """Green / amber / red, used consistently across every screen."""
    if ok:
        return GREEN
    return AMBER if marginal else RED
