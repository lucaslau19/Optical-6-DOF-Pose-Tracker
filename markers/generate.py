"""Print-ready target generation: ChArUco calibration board and single markers.

THE ONE THING THAT MATTERS HERE IS PHYSICAL SCALE.

Everything this project reports in millimetres traces back to the printed
size of these targets. If `config.yaml` says a square is 25.0 mm and the
printer scaled the page to 96%, every depth the tracker reports is 4% wrong
and nothing in the software will ever notice. So:

  1. Images are rendered at a whole number of pixels per square, at a known
     DPI, so the intended physical size is exact by construction.
  2. Every generated PNG is annotated with the dictionary, the nominal
     dimensions, and the size the user should *measure* on the print.
  3. The CLI prints a "verify with a ruler" step that is not optional.

This is the software equivalent of a length standard, and it is exactly the
kind of thing a V&V process would call out as a measurement-traceability
requirement.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.config import Config
from core.cv_compat import get_dictionary, make_charuco_board

MM_PER_INCH = 25.4


def mm_to_px(mm: float, dpi: int) -> float:
    return mm * dpi / MM_PER_INCH


def _round_to_multiple(value: float, multiple: int) -> int:
    """Round to the nearest positive whole multiple of `multiple`.

    Marker cells must land on whole pixels or the printed bits get uneven
    edges, which costs sub-pixel corner accuracy. Sizing every marker to an
    exact multiple of its cell count avoids resampling artefacts entirely.
    """
    n = max(1, int(round(value / multiple)))
    return n * multiple


def _annotate(image: np.ndarray, lines: list[str], dpi: int) -> np.ndarray:
    """Append a white caption strip below the target.

    A printed board with no label is unusable six months later -- you cannot
    tell a 25 mm board from a 30 mm one by eye, and using the wrong number
    silently scales every measurement.
    """
    scale = dpi / 300.0  # caption sized relative to the 300 dpi design point
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55 * scale
    thickness = max(1, int(round(1.4 * scale)))
    line_h = int(round(34 * scale))
    pad = int(round(20 * scale))

    strip_h = pad + line_h * len(lines)
    strip = np.full((strip_h, image.shape[1]), 255, dtype=np.uint8)

    for i, text in enumerate(lines):
        y = pad // 2 + line_h * (i + 1) - int(round(10 * scale))
        cv2.putText(strip, text, (pad, y), font, font_scale, 0, thickness, cv2.LINE_AA)

    return np.vstack([image, strip])


# --------------------------------------------------------------------------
# ChArUco calibration board
# --------------------------------------------------------------------------


def generate_charuco_board(cfg: Config, out_path: Path) -> dict[str, float]:
    """Render the ChArUco calibration board defined in config.yaml.

    Returns a dict of the *actual* physical dimensions implied by the pixel
    rounding, so the caller can tell the user what to measure.
    """
    ch = cfg.charuco
    board = make_charuco_board(
        ch.squares_x, ch.squares_y, ch.square_length_mm, ch.marker_length_mm, cfg.aruco.dictionary
    )

    # Whole pixels per chessboard square -> the grid lands on exact pixel
    # boundaries and the print has no resampling softness at the corners.
    # Those corners are the calibration measurement, so their sharpness is
    # not a cosmetic concern.
    square_px = int(round(mm_to_px(ch.square_length_mm, ch.render_dpi)))
    margin_px = int(round(mm_to_px(ch.margin_mm, ch.render_dpi)))

    content_w = square_px * ch.squares_x
    content_h = square_px * ch.squares_y
    out_size = (content_w + 2 * margin_px, content_h + 2 * margin_px)

    # borderBits=1: one black cell of border around each marker's payload,
    # which is what the detector expects and what the dictionary assumes.
    image = board.generateImage(out_size, marginSize=margin_px, borderBits=1)

    # What the print will actually measure, given integer-pixel rounding.
    actual_square_mm = square_px * MM_PER_INCH / ch.render_dpi
    actual_marker_mm = ch.marker_length_mm * (actual_square_mm / ch.square_length_mm)

    caption = [
        f"ChArUco {ch.squares_x}x{ch.squares_y}  dict={cfg.aruco.dictionary}"
        f"  square={actual_square_mm:.2f}mm  marker={actual_marker_mm:.2f}mm",
        f"PRINT AT 100% SCALE (no 'fit to page'). Then MEASURE one square: "
        f"it must be {actual_square_mm:.2f} mm.",
        "If it is not, put the measured value in charuco.square_length_mm and rerun.",
    ]
    image = _annotate(image, caption, ch.render_dpi)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)

    return {
        "square_mm": actual_square_mm,
        "marker_mm": actual_marker_mm,
        "board_width_mm": actual_square_mm * ch.squares_x,
        "board_height_mm": actual_square_mm * ch.squares_y,
        "pixels": (image.shape[1], image.shape[0]),
        "dpi": ch.render_dpi,
    }


# --------------------------------------------------------------------------
# Standalone single markers
# --------------------------------------------------------------------------


def generate_single_markers(cfg: Config, out_dir: Path) -> list[tuple[int, Path, float]]:
    """Render one print-ready PNG per marker ID listed in config.yaml.

    Returns [(marker_id, path, actual_side_mm), ...].
    """
    mk = cfg.markers
    dictionary = get_dictionary(cfg.aruco.dictionary)

    # markerSize is the payload grid (5 for DICT_5X5_*); +2 for the one-cell
    # black border on each side. Sizing the image to a whole multiple of the
    # total cell count keeps every bit exactly the same number of pixels.
    cells = int(dictionary.markerSize) + 2

    side_px = _round_to_multiple(mm_to_px(mk.marker_length_mm, mk.render_dpi), cells)
    margin_px = int(round(mm_to_px(mk.margin_mm, mk.render_dpi)))
    actual_side_mm = side_px * MM_PER_INCH / mk.render_dpi

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[int, Path, float]] = []

    for marker_id in mk.ids:
        # generateImageMarker is the 4.7+ name for the old aruco.drawMarker.
        marker = cv2.aruco.generateImageMarker(dictionary, int(marker_id), side_px, borderBits=1)

        # The white quiet zone is functional, not decorative: the detector
        # finds markers by their outer black contour, and without white
        # around it a marker printed to the edge of the paper (or butted up
        # against another marker) will not be found at all.
        padded = cv2.copyMakeBorder(
            marker, margin_px, margin_px, margin_px, margin_px, cv2.BORDER_CONSTANT, value=255
        )

        caption = [
            f"ArUco id={marker_id}  dict={cfg.aruco.dictionary}  side={actual_side_mm:.2f}mm",
            f"PRINT AT 100%. MEASURE the black square: {actual_side_mm:.2f} mm across.",
        ]
        annotated = _annotate(padded, caption, mk.render_dpi)

        path = out_dir / f"aruco_{cfg.aruco.dictionary}_id{marker_id:03d}.png"
        cv2.imwrite(str(path), annotated)
        results.append((int(marker_id), path, actual_side_mm))

    return results
