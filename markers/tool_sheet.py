"""Print-ready tool sheet: the physical realisation of the config geometry.

The sheet and config.yaml MUST agree, because the millimetres in config are
what make the reported pose metric. Generating the sheet from the same numbers
the tracker uses removes the most likely source of disagreement -- a hand-made
layout that does not quite match what was typed into the config.

It is still not proof. The printer can rescale the page, so the sheet carries
a 100 mm scale bar and the centre-to-centre distances to check with a ruler.
Measure, then correct config.yaml.

Tool frame is X right, Y DOWN and the page is X right, Y down, so the mapping
from tool millimetres to image pixels is a pure scale and offset with no axis
flip -- a small dividend from choosing the Y-down convention in Phase 1.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.config import Config
from core.cv_compat import get_dictionary
from markers.generate import MM_PER_INCH, _annotate, mm_to_px
from markers.tool import ToolGeometry

SCALE_BAR_MM = 100.0


def generate_tool_sheet(cfg: Config, geometry: ToolGeometry, out_path: Path) -> dict:
    """Render every member marker at its configured position on one sheet."""
    sheet = geometry.config.sheet
    dpi = sheet.render_dpi
    dictionary = get_dictionary(cfg.aruco.dictionary)

    # TRUE SCALE, deliberately -- unlike markers/generate.py, which rounds a
    # standalone marker up to a whole number of pixels per bit cell.
    #
    # That rounding is fine for a lone marker (the caption just reports the
    # size it came out at), but it is wrong here. The sheet has several
    # interacting dimensions -- marker size AND centre-to-centre spacing -- and
    # rounding the marker would scale the entire layout by the same factor to
    # keep them consistent. At 300 dpi that is a ~0.75% stretch: 60 mm spacing
    # would print at 60.45 mm, a systematic metric error in every pose.
    #
    # Rendering at exact scale instead means bit cells can land on fractional
    # pixel boundaries, so some cells are one pixel wider than others. That
    # costs nothing that matters: ArUco localises the marker's OUTER contour,
    # which is still exact, and bit decoding is thresholded and wholly
    # insensitive to a one-pixel cell difference at print resolution.
    px_per_mm = dpi / MM_PER_INCH
    side_px = int(round(mm_to_px(geometry.marker_length_mm, dpi)))
    actual_side_mm = side_px * MM_PER_INCH / dpi

    # Layout bounds come from the marker corners, so a rotated member is still
    # fully contained.
    lo, hi = geometry.extent_mm()
    margin = sheet.margin_mm
    origin_x_mm = lo[0] - margin
    origin_y_mm = lo[1] - margin
    width_mm = (hi[0] - lo[0]) + 2 * margin
    height_mm = (hi[1] - lo[1]) + 2 * margin

    def to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
        """Tool millimetres -> sheet pixels. No axis flip: both are Y-down."""
        return (
            int(round((x_mm - origin_x_mm) * px_per_mm)),
            int(round((y_mm - origin_y_mm) * px_per_mm)),
        )

    canvas_w = int(round(width_mm * px_per_mm))
    canvas_h = int(round(height_mm * px_per_mm))
    sheet_img = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

    for marker_id in geometry.ids:
        member = geometry.markers[marker_id]
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px, borderBits=1)

        if member.rotation_deg:
            # Rotating a bitmap at an arbitrary angle resamples it. NEAREST
            # keeps hard black/white edges (no grey fringe) at the cost of
            # slightly ragged diagonals -- the better trade for a target whose
            # corners get localised to sub-pixel precision. Multiples of 90
            # degrees are exact either way.
            M = cv2.getRotationMatrix2D((side_px / 2.0, side_px / 2.0), -member.rotation_deg, 1.0)
            marker = cv2.warpAffine(
                marker,
                M,
                (side_px, side_px),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )

        cx, cy = to_px(member.center[0], member.center[1])
        x0, y0 = cx - side_px // 2, cy - side_px // 2
        x1, y1 = x0 + side_px, y0 + side_px
        if x0 < 0 or y0 < 0 or x1 > canvas_w or y1 > canvas_h:
            raise ValueError(
                f"Member {marker_id} falls outside the sheet. Increase "
                "tool.sheet.margin_mm in config.yaml."
            )
        sheet_img[y0:y1, x0:x1] = marker

        # Label each marker with its ID and configured centre, so the printed
        # sheet is self-documenting against config.yaml.
        cv2.putText(
            sheet_img,
            f"id {marker_id} @ ({member.center[0]:.0f}, {member.center[1]:.0f})",
            (x0, y0 - int(8 * dpi / 300)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 * dpi / 300,
            0,
            max(1, int(dpi / 300)),
            cv2.LINE_AA,
        )

    _draw_origin_and_axes(sheet_img, to_px, px_per_mm, dpi)
    _draw_scale_bar(sheet_img, to_px, px_per_mm, dpi, lo, hi, margin)

    centres = ", ".join(
        f"{i}({geometry.markers[i].center[0]:.0f},{geometry.markers[i].center[1]:.0f})"
        for i in geometry.ids
    )
    caption = [
        f"TOOL '{geometry.name}'  dict={cfg.aruco.dictionary}  "
        f"marker={actual_side_mm:.2f}mm  centres(mm): {centres}",
        "PRINT AT 100% SCALE (no 'fit to page'). Then MEASURE the 100 mm scale bar",
        "and the marker centre-to-centre spacing, and correct tool.* in config.yaml.",
    ]
    sheet_img = _annotate(sheet_img, caption, dpi)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet_img)

    return {
        "marker_mm": actual_side_mm,
        "sheet_mm": (width_mm, height_mm),
        "pixels": (sheet_img.shape[1], sheet_img.shape[0]),
        "dpi": dpi,
    }


def _draw_origin_and_axes(img: np.ndarray, to_px, px_per_mm: float, dpi: int) -> None:
    """Mark the tool origin and the +X / +Y directions on the sheet.

    Worth printing: the origin is usually empty space (the centroid of the
    cluster), and without a mark there is nothing on the page to tell you
    where the reported pose is actually anchored -- which matters the moment
    Phase 3 measures a tip offset from it.
    """
    ox, oy = to_px(0.0, 0.0)
    arm = int(round(12.0 * px_per_mm))
    thick = max(1, int(round(dpi / 300)))
    font_scale = 0.5 * dpi / 300

    cv2.line(img, (ox - arm, oy), (ox + arm, oy), 0, thick, cv2.LINE_AA)
    cv2.line(img, (ox, oy - arm), (ox, oy + arm), 0, thick, cv2.LINE_AA)
    cv2.circle(img, (ox, oy), max(2, arm // 6), 0, thick, cv2.LINE_AA)

    cv2.arrowedLine(img, (ox, oy), (ox + 2 * arm, oy), 0, thick, cv2.LINE_AA, tipLength=0.2)
    cv2.arrowedLine(img, (ox, oy), (ox, oy + 2 * arm), 0, thick, cv2.LINE_AA, tipLength=0.2)
    cv2.putText(img, "+X", (ox + 2 * arm + 6, oy + 6), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, 0, thick, cv2.LINE_AA)
    cv2.putText(img, "+Y", (ox + 6, oy + 2 * arm + 20), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, 0, thick, cv2.LINE_AA)
    cv2.putText(img, "TOOL ORIGIN (0,0)  +Z into page",
                (ox - arm, oy - arm - int(10 * dpi / 300)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, thick, cv2.LINE_AA)


def _draw_scale_bar(img, to_px, px_per_mm: float, dpi: int, lo, hi, margin: float) -> None:
    """A 100 mm bar to check the print scale with a ruler.

    This is the sheet's own length standard. If the bar does not measure
    100.0 mm, the page was rescaled and every millimetre the tracker reports
    is wrong by the same factor -- a systematic error no amount of averaging
    will remove.
    """
    thick = max(1, int(round(dpi / 300)))
    y_mm = hi[1] + margin * 0.55
    x0_mm = lo[0]
    x1_mm = lo[0] + SCALE_BAR_MM

    x0, y0 = to_px(x0_mm, y_mm)
    x1, _ = to_px(x1_mm, y_mm)
    if x1 >= img.shape[1] or y0 >= img.shape[0]:
        return  # sheet too small for the bar; the caption still says to measure

    tick = int(round(2.5 * px_per_mm))
    cv2.line(img, (x0, y0), (x1, y0), 0, thick, cv2.LINE_AA)
    for x in (x0, x1):
        cv2.line(img, (x, y0 - tick), (x, y0 + tick), 0, thick, cv2.LINE_AA)
    # 10 mm minor ticks
    for i in range(1, int(SCALE_BAR_MM // 10)):
        xi, _ = to_px(x0_mm + 10.0 * i, y_mm)
        cv2.line(img, (xi, y0), (xi, y0 + tick // 2), 0, thick, cv2.LINE_AA)
    cv2.putText(
        img,
        f"{SCALE_BAR_MM:.0f} mm -- measure me",
        (x0, y0 - tick - int(6 * dpi / 300)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5 * dpi / 300,
        0,
        thick,
        cv2.LINE_AA,
    )
