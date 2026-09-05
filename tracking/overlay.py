"""Pose-specific drawing: axes, marker outlines, per-marker readouts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from core.hud import AMBER, BLACK, CYAN, GREEN, GREY, RED, WHITE, draw_text_panel
from core.intrinsics import CameraIntrinsics
from tracking.single_marker import MarkerPose

if TYPE_CHECKING:  # imported for typing only, to keep this module import-light
    from markers.tool import ToolGeometry
    from tracking.rigid_body import ToolPose


def draw_marker_axes(
    image: np.ndarray,
    marker: MarkerPose,
    intrinsics: CameraIntrinsics,
    axis_length_mm: float,
) -> None:
    """Draw the marker's coordinate frame: X red, Y green, Z blue.

    `drawFrameAxes` takes the distortion coefficients, so the drawn axes
    follow the same lens model as the pose solution -- draw them with a
    pinhole projection instead and they will visibly miss the marker near the
    frame edges, which looks like a pose error but is a drawing error.

    Z points INTO the marker face (away from the camera), so on a marker held
    square-on the blue axis is a dot at the centre, and it swings away from
    the viewer as the marker tilts. If blue ever swings towards you instead,
    the pose has flipped to the ambiguous solution -- the quickest visual
    check there is.
    """
    rvec, tvec = marker.pose.as_rvec_tvec()
    cv2.drawFrameAxes(
        image,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        rvec,
        tvec,
        float(axis_length_mm),
        3,
    )


def draw_marker_outline(image: np.ndarray, marker: MarkerPose) -> None:
    """Outline the marker and label it, colour-coded by pose trustworthiness."""
    pts = marker.corners.astype(np.int32).reshape(-1, 1, 2)
    colour = AMBER if marker.is_ambiguous else GREEN
    cv2.polylines(image, [pts], isClosed=True, color=colour, thickness=2, lineType=cv2.LINE_AA)

    # Mark corner 0 (top-left in the marker frame) so the marker's own
    # orientation is readable on screen -- useful when checking that the
    # object-point ordering matches the detector's.
    c0 = marker.corners[0].astype(int)
    cv2.circle(image, tuple(c0), 5, colour, -1, cv2.LINE_AA)

    centre = marker.corners.mean(axis=0).astype(int)
    cv2.putText(
        image,
        f"{marker.marker_id}",
        (int(centre[0]) - 10, int(centre[1]) + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        colour,
        2,
        cv2.LINE_AA,
    )


def draw_pose_readout(
    image: np.ndarray,
    markers: list[MarkerPose],
    origin: tuple[int, int] = (12, 12),
    max_markers: int = 4,
) -> None:
    """Numeric 6-DOF readout for the visible markers."""
    lines: list[tuple[str, tuple[int, int, int]]] = []

    if not markers:
        lines.append(("no markers detected", RED))
    for marker in markers[:max_markers]:
        x, y, z = marker.pose.t
        rx, ry, rz = marker.pose.as_euler_deg()
        colour = AMBER if marker.is_ambiguous else GREEN

        lines.append((f"id {marker.marker_id}   range {marker.distance_mm:7.1f} mm", colour))
        lines.append((f"   x {x:8.1f}   y {y:8.1f}   z {z:8.1f}  mm", WHITE))
        lines.append((f"   rx{rx:8.1f}  ry{ry:8.1f}  rz{rz:8.1f}  deg", WHITE))

        amb = "inf" if not np.isfinite(marker.ambiguity_ratio) else f"{marker.ambiguity_ratio:.2f}"
        amb_note = "  AMBIGUOUS -- pose may flip" if marker.is_ambiguous else ""
        lines.append(
            (
                f"   reproj {marker.reprojection_rms_px:.2f} px   amb {amb}{amb_note}",
                AMBER if marker.is_ambiguous else GREY,
            )
        )

    if len(markers) > max_markers:
        lines.append((f"   ... and {len(markers) - max_markers} more", GREY))

    draw_text_panel(image, lines, origin=origin)


# --------------------------------------------------------------------------
# Phase 2: rigid-body tool
# --------------------------------------------------------------------------


def draw_tool_axes(
    image: np.ndarray,
    tool: "ToolPose",
    intrinsics: CameraIntrinsics,
    axis_length_mm: float,
) -> None:
    """Draw ONE set of axes at the tool origin -- not per-marker axes.

    A single frame for the whole rigid body is the visible difference between
    Phase 1 and Phase 2, and it is what the tip offset (Phase 3) and the
    reference-frame maths (Phase 4) will be expressed in.
    """
    if tool.pose is None:
        return
    rvec, tvec = tool.pose.as_rvec_tvec()
    cv2.drawFrameAxes(
        image,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        rvec,
        tvec,
        float(axis_length_mm),
        4,
    )


def _project(points_tool: np.ndarray, tool: "ToolPose", intrinsics: CameraIntrinsics) -> np.ndarray:
    rvec, tvec = tool.pose.as_rvec_tvec()  # type: ignore[union-attr]
    projected, _ = cv2.projectPoints(
        np.asarray(points_tool, dtype=np.float64),
        rvec,
        tvec,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    )
    return projected.reshape(-1, 2)


def draw_tool_origin(
    image: np.ndarray, tool: "ToolPose", intrinsics: CameraIntrinsics
) -> None:
    """Reproject the tool origin as a dot.

    The point of this dot is stability: it should stay pinned to the same
    physical spot on the tool while markers are covered and uncovered. A dot
    is far easier to judge that on than three axis lines.
    """
    if tool.pose is None:
        return
    origin = _project(np.zeros((1, 3)), tool, intrinsics)[0]
    if not np.all(np.isfinite(origin)):
        return
    centre = (int(round(origin[0])), int(round(origin[1])))
    cv2.circle(image, centre, 9, BLACK, 3, cv2.LINE_AA)
    cv2.circle(image, centre, 9, WHITE, 1, cv2.LINE_AA)
    cv2.circle(image, centre, 3, WHITE, -1, cv2.LINE_AA)


def draw_tool_markers(
    image: np.ndarray,
    tool: "ToolPose",
    geometry: "ToolGeometry",
    intrinsics: CameraIntrinsics,
) -> None:
    """Outline detected member markers solid, and PREDICTED hidden ones dashed.

    The predicted outlines are the demonstration. Cover a marker with your
    hand and its ghost stays exactly where the fitted pose says it should be:
    the rigid body is still accounting for it, and you can see the pose has
    not shifted. Without this you can only judge stability from the axes,
    which is much harder to eyeball.
    """
    for marker_id in geometry.ids:
        detected = tool.detected_corners.get(marker_id)
        if detected is not None:
            pts = detected.astype(np.int32).reshape(-1, 1, 2)
            in_fit = marker_id in tool.marker_ids_used
            cv2.polylines(
                image, [pts], True, GREEN if in_fit else GREY, 2, cv2.LINE_AA
            )
            continue

        # Not detected this frame -- draw where the fitted pose predicts it.
        if tool.pose is None:
            continue
        predicted = _project(geometry.markers[marker_id].corners, tool, intrinsics)
        if not np.all(np.isfinite(predicted)):
            continue
        pts = predicted.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], True, AMBER, 1, cv2.LINE_AA)
        centre = predicted.mean(axis=0).astype(int)
        cv2.putText(
            image,
            f"{marker_id}?",
            (int(centre[0]) - 14, int(centre[1]) + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            AMBER,
            1,
            cv2.LINE_AA,
        )


def draw_tool_tip(
    image: np.ndarray,
    tool: "ToolPose",
    p_tip: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray | None:
    """Reproject the calibrated tip and mark it. Returns its camera-frame position.

    The tip is drawn as a crosshair with a stalk back to the tool origin, so
    it reads as a rigid extension of the body rather than a floating dot.

    This is the Phase 3 acceptance test made visible: with the tip replanted
    in its divot, this marker should stay pinned to the divot while the tool
    body swings around it. Any wander is the tip calibration error, shown
    directly in the image at the scale you care about.
    """
    if tool.pose is None:
        return None

    p_tip = np.asarray(p_tip, dtype=float).reshape(3)
    tip_cam = tool.pose.transform_points(p_tip)  # R @ p_tip + t

    pts = _project(np.vstack([np.zeros(3), p_tip]), tool, intrinsics)
    if not np.all(np.isfinite(pts)):
        return tip_cam
    origin_px = tuple(np.round(pts[0]).astype(int))
    tip_px = tuple(np.round(pts[1]).astype(int))

    cv2.line(image, origin_px, tip_px, BLACK, 4, cv2.LINE_AA)
    cv2.line(image, origin_px, tip_px, CYAN, 2, cv2.LINE_AA)

    x, y = tip_px
    for colour, thickness in ((BLACK, 4), (CYAN, 2)):
        cv2.line(image, (x - 13, y), (x + 13, y), colour, thickness, cv2.LINE_AA)
        cv2.line(image, (x, y - 13), (x, y + 13), colour, thickness, cv2.LINE_AA)
    cv2.circle(image, (x, y), 7, BLACK, 3, cv2.LINE_AA)
    cv2.circle(image, (x, y), 7, CYAN, 1, cv2.LINE_AA)

    return tip_cam


def draw_tool_readout(
    image: np.ndarray,
    tool: "ToolPose",
    geometry: "ToolGeometry",
    origin: tuple[int, int] = (12, 12),
    tip_cam: np.ndarray | None = None,
) -> None:
    """Tool status panel: marker count, pose, reprojection RMS, per-member RMS."""
    lines: list[tuple[str, tuple[int, int, int]]] = []
    max_rms = geometry.config.max_reprojection_rms_px

    if tool.pose is None:
        lines.append((f"{geometry.name}: TOOL LOST", RED))
        lines.append((f"  need >= {geometry.config.min_markers} of {geometry.ids}", GREY))
        if tool.marker_ids_visible_other:
            lines.append((f"  other markers seen: {tool.marker_ids_visible_other}", GREY))
        draw_text_panel(image, lines, origin=origin)
        return

    low = tool.status == "low_confidence"
    head_colour = AMBER if low else GREEN
    lines.append(
        (
            f"{geometry.name}: {'LOW CONFIDENCE' if low else 'TRACKING'}",
            head_colour,
        )
    )

    lines.append(
        (
            f"  markers used : {tool.n_markers_used} / {geometry.n_markers}   "
            f"{tool.marker_ids_used}",
            head_colour,
        )
    )

    rms_ok = tool.reprojection_rms_px <= max_rms
    lines.append(
        (
            f"  reproj RMS   : {tool.reprojection_rms_px:5.2f} px  "
            f"({tool.n_points} pts, limit {max_rms:.1f})",
            GREEN if rms_ok else RED,
        )
    )

    x, y, z = tool.pose.t
    rx, ry, rz = tool.pose.as_euler_deg()
    lines.append((f"  x {x:8.1f}   y {y:8.1f}   z {z:8.1f}  mm", WHITE))
    lines.append((f"  rx{rx:8.1f}  ry{ry:8.1f}  rz{rz:8.1f}  deg", WHITE))
    lines.append((f"  range        : {tool.distance_mm:7.1f} mm", WHITE))

    if tip_cam is not None:
        # With the tip planted in a divot, these three numbers should stay
        # constant while the tool body rotates. That is the Phase 3 test.
        lines.append(
            (
                f"  TIP in camera: [{tip_cam[0]:7.1f} {tip_cam[1]:7.1f} "
                f"{tip_cam[2]:7.1f}] mm",
                CYAN,
            )
        )

    if tool.is_ambiguous:
        amb = "inf" if not np.isfinite(tool.ambiguity_ratio) else f"{tool.ambiguity_ratio:.2f}"
        lines.append((f"  AMBIGUOUS (amb {amb}) -- coplanar tool, pose may flip", AMBER))

    # Per-member residuals: an outlier here means that member's configured
    # centre disagrees with where the camera sees it.
    if tool.per_marker_rms_px:
        worst_id = max(tool.per_marker_rms_px, key=lambda k: tool.per_marker_rms_px[k])
        worst = tool.per_marker_rms_px[worst_id]
        detail = "  ".join(
            f"{mid}:{v:.2f}" for mid, v in sorted(tool.per_marker_rms_px.items())
        )
        lines.append((f"  per-marker px: {detail}", GREY if worst <= max_rms else AMBER))
        if worst > max_rms and tool.n_markers_used > 1:
            lines.append(
                (f"  -> check tool.markers id {worst_id} position in config.yaml", AMBER)
            )

    draw_text_panel(image, lines, origin=origin)


def draw_status_bar(
    image: np.ndarray,
    fps: float,
    intrinsics: CameraIntrinsics,
    marker_length_mm: float,
    n_detected: int,
) -> None:
    """Bottom-left provenance strip.

    Which calibration is loaded and what marker size is assumed are the two
    things that most often explain a wrong number, so they stay on screen.
    """
    lines = [
        (f"{fps:5.1f} fps   markers {n_detected}", WHITE),
        (
            f"calib RMS {intrinsics.reprojection_rms_px:.3f} px   "
            f"marker {marker_length_mm:.1f} mm",
            GREY,
        ),
        ("q quit   p print pose to console", GREY),
    ]
    draw_text_panel(image, lines, origin=(12, image.shape[0] - 90))
