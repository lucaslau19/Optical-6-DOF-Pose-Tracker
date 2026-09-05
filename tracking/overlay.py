"""Pose-specific drawing: axes, marker outlines, per-marker readouts."""

from __future__ import annotations

import cv2
import numpy as np

from core.hud import AMBER, GREEN, GREY, RED, WHITE, draw_text_panel
from core.intrinsics import CameraIntrinsics
from tracking.single_marker import MarkerPose


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
