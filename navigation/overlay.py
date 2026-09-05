"""Navigation HUD: target marks in the scene, big readouts, tolerance colours."""

from __future__ import annotations

import cv2
import numpy as np

from core.config import NavigationConfig, TolerancesConfig
from core.hud import AMBER, BLACK, CYAN, GREEN, GREY, RED, WHITE, draw_text_panel
from core.intrinsics import CameraIntrinsics
from core.transforms import SE3
from navigation.guidance import STATUS_IN, STATUS_WARN, Guidance, perpendicular_basis
from navigation.relative import RelativePose

FONT = cv2.FONT_HERSHEY_SIMPLEX


def status_colour(status: str) -> tuple[int, int, int]:
    """Map a tolerance band to the project's standard green/amber/red."""
    if status == STATUS_IN:
        return GREEN
    if status == STATUS_WARN:
        return AMBER
    return RED


def _project_ref_points(
    points_ref: np.ndarray, T_cam_ref: SE3, intrinsics: CameraIntrinsics
) -> np.ndarray:
    """Project reference-frame points into the image via T_cam_ref."""
    rvec, tvec = T_cam_ref.as_rvec_tvec()
    projected, _ = cv2.projectPoints(
        np.asarray(points_ref, dtype=np.float64).reshape(-1, 3),
        rvec,
        tvec,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    )
    return projected.reshape(-1, 2)


def draw_target_in_scene(
    image: np.ndarray,
    T_cam_ref: SE3,
    nav: NavigationConfig,
    intrinsics: CameraIntrinsics,
    guidance: Guidance | None,
) -> None:
    """Draw the target point (and axis) into the video, anchored to the patient.

    The target is defined in the reference frame, so projecting it through
    T_cam_ref pins it to the physical reference body. Move the camera and it
    stays on the same spot of the bench -- which is the camera-independence
    property made visible without needing to read any numbers.
    """
    colour = GREEN if (guidance and guidance.all_in_tolerance) else AMBER

    if nav.target_axis is not None:
        # A finite segment of the trajectory, centred on the axis point.
        d = nav.target_axis.direction
        a = nav.target_axis.point
        ends = np.vstack([a - d * 60.0, a + d * 60.0])
        px = _project_ref_points(ends, T_cam_ref, intrinsics)
        if np.all(np.isfinite(px)):
            p0 = tuple(np.round(px[0]).astype(int))
            p1 = tuple(np.round(px[1]).astype(int))
            cv2.line(image, p0, p1, BLACK, 5, cv2.LINE_AA)
            cv2.line(image, p0, p1, colour, 2, cv2.LINE_AA)
            cv2.circle(image, p1, 6, colour, 2, cv2.LINE_AA)

    px = _project_ref_points(nav.target_point.reshape(1, 3), T_cam_ref, intrinsics)
    if not np.all(np.isfinite(px)):
        return
    centre = tuple(np.round(px[0]).astype(int))
    for radius, thickness in ((14, 3), (7, 2)):
        cv2.circle(image, centre, radius, BLACK, thickness + 2, cv2.LINE_AA)
        cv2.circle(image, centre, radius, colour, thickness, cv2.LINE_AA)
    cv2.putText(image, "TARGET", (centre[0] + 20, centre[1] + 5), FONT, 0.5, colour, 1, cv2.LINE_AA)


def draw_tip_to_target(
    image: np.ndarray,
    rel: RelativePose,
    nav: NavigationConfig,
    intrinsics: CameraIntrinsics,
    guidance: Guidance,
) -> None:
    """A line from the tip to the target -- the error, drawn to scale."""
    if rel.p_tip_ref is None or rel.reference.pose is None:
        return
    pts = _project_ref_points(
        np.vstack([rel.p_tip_ref, nav.target_point]), rel.reference.pose, intrinsics
    )
    if not np.all(np.isfinite(pts)):
        return
    p0 = tuple(np.round(pts[0]).astype(int))
    p1 = tuple(np.round(pts[1]).astype(int))
    cv2.line(image, p0, p1, status_colour(guidance.distance_status), 2, cv2.LINE_AA)


def draw_bullseye(
    image: np.ndarray,
    guidance: Guidance,
    tol: TolerancesConfig,
    origin: tuple[int, int],
    radius: int = 78,
) -> None:
    """Cross-section view down the trajectory: where the tip sits off-axis.

    This is the classic navigation display. The rings are the tolerance and
    the warning band, and the dot is the tip's perpendicular offset decomposed
    into the two directions across the axis. Steering a dot into a ring is a
    far more natural control task than reading a millimetre number and working
    out which way to move.
    """
    if not guidance.has_axis or guidance.offset_components is None:
        return

    cx, cy = origin
    display_max_mm = max(tol.offset_mm * tol.warn_factor * 2.0, 4.0)
    px_per_mm = radius / display_max_mm

    cv2.circle(image, (cx, cy), radius, (45, 45, 45), cv2.FILLED, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), radius, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), int(tol.offset_mm * tol.warn_factor * px_per_mm), AMBER, 1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), int(tol.offset_mm * px_per_mm), GREEN, 1, cv2.LINE_AA)
    cv2.line(image, (cx - radius, cy), (cx + radius, cy), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.line(image, (cx, cy - radius), (cx, cy + radius), (80, 80, 80), 1, cv2.LINE_AA)

    ox, oy = guidance.offset_components
    dot = np.array([ox, oy]) * px_per_mm
    dist = float(np.linalg.norm(dot))
    if dist > radius - 4:  # clamp to the rim so it never leaves the dial
        dot = dot / dist * (radius - 4)
    p = (int(round(cx + dot[0])), int(round(cy + dot[1])))
    colour = status_colour(guidance.offset_status)
    cv2.circle(image, p, 7, BLACK, -1, cv2.LINE_AA)
    cv2.circle(image, p, 6, colour, -1, cv2.LINE_AA)


def draw_navigation_readout(
    image: np.ndarray,
    rel: RelativePose,
    guidance: Guidance | None,
    nav: NavigationConfig,
    tol: TolerancesConfig,
    n_captured: int,
    last_gap_mm: float | None,
) -> None:
    """The main panel: status, tip in both frames, and the guidance numbers."""
    lines: list[tuple[str, tuple[int, int, int]]] = []

    if not rel.is_valid:
        colour = RED
        lines.append((rel.status_text, colour))
        if rel.status in ("reference_lost", "both_lost"):
            lines.append(("  the reference body must be visible to compute", GREY))
            lines.append(("  anything relative -- no fallback is possible", GREY))
        if rel.status in ("tool_lost", "both_lost"):
            lines.append(("  tool not tracked", GREY))
        draw_text_panel(image, lines, origin=(12, 12))
        return

    lines.append(("NAVIGATING", GREEN))
    lines.append(
        (
            f"  tool {rel.tool.n_markers_used} mk   reference "
            f"{rel.reference.n_markers_used} mk",
            GREY,
        )
    )

    if rel.is_ambiguous:
        # The relative pose inherits the weaknesses of both fits, so an
        # ambiguous body -- especially the reference, which sits still and
        # looks stable -- can quietly corrupt every number below.
        which = " + ".join(rel.ambiguous_bodies)
        lines.append((f"  AMBIGUOUS POSE ({which}) -- readings may be wrong", AMBER))
        lines.append(("  move closer or view the sheet more obliquely", AMBER))

    if rel.p_tip_ref is not None:
        x, y, z = rel.p_tip_ref
        # The camera-independence test reads straight off these two lines:
        # move the camera and the CAM row changes while the REF row does not.
        lines.append((f"  tip in REF : [{x:8.2f} {y:8.2f} {z:8.2f}] mm", CYAN))
    if rel.p_tip_cam is not None:
        x, y, z = rel.p_tip_cam
        lines.append((f"  tip in CAM : [{x:8.2f} {y:8.2f} {z:8.2f}] mm", GREY))

    if rel.T_ref_tool is not None:
        rx, ry, rz = rel.T_ref_tool.as_euler_deg()
        tx, ty, tz = rel.T_ref_tool.t
        lines.append((f"  T_ref_tool t [{tx:8.2f} {ty:8.2f} {tz:8.2f}] mm", GREY))
        lines.append((f"             r [{rx:8.2f} {ry:8.2f} {rz:8.2f}] deg", GREY))

    if guidance is not None:
        lines.append(("", WHITE))
        if guidance.has_axis:
            lines.append(
                (
                    f"  offset  {guidance.offset_mm:6.2f} mm   "
                    f"(tol {tol.offset_mm:.1f})",
                    status_colour(guidance.offset_status),
                )
            )
            if np.isfinite(guidance.angle_deg):
                lines.append(
                    (
                        f"  angle   {guidance.angle_deg:6.2f} deg  "
                        f"(tol {tol.angle_deg:.1f})",
                        status_colour(guidance.angle_status),
                    )
                )
            else:
                lines.append(("  angle   n/a -- no tip calibration", GREY))
            lines.append((f"  depth   {guidance.depth_mm:6.2f} mm along axis", WHITE))

    lines.append(("", WHITE))
    gap = f"{last_gap_mm:.2f} mm" if last_gap_mm is not None else "-"
    lines.append((f"  digitised: {n_captured}   last gap: {gap}", WHITE))
    lines.append(("d digitise | u undo | s save | q quit", WHITE))

    draw_text_panel(image, lines, origin=(12, 12))


def draw_distance_banner(
    image: np.ndarray, guidance: Guidance | None, tol: TolerancesConfig
) -> None:
    """The one number you steer by, big enough to read from across the bench."""
    h, w = image.shape[:2]
    if guidance is None:
        text, colour = "-- mm", GREY
    else:
        text = f"{guidance.distance_mm:.1f} mm"
        colour = status_colour(guidance.distance_status)

    scale, thickness = 2.1, 4
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = (w - tw) // 2
    y = h - 40

    panel = image.copy()
    cv2.rectangle(panel, (x - 26, y - th - 26), (x + tw + 26, y + 22), BLACK, cv2.FILLED)
    cv2.addWeighted(panel, 0.55, image, 0.45, 0, dst=image)
    cv2.rectangle(image, (x - 26, y - th - 26), (x + tw + 26, y + 22), colour, 2, cv2.LINE_AA)

    cv2.putText(image, text, (x, y), FONT, scale, BLACK, thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)
    label = "TIP TO TARGET"
    (lw, _), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.putText(
        image, label, (x + (tw - lw) // 2, y - th - 8), FONT, 0.5, colour, 1, cv2.LINE_AA
    )
