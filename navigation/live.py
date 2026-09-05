"""Live navigation: reference-relative pose, target guidance, digitising."""

from __future__ import annotations

import time

import cv2
import numpy as np

from calibration_pivot.tip import load_tip_if_available
from core.config import Config
from core.intrinsics import CameraIntrinsics
from core.video import camera
from markers.tool import ToolGeometry, load_tool_geometry
from navigation.digitize import PointLog
from navigation.guidance import compute_guidance
from navigation.overlay import (
    draw_bullseye,
    draw_distance_banner,
    draw_navigation_readout,
    draw_target_in_scene,
    draw_tip_to_target,
)
from navigation.relative import compute_relative_pose, resolve_tool_axis
from tracking.live import PRINT_INTERVAL_S, _FpsMeter
from tracking.overlay import (
    draw_tool_axes,
    draw_tool_markers,
    draw_tool_origin,
    draw_tool_tip,
)
from tracking.rigid_body import RigidBodyTracker


def load_reference_geometry(cfg: Config) -> ToolGeometry:
    """The patient reference body -- same class as the tool, different markers."""
    if cfg.reference is None:
        raise ValueError(
            "config.yaml has no `reference:` section, so there is no patient "
            "reference frame.\nAdd one (see the Phase 4 block in config.yaml) "
            "with its own marker IDs and their positions in millimetres."
        )
    return ToolGeometry(cfg.reference)


def _check_disjoint_ids(tool: ToolGeometry, reference: ToolGeometry) -> None:
    """Refuse to run if the two bodies claim the same marker ID.

    Both trackers scan the same detections, so a shared ID would be fed into
    both fits at once and the relative pose would be quietly meaningless --
    the kind of misconfiguration that produces plausible numbers rather than
    an obvious failure.
    """
    shared = sorted(set(tool.ids) & set(reference.ids))
    if shared:
        raise ValueError(
            f"tool and reference share marker id(s) {shared}.\n"
            "Each rigid body needs its own IDs -- give the reference a distinct "
            "range (e.g. 20-23) in config.yaml and reprint its sheet."
        )


def run_navigation(cfg: Config) -> None:
    """Phase 4: report the tool relative to the patient, and guide to a target."""
    if cfg.navigation is None:
        raise ValueError("config.yaml has no `navigation:` section -- nothing to guide to.")

    intrinsics = CameraIntrinsics.load(cfg.camera.intrinsics_file)
    intrinsics.validate_for(cfg.camera.frame_width, cfg.camera.frame_height)

    tool_geom = load_tool_geometry(cfg)
    ref_geom = load_reference_geometry(cfg)
    _check_disjoint_ids(tool_geom, ref_geom)

    tool_tracker = RigidBodyTracker(
        tool_geom, intrinsics, cfg.aruco.dictionary, cfg.aruco.corner_refinement
    )
    ref_tracker = RigidBodyTracker(
        ref_geom, intrinsics, cfg.aruco.dictionary, cfg.aruco.corner_refinement
    )

    tip, tip_warning = load_tip_if_available(cfg, tool_geom)
    p_tip = tip.p_tip if tip is not None else None
    tool_axis = resolve_tool_axis(cfg.navigation.tool_axis, p_tip)

    print(_instructions(cfg, tool_geom, ref_geom, tip, tip_warning, tool_axis))

    log = PointLog()
    fps = _FpsMeter()
    last_print = 0.0
    axis_len = max(20.0, tool_geom.marker_length_mm)

    with camera(
        cfg.camera.device_index, cfg.camera.frame_width, cfg.camera.frame_height, cfg.camera.fps
    ) as cap:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera read failed; stopping.")
                break

            tool = tool_tracker.process(frame)
            reference = ref_tracker.process(frame)
            rel = compute_relative_pose(tool, reference, p_tip)

            guidance = None
            if rel.is_valid and rel.p_tip_ref is not None:
                guidance = compute_guidance(
                    rel.p_tip_ref, rel.T_ref_tool, cfg.navigation, cfg.tolerances, tool_axis
                )

            display = frame.copy()

            # Both bodies drawn with the same machinery; the reference gets its
            # own axes so it is obvious which frame the numbers refer to.
            draw_tool_markers(display, reference, ref_geom, intrinsics)
            draw_tool_markers(display, tool, tool_geom, intrinsics)
            if reference.pose is not None:
                draw_tool_axes(display, reference, intrinsics, axis_len)
                draw_tool_origin(display, reference, intrinsics)
                draw_target_in_scene(
                    display, reference.pose, cfg.navigation, intrinsics, guidance
                )
            if tool.pose is not None:
                draw_tool_axes(display, tool, intrinsics, axis_len)
                draw_tool_origin(display, tool, intrinsics)
                if p_tip is not None:
                    draw_tool_tip(display, tool, p_tip, intrinsics)

            if guidance is not None:
                draw_tip_to_target(display, rel, cfg.navigation, intrinsics, guidance)
                if guidance.has_axis:
                    draw_bullseye(
                        display,
                        guidance,
                        cfg.tolerances,
                        (display.shape[1] - 104, 120),
                    )

            draw_navigation_readout(
                display, rel, guidance, cfg.navigation, cfg.tolerances,
                len(log.points), log.last_gap_mm,
            )
            draw_distance_banner(display, guidance, cfg.tolerances)

            cv2.imshow("Navigation -- reference-relative guidance", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                if rel.p_tip_ref is None:
                    print("  cannot digitise: need tool, reference and a tip calibration")
                else:
                    point = log.add(
                        rel.p_tip_ref, tool.n_markers_used, reference.n_markers_used
                    )
                    x, y, z = point.p_ref
                    msg = f"  {point.label}: [{x:8.2f} {y:8.2f} {z:8.2f}] mm (reference frame)"
                    gap = log.last_gap_mm
                    if gap is not None:
                        msg += f"   <- {gap:.2f} mm from {log.points[-2].label}"
                    print(msg)
            elif key == ord("u"):
                removed = log.undo()
                if removed:
                    print(f"  undo {removed.label} -> {len(log.points)} points")
            elif key == ord("s") and log.points:
                path = log.save(cfg.navigation.captured_points_file)
                print(f"  saved {len(log.points)} points -> {path}")

            now = time.time()
            if key == ord("p") or now - last_print >= PRINT_INTERVAL_S:
                last_print = now
                print(_console_line(rel, guidance, fps.value))
            else:
                fps.tick()

    if log.points:
        print(f"\nDigitised {len(log.points)} points (reference frame):")
        print(log.pairwise_summary())
        path = log.save(cfg.navigation.captured_points_file)
        print(f"Saved -> {path}")


def _console_line(rel, guidance, fps_value: float) -> str:
    if not rel.is_valid:
        return f"{rel.status_text}"
    parts = [f"tool {rel.tool.n_markers_used}mk ref {rel.reference.n_markers_used}mk"]
    if rel.p_tip_ref is not None:
        x, y, z = rel.p_tip_ref
        parts.append(f"tip_ref [{x:8.2f} {y:8.2f} {z:8.2f}] mm")
    if rel.p_tip_cam is not None:
        x, y, z = rel.p_tip_cam
        parts.append(f"tip_cam [{x:8.2f} {y:8.2f} {z:8.2f}] mm")
    if guidance is not None:
        parts.append(f"dist {guidance.distance_mm:6.2f} mm")
        if guidance.has_axis:
            parts.append(f"off {guidance.offset_mm:5.2f} mm")
            if np.isfinite(guidance.angle_deg):
                parts.append(f"ang {guidance.angle_deg:5.2f} deg")
    return " | ".join(parts)


def _instructions(cfg, tool_geom, ref_geom, tip, tip_warning, tool_axis) -> str:
    nav = cfg.navigation
    tol = cfg.tolerances
    lines = [
        "",
        "Navigation -- reference-relative guidance",
        "-" * 60,
        f"Tool      : {tool_geom.name}, ids {tool_geom.ids}",
        f"Reference : {ref_geom.name}, ids {ref_geom.ids}",
        "",
        "Pose is reported RELATIVE TO THE REFERENCE BODY:",
        "    T_ref_tool = T_cam_ref^-1 @ T_cam_tool",
        "The camera cancels, so you can pick the camera up and move it while",
        "the reference-frame numbers stay put. That is the thing to test.",
        "",
    ]
    if tip is not None:
        lines.append("Tip calibration:")
        lines.append(tip.summary())
        if tip_warning:
            lines.append(f"  WARNING: {tip_warning}")
    else:
        lines.append("NO TIP CALIBRATION -- run `python app.py pivot` first.")
        lines.append("  Without it there is no tip to guide, digitise, or measure.")
    lines.append("")

    x, y, z = nav.target_point
    lines.append(f"Target point : [{x:.1f} {y:.1f} {z:.1f}] mm in the reference frame")
    if nav.target_axis is not None:
        ax, ay, az = nav.target_axis.point
        dx, dy, dz = nav.target_axis.direction
        lines.append(f"Target axis  : through [{ax:.1f} {ay:.1f} {az:.1f}] mm")
        lines.append(f"               direction [{dx:.3f} {dy:.3f} {dz:.3f}]")
    if tool_axis is not None:
        source = "config" if cfg.navigation.tool_axis is not None else "derived from the tip"
        lines.append(
            f"Tool axis    : [{tool_axis[0]:.3f} {tool_axis[1]:.3f} {tool_axis[2]:.3f}] "
            f"({source})"
        )
    lines.append(
        f"Tolerances   : {tol.distance_mm:.1f} mm point, {tol.offset_mm:.1f} mm offset, "
        f"{tol.angle_deg:.1f} deg  (amber to {tol.warn_factor:.0f}x)"
    )
    lines.append("")
    lines.append("Keys:  d  digitise the current tip position")
    lines.append("       u  undo the last digitised point")
    lines.append("       s  save digitised points")
    lines.append("       p  print a line now")
    lines.append("       q  quit")
    lines.append("")
    return "\n".join(lines)
