"""Live pivot-calibration capture.

The operator plants the tool tip in a fixed divot and precesses the body
around it. This module decides which frames are worth keeping, shows whether
the capture is actually well-conditioned yet, and hands the collected poses to
the least-squares solver.

WHAT MAKES A FRAME WORTH KEEPING
--------------------------------
Three gates, each closing a specific failure:

  * enough markers -- a single-marker pose carries Phase 1's planar ambiguity
    straight into the tip solve, where it becomes a systematic error rather
    than visible flicker.
  * low reprojection RMS -- a blurred or poorly-fitted body pose contributes a
    wrong (R_i, t_i) that biases everything.
  * enough rotation since the last keep -- otherwise pausing to steady your
    hand silently adds 40 copies of one pose. Those inflate the frame count
    and *improve* the residual (identical frames agree with each other) while
    adding nothing to the conditioning. That is precisely the trap described
    in calibration_pivot/solve.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from calibration_pivot.solve import PivotResult, orientation_spread, solve_pivot
from calibration_pivot.tip import TipCalibration
from core.config import Config
from core.hud import (
    AMBER,
    GREEN,
    GREY,
    RED,
    WHITE,
    draw_orientation_wheel,
    draw_text_panel,
)
from core.intrinsics import CameraIntrinsics
from core.transforms import SE3, rotation_angle_between
from core.video import camera
from markers.tool import load_tool_geometry
from tracking.overlay import draw_tool_axes, draw_tool_markers, draw_tool_origin
from tracking.rigid_body import RigidBodyTracker


def run_pivot_calibration(cfg: Config) -> PivotResult | None:
    """Capture poses while the tool pivots, then solve for the tip offset."""
    intrinsics = CameraIntrinsics.load(cfg.camera.intrinsics_file)
    intrinsics.validate_for(cfg.camera.frame_width, cfg.camera.frame_height)

    geometry = load_tool_geometry(cfg)
    tracker = RigidBodyTracker(
        geometry=geometry,
        intrinsics=intrinsics,
        dictionary_name=cfg.aruco.dictionary,
        corner_refinement=cfg.aruco.corner_refinement,
    )
    pv = cfg.pivot

    print(_instructions(cfg, geometry))

    poses: list[SE3] = []
    last_kept: SE3 | None = None
    flash_until = 0.0
    spread = orientation_spread([])

    with camera(
        cfg.camera.device_index, cfg.camera.frame_width, cfg.camera.frame_height, cfg.camera.fps
    ) as cap:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera read failed; stopping.")
                break

            tool = tracker.process(frame)
            display = frame.copy()
            draw_tool_markers(display, tool, geometry, intrinsics)
            draw_tool_axes(display, tool, intrinsics, max(20.0, geometry.marker_length_mm))
            draw_tool_origin(display, tool, intrinsics)

            # -- decide whether this frame is worth keeping -----------------
            reasons: list[str] = []
            if tool.pose is None:
                reasons.append("tool not tracked")
            else:
                if tool.n_markers_used < pv.min_markers:
                    reasons.append(f"need >= {pv.min_markers} markers")
                if tool.reprojection_rms_px > pv.max_reprojection_rms_px:
                    reasons.append(f"rms > {pv.max_reprojection_rms_px:.1f} px")

            rotation_delta = (
                rotation_angle_between(last_kept, tool.pose)
                if (last_kept is not None and tool.pose is not None)
                else float("inf")
            )
            if tool.pose is not None and not reasons and rotation_delta < pv.min_rotation_deg:
                reasons.append(f"rotate more (+{pv.min_rotation_deg - rotation_delta:.0f} deg)")

            keepable = tool.pose is not None and not reasons
            if keepable:
                poses.append(tool.pose)
                last_kept = tool.pose
                spread = orientation_spread(poses)
                flash_until = time.time() + 0.08

            _draw_capture_hud(
                display, cfg, geometry, tool, poses, spread, reasons, rotation_delta
            )
            if time.time() < flash_until:
                cv2.rectangle(
                    display, (0, 0), (display.shape[1] - 1, display.shape[0] - 1), GREEN, 8
                )

            cv2.imshow("Pivot (tip) calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("Aborted; nothing saved.")
                return None
            if key == ord("u") and poses:
                poses.pop()
                last_kept = poses[-1] if poses else None
                spread = orientation_spread(poses)
                print(f"  undo -> {len(poses)} poses")
            elif key == ord("c"):
                if len(poses) < pv.min_frames:
                    print(
                        f"  need >= {pv.min_frames} poses to solve (have {len(poses)}); "
                        "keep pivoting, or 'q' to abort."
                    )
                    continue
                break

            if len(poses) >= pv.target_frames:
                print(f"\nReached the target of {pv.target_frames} poses.")
                break

    if len(poses) < pv.min_frames:
        print(f"Only {len(poses)} poses captured; not solving.")
        return None

    return _solve_and_save(cfg, geometry, poses)


def _solve_and_save(cfg: Config, geometry, poses: list[SE3]) -> PivotResult:
    print(f"\nSolving for the tip offset from {len(poses)} poses ...\n")
    result = solve_pivot(poses)
    print(result.report())

    # Save the raw poses next to the answer so the calibration can be
    # re-solved or audited later without re-pivoting -- same reasoning as
    # keeping the ChArUco frames in Phase 1.
    poses_path = Path(cfg.pivot.poses_file)
    poses_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        poses_path,
        R=np.array([p.R for p in poses]),
        t=np.array([p.t for p in poses]),
        tool_fingerprint=geometry.fingerprint(),
    )
    print(f"\nSaved {len(poses)} captured poses -> {poses_path}")

    tip = TipCalibration(
        p_tip=result.p_tip,
        tool_name=geometry.name,
        tool_fingerprint=geometry.fingerprint(),
        residual_rms_mm=result.residual_rms_mm,
        residual_max_mm=result.residual_max_mm,
        n_frames=result.n_frames,
        condition_number=result.condition_number,
        cone_half_angle_deg=result.spread.cone_half_angle_deg,
        metadata={
            "azimuth_coverage": round(float(result.spread.azimuth_coverage), 3),
            "pivot_in_camera_mm": [round(float(v), 3) for v in result.p_pivot],
            "intrinsics_file": str(cfg.camera.intrinsics_file),
        },
    )
    out_path = tip.save(cfg.tool.tip_calibration_file)
    print(f"Saved tip calibration    -> {out_path}")
    print("\nVerify it: run `python app.py track-tool`, replant the tip in the divot,")
    print("and rotate the tool about it. The drawn tip marker should stay put.")
    return result


def _instructions(cfg: Config, geometry) -> str:
    pv = cfg.pivot
    return f"""
Pivot (tip) calibration -- tool '{geometry.name}'
------------------------------------------------
Solves R_i @ p_tip + t_i = p_pivot for the tip offset, by least squares over
every captured pose. See calibration_pivot/solve.py for the derivation.

SETUP
  * Make a divot the tip cannot slide out of: a countersunk screw hole, the
    dimple in a door hinge, a V cut in stiff card. A flat dent is not enough --
    the tip must stay at ONE point.
  * DO NOT MOVE THE CAMERA once you start. The pivot point is solved in camera
    coordinates, so a bumped tripod invalidates every frame captured before it.

HOW TO PIVOT
  * Plant the tip and keep it planted. Precess the tool around a wide cone --
    aim for at least {pv.min_cone_half_angle_deg:.0f} deg of tilt -- and go ALL
    the way around, not just side to side.
  * Also spin the tool about its own axis as you go.
  * Pause briefly at varied orientations; frames are only kept once the tool
    has rotated {pv.min_rotation_deg:.0f} deg since the last one, so lingering
    costs you nothing.
  * Target {pv.target_frames} poses (minimum {pv.min_frames}).

Watch the WHEEL on the right: each dot is a captured orientation, radius is
tilt, angle is azimuth. You want a full ring at or outside the dashed target
circle -- a blob in the middle means you never tilted enough, an arc on one
side means you only swept half the cone. Frame count alone is not progress.

Keys:  u  undo the last capture
       c  solve and save
       q  abort
"""


def _draw_capture_hud(
    display,
    cfg: Config,
    geometry,
    tool,
    poses: list[SE3],
    spread,
    reasons: list[str],
    rotation_delta: float,
) -> None:
    pv = cfg.pivot
    n = len(poses)
    enough = n >= pv.min_frames

    lines: list[tuple[str, tuple[int, int, int]]] = [
        (
            f"poses captured : {n} / {pv.target_frames}   (min {pv.min_frames})",
            GREEN if enough else AMBER,
        )
    ]

    if tool.pose is None:
        lines.append(("tool           : NOT TRACKED", RED))
    else:
        lines.append(
            (
                f"tool           : {tool.n_markers_used} markers, "
                f"rms {tool.reprojection_rms_px:.2f} px",
                GREEN if not reasons else AMBER,
            )
        )

    if reasons:
        lines.append((f"holding        : {'; '.join(reasons)}", AMBER))
    else:
        lines.append(("capturing      : yes", GREEN))

    cone_ok = spread.cone_half_angle_deg >= pv.min_cone_half_angle_deg
    lines.append(
        (
            f"cone half-angle: {spread.cone_half_angle_deg:5.1f} deg "
            f"(target {pv.min_cone_half_angle_deg:.0f})",
            GREEN if cone_ok else AMBER,
        )
    )
    azimuth_ok = spread.azimuth_coverage >= 0.6
    lines.append(
        (
            f"azimuth swept  : {spread.azimuth_coverage * 100:3.0f}% "
            f"({int(spread.sectors_filled.sum())}/{len(spread.sectors_filled)} sectors)",
            GREEN if azimuth_ok else AMBER,
        )
    )

    # The live warning the whole screen exists for.
    if n >= 10 and not (cone_ok and azimuth_ok):
        lines.append(("ILL-CONDITIONED so far -- tilt further and go all", RED))
        lines.append(("the way around. Frame count is not progress.", RED))

    lines.append(("u undo | c solve+save | q abort", WHITE))
    draw_text_panel(display, lines, origin=(12, 12))

    wheel_r = 62
    wheel_x = display.shape[1] - wheel_r - 26
    wheel_y = wheel_r + 46
    draw_text_panel(
        display,
        [("orientation spread", WHITE)],
        origin=(wheel_x - wheel_r - 8, 12),
        width=2 * wheel_r + 16,
        scale=0.45,
    )
    draw_orientation_wheel(
        display,
        spread.tilts_deg,
        spread.azimuths_deg,
        (wheel_x, wheel_y),
        radius=wheel_r,
        target_cone_deg=pv.min_cone_half_angle_deg,
    )
