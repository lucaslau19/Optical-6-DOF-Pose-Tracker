"""Live tracking loops.

`run_single_marker_tracking` -- Phase 1, one pose per visible marker.
`run_tool_tracking`          -- Phase 2, one pose for a rigid marker cluster.
"""

from __future__ import annotations

import time

import cv2

from calibration_pivot.tip import load_tip_if_available
from core.config import Config
from core.intrinsics import CameraIntrinsics
from core.video import camera
from markers.tool import load_tool_geometry
from tracking.overlay import (
    draw_marker_axes,
    draw_marker_outline,
    draw_pose_readout,
    draw_status_bar,
    draw_tool_axes,
    draw_tool_markers,
    draw_tool_origin,
    draw_tool_readout,
    draw_tool_tip,
)
from tracking.rigid_body import RigidBodyTracker
from tracking.single_marker import SingleMarkerTracker

# Console printing is throttled: a 30 fps stream would otherwise scroll far
# too fast to read, and the terminal I/O itself starts costing frame time.
PRINT_INTERVAL_S = 0.5


class _FpsMeter:
    """Exponentially smoothed frame rate.

    Smoothed rather than instantaneous because a raw per-frame reciprocal
    jumps around so much that it is unreadable, and a dropped frame would
    show as a spike rather than the sustained rate that actually matters.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.value = 0.0
        self._last: float | None = None

    def tick(self) -> float:
        now = time.perf_counter()
        # The first call has no previous frame to measure against -- it would
        # otherwise report the interval since construction, which is camera
        # open time (absurdly low) or near zero (absurdly high).
        if self._last is None:
            self._last = now
            return 0.0
        dt = now - self._last
        self._last = now
        if dt > 0:
            inst = 1.0 / dt
            self.value = (
                inst
                if self.value == 0.0
                else self.alpha * inst + (1 - self.alpha) * self.value
            )
        return self.value


def run_single_marker_tracking(cfg: Config, *, marker_length_mm: float | None = None) -> None:
    """Open the camera, track markers, draw axes, print pose."""
    intrinsics = CameraIntrinsics.load(cfg.camera.intrinsics_file)
    intrinsics.validate_for(cfg.camera.frame_width, cfg.camera.frame_height)

    length = float(marker_length_mm or cfg.markers.marker_length_mm)
    tracker = SingleMarkerTracker(
        intrinsics=intrinsics,
        dictionary_name=cfg.aruco.dictionary,
        corner_refinement=cfg.aruco.corner_refinement,
        marker_length_mm=length,
    )

    print(f"\nLoaded intrinsics from {cfg.camera.intrinsics_file}")
    print(intrinsics.summary())
    print(f"\nTracking {cfg.aruco.dictionary} markers, {length:.1f} mm side.")
    print("Pose is T_cam_marker: translation is the marker centre in camera coordinates.")
    print("Axes drawn on each marker: X red (right), Y green (down), Z blue (into the face).")
    print("A marker held square-on to the camera reads rpy ~ (0, 0, 0).")
    print("Keys: q quit | p force a pose print\n")

    fps = _FpsMeter()
    last_print = 0.0

    with camera(
        cfg.camera.device_index, cfg.camera.frame_width, cfg.camera.frame_height, cfg.camera.fps
    ) as cap:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera read failed; stopping.")
                break

            markers = tracker.process(frame)
            display = frame.copy()

            # Axis glyph sized to the marker so it stays readable at any range.
            axis_len = length * 0.5
            for marker in markers:
                draw_marker_outline(display, marker)
                draw_marker_axes(display, marker, intrinsics, axis_len)

            draw_pose_readout(display, markers)
            draw_status_bar(display, fps.tick(), intrinsics, length, len(markers))

            cv2.imshow("6-DOF single-marker tracking", display)
            key = cv2.waitKey(1) & 0xFF

            now = time.time()
            force_print = key == ord("p")
            if markers and (force_print or now - last_print >= PRINT_INTERVAL_S):
                last_print = now
                for marker in markers:
                    flag = "  <-- ambiguous" if marker.is_ambiguous else ""
                    print(marker.format_line() + flag)

            if key in (ord("q"), 27):
                break


def run_tool_tracking(cfg: Config) -> None:
    """Phase 2: track a rigid multi-marker tool as one 6-DOF body."""
    intrinsics = CameraIntrinsics.load(cfg.camera.intrinsics_file)
    intrinsics.validate_for(cfg.camera.frame_width, cfg.camera.frame_height)

    geometry = load_tool_geometry(cfg)
    tracker = RigidBodyTracker(
        geometry=geometry,
        intrinsics=intrinsics,
        dictionary_name=cfg.aruco.dictionary,
        corner_refinement=cfg.aruco.corner_refinement,
    )

    print(f"\nLoaded intrinsics from {cfg.camera.intrinsics_file}")
    print(geometry.summary())

    # A calibrated tip is optional -- tool tracking works without one, it just
    # cannot draw the business end.
    tip, tip_warning = load_tip_if_available(cfg, geometry)
    if tip is not None:
        print(f"\nTip calibration ({cfg.tool.tip_calibration_file}):")
        print(tip.summary())
        if tip_warning:
            print(f"\n  WARNING: {tip_warning}")
    else:
        print(
            "\n  No tip calibration found -- run `python app.py pivot` to measure one.\n"
            "  Tool tracking works without it; the tip marker just is not drawn."
        )

    if geometry.is_coplanar():
        print(
            "\n  NOTE: this tool is coplanar (all markers on one flat sheet).\n"
            "  That keeps the two-fold planar pose ambiguity -- better conditioned\n"
            "  than a single marker, but not eliminated. A non-coplanar arrangement\n"
            "  (markers on a folded bracket, different z in config.yaml) removes it,\n"
            "  which is why real tracked instruments are not flat."
        )
    too_close = geometry.overlapping_pairs()
    if too_close:
        print(f"\n  WARNING: member markers printed too close to detect reliably: {too_close}")

    print("\nOne pose is fitted to ALL visible member corners at once (T_cam_tool).")
    print("Cover markers with your hand: the axes should stay put while >= 1 is visible.")
    print("Hidden members are drawn as amber predicted outlines.")
    print("Keys: q quit | p force a pose print\n")

    fps = _FpsMeter()
    last_print = 0.0
    axis_len = max(20.0, geometry.marker_length_mm)

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
            draw_tool_axes(display, tool, intrinsics, axis_len)
            draw_tool_origin(display, tool, intrinsics)

            tip_cam = None
            if tip is not None:
                tip_cam = draw_tool_tip(display, tool, tip.p_tip, intrinsics)

            draw_tool_readout(display, tool, geometry, tip_cam=tip_cam)
            draw_status_bar(
                display, fps.tick(), intrinsics, geometry.marker_length_mm, tool.n_markers_used
            )

            cv2.imshow(f"Rigid-body tool tracking -- {geometry.name}", display)
            key = cv2.waitKey(1) & 0xFF

            now = time.time()
            if key == ord("p") or now - last_print >= PRINT_INTERVAL_S:
                last_print = now
                line = tool.format_line()
                if tip_cam is not None:
                    # Printed so the tip position can be watched for constancy
                    # while the tool rotates about a planted tip -- the numeric
                    # form of the Phase 3 acceptance test.
                    line += (
                        f" | tip_cam [{tip_cam[0]:7.1f} {tip_cam[1]:7.1f} {tip_cam[2]:7.1f}] mm"
                    )
                print(line)

            if key in (ord("q"), 27):
                break
