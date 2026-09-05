"""Live single-marker tracking loop (Phase 1 demo)."""

from __future__ import annotations

import time

import cv2

from core.config import Config
from core.intrinsics import CameraIntrinsics
from core.video import camera
from tracking.overlay import (
    draw_marker_axes,
    draw_marker_outline,
    draw_pose_readout,
    draw_status_bar,
)
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
