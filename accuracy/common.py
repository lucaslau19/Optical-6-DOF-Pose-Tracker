"""Shared setup and HUD for the accuracy measurements."""

from __future__ import annotations

import numpy as np

from calibration_pivot.tip import load_tip_if_available
from core.config import Config
from core.hud import AMBER, GREEN, GREY, RED, WHITE, draw_text_panel
from core.intrinsics import CameraIntrinsics
from markers.tool import load_tool_geometry
from navigation.live import load_reference_geometry, _check_disjoint_ids
from navigation.relative import compute_relative_pose
from tracking.rigid_body import RigidBodyTracker


class MeasurementRig:
    """Everything the accuracy commands need: both bodies, intrinsics, tip.

    Every metric here is measured on `p_tip_ref` -- the tip expressed in the
    reference frame -- because that is the quantity the system actually
    delivers to a user. Measuring the camera-frame pose instead would flatter
    the result by leaving out the reference transform, which is a real part of
    the chain and carries real error.
    """

    def __init__(self, cfg: Config, *, require_tip: bool = True) -> None:
        self.cfg = cfg
        self.intrinsics = CameraIntrinsics.load(cfg.camera.intrinsics_file)
        self.intrinsics.validate_for(cfg.camera.frame_width, cfg.camera.frame_height)

        self.tool_geometry = load_tool_geometry(cfg)
        self.reference_geometry = load_reference_geometry(cfg)
        _check_disjoint_ids(self.tool_geometry, self.reference_geometry)

        self.tool_tracker = RigidBodyTracker(
            self.tool_geometry, self.intrinsics, cfg.aruco.dictionary, cfg.aruco.corner_refinement
        )
        self.reference_tracker = RigidBodyTracker(
            self.reference_geometry,
            self.intrinsics,
            cfg.aruco.dictionary,
            cfg.aruco.corner_refinement,
        )

        self.tip, tip_warning = load_tip_if_available(cfg, self.tool_geometry)
        if require_tip and self.tip is None:
            raise ValueError(
                "No tip calibration found, so there is no tip position to measure.\n"
                "Run `python app.py pivot` first."
            )
        self.tip_warning = tip_warning
        self.p_tip = self.tip.p_tip if self.tip is not None else None

        # Working distance is measured from the data rather than assumed, so
        # the recorded conditions describe what actually happened.
        self._distances: list[float] = []

    def process(self, frame):
        tool = self.tool_tracker.process(frame)
        reference = self.reference_tracker.process(frame)
        rel = compute_relative_pose(tool, reference, self.p_tip)
        if tool.pose is not None:
            self._distances.append(tool.distance_mm)
        return rel

    @property
    def working_distance_range(self) -> tuple[float | None, float | None]:
        if not self._distances:
            return None, None
        return float(min(self._distances)), float(max(self._distances))

    def apply_working_distance(self, conditions) -> None:
        lo, hi = self.working_distance_range
        conditions.working_distance_min_mm = lo
        conditions.working_distance_max_mm = hi


def draw_measurement_hud(
    display,
    rel,
    lines: list[tuple[str, tuple[int, int, int]]],
    *,
    ambiguity_warning: bool = True,
) -> None:
    """Status panel shared by the accuracy screens."""
    head: list[tuple[str, tuple[int, int, int]]] = []
    if not rel.is_valid:
        head.append((rel.status_text, RED))
        head.append(("  measurement paused -- both bodies must be visible", GREY))
    else:
        head.append(
            (
                f"tracking: tool {rel.tool.n_markers_used} mk, "
                f"reference {rel.reference.n_markers_used} mk",
                GREEN,
            )
        )
        if ambiguity_warning and rel.is_ambiguous:
            which = " + ".join(rel.ambiguous_bodies)
            head.append((f"  AMBIGUOUS ({which}) -- samples may be corrupted", AMBER))
        if rel.p_tip_ref is not None:
            x, y, z = rel.p_tip_ref
            head.append((f"  tip in REF: [{x:8.2f} {y:8.2f} {z:8.2f}] mm", WHITE))
    draw_text_panel(display, head + lines, origin=(12, 12))


def verdict_line(value_mm: float, thresholds=(0.5, 1.0, 2.0)) -> tuple[str, bool]:
    """Shared plain-language grading for a millimetre-scale error figure.

    Bands chosen for what this hardware can plausibly do: a webcam with printed
    paper fiducials is not a Polaris (which quotes ~0.25 mm RMS), so sub-mm is
    genuinely good here and anything past a couple of mm means something in
    the chain is wrong rather than merely noisy.
    """
    excellent, good, acceptable = thresholds
    if value_mm < excellent:
        return "excellent for a webcam-and-paper system", True
    if value_mm < good:
        return "good", True
    if value_mm < acceptable:
        return "acceptable, but worth chasing", True
    return "POOR -- something in the chain is wrong, not just noisy", False
