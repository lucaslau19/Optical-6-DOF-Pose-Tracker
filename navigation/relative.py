"""Pose relative to the patient reference frame.

THE ONE LINE THAT MATTERS
-------------------------
        T_ref_tool = T_cam_ref^-1 @ T_cam_tool

Both measurements are made in the camera frame, so when you compose them the
camera cancels -- read it in the convention from core/transforms.py:

        T_ref_tool = T_ref_cam @ T_cam_tool
                      ^^^^^^^     ^^^^^^^
                        `cam` cancels

The result depends only on where the tool is relative to the reference body.
Pick the camera up, move it to the other side of the bench, and T_cam_tool and
T_cam_ref both change completely while T_ref_tool does not change at all.

That is the defining feature of an optical navigation system. It is why a
surgeon can reposition the tracker mid-procedure, and why the patient's bone
carries its own marker array rather than the system assuming nothing moves.
It is also, satisfyingly, one matrix multiply.

WHY THE REFERENCE MUST BE VISIBLE
---------------------------------
If the reference body is not seen, T_cam_ref is unknown and NOTHING relative
can be computed. There is no sensible fallback: silently reusing the last
known T_cam_ref would produce confident, wrong numbers the moment the camera
moved, which is precisely the failure mode this whole phase exists to prevent.
So a missing reference is a hard, loud state -- REFERENCE LOST -- and the
guidance readout goes blank rather than stale.

ERROR PROPAGATION, HONESTLY
---------------------------
The relative pose carries the error of BOTH bodies. Two rigid-body fits, each
with its own noise, compose here -- so `p_tip_ref` is noisier than either
`T_cam_tool` or `T_cam_ref` alone. Nothing is wrong when the reference-frame
numbers jitter slightly more than the camera-frame ones; that is the price of
camera independence, and Phase 5 measures it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.transforms import SE3
from tracking.rigid_body import ToolPose


@dataclass
class RelativePose:
    """Tool pose expressed in the reference frame, plus the pieces behind it."""

    T_ref_tool: SE3 | None  # None when either body is not tracked
    p_tip_ref: np.ndarray | None  # (3,) tip in the REFERENCE frame, mm
    p_tip_cam: np.ndarray | None  # (3,) tip in the CAMERA frame, mm
    tool: ToolPose
    reference: ToolPose
    status: str  # "ok" | "tool_lost" | "reference_lost" | "both_lost"

    @property
    def is_valid(self) -> bool:
        return self.T_ref_tool is not None

    @property
    def is_ambiguous(self) -> bool:
        """True if EITHER body's pose is near the planar two-fold ambiguity.

        The relative pose inherits the weaknesses of both fits, so an
        ambiguous reference is just as damaging as an ambiguous tool -- and it
        is the less obvious of the two, because the reference usually sits
        still and looks reassuringly stable while being the thing everything
        is measured against.
        """
        return (
            (self.tool.pose is not None and self.tool.is_ambiguous)
            or (self.reference.pose is not None and self.reference.is_ambiguous)
        )

    @property
    def ambiguous_bodies(self) -> list[str]:
        names = []
        if self.tool.pose is not None and self.tool.is_ambiguous:
            names.append("tool")
        if self.reference.pose is not None and self.reference.is_ambiguous:
            names.append("reference")
        return names

    @property
    def status_text(self) -> str:
        return {
            "ok": "NAVIGATING",
            "tool_lost": "TOOL LOST",
            "reference_lost": "REFERENCE LOST",
            "both_lost": "TOOL + REFERENCE LOST",
        }[self.status]


def compute_relative_pose(
    tool: ToolPose,
    reference: ToolPose,
    p_tip: np.ndarray | None,
) -> RelativePose:
    """Express the tool (and its tip) in the reference frame.

    `p_tip` is the calibrated tip offset in the TOOL frame from Phase 3, or
    None if no pivot calibration has been done -- in which case the relative
    pose is still computed, there is simply no tip to place.
    """
    tool_ok = tool.pose is not None
    ref_ok = reference.pose is not None

    if not tool_ok and not ref_ok:
        status = "both_lost"
    elif not ref_ok:
        status = "reference_lost"
    elif not tool_ok:
        status = "tool_lost"
    else:
        status = "ok"

    if status != "ok":
        return RelativePose(None, None, None, tool, reference, status)

    # The line this whole phase is built around.
    T_ref_tool = reference.pose.inv() @ tool.pose  # type: ignore[union-attr]

    p_tip_ref = None
    p_tip_cam = None
    if p_tip is not None:
        p_tip = np.asarray(p_tip, dtype=float).reshape(3)
        # Same physical point, expressed two ways. Showing both side by side is
        # what makes the camera-independence test self-evident: move the
        # camera and the camera-frame numbers move while the reference-frame
        # numbers do not.
        p_tip_ref = T_ref_tool.transform_points(p_tip)
        p_tip_cam = tool.pose.transform_points(p_tip)  # type: ignore[union-attr]

    return RelativePose(T_ref_tool, p_tip_ref, p_tip_cam, tool, reference, "ok")


def resolve_tool_axis(configured: np.ndarray | None, p_tip: np.ndarray | None) -> np.ndarray | None:
    """The tool's long axis in the TOOL frame, as a unit vector.

    Defaults to the direction from the tool origin to the calibrated tip.
    That is the instrument's actual pointing direction, measured rather than
    declared, so it cannot drift out of agreement with the physical object the
    way a hand-typed vector in config.yaml would.
    """
    if configured is not None:
        return configured
    if p_tip is None:
        return None
    p_tip = np.asarray(p_tip, dtype=float).reshape(3)
    norm = float(np.linalg.norm(p_tip))
    if norm < 1e-6:
        # A tip sitting on the tool origin has no direction to offer.
        return None
    return p_tip / norm
