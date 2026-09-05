"""Target guidance: turn a relative pose into the numbers a surgeon reads.

Everything here lives in the REFERENCE frame, because that is the frame that
is attached to the patient. A target expressed in camera coordinates would
move every time the camera did, which would make it useless.

TWO KINDS OF GUIDANCE
---------------------
* To a POINT -- "get the tip here". One number: distance.
* Along an AXIS (a trajectory: think a drill or pin path) -- three numbers,
  because a trajectory constrains more than a point does:

      perpendicular offset : how far the tip is from the line, in mm
      angular deviation    : how far the tool's long axis is from the line's
                             direction, in degrees
      depth along axis     : how far along the line the tip has advanced

  Offset and angle are independent failure modes. You can be perfectly on the
  entry point while pointing 20 degrees wrong, which puts the far end of the
  trajectory badly off; or perfectly parallel but entering 5 mm to the side.
  A single "error" number would hide one of them, so both are always shown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.config import NavigationConfig, TolerancesConfig
from core.transforms import SE3

# Tolerance bands. Green inside `tolerance`, amber out to
# `tolerance * warn_factor`, red beyond.
STATUS_IN = "in"
STATUS_WARN = "warn"
STATUS_OUT = "out"


def band(value: float, tolerance: float, warn_factor: float) -> str:
    """Classify a magnitude against its tolerance band."""
    if not np.isfinite(value):
        return STATUS_OUT
    if value <= tolerance:
        return STATUS_IN
    if value <= tolerance * warn_factor:
        return STATUS_WARN
    return STATUS_OUT


def perpendicular_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to `direction`.

    Used to decompose the perpendicular offset into the two screen axes of the
    bullseye display. The helper vector is chosen to be the world axis least
    parallel to `direction`, so the cross product never degenerates.
    """
    d = np.asarray(direction, dtype=float).reshape(3)
    d = d / np.linalg.norm(d)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(d @ helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    return e1, e2


@dataclass
class Guidance:
    """Everything the navigation HUD needs for one frame."""

    # Point guidance
    distance_mm: float
    distance_status: str
    delta_ref: np.ndarray  # (3,) vector from tip to target, reference frame

    # Axis guidance (all NaN / None when no axis is configured)
    has_axis: bool = False
    offset_mm: float = float("nan")
    offset_status: str = STATUS_OUT
    offset_components: np.ndarray | None = None  # (2,) offset in the bullseye plane
    angle_deg: float = float("nan")
    angle_status: str = STATUS_OUT
    depth_mm: float = float("nan")

    @property
    def all_in_tolerance(self) -> bool:
        if not self.has_axis:
            return self.distance_status == STATUS_IN
        return (
            self.distance_status == STATUS_IN
            and self.offset_status == STATUS_IN
            and self.angle_status == STATUS_IN
        )


def compute_guidance(
    p_tip_ref: np.ndarray,
    T_ref_tool: SE3,
    nav: NavigationConfig,
    tol: TolerancesConfig,
    tool_axis_tool: np.ndarray | None,
) -> Guidance:
    """Distance to the target point, and trajectory error if an axis is set."""
    p_tip_ref = np.asarray(p_tip_ref, dtype=float).reshape(3)

    delta = nav.target_point - p_tip_ref
    distance = float(np.linalg.norm(delta))

    g = Guidance(
        distance_mm=distance,
        distance_status=band(distance, tol.distance_mm, tol.warn_factor),
        delta_ref=delta,
    )

    if nav.target_axis is None:
        return g

    axis_point = nav.target_axis.point
    d = nav.target_axis.direction  # already unit length

    # Decompose the tip's position relative to the axis into a component along
    # the axis (depth) and a component across it (offset).
    v = p_tip_ref - axis_point
    depth = float(v @ d)
    perpendicular = v - depth * d
    offset = float(np.linalg.norm(perpendicular))

    e1, e2 = perpendicular_basis(d)
    g.has_axis = True
    g.depth_mm = depth
    g.offset_mm = offset
    g.offset_status = band(offset, tol.offset_mm, tol.warn_factor)
    g.offset_components = np.array([float(perpendicular @ e1), float(perpendicular @ e2)])

    if tool_axis_tool is not None:
        # Rotate the tool's own long axis into the reference frame. Only the
        # rotation is used -- a direction has no position.
        u = T_ref_tool.R @ np.asarray(tool_axis_tool, dtype=float).reshape(3)
        u = u / np.linalg.norm(u)
        # Full 0-180 range rather than the acute angle: holding the instrument
        # backwards should read 180 deg, not 0. Folding it to the acute angle
        # would hide a real and serious mistake.
        angle = float(np.degrees(np.arccos(np.clip(float(u @ d), -1.0, 1.0))))
        g.angle_deg = angle
        g.angle_status = band(angle, tol.angle_deg, tol.warn_factor)

    return g
