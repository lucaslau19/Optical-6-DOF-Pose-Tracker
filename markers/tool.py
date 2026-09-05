"""Rigid tool geometry: configured marker centres -> 3D corner points.

WHAT A "TOOL" IS
----------------
Several ArUco markers rigidly fixed at known positions. Because the geometry
is known, every visible marker contributes four 3D<->2D correspondences to a
SINGLE pose fit. Consequences:

  * Occlusion tolerance. Cover a marker and the remaining ones still pin the
    pose down. Nothing needs to be re-initialised; the fit just has fewer
    points.
  * Better conditioning. Four markers 60 mm apart span far more image area
    than one 30 mm marker, so the same corner-localisation noise perturbs the
    pose much less -- especially the out-of-plane rotations, which are what a
    small single marker estimates worst.

This is exactly why commercial optical trackers (NDI Polaris, Intellijoint)
use a cluster of fiducials on every instrument rather than a single one.

THE TOOL FRAME
--------------
X right, Y DOWN, Z into the sheet -- the same convention as the marker frame
in tracking/single_marker.py. Keeping them identical means a one-marker tool
centred on the origin reports exactly the pose `track` reports for that
marker, which is a property worth having and worth testing.

Because the tool frame and the printed page agree on "Y is down", the sheet
generator in markers/tool_sheet.py can map tool millimetres to image pixels
with no axis flip at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from core.config import ToolConfig, ToolMemberConfig

# Corner offsets of a marker of side 1, in the marker's own plane, in the
# order cv2.aruco detectMarkers returns them (top-left, top-right,
# bottom-right, bottom-left as seen upright) under a Y-DOWN convention.
#
# Compare tracking/single_marker.marker_object_points, which is the same four
# points with Y up because SOLVEPNP_IPPE_SQUARE demands that ordering. Here we
# are building object points for a general solvePnP, so we are free to use the
# project's own frame directly -- and must, or the tool pose would disagree
# with the single-marker pose by 180 degrees.
_UNIT_CORNERS = np.array(
    [
        [-0.5, -0.5, 0.0],  # top-left
        [+0.5, -0.5, 0.0],  # top-right
        [+0.5, +0.5, 0.0],  # bottom-right
        [-0.5, +0.5, 0.0],  # bottom-left
    ],
    dtype=float,
)


@dataclass(frozen=True)
class ToolMarker:
    """One member marker, with its corners precomputed in the tool frame."""

    marker_id: int
    center: np.ndarray  # (3,) tool-frame position of the marker centre, mm
    rotation_deg: float  # in-plane rotation about the tool +Z axis
    corners: np.ndarray  # (4, 3) tool-frame corner positions, mm


def _member_corners(member: ToolMemberConfig, side_mm: float) -> np.ndarray:
    """Corner positions of one member, in the tool frame, in millimetres.

    The in-plane rotation is about the tool's +Z axis. Since +Z points INTO
    the sheet, a positive angle looks CLOCKWISE when you view the printed
    page -- a sign convention that is easy to get backwards, so the sheet
    generator draws the rotation it actually used.
    """
    corners = _UNIT_CORNERS * float(side_mm)
    if member.rotation_deg:
        R = Rotation.from_euler("z", member.rotation_deg, degrees=True).as_matrix()
        corners = corners @ R.T
    return corners + np.array([member.x, member.y, member.z], dtype=float)


class ToolGeometry:
    """The rigid body: which marker IDs belong to the tool, and where they are.

    Corner points are computed once at construction. They are fixed properties
    of the physical object, and recomputing them per frame would be pure waste
    in the tracking loop.
    """

    def __init__(self, cfg: ToolConfig) -> None:
        self.name = cfg.name
        self.marker_length_mm = float(cfg.marker_length_mm)
        self.config = cfg

        self.markers: dict[int, ToolMarker] = {}
        for member in cfg.members:
            self.markers[member.marker_id] = ToolMarker(
                marker_id=member.marker_id,
                center=np.array([member.x, member.y, member.z], dtype=float),
                rotation_deg=member.rotation_deg,
                corners=_member_corners(member, self.marker_length_mm),
            )

    # -- basic properties --------------------------------------------------

    @property
    def ids(self) -> list[int]:
        return sorted(self.markers)

    @property
    def n_markers(self) -> int:
        return len(self.markers)

    @property
    def all_corners(self) -> np.ndarray:
        """(4N, 3) every member corner, for extent and coplanarity checks."""
        return np.vstack([m.corners for m in self.markers.values()])

    @property
    def centroid(self) -> np.ndarray:
        return self.all_corners.mean(axis=0)

    def extent_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box of all corners: (min_xyz, max_xyz)."""
        pts = self.all_corners
        return pts.min(axis=0), pts.max(axis=0)

    # -- conditioning ------------------------------------------------------

    def is_coplanar(self, tol_mm: float = 0.5) -> bool:
        """Do all corner points lie in one plane, to within `tol_mm`?

        This decides which PnP initialiser is valid (see
        tracking/rigid_body.py) and whether the two-fold planar pose ambiguity
        applies at all, so it is a property of the tool worth knowing rather
        than an implementation detail.

        Implemented with an SVD of the mean-centred points: the smallest
        singular value is the RMS spread along the thinnest direction, i.e.
        the plane's thickness.
        """
        pts = self.all_corners
        if len(pts) < 4:
            return True
        centred = pts - pts.mean(axis=0)
        singular_values = np.linalg.svd(centred, compute_uv=False)
        return float(singular_values[-1]) <= tol_mm * np.sqrt(len(pts))

    def overlapping_pairs(self, clearance_mm: float | None = None) -> list[tuple[int, int]]:
        """Member pairs printed too close together to detect reliably.

        ArUco finds markers by their outer black contour, so each needs white
        space around it. Two markers butted together read as one blob and
        neither is found -- a failure that looks like "the detector is bad"
        rather than "the layout is wrong", hence checking it up front.
        """
        if clearance_mm is None:
            # Half a marker of white between neighbours is a comfortable
            # quiet zone at any realistic working distance.
            clearance_mm = 0.5 * self.marker_length_mm
        need = self.marker_length_mm + clearance_mm

        too_close: list[tuple[int, int]] = []
        ids = self.ids
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                delta = np.abs(self.markers[a].center - self.markers[b].center)
                # Axis-aligned squares only overlap if they are close in BOTH
                # x and y; far apart in either one is enough clearance.
                if delta[0] < need and delta[1] < need:
                    too_close.append((a, b))
        return too_close

    # -- the thing the tracker actually needs ------------------------------

    def object_points_for(self, marker_ids: list[int]) -> np.ndarray:
        """(4K, 3) tool-frame corner points for the given member IDs, in order.

        The caller stacks the matching detected image corners in the same
        order, and the two arrays are handed to a single cv2.solvePnP. That
        one call -- over every visible marker's corners at once -- is the
        entire idea of rigid-body tracking.
        """
        return np.vstack([self.markers[i].corners for i in marker_ids]).astype(np.float32)

    def summary(self) -> str:
        lo, hi = self.extent_mm()
        size = hi - lo
        planar = "coplanar" if self.is_coplanar() else "non-coplanar (3D)"
        lines = [
            f"Tool '{self.name}': {self.n_markers} markers, "
            f"{self.marker_length_mm:.1f} mm each, {planar}",
            f"  footprint : {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm",
            "  members   : id   centre (x, y, z) mm      rot",
        ]
        for marker_id in self.ids:
            m = self.markers[marker_id]
            lines.append(
                f"              {marker_id:<4d} ({m.center[0]:7.1f},{m.center[1]:7.1f},"
                f"{m.center[2]:7.1f})   {m.rotation_deg:5.1f} deg"
            )
        return "\n".join(lines)


def load_tool_geometry(cfg) -> ToolGeometry:
    """Build the tool from a loaded Config, with an actionable error if absent."""
    if cfg.tool is None:
        raise ValueError(
            "config.yaml has no `tool:` section, so there is no tool to track.\n"
            "Add one (see the Phase 2 block in config.yaml) with the marker IDs "
            "and their centre positions in millimetres."
        )
    return ToolGeometry(cfg.tool)
