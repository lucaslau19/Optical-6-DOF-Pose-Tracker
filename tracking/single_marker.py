"""Single-marker 6-DOF pose estimation via cv2.solvePnP.

This module deliberately does by hand what the removed
`cv2.aruco.estimatePoseSingleMarkers` used to do implicitly, because the
explicit form is the one that generalises. In Phase 2 a rigid tool is a set
of markers whose corners all live in ONE tool frame, and pose comes from a
single solvePnP over every visible corner at once. That is the same call as
below with a longer object-point array -- so writing it explicitly now means
Phase 2 is an extension rather than a rewrite.

MARKER FRAME CONVENTION
-----------------------
`detectMarkers` returns the four corners in a fixed order -- top-left,
top-right, bottom-right, bottom-left as seen in the marker's own upright
orientation. Matching object points for a marker of side s:

        (-s/2, +s/2, 0)   (+s/2, +s/2, 0)
              +-----------------+          Y
              |        ^ Y      |          ^
              |        |        |          |
              |        +--> X   |          +--> X      Z = X x Y, out of
              |     (origin at  |                      the marker face,
              |      centre)    |                      toward the camera.
              +-----------------+
        (-s/2, -s/2, 0)   (+s/2, -s/2, 0)

so the pose returned is T_cam_marker, and its translation is the position of
the marker's *centre* in camera coordinates.

THE PLANAR AMBIGUITY
--------------------
A single planar square viewed under perspective has TWO poses that reproject
almost identically -- the true one and one flipped about an axis in the
marker plane. When the marker is small, distant, or near fronto-parallel, the
two reprojection errors are nearly equal and the solver's choice between them
flickers frame to frame. On screen that looks like the axes suddenly folding
over. This is not a bug to be filtered away; it is a fundamental limitation
of single-marker tracking, and it is the reason real optical navigation tools
use a rigid cluster of several markers spread in 3D.

So rather than hide it, `estimate_pose` reports the *ambiguity ratio*
(second-best reprojection error / best). A ratio near 1.0 means the two
solutions are indistinguishable and the pose should not be trusted. Phase 2
fixes this properly; Phase 1 at least measures it honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.cv_compat import make_aruco_detector
from core.intrinsics import CameraIntrinsics
from core.transforms import SE3

# Above this ratio the two planar solutions are considered well separated and
# the chosen pose is trustworthy. Empirical, and reported on the HUD so it can
# be judged rather than taken on faith.
AMBIGUITY_OK_RATIO = 3.0


def marker_object_points(marker_length_mm: float) -> np.ndarray:
    """The four marker corners in the marker's own frame, in mm.

    Order matches cv2.aruco detectMarkers output. float32 because that is
    what solvePnP expects; a float64 array here is a silent no-op on some
    builds and a type error on others.
    """
    s = float(marker_length_mm) / 2.0
    return np.array(
        [
            [-s, +s, 0.0],  # top-left
            [+s, +s, 0.0],  # top-right
            [+s, -s, 0.0],  # bottom-right
            [-s, -s, 0.0],  # bottom-left
        ],
        dtype=np.float32,
    )


@dataclass
class MarkerPose:
    """A single marker's measured pose and the quality numbers behind it."""

    marker_id: int
    pose: SE3  # T_cam_marker
    corners: np.ndarray  # (4, 2) image points, sub-pixel
    reprojection_rms_px: float
    ambiguity_ratio: float  # second-best error / best error; >1, higher is better

    @property
    def distance_mm(self) -> float:
        """Range from camera origin to marker centre."""
        return self.pose.translation_norm

    @property
    def is_ambiguous(self) -> bool:
        return self.ambiguity_ratio < AMBIGUITY_OK_RATIO

    def format_line(self) -> str:
        x, y, z = self.pose.t
        rx, ry, rz = self.pose.as_euler_deg()
        return (
            f"id {self.marker_id:3d} | "
            f"xyz [{x:8.1f} {y:8.1f} {z:8.1f}] mm | "
            f"range {self.distance_mm:7.1f} mm | "
            f"rpy [{rx:7.1f} {ry:7.1f} {rz:7.1f}] deg | "
            f"reproj {self.reprojection_rms_px:5.2f} px | "
            f"amb {self.ambiguity_ratio:5.2f}"
        )


def estimate_pose(
    corners: np.ndarray,
    marker_length_mm: float,
    intrinsics: CameraIntrinsics,
) -> tuple[SE3, float, float]:
    """Solve T_cam_marker for one marker.

    Returns (pose, reprojection RMS in px, ambiguity ratio).

    SOLVEPNP_IPPE_SQUARE is the right flag here and not a detail: it is a
    closed-form solver specialised for a planar square with exactly four
    correspondences, and it is what estimatePoseSingleMarkers used internally.
    The iterative default would need an initial guess and can converge to the
    flipped solution.

    solvePnPGeneric (rather than solvePnP) is used so that BOTH planar
    solutions come back and the ambiguity can be quantified -- see the module
    docstring.
    """
    object_points = marker_object_points(marker_length_mm)
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)

    n_solutions, rvecs, tvecs, errors = cv2.solvePnPGeneric(
        object_points,
        image_points,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if n_solutions < 1:
        raise RuntimeError("solvePnP found no solution for a detected marker")

    # solvePnPGeneric returns solutions sorted by reprojection error, best first.
    pose = SE3.from_rvec_tvec(rvecs[0], tvecs[0])

    err = np.asarray(errors, dtype=float).ravel()
    best = float(err[0])
    if n_solutions > 1 and best > 1e-9:
        ambiguity = float(err[1]) / best
    else:
        ambiguity = float("inf")

    # solvePnPGeneric's error is an L2 norm over all points; convert to a
    # per-point RMS so the number is comparable to the calibration RMS.
    rms = best / np.sqrt(len(object_points))

    return pose, float(rms), ambiguity


class SingleMarkerTracker:
    """Detect ArUco markers in a frame and solve each one's 6-DOF pose."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        dictionary_name: str,
        corner_refinement: str,
        marker_length_mm: float,
    ) -> None:
        self.intrinsics = intrinsics
        self.marker_length_mm = float(marker_length_mm)
        self.detector = make_aruco_detector(dictionary_name, corner_refinement)

    def process(self, frame_bgr: np.ndarray) -> list[MarkerPose]:
        """Detect and pose every marker in a BGR frame, nearest first."""
        # Detection runs on grayscale: the marker code is binary, colour adds
        # nothing, and converting once here avoids OpenCV doing it internally
        # on every call.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return []

        results: list[MarkerPose] = []
        for corner_set, marker_id in zip(corners, ids.ravel()):
            pose, rms, ambiguity = estimate_pose(
                corner_set, self.marker_length_mm, self.intrinsics
            )
            results.append(
                MarkerPose(
                    marker_id=int(marker_id),
                    pose=pose,
                    corners=np.asarray(corner_set, dtype=float).reshape(4, 2),
                    reprojection_rms_px=rms,
                    ambiguity_ratio=ambiguity,
                )
            )

        results.sort(key=lambda m: m.distance_mm)
        return results
