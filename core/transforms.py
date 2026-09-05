"""Minimal SE(3) rigid-body transform helper.

=============================================================================
NAMING CONVENTION  (used everywhere in this repo -- documented once, here)
=============================================================================

    T_a_b  maps points expressed in frame b  ->  points expressed in frame a.

        p_a = T_a_b @ p_b

Read the subscripts right-to-left: "b into a". The nice property of this
convention is that composition cancels adjacent frames, so a chain reads
left-to-right like a sentence:

        T_a_c = T_a_b @ T_b_c            (b cancels)
        T_b_a = T_a_b.inv()

The pose of an object is the transform that takes points *out of* that
object's frame and into the observer's frame. So a marker's pose in the
camera frame is `T_cam_marker`, and its translation column is literally the
position of the marker origin as seen by the camera. That is exactly what
cv2.solvePnP gives you: rvec/tvec describe how to move object points into
camera coordinates, i.e. T_cam_object.

Later phases lean on this hard, e.g. the navigation transform is

        T_ref_tool = T_cam_ref.inv() @ T_cam_tool

which is invariant to camera motion -- the whole reason a patient reference
frame exists.

=============================================================================
UNITS
=============================================================================

Millimetres, everywhere, with no exceptions. Surgical navigation accuracy is
quoted in mm, and solvePnP returns translation in whatever unit the 3D object
points were given in -- so if the marker geometry is specified in mm, the pose
comes back in mm for free. Angles are degrees at the display boundary only;
internally they stay in radians / rotation matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SE3:
    """A rigid-body transform: rotation matrix + translation vector.

    Stored as (R, t) rather than a 4x4 matrix because that is the form
    OpenCV and the pivot-calibration least-squares system both want. The 4x4
    is available via `as_matrix()` when it is more convenient.

    Frozen (immutable) so a pose can be stashed in a history buffer without
    worrying that something downstream mutated it in place.
    """

    R: np.ndarray  # (3, 3) rotation, right-handed, det(R) = +1
    t: np.ndarray  # (3,) translation in mm

    def __post_init__(self) -> None:
        R = np.asarray(self.R, dtype=float).reshape(3, 3)
        t = np.asarray(self.t, dtype=float).reshape(3)
        # dataclass is frozen, so assign through object.__setattr__
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    # -- constructors ------------------------------------------------------

    @staticmethod
    def identity() -> "SE3":
        return SE3(np.eye(3), np.zeros(3))

    @staticmethod
    def from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> "SE3":
        """Build from the OpenCV pose representation.

        `rvec` is a Rodrigues (axis-angle) vector whose magnitude is the
        rotation angle in radians. scipy's `from_rotvec` uses the identical
        convention, so this needs no sign or ordering fixups.
        """
        R = Rotation.from_rotvec(np.asarray(rvec, dtype=float).reshape(3)).as_matrix()
        return SE3(R, np.asarray(tvec, dtype=float).reshape(3))

    @staticmethod
    def from_matrix(M: np.ndarray) -> "SE3":
        M = np.asarray(M, dtype=float)
        return SE3(M[:3, :3], M[:3, 3])

    # -- conversions -------------------------------------------------------

    def as_matrix(self) -> np.ndarray:
        """Homogeneous 4x4 form."""
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    def as_rvec_tvec(self) -> tuple[np.ndarray, np.ndarray]:
        """OpenCV form, for handing back to cv2.projectPoints / drawFrameAxes."""
        rvec = Rotation.from_matrix(self.R).as_rotvec()
        return rvec.reshape(3, 1), self.t.reshape(3, 1)

    def as_euler_deg(self, seq: str = "xyz") -> np.ndarray:
        """Euler angles in degrees, for human-readable printouts only.

        Never use Euler angles for maths -- they gimbal-lock and the ordering
        conventions are a well-known source of silent bugs. They exist here
        because "rx=12.4 deg" is easier to sanity-check on a HUD than a
        3x3 matrix.
        """
        return Rotation.from_matrix(self.R).as_euler(seq, degrees=True)

    def as_quat(self) -> np.ndarray:
        """Quaternion (x, y, z, w) -- scipy's ordering, w last."""
        return Rotation.from_matrix(self.R).as_quat()

    # -- algebra -----------------------------------------------------------

    def inv(self) -> "SE3":
        """Inverse transform. For a rotation, R^-1 == R.T (cheap and exact)."""
        Rt = self.R.T
        return SE3(Rt, -Rt @ self.t)

    def __matmul__(self, other: "SE3") -> "SE3":
        """Compose: `T_a_b @ T_b_c` -> `T_a_c`."""
        if not isinstance(other, SE3):
            return NotImplemented
        return SE3(self.R @ other.R, self.R @ other.t + self.t)

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        """Apply to an (N, 3) array of points (or a single (3,) point)."""
        pts = np.asarray(pts, dtype=float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)
        out = pts @ self.R.T + self.t
        return out[0] if single else out

    # -- metrics -----------------------------------------------------------

    @property
    def translation_norm(self) -> float:
        """Distance from the origin of the parent frame, in mm."""
        return float(np.linalg.norm(self.t))

    @property
    def rotation_angle_deg(self) -> float:
        """Total rotation angle (the axis-angle magnitude), in degrees."""
        return float(np.degrees(np.linalg.norm(Rotation.from_matrix(self.R).as_rotvec())))

    def __repr__(self) -> str:  # pragma: no cover - display only
        rx, ry, rz = self.as_euler_deg()
        x, y, z = self.t
        return (
            f"SE3(t=[{x:8.2f} {y:8.2f} {z:8.2f}] mm, "
            f"rpy=[{rx:7.2f} {ry:7.2f} {rz:7.2f}] deg)"
        )


def rotation_angle_between(a: SE3, b: SE3) -> float:
    """Angle in degrees between the orientations of two poses.

    Used later for repeatability/jitter numbers, where "how much did the
    orientation wobble" needs to be a single scalar rather than three
    interacting Euler angles.
    """
    R_rel = a.R.T @ b.R
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(R_rel).as_rotvec())))
