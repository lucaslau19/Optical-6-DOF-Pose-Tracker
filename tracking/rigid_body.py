"""Rigid-body (multi-marker) tool tracking.

THE ONE IDEA
------------
Every visible member marker contributes four 3D<->2D correspondences. They are
concatenated into ONE array and solved with ONE cv2.solvePnP:

    corners of marker 10  ->  4 points
    corners of marker 11  ->  4 points        all in the TOOL frame
    corners of marker 13  ->  4 points
    ---------------------------------
    12 correspondences  ->  solvePnP  ->  T_cam_tool

This is not the same as estimating each marker's pose and averaging. Averaging
poses is ill-defined for rotations, weights a badly-conditioned marker equally
with a good one, and throws away the fact that the markers constrain each
other. A single fit over all points minimises one reprojection cost and uses
the whole rigid body as evidence -- which is why the pose barely moves when a
marker disappears: the remaining points were already doing most of the work.

WHY THE POSE SURVIVES OCCLUSION
-------------------------------
Nothing is re-initialised when a marker vanishes. The correspondence set just
gets shorter. With 3 of 4 markers visible the geometry is still massively
over-determined (12 equations for 6 unknowns), so the fitted pose is nearly
unchanged. That continuity is the Phase 2 win condition, and it falls out of
the formulation rather than from any smoothing or filtering -- there is
deliberately no temporal filter here, because a filter would hide exactly the
behaviour we want to measure.

THE PLANAR AMBIGUITY, HONESTLY
------------------------------
A coplanar marker cluster -- markers printed on one flat sheet -- still has
the two-fold planar pose ambiguity described in tracking/single_marker.py. It
is much better conditioned than a single small marker, because the points
span far more image area, but it is not eliminated. Only a NON-COPLANAR
arrangement removes it, which is why real tracked instruments carry fiducials
on a bent or stepped bracket rather than a flat plate.

So: the ambiguity ratio is still measured and still reported, `z` is already
honoured in the tool config so a 3D arrangement works today, and the HUD says
which case you are in rather than implying the problem went away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from core.cv_compat import make_aruco_detector
from core.intrinsics import CameraIntrinsics
from core.transforms import SE3, rotation_angle_between
from markers.tool import ToolGeometry

# Below this ratio the two planar solutions are hard to tell apart. Same
# meaning as AMBIGUITY_OK_RATIO in single_marker.py.
AMBIGUITY_OK_RATIO = 3.0

# When the two planar solutions are nearly tied, prefer the one closer to the
# previous frame's orientation. This is display hysteresis, NOT a fix: it stops
# a genuinely ambiguous pose from flickering between two readings frame to
# frame, which is unwatchable on a HUD. It cannot make an ambiguous pose
# correct, so the low-confidence flag stays on regardless.
HYSTERESIS_MAX_ANGLE_DEG = 90.0


@dataclass
class ToolPose:
    """The result of one frame's tool fit."""

    pose: SE3 | None  # T_cam_tool, or None when the tool is lost
    status: str  # "tracking" | "low_confidence" | "lost"
    marker_ids_used: list[int] = field(default_factory=list)
    marker_ids_visible_other: list[int] = field(default_factory=list)
    reprojection_rms_px: float = float("nan")
    per_marker_rms_px: dict[int, float] = field(default_factory=dict)
    ambiguity_ratio: float = float("inf")
    n_points: int = 0
    detected_corners: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def is_tracking(self) -> bool:
        return self.pose is not None

    @property
    def n_markers_used(self) -> int:
        return len(self.marker_ids_used)

    @property
    def is_ambiguous(self) -> bool:
        return self.ambiguity_ratio < AMBIGUITY_OK_RATIO

    @property
    def distance_mm(self) -> float:
        return self.pose.translation_norm if self.pose else float("nan")

    def format_line(self) -> str:
        if self.pose is None:
            return f"TOOL LOST  (visible non-tool ids: {self.marker_ids_visible_other})"
        x, y, z = self.pose.t
        rx, ry, rz = self.pose.as_euler_deg()
        return (
            f"{self.n_markers_used}/{self.n_points // 4} mk | "
            f"xyz [{x:8.1f} {y:8.1f} {z:8.1f}] mm | "
            f"range {self.distance_mm:7.1f} mm | "
            f"rpy [{rx:7.1f} {ry:7.1f} {rz:7.1f}] deg | "
            f"rms {self.reprojection_rms_px:5.2f} px | "
            f"ids {self.marker_ids_used}"
        )


def _reprojection_rms(
    object_points: np.ndarray,
    image_points: np.ndarray,
    pose: SE3,
    intrinsics: CameraIntrinsics,
) -> tuple[float, np.ndarray]:
    """Overall RMS in pixels, plus the per-point error magnitudes."""
    rvec, tvec = pose.as_rvec_tvec()
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, intrinsics.camera_matrix, intrinsics.dist_coeffs
    )
    residuals = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    per_point = np.linalg.norm(residuals, axis=1)
    rms = float(np.sqrt(np.mean(per_point**2)))
    return rms, per_point


class RigidBodyTracker:
    """Detects the tool's member markers and fits one pose to all of them."""

    def __init__(
        self,
        geometry: ToolGeometry,
        intrinsics: CameraIntrinsics,
        dictionary_name: str,
        corner_refinement: str,
    ) -> None:
        self.geometry = geometry
        self.intrinsics = intrinsics
        self.detector = make_aruco_detector(dictionary_name, corner_refinement)

        self._coplanar = geometry.is_coplanar()
        # Previous pose, used only to break near-ties between the two planar
        # solutions and as the iterative refiner's starting point.
        self._previous_pose: SE3 | None = None

    # -- solving -----------------------------------------------------------

    def _initial_pose(
        self, object_points: np.ndarray, image_points: np.ndarray
    ) -> tuple[SE3 | None, float]:
        """Closed-form starting pose, plus the ambiguity ratio.

        SOLVEPNP_ITERATIVE is a local optimiser: given a poor starting point it
        happily converges to the wrong local minimum (typically the flipped
        planar solution). So the initial pose comes from a closed-form solver
        chosen for the geometry:

          * coplanar tool -> SOLVEPNP_IPPE, which is built for planar targets
            and returns BOTH solutions so the ambiguity can be measured.
          * non-coplanar  -> SOLVEPNP_SQPNP, a global solver for general 3D
            point sets with no planar ambiguity to worry about.
        """
        K, dist = self.intrinsics.camera_matrix, self.intrinsics.dist_coeffs

        if self._coplanar:
            try:
                n, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                    object_points, image_points, K, dist, flags=cv2.SOLVEPNP_IPPE
                )
            except cv2.error:
                return None, float("inf")
            if n < 1:
                return None, float("inf")

            candidates = [SE3.from_rvec_tvec(rvecs[i], tvecs[i]) for i in range(n)]
            err = np.asarray(errors, dtype=float).ravel()
            ambiguity = float(err[1] / err[0]) if n > 1 and err[0] > 1e-9 else float("inf")

            chosen = candidates[0]
            # Near-tie: prefer continuity with the previous frame so the HUD
            # does not flicker between two equally-good readings.
            if (
                n > 1
                and ambiguity < AMBIGUITY_OK_RATIO
                and self._previous_pose is not None
            ):
                deltas = [
                    rotation_angle_between(self._previous_pose, c) for c in candidates
                ]
                nearest = int(np.argmin(deltas))
                if deltas[nearest] < HYSTERESIS_MAX_ANGLE_DEG:
                    chosen = candidates[nearest]
            return chosen, ambiguity

        # Non-coplanar: no two-fold ambiguity, so a single global solve.
        flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_EPNP)
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist, flags=flag)
        if not ok:
            return None, float("inf")
        return SE3.from_rvec_tvec(rvec, tvec), float("inf")

    def _fit(self, object_points: np.ndarray, image_points: np.ndarray) -> tuple[SE3 | None, float]:
        """Initialise, then refine with SOLVEPNP_ITERATIVE."""
        initial, ambiguity = self._initial_pose(object_points, image_points)
        if initial is None:
            return None, ambiguity

        rvec, tvec = initial.as_rvec_tvec()
        # useExtrinsicGuess=True makes ITERATIVE start from the closed-form
        # solution and polish it by Levenberg-Marquardt on the true
        # reprojection error -- which is the quantity we actually care about,
        # and which the closed-form solvers only approximate.
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.intrinsics.camera_matrix,
            self.intrinsics.dist_coeffs,
            rvec.copy(),
            tvec.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            # Refinement failing is not fatal: the closed-form pose is still a
            # valid answer, just slightly less accurate.
            return initial, ambiguity
        return SE3.from_rvec_tvec(rvec, tvec), ambiguity

    # -- per-frame ---------------------------------------------------------

    def process(self, frame_bgr: np.ndarray) -> ToolPose:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            self._previous_pose = None
            return ToolPose(pose=None, status="lost")

        # Split what was seen into "belongs to this tool" and "everything
        # else". Foreign markers are not an error -- the calibration board and
        # the Phase 1 single markers share the dictionary -- they are just not
        # part of this rigid body, so they must not enter the fit.
        used_ids: list[int] = []
        detected: dict[int, np.ndarray] = {}
        other_ids: list[int] = []
        for corner_set, marker_id in zip(corners, ids.ravel()):
            marker_id = int(marker_id)
            if marker_id in self.geometry.markers:
                used_ids.append(marker_id)
                detected[marker_id] = np.asarray(corner_set, dtype=np.float32).reshape(4, 2)
            else:
                other_ids.append(marker_id)

        # Sort so the object and image point arrays are built in a stable,
        # matching order regardless of detection order.
        used_ids.sort()

        if len(used_ids) < max(1, self.geometry.config.min_markers):
            self._previous_pose = None
            return ToolPose(
                pose=None,
                status="lost",
                marker_ids_visible_other=sorted(other_ids),
                detected_corners=detected,
            )

        object_points = self.geometry.object_points_for(used_ids)
        image_points = np.vstack([detected[i] for i in used_ids]).astype(np.float32)

        pose, ambiguity = self._fit(object_points, image_points)
        if pose is None:
            self._previous_pose = None
            return ToolPose(
                pose=None,
                status="lost",
                marker_ids_visible_other=sorted(other_ids),
                detected_corners=detected,
            )

        rms, per_point = _reprojection_rms(object_points, image_points, pose, self.intrinsics)

        # Per-marker residuals. A single member with a much larger residual
        # than the rest is the signature of a mis-measured centre in
        # config.yaml -- the geometry says it is somewhere the image disagrees
        # with. This turns "the pose is a bit off" into "member 12 is wrong".
        per_marker: dict[int, float] = {}
        for k, marker_id in enumerate(used_ids):
            block = per_point[4 * k : 4 * k + 4]
            per_marker[marker_id] = float(np.sqrt(np.mean(block**2)))

        low_conf_at = self.geometry.config.low_confidence_markers
        status = "low_confidence" if len(used_ids) <= low_conf_at else "tracking"

        self._previous_pose = pose
        return ToolPose(
            pose=pose,
            status=status,
            marker_ids_used=used_ids,
            marker_ids_visible_other=sorted(other_ids),
            reprojection_rms_px=rms,
            per_marker_rms_px=per_marker,
            ambiguity_ratio=ambiguity,
            n_points=len(object_points),
            detected_corners=detected,
        )
