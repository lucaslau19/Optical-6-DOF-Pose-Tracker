"""ChArUco detection and camera-intrinsics estimation.

Pipeline, in the modern (OpenCV >= 4.7) API:

    CharucoDetector.detectBoard(gray)
        -> chessboard corners with unique IDs
    CharucoBoard.matchImagePoints(corners, ids)
        -> (3D object points in board frame, 2D image points)   [explicit!]
    cv2.calibrateCamera(objpoints, imgpoints, size)
        -> camera matrix, distortion, per-view rvec/tvec, overall RMS

The middle step is the one worth understanding. `matchImagePoints` is how
the board tells you the 3D coordinate of each corner it just found, in
millimetres, in the board's own frame. Nothing is implicit: if the board
geometry in config.yaml is wrong, the object points are wrong, and the whole
calibration is wrong in a way no residual will reveal (a scale error fits
perfectly -- it just fits the wrong scale). Hence the "measure your print"
insistence in markers/generate.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.intrinsics import CameraIntrinsics


@dataclass
class CharucoView:
    """One accepted calibration observation."""

    charuco_corners: np.ndarray  # (N, 1, 2) float32, image pixels
    charuco_ids: np.ndarray  # (N, 1) int32, board corner IDs
    marker_corners: list  # raw ArUco corners, kept for the on-screen overlay
    marker_ids: np.ndarray | None
    image_size: tuple[int, int]  # (width, height)

    @property
    def n_corners(self) -> int:
        return int(self.charuco_corners.shape[0])

    @property
    def centroid(self) -> np.ndarray:
        return self.charuco_corners.reshape(-1, 2).mean(axis=0)


@dataclass
class CalibrationResult:
    intrinsics: CameraIntrinsics
    per_view_rms_px: np.ndarray  # (n_views,)
    n_views: int
    n_points: int
    coverage_fraction: float  # fraction of the image area the board visited

    def report(self, outlier_rms_px: float = 1.0) -> str:
        """Human-readable calibration report, in the shape of a V&V summary."""
        lines = [
            "Camera calibration result",
            "=" * 60,
            self.intrinsics.summary(),
            "",
            f"  views used     : {self.n_views}",
            f"  corner points  : {self.n_points}",
            f"  image coverage : {self.coverage_fraction * 100:.0f}% of frame area",
            f"  per-view RMS   : min {self.per_view_rms_px.min():.3f} / "
            f"median {np.median(self.per_view_rms_px):.3f} / "
            f"max {self.per_view_rms_px.max():.3f} px",
        ]

        outliers = np.flatnonzero(self.per_view_rms_px > outlier_rms_px)
        if outliers.size:
            lines.append("")
            lines.append(f"  {outliers.size} view(s) above {outlier_rms_px:.2f} px:")
            for i in outliers:
                lines.append(f"    view {int(i):3d}: {self.per_view_rms_px[i]:.3f} px")
            lines.append(
                "  High-residual views are usually motion blur or a board seen too\n"
                "  obliquely. Consider deleting those frames and re-running with\n"
                "  `--from-images`."
            )

        lines.append("")
        lines.append(_interpret_rms(self.intrinsics.reprojection_rms_px))
        return "\n".join(lines)


def _interpret_rms(rms: float) -> str:
    """Turn the RMS number into a pass/fail judgement.

    A reprojection RMS is a residual, not an accuracy: it says how well the
    model fits the data it was fitted to, and it can be made small by using
    too few, too-similar views. It is a necessary-but-not-sufficient check.
    Real accuracy gets measured against known geometry in Phase 5.
    """
    if not np.isfinite(rms):
        return "  VERDICT: no RMS available."
    if rms < 0.3:
        verdict = "excellent"
    elif rms < 0.5:
        verdict = "good -- typical of a solid webcam calibration"
    elif rms < 1.0:
        verdict = "acceptable, but worth a retake with sharper, better-spread views"
    else:
        verdict = "POOR -- do not trust poses from this; recalibrate"
    return (
        f"  VERDICT: reprojection RMS {rms:.3f} px is {verdict}.\n"
        "  Note: low RMS means the model fits these views, NOT that pose is\n"
        "  accurate. Views must also span the frame and a range of tilts, and\n"
        "  the printed board must measure what config.yaml claims."
    )


def detect_charuco(
    detector,
    gray: np.ndarray,
    min_corners: int,
) -> CharucoView | None:
    """Detect the ChArUco board in a grayscale frame.

    Returns None if the board is absent or too partially visible to be worth
    keeping. `detectBoard` returns None (not an empty array) for a frame with
    no board, which is why every access is guarded.
    """
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    if charuco_ids is None or len(charuco_ids) < min_corners:
        return None

    h, w = gray.shape[:2]
    return CharucoView(
        charuco_corners=np.asarray(charuco_corners, dtype=np.float32),
        charuco_ids=np.asarray(charuco_ids, dtype=np.int32),
        marker_corners=list(marker_corners) if marker_corners is not None else [],
        marker_ids=marker_ids,
        image_size=(w, h),
    )


# Resolution of the coverage map. Shared by the report and the live capture
# HUD so the number printed at the end matches the grid drawn during capture.
COVERAGE_GRID = 8


def corner_occupancy(
    views: list[CharucoView], image_size: tuple[int, int], grid: int = COVERAGE_GRID
) -> np.ndarray:
    """Boolean (grid, grid) map of where detected corners have ever landed.

    Distortion coefficients are only constrained where you actually put the
    board. A dataset shot entirely in the middle of the frame yields a
    beautiful RMS and a distortion model that is pure extrapolation at the
    edges -- exactly where lens distortion is worst. This map makes that
    failure mode visible instead of invisible, both live and in the report.
    """
    w, h = image_size
    occupied = np.zeros((grid, grid), dtype=bool)
    for view in views:
        pts = view.charuco_corners.reshape(-1, 2)
        cols = np.clip((pts[:, 0] / w * grid).astype(int), 0, grid - 1)
        rows = np.clip((pts[:, 1] / h * grid).astype(int), 0, grid - 1)
        occupied[rows, cols] = True
    return occupied


def calibrate_from_views(
    board,
    views: list[CharucoView],
    image_size: tuple[int, int],
    *,
    flags: int = 0,
) -> CalibrationResult:
    """Fit camera intrinsics from a set of ChArUco observations."""
    if len(views) < 3:
        raise ValueError(
            f"Need at least 3 views to fit intrinsics (got {len(views)}). "
            "In practice aim for 15-25."
        )

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []

    for view in views:
        # The explicit 3D<->2D correspondence. objp is in board coordinates
        # (millimetres, z = 0 because the board is planar).
        objp, imgp = board.matchImagePoints(view.charuco_corners, view.charuco_ids)
        if objp is None or len(objp) < 4:
            continue
        object_points.append(objp)
        image_points.append(imgp)

    if len(object_points) < 3:
        raise ValueError("Too few views survived point matching to calibrate.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=flags,
    )

    # Per-view residuals. calibrateCamera only hands back the pooled RMS, but
    # a single bad view (blur, a mis-detection) can dominate it, and you want
    # to know *which* view to discard rather than just that one exists.
    per_view = np.zeros(len(object_points), dtype=float)
    total_sq_err = 0.0
    total_pts = 0
    for i, (objp, imgp) in enumerate(zip(object_points, image_points)):
        projected, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = projected.reshape(-1, 2) - imgp.reshape(-1, 2)
        sq = float(np.sum(err**2))
        per_view[i] = np.sqrt(sq / len(objp))
        total_sq_err += sq
        total_pts += len(objp)

    intrinsics = CameraIntrinsics(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        reprojection_rms_px=float(rms),
        metadata={
            "n_views": len(object_points),
            "n_points": int(total_pts),
            "board_squares": [int(board.getChessboardSize()[0]), int(board.getChessboardSize()[1])],
            "square_length_mm": float(board.getSquareLength()),
            "marker_length_mm": float(board.getMarkerLength()),
            "units": "mm",
            "opencv_version": cv2.__version__,
        },
    )

    return CalibrationResult(
        intrinsics=intrinsics,
        per_view_rms_px=per_view,
        n_views=len(object_points),
        n_points=int(total_pts),
        coverage_fraction=float(corner_occupancy(views, image_size).mean()),
    )
