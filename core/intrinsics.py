"""Camera intrinsics: the pinhole model + lens distortion, with disk I/O.

Intrinsics are stored as YAML rather than .npz so they are diffable and
reviewable in a pull request. A calibration is a measurement, and a
measurement without its provenance is not much use -- so the file also
carries the reprojection RMS, the number and size of the views it was fitted
from, and the board geometry used. If a tracking result later looks wrong,
the first question is always "which calibration is this?", and this file
answers it.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class CameraIntrinsics:
    """Pinhole camera matrix + distortion coefficients.

        camera_matrix = [[fx,  0, cx],
                         [ 0, fy, cy],
                         [ 0,  0,  1]]

    fx/fy are focal lengths in *pixels*; cx/cy the principal point in pixels.
    `dist_coeffs` is OpenCV's plain-radial-tangential vector
    (k1, k2, p1, p2, k3[, ...]).

    `image_size` is not decoration: intrinsics are only valid at the
    resolution they were calibrated at, because fx, fy, cx, cy are all in
    pixel units. Calibrate at 1280x720 and stream at 640x480 and every pose
    will be wrong by roughly 2x with no error message anywhere. `validate_for`
    below is the guard against that.
    """

    camera_matrix: np.ndarray  # (3, 3)
    dist_coeffs: np.ndarray  # (N,) typically 5
    image_size: tuple[int, int]  # (width, height) in pixels
    reprojection_rms_px: float = float("nan")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=float).reshape(3, 3)
        self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=float).ravel()
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))

    # -- convenience accessors --------------------------------------------

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    @property
    def fov_deg(self) -> tuple[float, float]:
        """Horizontal and vertical field of view, in degrees.

        A quick sanity check on a fresh calibration: a typical webcam is
        55-75 deg horizontally. A wildly different number means the fit did
        not converge on anything physical.
        """
        w, h = self.image_size
        fov_x = 2.0 * np.degrees(np.arctan(0.5 * w / self.fx))
        fov_y = 2.0 * np.degrees(np.arctan(0.5 * h / self.fy))
        return float(fov_x), float(fov_y)

    # -- validation --------------------------------------------------------

    def validate_for(self, width: int, height: int) -> None:
        """Raise if these intrinsics do not match the live frame size."""
        if (int(width), int(height)) != self.image_size:
            raise ValueError(
                f"Intrinsics were calibrated at {self.image_size[0]}x{self.image_size[1]} "
                f"but the camera is delivering {width}x{height}. Pose would be wrong by "
                "roughly the ratio of the resolutions.\n"
                "Fix: set camera.frame_width/frame_height in config.yaml to the "
                "calibrated size, or re-run `calibrate` at the new resolution."
            )

    # -- I/O ---------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "image_width": self.image_size[0],
            "image_height": self.image_size[1],
            "reprojection_rms_px": float(self.reprojection_rms_px),
            "metadata": {
                "saved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                **self.metadata,
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)
        return path

    @staticmethod
    def load(path: str | Path) -> "CameraIntrinsics":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"No camera intrinsics at {path}.\n"
                "Run `python app.py calibrate` first -- pose estimation is "
                "meaningless without them."
            )
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        return CameraIntrinsics(
            camera_matrix=np.array(doc["camera_matrix"], dtype=float),
            dist_coeffs=np.array(doc["dist_coeffs"], dtype=float),
            image_size=(int(doc["image_width"]), int(doc["image_height"])),
            reprojection_rms_px=float(doc.get("reprojection_rms_px", float("nan"))),
            metadata=doc.get("metadata") or {},
        )

    def summary(self) -> str:
        fov_x, fov_y = self.fov_deg
        dist = ", ".join(f"{c:+.5f}" for c in self.dist_coeffs)
        return (
            f"  resolution     : {self.image_size[0]} x {self.image_size[1]} px\n"
            f"  focal length   : fx={self.fx:.2f}  fy={self.fy:.2f} px\n"
            f"  principal point: cx={self.cx:.2f}  cy={self.cy:.2f} px\n"
            f"  field of view  : {fov_x:.1f} deg horizontal, {fov_y:.1f} deg vertical\n"
            f"  distortion     : [{dist}]\n"
            f"  reproj. RMS    : {self.reprojection_rms_px:.4f} px"
        )
