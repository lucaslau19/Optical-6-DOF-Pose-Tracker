"""Pure statistics for the accuracy metrics -- no camera, no I/O, no OpenCV.

Separated out so every number the project reports can be tested against
analytic cases. If a metric is wrong here, it is wrong everywhere.

A NOTE ON VOCABULARY, because these words get used loosely and they are not
interchangeable:

    PRECISION (repeatability) -- how tightly repeated measurements agree with
        EACH OTHER. Needs no ground truth. Static jitter and touch
        repeatability are precision metrics.

    TRUENESS (bias) -- how close measurements are to the TRUE value. Needs
        ground truth: a ruler, a caliper, a printed grid. Distance error and
        registration residual are trueness metrics.

    ACCURACY -- both together.

A system can be beautifully precise and completely untrue: if the printed
marker geometry is 2% small, every measurement is consistently 2% short and
the repeatability looks superb. That is exactly why this project reports the
MEAN SIGNED distance error separately from the mean absolute error -- a
non-zero signed mean is the signature of a scale or geometry error, and no
amount of averaging will remove it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.transforms import SE3


# --------------------------------------------------------------------------
# Scalar summaries
# --------------------------------------------------------------------------


@dataclass
class Spread:
    """Summary of a 1-D sample."""

    n: int
    mean: float
    std: float
    rms: float
    min: float
    max: float
    peak_to_peak: float

    def format(self, unit: str = "mm", width: int = 8) -> str:
        return (
            f"mean {self.mean:{width}.4f}  std {self.std:{width}.4f}  "
            f"p-p {self.peak_to_peak:{width}.4f}  {unit}"
        )


def describe(values: np.ndarray) -> Spread:
    """Summarise a 1-D sample.

    `std` uses ddof=1 (the sample standard deviation). With N in the hundreds
    the difference from ddof=0 is negligible, but ddof=1 is the correct
    estimator for a sample drawn from a larger population, which is what a
    finite run of frames is.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("describe() needs at least one sample")
    return Spread(
        n=int(v.size),
        mean=float(v.mean()),
        std=float(v.std(ddof=1)) if v.size > 1 else 0.0,
        rms=float(np.sqrt(np.mean(v**2))),
        min=float(v.min()),
        max=float(v.max()),
        peak_to_peak=float(v.max() - v.min()),
    )


# --------------------------------------------------------------------------
# Point scatter
# --------------------------------------------------------------------------


@dataclass
class PointScatter:
    """How tightly a set of 3-D points clusters about its own centroid."""

    n: int
    centroid: np.ndarray  # (3,)
    per_axis_std: np.ndarray  # (3,) sample std of x, y, z
    per_axis_peak_to_peak: np.ndarray  # (3,)
    rms_3d: float  # RMS distance from the centroid
    max_3d: float  # worst single deviation
    distances: np.ndarray  # (n,) distance of each point from the centroid


def point_scatter(points: np.ndarray) -> PointScatter:
    """Scatter of an (N, 3) point set about its centroid, in the input units.

    `rms_3d` is the headline: sqrt(mean(||p_i - centroid||^2)). It is reported
    rather than a per-axis std because a surgeon cares how far the tip is from
    where it should be, not how that error decomposes onto axes -- and because
    the three axes are not independent (depth is always the worst for a
    monocular system, and quoting only the best axis would flatter the result).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 2:
        raise ValueError("point_scatter() needs at least 2 points")
    centroid = pts.mean(axis=0)
    deltas = pts - centroid
    distances = np.linalg.norm(deltas, axis=1)
    return PointScatter(
        n=len(pts),
        centroid=centroid,
        per_axis_std=pts.std(axis=0, ddof=1),
        per_axis_peak_to_peak=pts.max(axis=0) - pts.min(axis=0),
        rms_3d=float(np.sqrt(np.mean(distances**2))),
        max_3d=float(distances.max()),
        distances=distances,
    )


# --------------------------------------------------------------------------
# Rotations
# --------------------------------------------------------------------------


def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    """The chordal (Frobenius) mean of a set of rotation matrices.

    Averaging rotation matrices elementwise gives a matrix that is not a
    rotation, so the elementwise mean is projected back onto SO(3) by taking
    the closest orthonormal matrix via SVD. The determinant guard prevents
    that projection landing on a reflection (det = -1), which is orthonormal
    but is a mirror, not a rotation.

    Valid for tightly clustered rotations, which is exactly the case here --
    a static tool wobbling by a fraction of a degree.
    """
    Rs = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    U, _s, Vt = np.linalg.svd(Rs.sum(axis=0))
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    return U @ D @ Vt


def rotation_deviations_deg(rotations: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Angle of each rotation relative to `reference`, in degrees.

    A single angle per sample rather than three Euler components, because
    Euler angles interact and "0.3 deg of wobble" is the number that means
    something.

    Computed as atan2(sin, cos) rather than the textbook
    arccos((trace - 1) / 2). That matters here specifically: this function
    exists to measure SUB-DEGREE jitter, and arccos is at its worst exactly
    there. Near zero rotation cos is ~1, so rounding to 1 - eps becomes
    arccos(1 - eps) ~ sqrt(2 eps) -- the error is AMPLIFIED to about 1e-6 deg
    for identical matrices, which is the same order as the signal being
    measured. The atan2 form reads the angle from the skew-symmetric part
    (whose norm is sin) against the trace (cos), and stays accurate at both
    0 and pi.
    """
    Rs = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    rel = np.einsum("ij,njk->nik", np.asarray(reference, dtype=float).T, Rs)

    # Skew-symmetric part: its vector norm is sin(angle).
    sin_vec = 0.5 * np.stack(
        [
            rel[:, 2, 1] - rel[:, 1, 2],
            rel[:, 0, 2] - rel[:, 2, 0],
            rel[:, 1, 0] - rel[:, 0, 1],
        ],
        axis=1,
    )
    sin_component = np.linalg.norm(sin_vec, axis=1)
    cos_component = (np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0
    return np.degrees(np.arctan2(sin_component, cos_component))


# --------------------------------------------------------------------------
# Distance trueness
# --------------------------------------------------------------------------


@dataclass
class DistanceAccuracy:
    """Measured vs known distances between digitised point pairs."""

    true_mm: np.ndarray
    measured_mm: np.ndarray
    signed_error_mm: np.ndarray  # measured - true
    absolute_error_mm: np.ndarray
    mean_absolute_error_mm: float
    rms_absolute_error_mm: float
    mean_signed_error_mm: float
    std_signed_error_mm: float
    scale_factor: float  # least-squares measured/true through the origin
    scale_error_percent: float

    @property
    def n(self) -> int:
        return len(self.true_mm)

    def bias_note(self) -> str:
        """Plain-language reading of the signed mean -- the important one."""
        mean = self.mean_signed_error_mm
        std = self.std_signed_error_mm
        # With few pairs the signed mean is itself noisy, so only call it a
        # bias when it is large compared with the scatter of the pairs.
        significant = self.n >= 3 and abs(mean) > (std / np.sqrt(self.n))

        if not significant:
            return (
                f"Mean signed error {mean:+.3f} mm is small relative to the scatter "
                f"(std {std:.3f} mm over {self.n} pairs):\n"
                "  no clear systematic bias -- the error looks random."
            )
        direction = "LONG" if mean > 0 else "SHORT"
        return (
            f"Mean signed error {mean:+.3f} mm is a SYSTEMATIC BIAS: measurements\n"
            f"  read consistently {direction} by ~{abs(self.scale_error_percent):.2f}% "
            f"(scale factor {self.scale_factor:.5f}).\n"
            "  This is NOT random noise and averaging will not remove it. The usual\n"
            "  cause is printed geometry that does not match config.yaml -- re-measure\n"
            "  the marker size and the centre-to-centre spacing on your sheets, and\n"
            "  check the ChArUco square size used for the camera calibration."
        )


def distance_accuracy(true_mm: np.ndarray, measured_mm: np.ndarray) -> DistanceAccuracy:
    """Compare measured pair distances against known ones."""
    t = np.asarray(true_mm, dtype=float).ravel()
    m = np.asarray(measured_mm, dtype=float).ravel()
    if t.size != m.size or t.size == 0:
        raise ValueError("true and measured distance arrays must be the same non-zero length")

    signed = m - t
    absolute = np.abs(signed)

    # Least-squares scale through the origin: minimises ||m - s*t||, i.e.
    # s = (t.m)/(t.t). Fitted through the origin because a scale error is
    # multiplicative -- a 1% short ruler is 1% short at every length -- and an
    # intercept would absorb exactly the bias we are trying to expose.
    denom = float(t @ t)
    scale = float(t @ m) / denom if denom > 0 else float("nan")

    return DistanceAccuracy(
        true_mm=t,
        measured_mm=m,
        signed_error_mm=signed,
        absolute_error_mm=absolute,
        mean_absolute_error_mm=float(absolute.mean()),
        rms_absolute_error_mm=float(np.sqrt(np.mean(signed**2))),
        mean_signed_error_mm=float(signed.mean()),
        std_signed_error_mm=float(signed.std(ddof=1)) if signed.size > 1 else 0.0,
        scale_factor=scale,
        scale_error_percent=(scale - 1.0) * 100.0,
    )


# --------------------------------------------------------------------------
# Rigid registration (Kabsch / Procrustes)
# --------------------------------------------------------------------------


@dataclass
class RegistrationResult:
    """Rigid fit of measured points onto known coordinates."""

    transform: SE3  # maps measured -> known
    residuals_mm: np.ndarray  # (N,) per-point distance after the fit
    rms_mm: float
    max_mm: float
    n: int


def kabsch(measured: np.ndarray, known: np.ndarray, *, allow_scale: bool = False) -> RegistrationResult:
    """Best-fit rigid transform taking `measured` onto `known`.

    This is the closest analogue in the project to a real navigation system's
    quoted accuracy: touch a set of landmarks whose true coordinates are
    known, fit a single rigid transform, and report how far off each landmark
    still is. The residual cannot be reduced by choosing a better transform --
    it is the part of the error that is not a rigid misalignment, i.e. the
    genuine distortion of the measurement.

    Solved in closed form: remove both centroids, take the SVD of the
    cross-covariance, and read off the rotation. The determinant guard is not
    optional -- without it a noisy or near-degenerate point set can produce a
    REFLECTION, which fits beautifully and is physically impossible.

    `allow_scale` additionally fits a uniform scale (Umeyama). Off by default,
    because letting the fit absorb a scale error would hide exactly the
    printed-geometry mistake this phase exists to detect; turn it on only to
    *measure* that scale.

    ON PLANAR LANDMARK SETS: a printed grid is coplanar, and a coplanar set
    constrains the out-of-plane direction only weakly -- the same conditioning
    theme as the coplanar marker bodies elsewhere in this project. It is also
    why the determinant guard cannot be *tested* on a planar set: for coplanar
    points a mirror is achievable by a proper rotation (flip the plane over),
    so reflection and rotation are indistinguishable. A registration target
    with some points out of plane gives a meaningfully stronger check.
    """
    P = np.asarray(measured, dtype=float).reshape(-1, 3)
    Q = np.asarray(known, dtype=float).reshape(-1, 3)
    if P.shape != Q.shape:
        raise ValueError(f"point sets differ in shape: {P.shape} vs {Q.shape}")
    if len(P) < 3:
        raise ValueError(
            f"rigid registration needs at least 3 non-collinear points (got {len(P)})"
        )

    p_centroid = P.mean(axis=0)
    q_centroid = Q.mean(axis=0)
    Pc = P - p_centroid
    Qc = Q - q_centroid

    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T

    scale = 1.0
    if allow_scale:
        variance = float((Pc**2).sum())
        if variance > 0:
            scale = float((S @ np.diag(D)).sum() / variance)
        R = R * scale

    t = q_centroid - R @ p_centroid
    transform = SE3(R, t)

    residual_vectors = transform.transform_points(P) - Q
    residuals = np.linalg.norm(residual_vectors, axis=1)

    return RegistrationResult(
        transform=transform,
        residuals_mm=residuals,
        rms_mm=float(np.sqrt(np.mean(residuals**2))),
        max_mm=float(residuals.max()),
        n=len(P),
    )
