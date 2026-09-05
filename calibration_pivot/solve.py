"""Pivot calibration: solve the tool tip offset by least squares.

THE PROBLEM
-----------
The tool's tip is a physical point on the instrument. We need its position in
the TOOL frame, `p_tip`, because that is the number that turns a tracked body
pose into "where the sharp end actually is" -- the only thing a surgeon cares
about. It cannot be measured with a ruler on a marker sheet: the tip is not
on the markers, and the tool origin is an abstraction at the cluster centroid.

So we measure it kinematically. Plant the tip in a fixed divot and pivot the
tool about it. The tip stays at one point in space; the body swings around it.

THE MATHS
---------
For frame i, the tracker gives T_cam_tool = (R_i, t_i). The tip, expressed in
camera coordinates, is R_i @ p_tip + t_i. Because the tip is planted, that is
the SAME camera-frame point every frame -- call it `p_pivot`:

        R_i @ p_tip + t_i = p_pivot                    for all i

Two unknown 3-vectors, so six unknowns in total. Rearranged so the unknowns
are on one side:

        R_i @ p_tip - p_pivot = -t_i

which in block form is linear in x = [p_tip; p_pivot]:

        [ R_i | -I ] x = -t_i                          (3 equations per frame)

Stack every frame:

        [ R_0 | -I ]        [ -t_0 ]
        [ R_1 | -I ]  x  =  [ -t_1 ]        A x = b,  A is (3N x 6)
        [  ...      ]       [  ... ]

and solve by linear least squares. No iteration, no initial guess, no local
minima -- the geometry made it linear, which is the elegant part.

WHY IT NEEDS WIDE ANGLES
------------------------
If every R_i were identical, A would have rank 3, not 6: sliding the tip
along a fixed direction while moving the pivot the same way fits equally
well, so p_tip is unrecoverable. Orientation variety is literally what makes
the system solvable, and *how much* variety sets the conditioning. This is
the number one failure mode of pivot calibration, which is why the capture
screen measures the cone angle live instead of just counting frames.

THE RESIDUAL IS THE QUALITY METRIC
----------------------------------
    r_i = R_i @ p_tip + t_i - p_pivot

is how far the reconstructed tip wandered on frame i, in millimetres. Its RMS
is a direct, physical accuracy figure -- unlike Phase 1's reprojection RMS,
which is in pixels and only says the model fits its own data. It absorbs
everything: tracking noise, a tip that slipped in the divot, a flexing tool,
and a wrong marker geometry.

ASSUMPTION: THE CAMERA MUST NOT MOVE
------------------------------------
`p_pivot` is solved in the CAMERA frame, so it is only constant if the camera
is stationary for the whole capture. Bump the tripod and the maths is
answering a different question. Phase 4's patient reference frame is exactly
what lifts this restriction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.transforms import SE3

# Azimuth bins used to judge how far around the cone the operator actually
# went. Precessing through only one side of the cone leaves the fit poorly
# constrained even if the tilt angle looks generous.
AZIMUTH_SECTORS = 12


@dataclass
class OrientationSpread:
    """How much orientation variety a set of captured poses actually covers."""

    cone_half_angle_deg: float  # widest tilt of the tool axis from its mean
    azimuth_coverage: float  # fraction of azimuth sectors visited, 0..1
    max_pairwise_angle_deg: float  # widest angle between any two tool axes
    sectors_filled: np.ndarray  # (AZIMUTH_SECTORS,) bool, for the HUD wheel
    tilts_deg: np.ndarray  # per-frame tilt from the mean direction
    azimuths_deg: np.ndarray  # per-frame azimuth about the mean direction


def orientation_spread(poses: list[SE3]) -> OrientationSpread:
    """Measure the cone the tool axis swept during capture.

    Uses the tool's own +Z axis (the sheet normal) expressed in camera
    coordinates. Precessing about a planted tip sweeps that axis around a
    cone, so tilt-from-mean and azimuth-about-mean are the natural polar
    coordinates for "did they actually move it around enough".
    """
    if not poses:
        empty = np.zeros(0)
        return OrientationSpread(0.0, 0.0, 0.0, np.zeros(AZIMUTH_SECTORS, bool), empty, empty)

    # Tool +Z in camera coordinates is the third column of R.
    axes = np.array([p.R[:, 2] for p in poses], dtype=float)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)

    mean_dir = axes.mean(axis=0)
    norm = np.linalg.norm(mean_dir)
    if norm < 1e-9:
        # Axes cancelled out entirely (spread over more than a hemisphere).
        # That is wonderful conditioning, not a failure -- fall back to the
        # first axis as the reference direction so the polar maths still works.
        mean_dir = axes[0]
    else:
        mean_dir = mean_dir / norm

    dots = np.clip(axes @ mean_dir, -1.0, 1.0)
    tilts = np.degrees(np.arccos(dots))

    # Orthonormal basis spanning the plane perpendicular to mean_dir, so
    # azimuth is well defined. Pick whichever world axis is least parallel to
    # mean_dir to avoid a degenerate cross product.
    helper = np.array([1.0, 0.0, 0.0])
    if abs(mean_dir @ helper) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(mean_dir, helper)
    u /= np.linalg.norm(u)
    v = np.cross(mean_dir, u)

    azimuths = np.degrees(np.arctan2(axes @ v, axes @ u))

    sectors = np.zeros(AZIMUTH_SECTORS, dtype=bool)
    # Only frames with a meaningful tilt have a meaningful azimuth: the
    # azimuth of an axis sitting near the cone's centre is pure noise, and
    # counting those makes a narrow sweep look like full coverage. The
    # threshold is relative to the cone actually achieved, so it adapts
    # instead of assuming a scale.
    meaningful = tilts > max(3.0, 0.35 * float(tilts.max()))
    if np.any(meaningful):
        idx = ((azimuths[meaningful] + 180.0) / 360.0 * AZIMUTH_SECTORS).astype(int)
        sectors[np.clip(idx, 0, AZIMUTH_SECTORS - 1)] = True

    # Widest angle between any two tool axes. Vectorised as a dot-product
    # matrix rather than pairwise SE3 comparisons, which would be O(N^2)
    # scipy calls for no extra information.
    gram = np.clip(axes @ axes.T, -1.0, 1.0)
    max_pairwise = float(np.degrees(np.arccos(gram.min())))

    return OrientationSpread(
        cone_half_angle_deg=float(tilts.max()),
        azimuth_coverage=float(sectors.mean()),
        max_pairwise_angle_deg=max_pairwise,
        sectors_filled=sectors,
        tilts_deg=tilts,
        azimuths_deg=azimuths,
    )


@dataclass
class PivotResult:
    """Solved tip offset plus everything needed to judge whether to trust it."""

    p_tip: np.ndarray  # (3,) tip position in the TOOL frame, mm
    p_pivot: np.ndarray  # (3,) divot position in the CAMERA frame, mm
    residual_rms_mm: float
    residual_max_mm: float
    per_frame_residual_mm: np.ndarray
    n_frames: int
    condition_number: float
    spread: OrientationSpread

    @property
    def tip_distance_mm(self) -> float:
        """How far the tip is from the tool origin -- a quick sanity check.

        Compare it against the physical instrument with a ruler. If the tool
        origin is the centroid of the marker sheet and the tip is 120 mm away
        on the real object, this number had better be about 120.
        """
        return float(np.linalg.norm(self.p_tip))

    def conditioning_verdict(self) -> tuple[str, bool]:
        """Judge the orientation variety, independently of the residual.

        THIS IS A SEPARATE CHECK FROM THE RESIDUAL, and both must pass.

        The residual is essentially blind to conditioning. Measured on
        synthetic captures at a fixed noise level, it sits at ~0.66 mm whether
        the tool swept a 60 degree cone or a 2 degree one -- because it only
        asks "do these frames agree with each other", and a set of nearly
        identical frames agrees with itself beautifully. Over that same sweep
        the actual tip error grew ~20x as the cone shrank. So a capture can
        report an *excellent* residual while the tip offset is barely
        determined, and the residual alone would pass it.

        The gate is therefore the CONDITION NUMBER of A, which measures the
        thing we actually care about: how much the solution moves for a given
        perturbation of the data. It captures both failure modes at once -- a
        narrow cone and a cone swept on one side only -- where the cone angle
        and azimuth coverage each miss one. Those two are kept as the
        human-readable explanation of *why*, and to tell the operator what to
        do differently.

        Thresholds come from the synthetic sweep in the docstring above:
        a 30-60 degree full cone lands at cond 2-5, a 15-25 degree cone or a
        half-swept azimuth at 5-10, and anything degenerate climbs past 10.
        """
        cond = self.condition_number
        s = self.spread

        if cond < 5.0:
            return "well conditioned", True
        if cond < 10.0:
            return "acceptable, wider angles would be better", True

        # Ill-conditioned: say which of the two mistakes was made.
        if s.cone_half_angle_deg < 20.0:
            reason = (
                f"the tool only tilted {s.cone_half_angle_deg:.0f} deg from its mean "
                "direction -- tip the tool over much further"
            )
        else:
            reason = (
                f"only {s.azimuth_coverage * 100:.0f}% of the azimuth was swept -- "
                "precess all the way around the cone, not just one side"
            )
        return f"ILL-CONDITIONED (cond {cond:.0f}): {reason}", False

    def residual_verdict(self) -> tuple[str, bool]:
        rms = self.residual_rms_mm
        if rms < 1.0:
            return "excellent", True
        if rms < 2.0:
            return "good", True
        if rms < 3.0:
            return "marginal -- usable, but a firmer divot would help", True
        return "POOR -- do not trust this tip; recapture", False

    def report(self) -> str:
        s = self.spread
        cond_text, cond_ok = self.conditioning_verdict()
        res_text, res_ok = self.residual_verdict()

        lines = [
            "Pivot (tip) calibration result",
            "=" * 62,
            f"  tip in TOOL frame : x {self.p_tip[0]:8.2f}   y {self.p_tip[1]:8.2f}   "
            f"z {self.p_tip[2]:8.2f}   mm",
            f"  |tip| from origin : {self.tip_distance_mm:.2f} mm  "
            "(check this against the real tool with a ruler)",
            f"  divot in CAM frame: x {self.p_pivot[0]:8.2f}   y {self.p_pivot[1]:8.2f}   "
            f"z {self.p_pivot[2]:8.2f}   mm",
            "",
            f"  frames used       : {self.n_frames}",
            f"  residual RMS      : {self.residual_rms_mm:.3f} mm",
            f"  residual max      : {self.residual_max_mm:.3f} mm",
            "",
            "  orientation spread (what makes the system solvable at all)",
            f"    cone half-angle : {s.cone_half_angle_deg:.1f} deg",
            f"    azimuth covered : {s.azimuth_coverage * 100:.0f}% "
            f"({int(s.sectors_filled.sum())}/{AZIMUTH_SECTORS} sectors)",
            f"    widest angle    : {s.max_pairwise_angle_deg:.1f} deg between any two poses",
            f"    condition number: {self.condition_number:.1f}",
            f"    -> {cond_text}",
            "",
            f"  VERDICT: residual RMS {self.residual_rms_mm:.3f} mm is {res_text}.",
        ]

        if not cond_ok:
            lines.append(
                "  WARNING: the residual above is NOT trustworthy on its own -- a\n"
                "  capture with little orientation variety fits itself well while\n"
                "  leaving the tip offset under-determined. Recapture with a wider,\n"
                "  fuller cone."
            )
        elif not res_ok:
            lines.append(
                "  The geometry was varied enough, so a high residual points at the\n"
                "  physical setup: a slipping tip, a divot that is not a point, a\n"
                "  flexing tool, or wrong tool.markers millimetres in config.yaml."
            )
        return "\n".join(lines)


def solve_pivot(poses: list[SE3]) -> PivotResult:
    """Least-squares solve for [p_tip; p_pivot] over all captured poses."""
    n = len(poses)
    if n < 3:
        raise ValueError(
            f"Pivot calibration needs at least 3 poses (got {n}). "
            "In practice aim for 50-150 spread over a wide cone."
        )

    # Build A (3N x 6) and b (3N,). Written as an explicit loop over frames
    # because the block structure is the whole point and should be readable.
    A = np.zeros((3 * n, 6), dtype=float)
    b = np.zeros(3 * n, dtype=float)
    for i, pose in enumerate(poses):
        rows = slice(3 * i, 3 * i + 3)
        A[rows, 0:3] = pose.R
        A[rows, 3:6] = -np.eye(3)
        b[rows] = -pose.t

    # rcond=None uses the machine-precision default cutoff, which is what we
    # want: a genuinely rank-deficient A (no orientation variety) should be
    # caught and reported by the conditioning check, not silently regularised
    # away here into a confident-looking wrong answer.
    x, _residuals, _rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

    p_tip = x[:3]
    p_pivot = x[3:]

    # Per-frame residual: how far this frame's reconstructed tip sits from the
    # common pivot point, in mm.
    per_frame = np.array(
        [float(np.linalg.norm(pose.R @ p_tip + pose.t - p_pivot)) for pose in poses]
    )

    condition_number = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float("inf")

    return PivotResult(
        p_tip=p_tip,
        p_pivot=p_pivot,
        residual_rms_mm=float(np.sqrt(np.mean(per_frame**2))),
        residual_max_mm=float(per_frame.max()),
        per_frame_residual_mm=per_frame,
        n_frames=n,
        condition_number=condition_number,
        spread=orientation_spread(poses),
    )
