"""Metrics 2-4: the touch-based measurements.

    repeatability -- PRECISION. Touch ONE divot many times, from varied
                     angles. Scatter about the centroid.
    distance      -- TRUENESS. Digitise pairs a KNOWN distance apart.
                     Measured vs true, with the signed bias called out.
    registration  -- TRUENESS. Digitise landmarks with KNOWN coordinates,
                     rigidly fit, report the residual.

These are harder -- and more honest -- than static jitter, because they
include everything a real user does: seating the tip in a divot, approaching
from a different angle each time, and their own hand. Expect them to be
several times worse than the jitter figure. If they are not, the jitter run
probably was not really static.
"""

from __future__ import annotations

import cv2
import numpy as np

from accuracy.common import MeasurementRig, draw_measurement_hud, verdict_line
from accuracy.session import ResultsWriter, build_conditions
from accuracy.stats import distance_accuracy, kabsch, point_scatter
from core.config import Config
from core.hud import AMBER, CYAN, GREEN, GREY, RED, WHITE
from core.video import camera


def _capture_loop(cfg: Config, rig: MeasurementRig, title: str, prompt_fn, n_needed: int):
    """Shared 'touch a point and press SPACE' loop.

    `prompt_fn(k)` returns the HUD lines for capture k. Returns the list of
    captured tip positions in the reference frame, or None if aborted.
    """
    captured: list[np.ndarray] = []
    quality: list[tuple[int, int, bool]] = []

    with camera(
        cfg.camera.device_index, cfg.camera.frame_width, cfg.camera.frame_height, cfg.camera.fps
    ) as cap:
        while len(captured) < n_needed:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera read failed; stopping.")
                break

            rel = rig.process(frame)
            display = frame.copy()

            lines = prompt_fn(len(captured))
            lines.append(("SPACE capture | u undo | q abort", GREY))
            draw_measurement_hud(display, rel, lines)

            cv2.imshow(title, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("Aborted; nothing saved.")
                return None, None
            if key == ord(" "):
                if rel.p_tip_ref is None:
                    print("  cannot capture: tool and reference must both be visible")
                else:
                    captured.append(rel.p_tip_ref.copy())
                    quality.append(
                        (rel.tool.n_markers_used, rel.reference.n_markers_used, rel.is_ambiguous)
                    )
                    x, y, z = captured[-1]
                    flag = "  <- AMBIGUOUS pose" if rel.is_ambiguous else ""
                    print(
                        f"  capture {len(captured):2d}: [{x:8.2f} {y:8.2f} {z:8.2f}] mm{flag}"
                    )
            elif key == ord("u") and captured:
                captured.pop()
                quality.pop()
                print(f"  undo -> {len(captured)} captures")

    return captured, quality


# --------------------------------------------------------------------------
# 2. Point repeatability
# --------------------------------------------------------------------------


def run_repeatability(cfg: Config, *, touches: int | None = None, lighting: str | None = None) -> int:
    n = touches or cfg.accuracy.repeatability_touches
    rig = MeasurementRig(cfg)

    print(
        f"""
Point repeatability -- precision under real use
-----------------------------------------------
Touch the SAME physical divot {n} times. Between touches, LIFT the tool away
and come back from a DIFFERENT ANGLE -- rotate it, approach from the other
side, change how far away you hold it.

That variation is the point. Static jitter measures the chain's noise with
nothing moving; this measures what you actually get when a person seats a tip
in a hole, which includes tip seating, approach angle and hand tremor.

SETUP
  * One divot the tip seats into repeatably (countersunk hole, hinge dimple).
  * Reference body fixed and visible throughout. Camera may stay still.
Keys: SPACE capture | u undo | q abort
"""
    )

    def prompt(k):
        return [
            (f"touches: {k} / {n}", GREEN if k >= n else AMBER),
            ("seat the tip in the SAME divot, from a NEW angle each time", WHITE),
        ]

    captured, quality = _capture_loop(cfg, rig, "Accuracy -- point repeatability", prompt, n)
    if not captured or len(captured) < 3:
        print("Need at least 3 touches.")
        return 1

    pts = np.array(captured)
    scatter = point_scatter(pts)
    verdict_text, ok = verdict_line(scatter.rms_3d, thresholds=(0.5, 1.0, 2.0))
    n_amb = sum(1 for q in quality if q[2])

    print(
        "\n".join(
            [
                "",
                "Point repeatability (precision, touch-based)",
                "=" * 62,
                f"  Method: the same physical divot touched {scatter.n} times from varied",
                "  tool angles; scatter of the captured tip positions about their centroid.",
                "",
                f"  centroid       : [{scatter.centroid[0]:8.2f} {scatter.centroid[1]:8.2f} "
                f"{scatter.centroid[2]:8.2f}] mm (reference frame)",
                f"  std per axis   : x {scatter.per_axis_std[0]:7.4f}  "
                f"y {scatter.per_axis_std[1]:7.4f}  z {scatter.per_axis_std[2]:7.4f}  mm",
                f"  3D RMS         : {scatter.rms_3d:7.4f} mm      <- the headline number",
                f"  3D worst       : {scatter.max_3d:7.4f} mm",
                "",
                f"  VERDICT: touch repeatability {scatter.rms_3d:.3f} mm RMS is {verdict_text}.",
                "  This should be noticeably WORSE than static jitter -- it includes tip",
                "  seating, approach angle and your hand. If it is not, check that the",
                "  jitter run was genuinely static.",
            ]
            + (
                [
                    "",
                    f"  WARNING: {n_amb} of {scatter.n} captures had an AMBIGUOUS body pose.",
                ]
                if n_amb
                else []
            )
        )
    )

    writer = ResultsWriter(cfg.accuracy.results_dir, "repeatability")
    writer.write_samples(
        ["touch", "tip_ref_x_mm", "tip_ref_y_mm", "tip_ref_z_mm", "dist_from_centroid_mm",
         "tool_markers", "reference_markers", "ambiguous"],
        [
            [i + 1, *pts[i], scatter.distances[i], quality[i][0], quality[i][1], quality[i][2]]
            for i in range(len(pts))
        ],
    )
    conditions = build_conditions(
        cfg, "repeatability", rig.intrinsics, rig.tool_geometry,
        reference_geometry=rig.reference_geometry, tip=rig.tip, lighting_note=lighting,
    )
    rig.apply_working_distance(conditions)
    writer.write_summary(
        conditions,
        {
            "n_touches": scatter.n,
            "rms_3d_mm": scatter.rms_3d,
            "max_3d_mm": scatter.max_3d,
            "std_xyz_mm": scatter.per_axis_std,
            "centroid_mm": scatter.centroid,
            "n_ambiguous": n_amb,
        },
        verdict=f"touch repeatability {scatter.rms_3d:.3f} mm RMS -- {verdict_text}",
    )
    print()
    print(conditions.format())
    print()
    print(writer.report_paths())
    return 0 if ok else 1


# --------------------------------------------------------------------------
# 3. Point-to-point distance accuracy
# --------------------------------------------------------------------------


def run_distance(
    cfg: Config,
    *,
    true_mm: float | None = None,
    pairs: int | None = None,
    lighting: str | None = None,
) -> int:
    if true_mm is not None:
        n_pairs = pairs or 1
        trues = [float(true_mm)] * n_pairs
    else:
        trues = list(cfg.accuracy.distance_pairs_mm)
        if pairs:
            trues = trues[:pairs]
    if not trues:
        raise ValueError(
            "No true distances configured. Set accuracy.distance_pairs_mm in "
            "config.yaml, or pass --true-mm."
        )

    rig = MeasurementRig(cfg)
    print(
        f"""
Point-to-point distance accuracy -- TRUENESS
--------------------------------------------
Digitise {len(trues)} pair(s) of points whose separation you KNOW:
    {', '.join(f'{t:.1f} mm' for t in trues)}

Use ruler graduations, a caliper set to a value, two drilled holes, or the
printed 100 mm scale bar on your marker sheets. Measure the true distance with
something you trust -- it is the reference this whole number rests on.

For each pair: touch point A, SPACE; touch point B, SPACE.

This is the first metric with GROUND TRUTH, so it is the first that can catch
a systematic error. Watch the mean SIGNED error in the report: a consistent
sign means a scale problem in your printed geometry, and no amount of
averaging will remove it.
Keys: SPACE capture | u undo | q abort
"""
    )

    n_needed = 2 * len(trues)

    def prompt(k):
        pair_idx = k // 2
        which = "A" if k % 2 == 0 else "B"
        return [
            (f"pair {pair_idx + 1} / {len(trues)}   true separation "
             f"{trues[min(pair_idx, len(trues) - 1)]:.1f} mm", CYAN),
            (f"touch point {which} and press SPACE", WHITE),
            (f"captured {k} / {n_needed}", GREY),
        ]

    captured, quality = _capture_loop(
        cfg, rig, "Accuracy -- distance to known points", prompt, n_needed
    )
    if not captured or len(captured) < 2:
        print("Need at least one complete pair.")
        return 1

    n_complete = len(captured) // 2
    pts = np.array(captured[: 2 * n_complete])
    measured = np.array(
        [float(np.linalg.norm(pts[2 * i + 1] - pts[2 * i])) for i in range(n_complete)]
    )
    true_arr = np.array(trues[:n_complete])

    da = distance_accuracy(true_arr, measured)
    verdict_text, ok = verdict_line(da.rms_absolute_error_mm, thresholds=(0.5, 1.0, 2.0))

    rows = [
        f"    {i + 1:2d}   {da.true_mm[i]:8.2f}   {da.measured_mm[i]:8.2f}   "
        f"{da.signed_error_mm[i]:+8.3f}   {da.absolute_error_mm[i]:8.3f}"
        for i in range(da.n)
    ]
    print(
        "\n".join(
            [
                "",
                "Point-to-point distance accuracy (trueness, vs known distances)",
                "=" * 62,
                f"  Method: {da.n} pair(s) of points digitised with the calibrated tip;",
                "  measured separation compared against an independently known value.",
                "",
                "    pair    true(mm)   meas(mm)    signed     |error|",
                *rows,
                "",
                f"  mean |error|      : {da.mean_absolute_error_mm:7.3f} mm",
                f"  RMS  |error|      : {da.rms_absolute_error_mm:7.3f} mm   <- headline",
                f"  mean SIGNED error : {da.mean_signed_error_mm:+7.3f} mm  "
                f"(std {da.std_signed_error_mm:.3f})",
                f"  fitted scale      : {da.scale_factor:.5f}  "
                f"({da.scale_error_percent:+.3f}%)",
                "",
                "  " + da.bias_note(),
                "",
                f"  VERDICT: distance error {da.rms_absolute_error_mm:.3f} mm RMS is "
                f"{verdict_text}.",
            ]
        )
    )

    writer = ResultsWriter(cfg.accuracy.results_dir, "distance")
    writer.write_samples(
        ["pair", "true_mm", "measured_mm", "signed_error_mm", "abs_error_mm",
         "a_x_mm", "a_y_mm", "a_z_mm", "b_x_mm", "b_y_mm", "b_z_mm"],
        [
            [i + 1, da.true_mm[i], da.measured_mm[i], da.signed_error_mm[i],
             da.absolute_error_mm[i], *pts[2 * i], *pts[2 * i + 1]]
            for i in range(da.n)
        ],
    )
    conditions = build_conditions(
        cfg, "distance", rig.intrinsics, rig.tool_geometry,
        reference_geometry=rig.reference_geometry, tip=rig.tip, lighting_note=lighting,
    )
    rig.apply_working_distance(conditions)
    writer.write_summary(
        conditions,
        {
            "n_pairs": da.n,
            "true_mm": da.true_mm,
            "measured_mm": da.measured_mm,
            "mean_absolute_error_mm": da.mean_absolute_error_mm,
            "rms_absolute_error_mm": da.rms_absolute_error_mm,
            "mean_signed_error_mm": da.mean_signed_error_mm,
            "std_signed_error_mm": da.std_signed_error_mm,
            "scale_factor": da.scale_factor,
            "scale_error_percent": da.scale_error_percent,
        },
        verdict=f"distance error {da.rms_absolute_error_mm:.3f} mm RMS -- {verdict_text}",
    )
    print()
    print(conditions.format())
    print()
    print(writer.report_paths())
    return 0 if ok else 1


# --------------------------------------------------------------------------
# 4. Multi-point registration residual
# --------------------------------------------------------------------------


def run_registration(cfg: Config, *, lighting: str | None = None) -> int:
    known = np.asarray(cfg.accuracy.registration_points_mm, dtype=float).reshape(-1, 3)
    if len(known) < 3:
        raise ValueError(
            "accuracy.registration_points_mm needs at least 3 points "
            f"(got {len(known)}). Set them in config.yaml."
        )

    rig = MeasurementRig(cfg)
    coplanar = np.linalg.matrix_rank(known - known.mean(axis=0), tol=1e-6) < 3

    listing = "\n".join(
        f"    {i + 1:2d}. [{p[0]:7.1f} {p[1]:7.1f} {p[2]:7.1f}] mm" for i, p in enumerate(known)
    )
    print(
        f"""
Multi-point registration residual -- TRUENESS
---------------------------------------------
The closest analogue here to a real navigation system's accuracy spec.

Digitise {len(known)} landmarks whose true coordinates are known, IN THIS ORDER:
{listing}

A single rigid transform is then fitted from your measured points onto the
known ones, and the residual is what remains. That residual cannot be reduced
by choosing a better transform -- it is the part of the error that is NOT a
rigid misalignment, i.e. the genuine distortion of the measurement.

SETUP
  * A target whose landmark coordinates you trust: a printed grid, graph paper
    with marked intersections, or holes drilled at measured spacing.
  * Keep the reference body fixed and visible throughout.
{"  * NOTE: your landmarks are COPLANAR, which constrains the out-of-plane" if coplanar else ""}
{"    direction only weakly. Adding points at a known height would make this" if coplanar else ""}
{"    a meaningfully stronger check." if coplanar else ""}
Keys: SPACE capture | u undo | q abort
"""
    )

    def prompt(k):
        p = known[min(k, len(known) - 1)]
        return [
            (f"landmark {k + 1} / {len(known)}", CYAN),
            (f"touch [{p[0]:.1f} {p[1]:.1f} {p[2]:.1f}] mm and press SPACE", WHITE),
        ]

    captured, quality = _capture_loop(
        cfg, rig, "Accuracy -- registration", prompt, len(known)
    )
    if not captured or len(captured) < 3:
        print("Need at least 3 landmarks.")
        return 1

    n = len(captured)
    measured = np.array(captured)
    reg = kabsch(measured, known[:n])
    reg_scaled = kabsch(measured, known[:n], allow_scale=True)
    verdict_text, ok = verdict_line(reg.rms_mm, thresholds=(0.5, 1.0, 2.0))

    rows = [
        f"    {i + 1:2d}   [{known[i][0]:7.1f} {known[i][1]:7.1f} {known[i][2]:7.1f}]   "
        f"{reg.residuals_mm[i]:8.3f}"
        for i in range(n)
    ]
    scale_pct = (np.linalg.norm(reg_scaled.transform.R[:, 0]) - 1.0) * 100.0

    print(
        "\n".join(
            [
                "",
                "Registration residual (trueness, rigid fit to known landmarks)",
                "=" * 62,
                f"  Method: {n} landmarks digitised with the calibrated tip; a single rigid",
                "  transform (Kabsch) fitted measured -> known; residual after the fit.",
                "",
                "    pt        known (mm)              residual(mm)",
                *rows,
                "",
                f"  registration RMS  : {reg.rms_mm:7.3f} mm   <- headline",
                f"  worst landmark    : {reg.max_mm:7.3f} mm",
                "",
                f"  if a uniform scale is also fitted: RMS {reg_scaled.rms_mm:.3f} mm "
                f"at scale {1 + scale_pct / 100:.5f} ({scale_pct:+.3f}%)",
                "  A large drop when scale is allowed means your printed geometry is",
                "  mis-scaled rather than distorted -- re-measure the sheets.",
                "",
                f"  VERDICT: registration residual {reg.rms_mm:.3f} mm RMS is {verdict_text}.",
            ]
            + (
                ["", "  NOTE: coplanar landmarks -- out-of-plane error is weakly constrained."]
                if coplanar
                else []
            )
        )
    )

    writer = ResultsWriter(cfg.accuracy.results_dir, "registration")
    writer.write_samples(
        ["landmark", "known_x_mm", "known_y_mm", "known_z_mm",
         "measured_x_mm", "measured_y_mm", "measured_z_mm", "residual_mm"],
        [[i + 1, *known[i], *measured[i], reg.residuals_mm[i]] for i in range(n)],
    )
    conditions = build_conditions(
        cfg, "registration", rig.intrinsics, rig.tool_geometry,
        reference_geometry=rig.reference_geometry, tip=rig.tip, lighting_note=lighting,
        notes="coplanar landmark set" if coplanar else "non-coplanar landmark set",
    )
    rig.apply_working_distance(conditions)
    writer.write_summary(
        conditions,
        {
            "n_landmarks": n,
            "rms_mm": reg.rms_mm,
            "max_mm": reg.max_mm,
            "rms_with_scale_mm": reg_scaled.rms_mm,
            "fitted_scale_percent": scale_pct,
            "coplanar_landmarks": bool(coplanar),
        },
        verdict=f"registration residual {reg.rms_mm:.3f} mm RMS -- {verdict_text}",
    )
    print()
    print(conditions.format())
    print()
    print(writer.report_paths())
    return 0 if ok else 1
