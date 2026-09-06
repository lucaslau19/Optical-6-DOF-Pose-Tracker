"""Metric 1: static jitter -- precision of a single held pose.

METHOD
------
Tool and reference both completely static. Collect N consecutive frames and
record, for each, the tip position in the reference frame and the tool's
orientation relative to the reference. Report how much those wandered.

WHAT IT MEASURES
----------------
Precision only, and of the easiest possible case. Nothing moves, so no ground
truth is needed and none is used: this is pure repeatability of the
measurement chain -- detector corner noise, the two PnP fits, and the
reference transform composing them.

WHAT IT DOES *NOT* MEASURE
--------------------------
Anything about trueness. A system with a 2 mm systematic offset would score
perfectly here, because every frame would be wrong by the same 2 mm. It also
excludes everything that varies in real use: approach angle, re-seating the
tip, the operator's hand. Static jitter is therefore the most flattering
number this project produces, and it is reported first precisely so that the
harder numbers that follow can be compared against it.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from accuracy.common import MeasurementRig, draw_measurement_hud, verdict_line
from accuracy.session import ResultsWriter, build_conditions
from accuracy.stats import describe, mean_rotation, point_scatter, rotation_deviations_deg
from core.config import Config
from core.hud import AMBER, GREEN, GREY, WHITE
from core.video import camera
from navigation.relative import RelativePose


def run_jitter(cfg: Config, *, frames: int | None = None, lighting: str | None = None) -> int:
    n_target = frames or cfg.accuracy.jitter_frames
    rig = MeasurementRig(cfg)

    print(_instructions(rig, n_target))

    tip_ref: list[np.ndarray] = []
    tip_cam: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    n_ambiguous = 0

    with camera(
        cfg.camera.device_index, cfg.camera.frame_width, cfg.camera.frame_height, cfg.camera.fps
    ) as cap:
        started = False
        while len(tip_ref) < n_target:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera read failed; stopping.")
                break

            rel = rig.process(frame)
            display = frame.copy()

            if started and rel.is_valid and rel.p_tip_ref is not None:
                tip_ref.append(rel.p_tip_ref.copy())
                tip_cam.append(rel.p_tip_cam.copy())
                rotations.append(rel.T_ref_tool.R.copy())
                if rel.is_ambiguous:
                    n_ambiguous += 1

            progress = len(tip_ref) / n_target
            lines = [
                (
                    f"frames: {len(tip_ref)} / {n_target}   ({progress * 100:4.0f}%)",
                    GREEN if started else AMBER,
                ),
                (
                    "SPACE to start" if not started else "collecting -- do not touch anything",
                    WHITE if not started else GREEN,
                ),
                ("q abort", GREY),
            ]
            if started and len(tip_ref) > 10:
                live = point_scatter(np.array(tip_ref))
                lines.insert(1, (f"live RMS: {live.rms_3d:6.3f} mm", WHITE))
            draw_measurement_hud(display, rel, lines)

            cv2.imshow("Accuracy -- static jitter", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("Aborted; nothing saved.")
                return 1
            if key == ord(" "):
                started = True
                print(f"Collecting {n_target} frames -- hold everything still ...")

    if len(tip_ref) < 10:
        print(f"Only {len(tip_ref)} frames collected; need at least 10.")
        return 1

    return _report(cfg, rig, np.array(tip_ref), np.array(tip_cam), np.array(rotations),
                   n_ambiguous, lighting)


def _report(cfg, rig, tip_ref, tip_cam, rotations, n_ambiguous, lighting) -> int:
    scatter_ref = point_scatter(tip_ref)
    scatter_cam = point_scatter(tip_cam)

    mean_R = mean_rotation(rotations)
    angles = rotation_deviations_deg(rotations, mean_R)
    angle_stats = describe(angles)

    dist_stats = describe(scatter_ref.distances)
    verdict_text, ok = verdict_line(scatter_ref.rms_3d, thresholds=(0.15, 0.4, 1.0))

    lines = [
        "",
        "Static jitter (precision, no ground truth)",
        "=" * 62,
        "  Method: tool and reference held completely static; "
        f"{scatter_ref.n} consecutive",
        "  frames; scatter of the tip position in the REFERENCE frame.",
        "",
        f"  tip position, reference frame  (n = {scatter_ref.n})",
        f"    std per axis   : x {scatter_ref.per_axis_std[0]:7.4f}  "
        f"y {scatter_ref.per_axis_std[1]:7.4f}  z {scatter_ref.per_axis_std[2]:7.4f}  mm",
        f"    peak-to-peak   : x {scatter_ref.per_axis_peak_to_peak[0]:7.4f}  "
        f"y {scatter_ref.per_axis_peak_to_peak[1]:7.4f}  "
        f"z {scatter_ref.per_axis_peak_to_peak[2]:7.4f}  mm",
        f"    3D RMS         : {scatter_ref.rms_3d:7.4f} mm      <- the headline number",
        f"    3D worst       : {scatter_ref.max_3d:7.4f} mm",
        "",
        "  tool orientation, relative to the reference",
        f"    std            : {angle_stats.std:7.4f} deg",
        f"    peak-to-peak   : {angle_stats.peak_to_peak:7.4f} deg",
        f"    worst          : {angle_stats.max:7.4f} deg from the mean orientation",
        "",
        "  for comparison, the same tip in the CAMERA frame",
        f"    3D RMS         : {scatter_cam.rms_3d:7.4f} mm",
    ]

    # The reference frame composes two rigid-body fits instead of one, so it
    # is expected to be noisier. Saying so pre-empts the reasonable question
    # "why is the useful number worse than the raw one?".
    if scatter_cam.rms_3d > 0:
        ratio = scatter_ref.rms_3d / scatter_cam.rms_3d
        lines.append(
            f"    -> the reference-frame figure is {ratio:.2f}x the camera-frame one; "
            "above 1x is"
        )
        lines.append(
            "       expected, since the relative pose composes TWO body fits, not one."
        )

    if n_ambiguous:
        lines.append("")
        lines.append(
            f"  WARNING: {n_ambiguous} of {scatter_ref.n} frames had an AMBIGUOUS body pose."
        )
        lines.append("  Coplanar sheets near fronto-parallel; move closer or more oblique.")

    lines.append("")
    lines.append(f"  VERDICT: static 3D jitter {scatter_ref.rms_3d:.3f} mm RMS is {verdict_text}.")
    lines.append(
        "  Note this is PRECISION under the easiest possible conditions -- nothing\n"
        "  moved. It says nothing about trueness, and excludes the approach-angle\n"
        "  and re-seating variation that `accuracy repeatability` captures."
    )
    report = "\n".join(lines)
    print(report)

    writer = ResultsWriter(cfg.accuracy.results_dir, "jitter")
    writer.write_samples(
        ["frame", "tip_ref_x_mm", "tip_ref_y_mm", "tip_ref_z_mm",
         "dist_from_centroid_mm", "tip_cam_x_mm", "tip_cam_y_mm", "tip_cam_z_mm",
         "orientation_dev_deg"],
        [
            [i, *tip_ref[i], scatter_ref.distances[i], *tip_cam[i], angles[i]]
            for i in range(len(tip_ref))
        ],
    )
    conditions = build_conditions(
        cfg, "jitter", rig.intrinsics, rig.tool_geometry,
        reference_geometry=rig.reference_geometry, tip=rig.tip, lighting_note=lighting,
    )
    rig.apply_working_distance(conditions)
    writer.write_summary(
        conditions,
        {
            "n_frames": scatter_ref.n,
            "tip_ref_rms_3d_mm": scatter_ref.rms_3d,
            "tip_ref_max_3d_mm": scatter_ref.max_3d,
            "tip_ref_std_xyz_mm": scatter_ref.per_axis_std,
            "tip_ref_peak_to_peak_xyz_mm": scatter_ref.per_axis_peak_to_peak,
            "tip_cam_rms_3d_mm": scatter_cam.rms_3d,
            "orientation_std_deg": angle_stats.std,
            "orientation_peak_to_peak_deg": angle_stats.peak_to_peak,
            "distance_from_centroid_mean_mm": dist_stats.mean,
            "n_ambiguous_frames": n_ambiguous,
        },
        verdict=f"static 3D jitter {scatter_ref.rms_3d:.3f} mm RMS -- {verdict_text}",
    )
    print()
    print(conditions.format())
    print()
    print(writer.report_paths())
    return 0 if ok else 1


def _instructions(rig, n_target: int) -> str:
    return f"""
Static jitter -- precision of a held pose
-----------------------------------------
Measures how much the reported tip position wanders while NOTHING moves. This
is the measurement chain's own noise floor.

SETUP
  * Tool and reference both clamped, propped or taped down. Neither may move.
  * Camera on a tripod or otherwise fixed. Do not touch the bench.
  * Frame both bodies comfortably, at a typical working distance.
  * Then take your hands off everything and press SPACE.

Collecting {n_target} frames (~{n_target / 30:.0f} s at 30 fps).
Keys: SPACE start | q abort
"""
