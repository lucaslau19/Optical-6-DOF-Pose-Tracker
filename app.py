#!/usr/bin/env python
"""Optical 6-DOF pose tracker -- command line entry point.

    python app.py check                # verify install, camera, intrinsics
    python app.py generate-markers     # print-ready ChArUco board + markers
    python app.py calibrate            # camera intrinsics from a ChArUco board
    python app.py track                # live single-marker 6-DOF pose
    python app.py generate-tool-sheet  # print-ready rigid-tool marker sheet
    python app.py track-tool           # live rigid-body tool pose (occlusion tolerant)
    python app.py pivot                # pivot (tip) calibration -> tip offset
    python app.py navigate             # reference-relative pose + target guidance
    python app.py accuracy jitter      # precision: static jitter
    python app.py accuracy repeatability  # precision: repeated touches of one point
    python app.py accuracy distance    # trueness: measured vs known separations
    python app.py accuracy registration   # trueness: rigid fit to known landmarks
    python app.py accuracy headline    # aggregate the latest results

Run `python app.py <command> --help` for per-command options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `python app.py` work from any directory without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import Config, load_config, resolve_path  # noqa: E402


def _load(args: argparse.Namespace) -> Config:
    return load_config(args.config)


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    """Verify the environment before anything else can confuse the issue."""
    import cv2

    from core.cv_compat import opencv_version, require_modern_aruco
    from core.intrinsics import CameraIntrinsics

    print("Environment")
    print("=" * 60)
    print(f"  python        : {sys.version.split()[0]}")
    print(f"  opencv        : {cv2.__version__}  {opencv_version()}")

    try:
        require_modern_aruco()
        print("  cv2.aruco     : OK (ArucoDetector / CharucoDetector available)")
    except RuntimeError as exc:
        print(f"  cv2.aruco     : FAIL\n{exc}")
        return 1

    cfg = _load(args)
    print(f"  config        : {cfg.source_path}")
    print(
        f"  charuco board : {cfg.charuco.squares_x}x{cfg.charuco.squares_y} squares, "
        f"{cfg.charuco.square_length_mm} mm  "
        f"({cfg.charuco.board_width_mm:.0f} x {cfg.charuco.board_height_mm:.0f} mm)"
    )
    print(f"  marker size   : {cfg.markers.marker_length_mm} mm")

    intr_path = cfg.camera.intrinsics_file
    if intr_path.is_file():
        intrinsics = CameraIntrinsics.load(intr_path)
        print(f"  intrinsics    : {intr_path}")
        print(intrinsics.summary())
        if intrinsics.image_size != (cfg.camera.frame_width, cfg.camera.frame_height):
            print(
                "  WARNING: intrinsics resolution does not match camera.frame_* "
                "in config.yaml -- `track` will refuse to run."
            )
    else:
        print(f"  intrinsics    : NOT FOUND at {intr_path} -- run `calibrate` first")

    if args.list_cameras:
        from core.video import probe_cameras

        print("\nCamera device survey")
        print("=" * 60)
        print("  (probing indices 0-3; this takes a few seconds per device)")
        results = probe_cameras(range(4), cfg.camera.frame_width, cfg.camera.frame_height)
        working = [r for r in results if r["status"] == "OK"]
        if not results:
            print("  no camera-like devices found at indices 0-3")
        for r in results:
            if r["status"] == "OK":
                size = f"{r['size'][0]}x{r['size'][1]}"
                print(f"  index {r['index']}  {r['backend']:10s}  OK, delivering {size}")
            else:
                print(f"  index {r['index']}  {r['backend']:10s}  {r['status']}")
        if working:
            print(
                f"\n  Set camera.device_index to one of: "
                f"{sorted({r['index'] for r in working})}"
            )
            print(
                "  Note: virtual cameras (NVIDIA Broadcast, OBS, Teams effects) also\n"
                "  show up here. If a pose looks wrong, make sure you picked the real one."
            )
        else:
            print("\n  No device delivered a frame. Close other apps using the camera.")
            return 1

    if args.camera:
        from core.video import open_camera

        try:
            # strict=False: report what the camera actually does rather than
            # refusing, since diagnosing a mismatch is the point of `check`.
            cap = open_camera(
                cfg.camera.device_index,
                cfg.camera.frame_width,
                cfg.camera.frame_height,
                cfg.camera.fps,
                strict=False,
            )
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                print(f"\n  camera {cfg.camera.device_index}      : OK, delivering {w}x{h}")
                if (w, h) != (cfg.camera.frame_width, cfg.camera.frame_height):
                    print(
                        f"  WARNING: config.yaml asks for {cfg.camera.frame_width}x"
                        f"{cfg.camera.frame_height} but this camera gives {w}x{h}.\n"
                        "  `calibrate` and `track` will refuse to run until they agree --\n"
                        "  set camera.frame_width/frame_height to the size above."
                    )
            else:
                print(f"\n  camera {cfg.camera.device_index}      : opened but returned no frame")
                return 1
        except RuntimeError as exc:
            print(f"\n  camera        : FAIL --\n{exc}")
            return 1

    return 0


# --------------------------------------------------------------------------
# generate-markers
# --------------------------------------------------------------------------


def cmd_generate_markers(args: argparse.Namespace) -> int:
    from markers.generate import generate_charuco_board, generate_single_markers

    cfg = _load(args)
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.markers_only:
        board_path = out_dir / (
            f"charuco_{cfg.charuco.squares_x}x{cfg.charuco.squares_y}"
            f"_{cfg.charuco.square_length_mm:g}mm_{cfg.aruco.dictionary}.png"
        )
        info = generate_charuco_board(cfg, board_path)
        print("ChArUco calibration board")
        print("-" * 60)
        print(f"  file        : {board_path}")
        print(f"  image       : {info['pixels'][0]} x {info['pixels'][1]} px @ {info['dpi']} dpi")
        print(
            f"  printed size: {info['board_width_mm']:.1f} x {info['board_height_mm']:.1f} mm "
            "(pattern area, excluding the white margin)"
        )
        print(f"  square      : {info['square_mm']:.2f} mm")
        print(f"  marker      : {info['marker_mm']:.2f} mm")

    if not args.board_only:
        results = generate_single_markers(cfg, out_dir / "markers")
        print()
        print("Single ArUco markers")
        print("-" * 60)
        for marker_id, path, side_mm in results:
            print(f"  id {marker_id:3d} : {side_mm:.2f} mm   {path}")

    print(
        """
NEXT: print, then MEASURE.
  1. Print at 100% / "Actual size". Turn OFF "Fit to page" and "Scale to fit"
     -- these silently resize the page and every millimetre this project
     reports would inherit that error.
  2. Print on matte paper if you can. Glossy paper produces specular
     highlights that wash out marker cells and cost corner accuracy.
  3. Tape the board FLAT to something rigid (clipboard, foam board, glass).
     A curled print violates the planar model and biases the calibration.
  4. Measure a chessboard square with a ruler or calipers and compare with
     the printed size above. If it differs, put the MEASURED value in
     charuco.square_length_mm (and scale marker_length_mm by the same
     factor) in config.yaml. Do the same for markers.marker_length_mm.
"""
    )
    return 0


# --------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------


def cmd_calibrate(args: argparse.Namespace) -> int:
    from calibration.capture import run_calibration_from_images, run_interactive_calibration

    cfg = _load(args)

    if args.from_images is not None:
        images_dir = resolve_path(args.from_images) if args.from_images else None
        run_calibration_from_images(cfg, images_dir)
        return 0

    result = run_interactive_calibration(cfg, auto=args.auto)
    return 0 if result is not None else 1


# --------------------------------------------------------------------------
# track
# --------------------------------------------------------------------------


def cmd_track(args: argparse.Namespace) -> int:
    from tracking.live import run_single_marker_tracking

    cfg = _load(args)
    run_single_marker_tracking(cfg, marker_length_mm=args.marker_size)
    return 0


# --------------------------------------------------------------------------
# generate-tool-sheet  /  track-tool   (Phase 2)
# --------------------------------------------------------------------------


def cmd_generate_tool_sheet(args: argparse.Namespace) -> int:
    from markers.tool import ToolGeometry, load_tool_geometry
    from markers.tool_sheet import generate_tool_sheet

    cfg = _load(args)
    out_dir = resolve_path(args.out)

    bodies: list[tuple[str, ToolGeometry]] = []
    if args.body in ("tool", "both"):
        bodies.append(("tool", load_tool_geometry(cfg)))
    if args.body in ("reference", "both"):
        if cfg.reference is None:
            if args.body == "reference":
                raise ValueError("config.yaml has no `reference:` section to generate.")
            print("  (no `reference:` section in config.yaml -- skipping that sheet)\n")
        else:
            bodies.append(("reference", ToolGeometry(cfg.reference)))

    # Two bodies sharing a marker ID would be claimed by both trackers at
    # once, so catch it here rather than after the sheets are printed.
    if len(bodies) == 2:
        shared = sorted(set(bodies[0][1].ids) & set(bodies[1][1].ids))
        if shared:
            raise ValueError(
                f"tool and reference share marker id(s) {shared}. Give each body "
                "its own IDs in config.yaml before printing."
            )

    for role, geometry in bodies:
        print(geometry.summary())

        too_close = geometry.overlapping_pairs()
        if too_close:
            print(
                f"\n  WARNING: these member pairs are too close to leave a usable quiet\n"
                f"  zone between them, and may fail to detect: {too_close}\n"
                "  Space them further apart in config.yaml."
            )

        out_path = out_dir / f"{role}_{geometry.name.replace(' ', '_').lower()}.png"
        info = generate_tool_sheet(cfg, geometry, out_path)

        print(f"\n{role.capitalize()} sheet")
        print("-" * 60)
        print(f"  file        : {out_path}")
        print(f"  image       : {info['pixels'][0]} x {info['pixels'][1]} px @ {info['dpi']} dpi")
        print(f"  printed size: {info['sheet_mm'][0]:.1f} x {info['sheet_mm'][1]:.1f} mm")
        print(f"  marker      : {info['marker_mm']:.2f} mm")
        print()

    print(
        """
NEXT: print, then MEASURE -- the config millimetres are what make the pose metric.
  1. Print at 100% / "Actual size". Turn OFF "Fit to page".
  2. Measure the printed 100 mm scale bar. If it is not 100.0 mm, the page was
     rescaled: every distance the tracker reports inherits that error.
  3. Measure marker CENTRE-TO-CENTRE spacing and compare with the labels on the
     sheet. Correct tool.markers x/y in config.yaml to what you measured.
  4. Mount it FLAT and rigid. A tool that flexes is not a rigid body, and the
     whole method assumes it is.
  5. The REFERENCE sheet must be fixed to whatever plays the part of the
     patient, and must not move relative to it during use. The tool sheet is
     the thing you pick up.
"""
    )
    return 0


def cmd_track_tool(args: argparse.Namespace) -> int:
    from tracking.live import run_tool_tracking

    cfg = _load(args)
    run_tool_tracking(cfg)
    return 0


# --------------------------------------------------------------------------
# pivot   (Phase 3)
# --------------------------------------------------------------------------


def cmd_pivot(args: argparse.Namespace) -> int:
    from calibration_pivot.capture import run_pivot_calibration
    from calibration_pivot.solve import solve_pivot

    cfg = _load(args)

    if args.replay:
        # Re-solve from previously captured poses. A calibration is a
        # measurement; being able to re-run it without repeating the physical
        # procedure is what makes it auditable.
        import numpy as np

        from calibration_pivot.tip import TipCalibration
        from core.transforms import SE3
        from markers.tool import load_tool_geometry

        path = cfg.pivot.poses_file
        if not path.is_file():
            raise FileNotFoundError(f"No saved pivot poses at {path}. Run `pivot` first.")
        data = np.load(path)
        poses = [SE3(R, t) for R, t in zip(data["R"], data["t"])]
        print(f"Re-solving from {len(poses)} saved poses in {path}\n")

        geometry = load_tool_geometry(cfg)
        saved_fp = str(data["tool_fingerprint"]) if "tool_fingerprint" in data else ""
        if saved_fp and saved_fp != geometry.fingerprint():
            print(
                f"  WARNING: these poses were captured against tool geometry "
                f"{saved_fp},\n  but config.yaml now describes "
                f"{geometry.fingerprint()}. The tool frame has moved,\n"
                "  so this re-solve does not describe your current tool.\n"
            )

        result = solve_pivot(poses)
        print(result.report())

        tip = TipCalibration(
            p_tip=result.p_tip,
            tool_name=geometry.name,
            tool_fingerprint=geometry.fingerprint(),
            residual_rms_mm=result.residual_rms_mm,
            residual_max_mm=result.residual_max_mm,
            n_frames=result.n_frames,
            condition_number=result.condition_number,
            cone_half_angle_deg=result.spread.cone_half_angle_deg,
            metadata={"resolved_from": str(path)},
        )
        print(f"\nSaved tip calibration -> {tip.save(cfg.tool.tip_calibration_file)}")
        return 0

    result = run_pivot_calibration(cfg)
    return 0 if result is not None else 1


# --------------------------------------------------------------------------
# navigate   (Phase 4)
# --------------------------------------------------------------------------


def cmd_navigate(args: argparse.Namespace) -> int:
    from navigation.live import run_navigation

    cfg = _load(args)
    run_navigation(cfg)
    return 0


# --------------------------------------------------------------------------
# accuracy   (Phase 5)
# --------------------------------------------------------------------------


def cmd_accuracy(args: argparse.Namespace) -> int:
    cfg = _load(args)
    lighting = getattr(args, "lighting", None)

    if args.accuracy_command == "jitter":
        from accuracy.jitter import run_jitter

        return run_jitter(cfg, frames=args.frames, lighting=lighting)

    if args.accuracy_command == "repeatability":
        from accuracy.points import run_repeatability

        return run_repeatability(cfg, touches=args.touches, lighting=lighting)

    if args.accuracy_command == "distance":
        from accuracy.points import run_distance

        return run_distance(cfg, true_mm=args.true_mm, pairs=args.pairs, lighting=lighting)

    if args.accuracy_command == "registration":
        from accuracy.points import run_registration

        return run_registration(cfg, lighting=lighting)

    if args.accuracy_command == "headline":
        from accuracy.headline import run_headline

        return run_headline(cfg)

    raise ValueError(f"unknown accuracy subcommand {args.accuracy_command!r}")


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to config.yaml (default: the one next to app.py)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="verify install, config, intrinsics and camera")
    p_check.add_argument(
        "--camera", action="store_true", help="also open the configured camera and grab a frame"
    )
    p_check.add_argument(
        "--list-cameras",
        action="store_true",
        help="survey device indices 0-3 and report which actually deliver frames",
    )
    p_check.set_defaults(func=cmd_check)

    p_gen = sub.add_parser(
        "generate-markers", help="render print-ready ChArUco board and ArUco markers"
    )
    p_gen.add_argument("--out", default="output/print", help="output directory")
    p_gen.add_argument("--board-only", action="store_true", help="skip the single markers")
    p_gen.add_argument("--markers-only", action="store_true", help="skip the ChArUco board")
    p_gen.set_defaults(func=cmd_generate_markers)

    p_cal = sub.add_parser("calibrate", help="estimate camera intrinsics from a ChArUco board")
    p_cal.add_argument(
        "--auto",
        action="store_true",
        help="start with auto-capture on (captures when the board is held still in a new position)",
    )
    p_cal.add_argument(
        "--from-images",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help="re-fit from saved frames instead of the live camera "
        "(default: calibration.images_dir from config.yaml)",
    )
    p_cal.set_defaults(func=cmd_calibrate)

    p_track = sub.add_parser("track", help="live single-marker 6-DOF pose with drawn axes")
    p_track.add_argument(
        "--marker-size",
        type=float,
        default=None,
        metavar="MM",
        help="override markers.marker_length_mm for this run",
    )
    p_track.set_defaults(func=cmd_track)

    p_sheet = sub.add_parser(
        "generate-tool-sheet",
        help="render a print-ready sheet of the tool's markers at their configured positions",
    )
    p_sheet.add_argument("--out", default="output/print", help="output directory")
    p_sheet.add_argument(
        "--body",
        choices=["tool", "reference", "both"],
        default="both",
        help="which rigid body's sheet to render (default: both)",
    )
    p_sheet.set_defaults(func=cmd_generate_tool_sheet)

    p_tool = sub.add_parser(
        "track-tool",
        help="live rigid-body tool pose from all visible member markers (occlusion tolerant)",
    )
    p_tool.set_defaults(func=cmd_track_tool)

    p_pivot = sub.add_parser(
        "pivot",
        help="pivot (tip) calibration: solve the tool tip offset by least squares",
    )
    p_pivot.add_argument(
        "--replay",
        action="store_true",
        help="re-solve from the saved poses instead of capturing again",
    )
    p_pivot.set_defaults(func=cmd_pivot)

    p_nav = sub.add_parser(
        "navigate",
        help="reference-relative pose, target guidance and point digitising",
    )
    p_nav.set_defaults(func=cmd_navigate)

    p_acc = sub.add_parser(
        "accuracy",
        help="accuracy characterisation: jitter, repeatability, distance, registration",
        description="Quantify precision (jitter, repeatability) and trueness "
        "(distance, registration). Each run writes raw samples and a summary to "
        "the results directory.",
    )
    acc_sub = p_acc.add_subparsers(dest="accuracy_command", required=True)

    # --lighting belongs on each measurement subcommand, not on the group, so
    # that the natural `accuracy jitter --lighting "..."` ordering works.
    # Options placed on the group parser would have to precede the subcommand
    # name, which nobody types.
    lighting_opt = argparse.ArgumentParser(add_help=False)
    lighting_opt.add_argument(
        "--lighting",
        default=None,
        metavar="NOTE",
        help='lighting description recorded with the result, e.g. "office fluorescent"',
    )

    a_jit = acc_sub.add_parser(
        "jitter", parents=[lighting_opt], help="static jitter of a held pose (precision)"
    )
    a_jit.add_argument("--frames", type=int, default=None, help="frames to collect")

    a_rep = acc_sub.add_parser(
        "repeatability",
        parents=[lighting_opt],
        help="repeatability of touching one divot (precision)",
    )
    a_rep.add_argument("--touches", type=int, default=None, help="number of touches")

    a_dist = acc_sub.add_parser(
        "distance",
        parents=[lighting_opt],
        help="measured vs known point separations (trueness)",
    )
    a_dist.add_argument(
        "--true-mm", type=float, default=None, metavar="MM",
        help="true separation for every pair (overrides accuracy.distance_pairs_mm)",
    )
    a_dist.add_argument("--pairs", type=int, default=None, help="number of pairs")

    acc_sub.add_parser(
        "registration",
        parents=[lighting_opt],
        help="rigid fit to known landmarks, residual RMS (trueness)",
    )
    acc_sub.add_parser("headline", help="aggregate the latest results into one paragraph")

    p_acc.set_defaults(func=cmd_accuracy)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        # These are the "you configured something wrong" errors. A traceback
        # adds nothing for them, so print the message and exit non-zero.
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
