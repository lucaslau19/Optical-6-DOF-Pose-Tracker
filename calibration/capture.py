"""Interactive camera-calibration capture, plus offline re-calibration.

Two entry points:

  run_interactive_calibration  -- live webcam, operator captures views
  run_calibration_from_images  -- re-fit from previously saved frames

The second exists because calibration is a measurement you may need to
repeat or audit. Keeping the raw frames means a calibration can be re-run
with different settings, or a suspect view dropped, without asking someone
to wave a board around again.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from calibration.charuco import (
    COVERAGE_GRID,
    CalibrationResult,
    CharucoView,
    calibrate_from_views,
    corner_occupancy,
    detect_charuco,
)
from core.config import Config
from core.cv_compat import make_charuco_board, make_charuco_detector
from core.hud import AMBER, CYAN, GREEN, GREY, RED, WHITE, draw_coverage_grid, draw_text_panel
from core.video import camera

# Auto-capture thresholds, in pixels.
#   NOVELTY: how far the board centroid must be from every stored view before
#     a new one is worth keeping -- near-duplicate views inflate the view
#     count without adding information, and a calibration fitted from 30
#     nearly identical views has a lovely RMS and terrible extrapolation.
#   STABILITY: frame-to-frame centroid motion below which the board counts as
#     held still. Motion blur smears the chessboard saddle points, which is
#     the single most common way to poison a calibration set.
AUTO_NOVELTY_PX = 70.0
AUTO_STABILITY_PX = 1.5
AUTO_COOLDOWN_S = 0.7


def _novelty_px(view: CharucoView, stored: list[CharucoView]) -> float:
    """Distance from this board position to the nearest already-stored one."""
    if not stored:
        return float("inf")
    c = view.centroid
    return float(min(np.linalg.norm(c - s.centroid) for s in stored))


def _finalise(
    cfg: Config,
    board,
    views: list[CharucoView],
    image_size: tuple[int, int],
) -> CalibrationResult:
    """Fit, report and save."""
    print(f"\nFitting intrinsics from {len(views)} views ...")
    result = calibrate_from_views(board, views, image_size)

    print()
    print(result.report(outlier_rms_px=cfg.calibration.outlier_rms_px))

    if result.coverage_fraction < 0.5:
        print(
            "\n  WARNING: the board only visited "
            f"{result.coverage_fraction * 100:.0f}% of the frame. The distortion\n"
            "  model is extrapolating where the board never went -- usually the\n"
            "  frame corners, which is exactly where distortion is largest.\n"
            "  Recapture with the board pushed into all four corners."
        )

    out_path = cfg.camera.intrinsics_file
    result.intrinsics.save(out_path)
    print(f"\nSaved intrinsics -> {out_path}")
    return result


# --------------------------------------------------------------------------
# Live capture
# --------------------------------------------------------------------------


def run_interactive_calibration(cfg: Config, *, auto: bool = False) -> CalibrationResult | None:
    """Live ChArUco capture loop. Returns the result, or None if aborted."""
    board = make_charuco_board(
        cfg.charuco.squares_x,
        cfg.charuco.squares_y,
        cfg.charuco.square_length_mm,
        cfg.charuco.marker_length_mm,
        cfg.aruco.dictionary,
    )
    detector = make_charuco_detector(board)

    images_dir = cfg.calibration.images_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    views: list[CharucoView] = []
    saved_frames: list[Path] = []
    auto_mode = auto
    last_centroid: np.ndarray | None = None
    last_capture_t = 0.0
    flash_until = 0.0

    print(_capture_instructions(cfg))

    with camera(
        cfg.camera.device_index, cfg.camera.frame_width, cfg.camera.frame_height, cfg.camera.fps
    ) as cap:
        image_size = (cfg.camera.frame_width, cfg.camera.frame_height)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera read failed; stopping.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            view = detect_charuco(detector, gray, cfg.calibration.min_corners_per_view)
            display = frame.copy()

            n_corners = 0
            novelty = 0.0
            stable = False

            if view is not None:
                n_corners = view.n_corners
                if view.marker_ids is not None and len(view.marker_corners):
                    cv2.aruco.drawDetectedMarkers(display, view.marker_corners, view.marker_ids)
                cv2.aruco.drawDetectedCornersCharuco(
                    display, view.charuco_corners, view.charuco_ids, CYAN
                )

                centroid = view.centroid
                stable = (
                    last_centroid is not None
                    and float(np.linalg.norm(centroid - last_centroid)) < AUTO_STABILITY_PX
                )
                last_centroid = centroid
                novelty = _novelty_px(view, views)

                if (
                    auto_mode
                    and stable
                    and novelty > AUTO_NOVELTY_PX
                    and (time.time() - last_capture_t) > AUTO_COOLDOWN_S
                ):
                    _store_view(view, frame, views, saved_frames, images_dir)
                    last_capture_t = time.time()
                    flash_until = time.time() + 0.15
            else:
                last_centroid = None

            _draw_capture_hud(
                display,
                cfg,
                views,
                image_size,
                n_corners=n_corners,
                detected=view is not None,
                auto_mode=auto_mode,
                novelty=novelty,
                stable=stable,
            )

            if time.time() < flash_until:
                cv2.rectangle(
                    display, (0, 0), (display.shape[1] - 1, display.shape[0] - 1), WHITE, 14
                )

            cv2.imshow("Camera calibration -- ChArUco", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q / Esc
                print("Aborted; nothing saved.")
                return None
            if key == ord(" "):
                if view is None:
                    print(
                        f"  (no capture: need >= {cfg.calibration.min_corners_per_view} "
                        "ChArUco corners)"
                    )
                else:
                    _store_view(view, frame, views, saved_frames, images_dir)
                    flash_until = time.time() + 0.15
            elif key == ord("a"):
                auto_mode = not auto_mode
                print(f"  auto-capture {'ON' if auto_mode else 'OFF'}")
            elif key == ord("u") and views:
                views.pop()
                removed = saved_frames.pop()
                removed.unlink(missing_ok=True)
                print(f"  undo -> {len(views)} views")
            elif key == ord("c"):
                if len(views) < cfg.calibration.min_views:
                    print(
                        f"  need >= {cfg.calibration.min_views} views to calibrate "
                        f"(have {len(views)}); press 'c' again after capturing more, "
                        "or 'q' to abort."
                    )
                    continue
                break

    if not views:
        return None
    return _finalise(cfg, board, views, image_size)


def _store_view(
    view: CharucoView,
    frame: np.ndarray,
    views: list[CharucoView],
    saved_frames: list[Path],
    images_dir: Path,
) -> None:
    path = images_dir / f"calib_{len(views):03d}.png"
    cv2.imwrite(str(path), frame)
    views.append(view)
    saved_frames.append(path)
    print(f"  captured view {len(views):3d}  ({view.n_corners} corners)  -> {path.name}")


def _capture_instructions(cfg: Config) -> str:
    ch = cfg.charuco
    return f"""
ChArUco camera calibration
--------------------------
Board: {ch.squares_x}x{ch.squares_y} squares, {ch.square_length_mm} mm square,
       {ch.marker_length_mm} mm marker, dict {cfg.aruco.dictionary}
       (these MUST match your printed board -- measure it)

Keys:  SPACE  capture the current view
       a      toggle auto-capture (captures when held still in a new position)
       u      undo the last capture
       c      calibrate and save
       q      abort

How to get a good calibration -- this is the part that decides your accuracy:
  * Capture {cfg.calibration.min_views}-25 views.
  * Fill the coverage grid: push the board into all four FRAME CORNERS, not
    just the middle. Distortion is largest at the edges and is unconstrained
    anywhere the board never went.
  * Vary the tilt: roughly +/-30-45 deg about both axes. Views that are all
    fronto-parallel cannot separate focal length from distance, and the fit
    becomes ill-conditioned.
  * Vary the distance: some near (board filling the frame), some far.
  * Hold still at each capture. Motion blur moves the chessboard corners and
    is the most common cause of a bad calibration set.
  * Keep the board flat -- tape it to something rigid. A curled print is a
    systematic error the planar model cannot represent.
"""


def _draw_capture_hud(
    display: np.ndarray,
    cfg: Config,
    views: list[CharucoView],
    image_size: tuple[int, int],
    *,
    n_corners: int,
    detected: bool,
    auto_mode: bool,
    novelty: float,
    stable: bool,
) -> None:
    n = len(views)
    enough = n >= cfg.calibration.min_views

    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"views captured : {n} / {cfg.calibration.min_views} min", GREEN if enough else AMBER),
        (
            f"charuco corners: {n_corners}" if detected else "charuco corners: board not detected",
            GREEN if detected else RED,
        ),
        (f"auto-capture   : {'ON' if auto_mode else 'off'}", GREEN if auto_mode else GREY),
    ]
    if auto_mode and detected:
        lines.append(
            (
                f"  hold still   : {'steady' if stable else 'moving'}",
                GREEN if stable else AMBER,
            )
        )
        novel = novelty > AUTO_NOVELTY_PX
        novelty_txt = "inf" if not np.isfinite(novelty) else f"{novelty:.0f} px"
        lines.append(
            (
                f"  new position : {novelty_txt}",
                GREEN if novel else GREY,
            )
        )
    lines.append(("SPACE capture | a auto | u undo | c calibrate | q quit", WHITE))

    draw_text_panel(display, lines, origin=(12, 12))

    # Coverage map, top-right. Sized explicitly so the label cannot overrun
    # the frame edge and the grid clears the panel below it.
    cell = 14
    panel_w = 148
    panel_x = display.shape[1] - panel_w - 14
    grid_x = panel_x + (panel_w - COVERAGE_GRID * cell) // 2
    draw_text_panel(
        display, [("frame coverage", WHITE)], origin=(panel_x, 12), width=panel_w, scale=0.45
    )
    draw_coverage_grid(display, corner_occupancy(views, image_size), (grid_x, 62), cell=cell)


# --------------------------------------------------------------------------
# Offline re-calibration
# --------------------------------------------------------------------------


def run_calibration_from_images(cfg: Config, images_dir: Path | None = None) -> CalibrationResult:
    """Re-fit intrinsics from saved frames, with no camera attached."""
    images_dir = images_dir or cfg.calibration.images_dir
    paths = sorted(
        p for p in images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    board = make_charuco_board(
        cfg.charuco.squares_x,
        cfg.charuco.squares_y,
        cfg.charuco.square_length_mm,
        cfg.charuco.marker_length_mm,
        cfg.aruco.dictionary,
    )
    detector = make_charuco_detector(board)

    views: list[CharucoView] = []
    image_size: tuple[int, int] | None = None

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  {path.name}: unreadable, skipped")
            continue
        h, w = image.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif (w, h) != image_size:
            # Mixing resolutions silently produces meaningless intrinsics,
            # since fx/fy/cx/cy are all in pixels.
            print(f"  {path.name}: {w}x{h} differs from {image_size}, skipped")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        view = detect_charuco(detector, gray, cfg.calibration.min_corners_per_view)
        if view is None:
            print(f"  {path.name}: board not found / too few corners, skipped")
            continue
        views.append(view)
        print(f"  {path.name}: {view.n_corners} corners")

    if image_size is None or len(views) < 3:
        raise ValueError(f"Only {len(views)} usable views in {images_dir}; need at least 3.")

    return _finalise(cfg, board, views, image_size)
