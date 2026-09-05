"""Webcam capture helper.

Thin, but it absorbs the things that bite on Windows:

* A requested resolution is only a *request*. The driver may hand back
  something else, and silently accepting that invalidates the intrinsics
  (see CameraIntrinsics.validate_for), so the mismatch has to be caught.

* `cap.get(CAP_PROP_FRAME_WIDTH)` IS NOT AUTHORITATIVE. Depending on the
  backend and on whether the device is contended, it can report 0x0 -- or a
  stale format -- on a capture that then goes on to deliver perfectly good
  frames. The only trustworthy source of the real frame size is a frame.
  So this module warms the camera up, reads an actual frame, and takes the
  size from `frame.shape`.

* Backends differ per machine. DirectShow is usually right on Windows, but
  when it fails MSMF often works (and vice versa), so we try in order rather
  than committing to one.

* Virtual cameras (NVIDIA Broadcast, OBS, Teams effects) install themselves
  as extra device indices and will happily open while delivering nothing
  useful. `probe_cameras` exists so the user can see what is actually there.
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterable, Iterator

import cv2

# Opening a camera can legitimately take a couple of seconds on Windows while
# the backend enumerates devices, so this is generous. It exists because when
# there is NO usable camera, cv2.VideoCapture does not return an unopened
# object -- it can block indefinitely inside the driver. A CLI that hangs with
# no output is far harder to diagnose than one that says what is wrong.
OPEN_TIMEOUT_S = 10.0

# How many reads to allow before declaring a capture dead. The first frames
# from a webcam are routinely dropped or black while exposure and white
# balance settle, and a contended device can need a moment to hand over.
WARMUP_READS = 20
WARMUP_SLEEP_S = 0.1


@contextmanager
def _quiet_opencv() -> Iterator[None]:
    """Silence OpenCV's native videoio logging for the duration of a block.

    Probing backends and indices that do not exist makes OpenCV print WARN and
    ERROR lines from native code. They are expected here -- trying things that
    may fail is the whole point -- and they bury this module's own, more
    useful, per-backend report. Set OPENCV_VIDEOIO_DEBUG=1 in the environment
    to see them anyway.
    """
    logging_mod = getattr(getattr(cv2, "utils", None), "logging", None)
    if logging_mod is None:
        yield
        return
    previous = logging_mod.getLogLevel()
    logging_mod.setLogLevel(logging_mod.LOG_LEVEL_SILENT)
    try:
        yield
    finally:
        logging_mod.setLogLevel(previous)


def _backend_candidates() -> list[tuple[str, int]]:
    """Capture backends to try, best first.

    On Windows DirectShow is usually the most cooperative about setting
    resolution on consumer webcams, but MSMF works where DSHOW does not on
    some machines -- so try both rather than betting on one. Elsewhere,
    CAP_ANY lets OpenCV pick (V4L2 on Linux, AVFoundation on macOS).
    """
    if sys.platform == "win32":
        return [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF), ("CAP_ANY", cv2.CAP_ANY)]
    return [("CAP_ANY", cv2.CAP_ANY)]


def _open_with_timeout(device_index: int, backend: int) -> cv2.VideoCapture | None:
    """Construct a VideoCapture, giving up if the driver never returns.

    cv2.VideoCapture offers no timeout of its own and cannot be interrupted,
    so the construction runs on a daemon thread that we simply abandon if it
    overruns. Abandoning a thread is not tidy, but the alternative is a CLI
    that hangs forever with no message, and the process is about to exit
    anyway.
    """
    box: dict[str, object] = {}

    def _work() -> None:
        try:
            box["cap"] = cv2.VideoCapture(device_index, backend)
        except Exception as exc:  # pragma: no cover - driver-specific
            box["err"] = exc

    thread = threading.Thread(target=_work, daemon=True, name="camera-open")
    thread.start()
    thread.join(OPEN_TIMEOUT_S)

    if thread.is_alive():
        raise RuntimeError(
            f"Opening camera index {device_index} timed out after "
            f"{OPEN_TIMEOUT_S:.0f} s -- the driver never returned.\n"
            "This usually means there is no camera at that index, or another "
            "application is holding it. Check camera.device_index in "
            "config.yaml (try 0, 1, 2), close Teams/Zoom, and on Windows check "
            "Settings > Privacy & security > Camera."
        )
    if "err" in box:
        raise RuntimeError(f"Opening camera index {device_index} failed: {box['err']}")
    return box.get("cap")  # type: ignore[return-value]


def _configure(cap: cv2.VideoCapture, width: int, height: int, fps: int) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    # A 1-frame buffer keeps the displayed pose current. Without it the driver
    # queues frames and the overlay lags behind the real tool, which for a
    # navigation display is worse than a lower frame rate. DirectShow refuses
    # this property (set() returns False) -- harmless, so it is not checked.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def _warmup(cap: cv2.VideoCapture, reads: int = WARMUP_READS):
    """Read until a real frame arrives; return it, or None if none ever does.

    This doubles as the resolution check. A capture that reports 1280x720 via
    get() but never yields a frame is useless, and one that reports 0x0 while
    yielding perfect 1280x720 frames is fine -- only the frame can tell them
    apart.
    """
    for _ in range(reads):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return frame
        time.sleep(WARMUP_SLEEP_S)
    return None


def open_camera(
    device_index: int,
    width: int,
    height: int,
    fps: int = 30,
    *,
    strict: bool = True,
) -> cv2.VideoCapture:
    """Open a webcam and confirm, from an actual frame, that it works.

    Tries each backend in turn. A backend that cannot open the device, cannot
    deliver a frame, or (when `strict`) delivers the wrong resolution is
    released and the next one is tried. Only if all of them fail does this
    raise -- with a per-backend account of what went wrong, because "it
    didn't work" is not an actionable message when three things could be
    responsible.

    With `strict=False` the first backend that delivers any frame at all
    wins; `check` uses that to report what the camera is really doing.
    """
    attempts: list[str] = []

    with _quiet_opencv():
        for name, backend in _backend_candidates():
            cap = _open_with_timeout(device_index, backend)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                attempts.append(f"  {name}: device did not open")
                continue

            _configure(cap, width, height, fps)
            frame = _warmup(cap)

            if frame is None:
                cap.release()
                attempts.append(f"  {name}: opened, but delivered no frames")
                continue

            actual = (frame.shape[1], frame.shape[0])
            if strict and actual != (width, height):
                cap.release()
                attempts.append(f"  {name}: delivered {actual[0]}x{actual[1]}")
                continue

            return cap

    raise RuntimeError(
        f"Could not get {width}x{height} frames from camera index {device_index}.\n"
        "Tried:\n" + "\n".join(attempts) + "\n\n"
        "Things to check, in order:\n"
        "  1. Run `python app.py check --list-cameras` to see which indices\n"
        "     actually deliver frames, then set camera.device_index to one.\n"
        "     Virtual cameras (NVIDIA Broadcast, OBS, Teams effects) appear as\n"
        "     extra indices and may open without producing usable video.\n"
        "  2. Close anything else using the camera -- Teams, Zoom, Slack and\n"
        "     browser tabs hold it exclusively.\n"
        "  3. If a backend reported a different resolution, set\n"
        "     camera.frame_width/frame_height in config.yaml to that format.\n"
        "  4. Windows: Settings > Privacy & security > Camera."
    )


def probe_cameras(
    indices: Iterable[int] = range(4), width: int = 1280, height: int = 720
) -> list[dict]:
    """Report which device indices actually deliver frames, and at what size.

    Diagnostic for the common Windows situation of several camera-like
    devices where only one is the real webcam.
    """
    found: list[dict] = []
    with _quiet_opencv():
        for index in indices:
            opened_any = False
            for name, backend in _backend_candidates():
                try:
                    cap = _open_with_timeout(index, backend)
                except RuntimeError:
                    found.append({"index": index, "backend": name, "status": "open timed out"})
                    continue
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    continue

                opened_any = True
                _configure(cap, width, height, fps=30)
                # Fewer reads than a real open: this is a survey, not a session.
                frame = _warmup(cap, reads=8)
                cap.release()

                if frame is None:
                    found.append({"index": index, "backend": name, "status": "no frames"})
                else:
                    found.append(
                        {
                            "index": index,
                            "backend": name,
                            "status": "OK",
                            "size": (frame.shape[1], frame.shape[0]),
                        }
                    )

            # Camera indices are enumerated densely in practice, so the first
            # index that no backend can even open marks the end of the list.
            # Stopping here keeps the survey quick and avoids backends printing
            # their own "index out of range" errors from native code.
            if not opened_any:
                break
    return found


@contextmanager
def camera(
    device_index: int, width: int, height: int, fps: int = 30, *, strict: bool = True
) -> Iterator[cv2.VideoCapture]:
    """Context manager that always releases the device and closes windows.

    A webcam left open by a crashed script stays locked until the process
    dies, so the release is worth guaranteeing.
    """
    cap = open_camera(device_index, width, height, fps, strict=strict)
    try:
        yield cap
    finally:
        cap.release()
        cv2.destroyAllWindows()


def frames(cap: cv2.VideoCapture) -> Iterator["cv2.typing.MatLike"]:
    """Yield frames until the stream ends or a read fails."""
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        yield frame
