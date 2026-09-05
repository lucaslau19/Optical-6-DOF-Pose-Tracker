"""OpenCV version guard and modern-ArUco-API factories.

WHY THIS FILE EXISTS
--------------------
The cv2.aruco API was restructured in OpenCV 4.7. The old free functions
(`cv2.aruco.detectMarkers`, `cv2.aruco.Dictionary_get`,
`cv2.aruco.estimatePoseSingleMarkers`, ...) were replaced by the
`ArucoDetector` / `CharucoDetector` / `CharucoBoard` classes, and
`estimatePoseSingleMarkers` was removed outright in 4.9.

Most ArUco tutorials online still use the removed API, which is why this
project centralises every construction call here: there is exactly one place
that touches the version-sensitive surface, it fails with an actionable
message on an old install, and nothing else in the codebase has to care.

`estimatePoseSingleMarkers` is deliberately *not* reimplemented as a helper.
It was removed for a good reason -- it hid the object-point definition, which
is precisely the thing that has to be explicit once markers get combined into
a rigid tool (Phase 2). See `tracking/single_marker.py`.
"""

from __future__ import annotations

import cv2

# 4.7.0 introduced ArucoDetector/CharucoDetector and the new CharucoBoard
# constructor. Everything below assumes that surface.
MIN_OPENCV = (4, 7, 0)


def opencv_version() -> tuple[int, int, int]:
    """Installed OpenCV version as a (major, minor, patch) integer tuple."""
    parts = cv2.__version__.split(".")
    nums: list[int] = []
    for p in parts[:3]:
        # Trailing components can carry suffixes on dev builds ("4.9.0-dev").
        digits = "".join(ch for ch in p if ch.isdigit())
        nums.append(int(digits) if digits else 0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)  # type: ignore[return-value]


def require_modern_aruco() -> tuple[int, int, int]:
    """Assert a usable OpenCV, and that it is the *contrib* build.

    `cv2.aruco` lives in opencv-contrib-python. Installing plain
    `opencv-python` (or having both installed, where the plain one can win)
    produces an AttributeError deep inside a detector call; catching it here
    with a clear message saves a genuinely confusing debugging session.
    """
    version = opencv_version()
    if version < MIN_OPENCV:
        raise RuntimeError(
            f"OpenCV {cv2.__version__} is too old for the ArucoDetector API "
            f"(need >= {'.'.join(map(str, MIN_OPENCV))}).\n"
            "Fix with:  pip install --upgrade 'opencv-contrib-python>=4.10'"
        )
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is unavailable -- the contrib modules are not installed.\n"
            "Fix with:  pip uninstall -y opencv-python opencv-python-headless\n"
            "           pip install 'opencv-contrib-python>=4.10'"
        )
    if not hasattr(cv2.aruco, "ArucoDetector"):
        raise RuntimeError(
            f"cv2.aruco in OpenCV {cv2.__version__} has no ArucoDetector class. "
            "This usually means a stale mixed install; reinstall with:\n"
            "  pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python\n"
            "  pip install 'opencv-contrib-python>=4.10'"
        )
    return version


def get_dictionary(name: str):
    """Look up a predefined ArUco dictionary by its constant name.

    e.g. "DICT_5X5_100" -> cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    """
    require_modern_aruco()
    if not hasattr(cv2.aruco, name):
        available = sorted(a for a in dir(cv2.aruco) if a.startswith("DICT_"))
        raise ValueError(
            f"Unknown ArUco dictionary '{name}'. Available:\n  " + "\n  ".join(available)
        )
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_detector_parameters(corner_refinement: str = "CORNER_REFINE_SUBPIX"):
    """Detector parameters tuned for pose estimation rather than raw detection rate.

    Corner refinement is the setting that matters. Without it, marker corners
    are reported at integer pixel positions and the resulting pose jitters by
    millimetres frame-to-frame; with sub-pixel refinement the same static
    marker settles to a fraction of that. Phase 5 measures this directly.
    """
    require_modern_aruco()
    params = cv2.aruco.DetectorParameters()

    if not hasattr(cv2.aruco, corner_refinement):
        raise ValueError(
            f"Unknown corner refinement method '{corner_refinement}'. "
            "Expected one of CORNER_REFINE_NONE, CORNER_REFINE_SUBPIX, "
            "CORNER_REFINE_CONTOUR, CORNER_REFINE_APRILTAG."
        )
    params.cornerRefinementMethod = getattr(cv2.aruco, corner_refinement)

    # Half-width of the sub-pixel search window, in pixels. 5 suits markers of
    # roughly 40-150 px on screen; too large and neighbouring image structure
    # pulls the corner off, too small and it cannot escape the initial guess.
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.01

    return params


def make_aruco_detector(dictionary_name: str, corner_refinement: str):
    """Build a `cv2.aruco.ArucoDetector` (the 4.7+ replacement for detectMarkers)."""
    dictionary = get_dictionary(dictionary_name)
    params = make_detector_parameters(corner_refinement)
    return cv2.aruco.ArucoDetector(dictionary, params)


def make_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    dictionary_name: str,
):
    """Build a `cv2.aruco.CharucoBoard` with dimensions in millimetres.

    Board geometry is defined in mm, so every 3D quantity downstream
    (calibration object points, pose translations) comes out in mm without a
    conversion step -- see the units note in core/transforms.py.

    Note the 4.7+ constructor signature: the board size is a single (x, y)
    tuple, where the pre-4.7 constructor took two separate ints.
    """
    dictionary = get_dictionary(dictionary_name)
    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        float(square_length_mm),
        float(marker_length_mm),
        dictionary,
    )


def make_charuco_detector(board):
    """Build a `cv2.aruco.CharucoDetector`.

    This one call does marker detection, board-aware interpolation of the
    chessboard corners, and refinement of markers that were initially missed
    but are predicted by the board layout -- what used to be `detectMarkers`
    followed by `interpolateCornersCharuco` and `refineDetectedMarkers`.
    """
    require_modern_aruco()
    return cv2.aruco.CharucoDetector(board)
