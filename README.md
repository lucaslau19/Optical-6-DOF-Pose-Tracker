# Optical 6-DOF Pose Tracker

A single-webcam optical navigation demonstrator, built the way a real surgical
navigation system is built: calibrate the camera, track a rigid tool, calibrate
its tip, express everything relative to a patient reference frame, guide to a
target — and then **measure how accurate it actually is**.

> **Status: Phase 1 complete.** Camera calibration and live single-marker 6-DOF
> pose are working. Phases 2–5 are described in the roadmap below.

<!-- DEMO GIF GOES HERE
     Suggested: ~8 s loop of `python app.py track` showing the drawn axes
     following a marker through tilt and range, with the numeric readout
     visible. Record with ScreenToGif / OBS, drop in docs/demo.gif, then:
     ![Live 6-DOF tracking](docs/demo.gif)
-->

---

## The problem

Surgical navigation answers one question continuously: **where is the
instrument, relative to the patient?** Optical systems answer it by watching
retro-reflective or printed fiducials with a calibrated camera and solving for
pose.

Commercial systems use a stereo infrared tracker. This project asks how far you
can get with one ordinary webcam and printed ArUco fiducials — and, more
importantly, how you would *know* how far you got. The interesting engineering
is not "draw axes on a marker"; it is the chain of transforms, the calibration
that underpins them, and the error budget.

## The approach

```
  printed ChArUco board ──► camera intrinsics (fx, fy, cx, cy, distortion)
                                      │
  printed ArUco marker ──► detection ─┴─► solvePnP ──► T_cam_marker   [Phase 1]
                                                          │
                        rigid marker cluster ─────────────┴─► T_cam_tool [Phase 2]
                                                          │
                        pivot calibration ────────────────┴─► tip position [Phase 3]
                                                          │
                        reference marker ─────────────────┴─► T_ref_tool  [Phase 4]
                                                          │
                        known geometry ───────────────────┴─► accuracy, mm [Phase 5]
```

### Conventions

One convention, used everywhere, documented in full at the top of
[core/transforms.py](core/transforms.py):

- **`T_a_b` maps points in frame b into frame a**, so `p_a = T_a_b @ p_b`, and
  chains cancel: `T_a_c = T_a_b @ T_b_c`.
- An object's *pose* is the transform out of its own frame into the observer's,
  so a marker's pose in the camera is `T_cam_marker` — and its translation is
  literally the marker centre in camera coordinates. This is exactly what
  `cv2.solvePnP` returns.
- **Millimetres and degrees**, everywhere. Marker geometry is specified in mm,
  so `solvePnP` returns translations in mm with no conversion step.

### OpenCV API note

The `cv2.aruco` API was restructured in **OpenCV 4.7**, and
`estimatePoseSingleMarkers` was **removed in 4.9**. Most tutorials online still
use the old API. This project uses the current classes throughout —
`ArucoDetector`, `CharucoDetector`, the `(x, y)`-tuple `CharucoBoard`
constructor, `CharucoBoard.matchImagePoints` — and does pose estimation with
`cv2.solvePnP` over **explicit 3D object points**.

That last choice is not just compliance with the new API. Explicit object points
are what make Phase 2 a small extension rather than a rewrite: a rigid tool is
just a longer object-point array fed to the same solver. All version-sensitive
construction is isolated in [core/cv_compat.py](core/cv_compat.py), which fails
with an actionable message on an old or non-contrib install.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

python app.py check --camera        # verify OpenCV, config and webcam
python app.py check --list-cameras  # if the camera misbehaves: what's actually there
python app.py generate-markers      # writes printable PNGs to output/print/
#   ... print them at 100% scale, measure, update config.yaml ...
python app.py calibrate             # live ChArUco capture -> output/camera_intrinsics.yaml
python app.py track                 # live 6-DOF pose with drawn axes
```

### 1. Generate and print the targets

```bash
python app.py generate-markers
```

Writes `output/print/charuco_7x5_25mm_DICT_5X5_100.png` and
`output/print/markers/aruco_*.png`.

**Print at 100% / "Actual size".** Turn off "Fit to page" — it silently rescales
the page, and since every millimetre this system reports is derived from the
printed square size, a 4% print scaling becomes a 4% error in every depth
measurement, with nothing in the software to catch it.

Then **measure a chessboard square** with a ruler or calipers and compare it to
the size printed in the caption on the sheet. If they differ, put the *measured*
value into `charuco.square_length_mm` in [config.yaml](config.yaml) and scale
`marker_length_mm` by the same factor. This is the project's length standard;
everything downstream inherits it.

Finally, **tape the board flat** to something rigid. A curled print violates the
planar model that calibration assumes.

### 2. Calibrate the camera

```bash
python app.py calibrate          # SPACE to capture, c to calibrate, q to abort
python app.py calibrate --auto   # auto-capture when held still in a new pose
```

Camera calibration estimates the pinhole intrinsics (focal lengths and principal
point, in pixels) and the lens distortion coefficients. Without them, pixel
coordinates cannot be turned into rays, and pose is meaningless.

A **ChArUco** board is used rather than a plain chessboard because it combines
the sub-pixel accuracy of chessboard saddle-point corners with uniquely
identified corners from the embedded ArUco markers — so it still works when the
board is partly out of frame or occluded, which a plain chessboard does not.

The capture screen shows a live **frame-coverage grid**. Fill it. Lens
distortion is largest at the image edges and is completely unconstrained
anywhere the board never went, so a dataset shot only in the middle of the frame
produces a beautiful reprojection residual and a distortion model that is pure
extrapolation exactly where it matters.

The result is written to `output/camera_intrinsics.yaml` along with the RMS, view
count, and board geometry it was fitted from — a calibration without its
provenance is not a measurement.

**On reading the RMS:** reprojection RMS is a *residual*, not an accuracy. It
says how well the model fits the views it was fitted to, and it can be driven
low by using too few, too-similar views. Below ~0.5 px is a good webcam
calibration; above ~1 px, recalibrate. But it is necessary, not sufficient —
real accuracy is measured against known geometry in Phase 5.

To re-fit from the saved frames without re-shooting (e.g. after fixing a
mis-measured board dimension):

```bash
python app.py calibrate --from-images
```

### 3. Track

```bash
python app.py track
python app.py track --marker-size 30    # override the configured marker size
```

Draws each marker's frame — **X red, Y green, Z blue**, with Z out of the marker
face toward the camera — and prints the 6-DOF pose to the console. `q` quits,
`p` forces a print.

The readout includes two quality numbers per marker:

- **reproj** — per-point reprojection RMS of the pose fit, in pixels.
- **amb** — the *ambiguity ratio*: the second-best planar solution's
  reprojection error divided by the best one's.

That second number is the honest part. A single planar square has **two** poses
that reproject almost identically — the true one and one flipped about an axis
in the marker plane. When the marker is small, distant, or nearly
fronto-parallel, the two errors converge, the solver's choice flickers frame to
frame, and the drawn axes visibly fold over. A ratio near 1.0 means the pose
should not be trusted; the marker outline turns amber when it does.

This is not a bug to be filtered away — it is the fundamental limitation of
single-marker tracking, and it is precisely why real optical navigation tools
use a rigid cluster of several fiducials spread in 3D. Phase 2 fixes it
properly. Phase 1 at least measures it.

---

## Repository layout

| Path | What lives there |
|---|---|
| [core/transforms.py](core/transforms.py) | `SE3` helper; **the transform convention is documented here** |
| [core/cv_compat.py](core/cv_compat.py) | OpenCV version guard, modern ArUco/ChArUco factories |
| [core/intrinsics.py](core/intrinsics.py) | `CameraIntrinsics` + YAML I/O with provenance |
| [core/config.py](core/config.py) | typed `config.yaml` loading |
| [core/video.py](core/video.py), [core/hud.py](core/hud.py) | camera capture, shared overlay primitives |
| [markers/generate.py](markers/generate.py) | print-ready ChArUco board + ArUco marker PNGs |
| [calibration/charuco.py](calibration/charuco.py) | ChArUco detection, `calibrateCamera`, per-view residuals |
| [calibration/capture.py](calibration/capture.py) | live capture loop and offline re-fit |
| [tracking/single_marker.py](tracking/single_marker.py) | `solvePnP` pose + planar-ambiguity metric |
| [tracking/live.py](tracking/live.py), [tracking/overlay.py](tracking/overlay.py) | live tracking loop and pose overlay |
| [app.py](app.py) | CLI entry point |
| [config.yaml](config.yaml) | all geometry, camera and tolerance settings |

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| **1** | Scaffolding, ChArUco board generation, camera calibration, single-marker 6-DOF pose | ✅ done |
| **2** | Rigid-body tool tracking: coplanar marker cluster with known geometry, one `solvePnP` over all visible corners, robust to partial occlusion | planned |
| **3** | Pivot calibration: solve the tip offset from `[R_i \| -I][p_tip; p_pivot] = -t_i` by least squares over all frames, report residual RMS | planned |
| **4** | Patient reference frame (`T_ref_tool = T_cam_ref⁻¹ · T_cam_tool`), target point/axis, live distance and angular-deviation HUD with tolerance feedback | planned |
| **5** | Accuracy characterisation: static jitter (std over N frames) and point accuracy (RMS in mm against a known grid), plus results write-up | planned |

---

## Results

<!-- Fill in after Phase 5. Suggested table:

| Metric | Result | Conditions |
|---|---|---|
| Calibration reprojection RMS | _x.xx px_ | N views, 1280x720 |
| Static positional jitter (1σ) | _x.xx mm_ | 300 frames, tool at 400 mm |
| Static angular jitter (1σ) | _x.xx °_ | 300 frames, tool at 400 mm |
| Tip point accuracy (RMS) | _x.xx mm_ | N grid points, 250–600 mm range |
| Pivot calibration residual | _x.xx mm_ | N frames |
-->

_To be measured in Phase 5._ The numbers that matter for a navigation system are
tip **accuracy** (RMS against known geometry) and **jitter** (repeatability at
rest) — not the calibration residual, which only reports self-consistency.

## Troubleshooting the camera

Windows webcam capture is the least reliable part of this stack, so
[core/video.py](core/video.py) treats it defensively.

**"Camera delivered 0x0" / "opened, but delivered no frames".** OpenCV's
`cap.get(CAP_PROP_FRAME_WIDTH)` is not authoritative — depending on the backend
and whether the device is contended, it can report `0x0` on a capture that then
delivers perfectly good frames, or report a valid size on one that delivers
nothing. So the resolution is taken from an actual grabbed frame, after a
warm-up read loop (the first frames off a webcam are routinely dropped or black
while exposure settles).

**Backend fallback.** DirectShow, MSMF and `CAP_ANY` are tried in order, and a
backend that opens but yields no frames is discarded rather than fatal. This is
not theoretical: on the machine this was developed on, `CAP_DSHOW` on index 0
intermittently opens and delivers nothing while `CAP_MSMF` on the same index
works fine.

**Wrong device.** Virtual cameras (NVIDIA Broadcast, OBS, Teams effects) install
themselves as extra indices, so index 0 is not necessarily the real webcam:

```bash
python app.py check --list-cameras
```

surveys the indices and reports which actually deliver frames, and at what size.
Put a working one in `camera.device_index`.

**Resolution mismatch.** Intrinsics are only valid at the resolution they were
calibrated at, because fx, fy, cx, cy are all in pixels — calibrate at 1280×720
and stream at 640×480 and every pose is wrong by roughly 2× with no error
anywhere. `track` refuses to run on a mismatch rather than producing plausible
nonsense.

## Known limitations

- **Monocular.** Depth comes entirely from the known physical size of the
  fiducials, so any error in the printed marker size scales the whole depth
  estimate. A stereo rig would not have this dependency.
- **Planar ambiguity** on single markers (see above); addressed in Phase 2.
- **Rolling shutter.** Most webcams distort fast motion; poses during rapid
  movement are less trustworthy than static ones, which is one reason accuracy
  is characterised statically.
- **Printed paper fiducials** are not sterile, not retro-reflective, and are
  sensitive to lighting and specular glare. This is a demonstrator of the
  *mathematics and workflow*, not a clinical device.

## Requirements

Python 3.10+, `opencv-contrib-python>=4.10,<5`, NumPy, SciPy, PyYAML — see
[requirements.txt](requirements.txt).
