# Optical 6-DOF Pose Tracker

A single-webcam optical navigation demonstrator, built the way a real surgical
navigation system is built: calibrate the camera, track a rigid tool, calibrate
its tip, express everything relative to a patient reference frame, guide to a
target — and then **measure how accurate it actually is**.


---

## The problem

After my work at Intellijoint Surgical I realized surgical navigation answers one question continuously: **where is the
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

python app.py generate-tool-sheet   # print-ready rigid-tool marker sheet
python app.py track-tool            # one pose for the whole tool, survives occlusion
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

Draws each marker's frame — **X red (right), Y green (down), Z blue (into the
marker face, away from the camera)** — and shows the full 6-DOF pose both as an
on-screen readout and on the console. `q` quits, `p` forces a print.

The axes tilt in 3D with the marker: held square-on, blue is a dot at the centre
and the pose reads `rpy ≈ (0, 0, 0)`; tilt the marker and blue swings away from
you.

That frame — X right, Y down, Z away — matches the **camera's** own axis
convention, which is why a square-on marker reads zero rather than 180°. It is a
deliberate deviation from `cv2.aruco`'s default (X right, Y up, Z *toward* the
camera). If you compare against an OpenCV tutorial and Z points the other way,
that is why; [tracking/single_marker.py](tracking/single_marker.py) documents the
one-line revert.

Note that `SOLVEPNP_IPPE_SQUARE` *requires* OpenCV's object-point ordering, so
the solve happens in the ArUco frame and the result is converted afterwards
(`T_cam_marker = T_cam_ippe @ T_ippe_marker`) rather than by re-signing the
object points — doing the latter reverses the winding and the solver can return
a mirrored solution.

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

### 4. Track a rigid tool (Phase 2)

```bash
python app.py generate-tool-sheet   # sheet built FROM config.yaml
python app.py track-tool
```

A **tool** is several markers rigidly fixed at known positions
([config.yaml](config.yaml) `tool:`). Every visible member contributes four
3D↔2D correspondences, and they are concatenated into **one** `solvePnP`:

```
  corners of marker 10  ->  4 points
  corners of marker 11  ->  4 points     all in the TOOL frame
  corners of marker 13  ->  4 points
  ---------------------------------
  12 correspondences  ->  solvePnP  ->  T_cam_tool
```

This is deliberately *not* "estimate each marker's pose and average them".
Averaging rotations is ill-defined, weights a badly-conditioned marker the same
as a good one, and discards the fact that the markers constrain each other. One
fit over all points minimises a single reprojection cost and uses the whole
rigid body as evidence.

**Why the pose survives occlusion.** Nothing is re-initialised when a marker
disappears — the correspondence set just gets shorter. With 3 of 4 markers
visible there are still 12 equations for 6 unknowns, so the fit barely moves.
There is deliberately **no temporal filter**: a filter would hide precisely the
behaviour this phase exists to demonstrate.

The HUD makes it visible. Detected members outline green; hidden ones are drawn
as **amber predicted outlines** from the fitted pose, so when you cover a marker
its ghost stays sitting on top of it. A dot at the tool origin is the easiest
thing to judge stability on.

Also shown: markers used (`k / 4`), reprojection RMS in px, and a **per-marker
RMS** — an outlier there means that member's configured position disagrees with
where the camera sees it, which turns "the pose is a bit off" into "member 13 is
wrong". Note every residual rises when one member is wrong, since the fit is
pulled; read the worst as the suspect.

**Honest caveat:** a coplanar tool (markers on one flat sheet) *still* has the
two-fold planar ambiguity from Phase 1 — much better conditioned, because the
points span far more image area, but not eliminated. Only a **non-coplanar**
arrangement removes it. `z` is already honoured in the tool config, so mounting
two markers on a raised step works today. This is exactly why real tracked
instruments are not flat.

### 5. Calibrate the tip (Phase 3)

```bash
python app.py pivot            # plant the tip in a divot and precess
python app.py track-tool       # tip now drawn as a cyan crosshair
```

The tip is the only part of a pointer anyone cares about, and it cannot be
measured with a ruler — it is not on the markers, and the tool origin is an
abstraction at the cluster centroid. So it is measured **kinematically**.

Plant the tip in a fixed divot and pivot the body around it. The tip stays at
one point in space, so for every frame:

```
  R_i @ p_tip + t_i = p_pivot          (tip is planted, so p_pivot is constant)
  R_i @ p_tip - p_pivot = -t_i         (unknowns on one side)
  [ R_i | -I ] x = -t_i                 x = [p_tip; p_pivot], 6 unknowns
```

Stack all frames into a `(3N × 6)` system and solve by linear least squares.
No iteration, no initial guess, no local minima — the geometry made it linear,
which is the elegant part.

**The residual is a real accuracy figure.** `r_i = R_i p_tip + t_i − p_pivot` is
how far the reconstructed tip wandered on frame i, **in millimetres** — unlike
the pixel reprojection RMS of Phase 1. It absorbs tracking noise, a slipping
tip, a flexing tool and wrong marker geometry all at once.

**But the residual alone is not enough, and this is the interesting part.** If
every `R_i` were identical, the system would have rank 3 instead of 6 and
`p_tip` would be unrecoverable — orientation variety is literally what makes it
solvable. Yet a set of near-identical frames agrees with *itself* beautifully.
Measured on synthetic captures at a fixed noise level:

| Cone swept | Residual RMS | True tip error |
|---|---|---|
| 45° | 0.80 mm | **0.44 mm** |
| 3° | 0.79 mm | **5.04 mm** |

The residual cannot tell those apart. The **condition number** of the stacked
matrix can, and it catches both failure modes — too narrow a cone, and a cone
swept on one side only. So conditioning is gated separately, and the capture
screen shows an **orientation wheel** live: radius is tilt, angle is azimuth,
and you want a full ring rather than a blob or an arc. Frame count is not
progress.

**Verification** is direct: `track-tool` draws the tip as a cyan crosshair and
prints its camera-frame position. Replant the tip and rotate the tool around it
— the crosshair should stay pinned to the divot while the body swings. Its
wander *is* the calibration error.

One constraint: **the camera must not move during pivot capture**, because
`p_pivot` is solved in camera coordinates. Phase 4's patient reference frame is
exactly what lifts this.

### 6. Navigate (Phase 4)

```bash
python app.py generate-tool-sheet --body reference   # print + measure it too
python app.py navigate
```

A second rigid body — the **patient reference** — is fixed to whatever is being
navigated on. Pose is then reported relative to *it* rather than to the camera:

```
  T_ref_tool = T_cam_ref⁻¹ · T_cam_tool
             = T_ref_cam · T_cam_tool        ← `cam` cancels
```

Both measurements are made in the camera frame, so composing them cancels the
camera out. Pick the camera up, move it across the bench, and `T_cam_tool` and
`T_cam_ref` both change completely while `T_ref_tool` does not change at all.

That is the defining feature of an optical navigation system — it is why a
surgeon can reposition the tracker mid-procedure, and why the patient's bone
carries its own marker array. It is also, satisfyingly, one matrix multiply.

Measured across five very different camera poses:

| | Tip position spread |
|---|---|
| Camera frame | 120.6 mm |
| **Reference frame** | **5.2 mm** |
| Reference frame, well-conditioned views | **1.7 mm** |

**If the reference is not visible, nothing relative can be computed** — and
there is deliberately no fallback to the last known `T_cam_ref`. Reusing a stale
one would give confident, wrong answers the instant the camera moved, which is
precisely the failure this phase exists to prevent. So it is a loud
`REFERENCE LOST` state and the guidance goes blank rather than stale.

**Guidance.** The HUD shows tip-to-target distance in a large banner, and for a
trajectory (`navigation.target.axis`) also the perpendicular **offset**, the
**angular deviation** of the tool's long axis, and the **depth** along the axis.
Offset and angle are independent failure modes — you can be dead on the entry
point while pointing 20° wrong — so both are always shown rather than collapsed
into one "error". A bullseye dial shows the off-axis error as a dot to steer
into a ring, which is a far more natural control task than reading millimetres.

The tool's long axis defaults to the direction from the tool origin to the
**calibrated tip**, so it comes free from Phase 3 and cannot drift out of
agreement with the physical instrument the way a hand-typed vector would.

**Digitising.** Press `d` to record the tip position in the reference frame.
Touch two marks a known distance apart and the console prints the measured gap —
the first number with the *whole chain* in it. Two points 40 mm apart, digitised
from different camera poses, measure 39.889 mm synthetically.

**Honest caveat:** the relative pose carries the error of *two* rigid-body fits,
so reference-frame numbers jitter somewhat more than camera-frame ones. And
because both default bodies are coplanar sheets, the planar ambiguity still
applies — to the **reference** as much as the tool, and that one is the more
dangerous because a stationary reference looks reassuringly stable while being
the thing everything is measured against. The HUD flags either body going
ambiguous.

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
| [markers/tool.py](markers/tool.py) | rigid tool geometry: configured centres → 3D corner points |
| [markers/tool_sheet.py](markers/tool_sheet.py) | print-ready tool sheet generated from the same config the tracker uses |
| [tracking/single_marker.py](tracking/single_marker.py) | `solvePnP` pose + planar-ambiguity metric |
| [tracking/rigid_body.py](tracking/rigid_body.py) | **one** `solvePnP` over all visible member corners |
| [calibration_pivot/solve.py](calibration_pivot/solve.py) | the pivot least-squares system, residual and conditioning |
| [calibration_pivot/capture.py](calibration_pivot/capture.py) | live pivot capture with the orientation wheel |
| [calibration_pivot/tip.py](calibration_pivot/tip.py) | tip storage with a tool-geometry fingerprint |
| [navigation/relative.py](navigation/relative.py) | `T_ref_tool = T_cam_ref⁻¹ · T_cam_tool` — the camera-independence step |
| [navigation/guidance.py](navigation/guidance.py) | distance, perpendicular offset, angular deviation, tolerance bands |
| [navigation/overlay.py](navigation/overlay.py), [navigation/live.py](navigation/live.py) | navigation HUD and live loop |
| [navigation/digitize.py](navigation/digitize.py) | digitised points and pairwise distances |
| [accuracy/stats.py](accuracy/stats.py) | scatter, rotation means, distance trueness, Kabsch — pure maths, unit-tested |
| [accuracy/jitter.py](accuracy/jitter.py), [accuracy/points.py](accuracy/points.py) | the four measurement procedures |
| [accuracy/session.py](accuracy/session.py) | raw-sample CSV + summary YAML with recorded conditions |
| [accuracy/headline.py](accuracy/headline.py) | aggregates the latest results into one citable paragraph |
| [assets/](assets/) | committed print-ready ChArUco board, tool and reference sheets |
| [tracking/live.py](tracking/live.py), [tracking/overlay.py](tracking/overlay.py) | live tracking loops and pose overlays |
| [app.py](app.py) | CLI entry point |
| [config.yaml](config.yaml) | all geometry, camera and tolerance settings |
| [TESTING.md](TESTING.md) | per-phase acceptance steps |

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| **1** | Scaffolding, ChArUco board generation, camera calibration, single-marker 6-DOF pose | ✅ done |
| **2** | Rigid-body tool tracking: marker cluster with known geometry, one `solvePnP` over all visible corners, robust to partial occlusion | ✅ done |
| **3** | Pivot calibration: solve the tip offset from `[R_i \| -I][p_tip; p_pivot] = -t_i` by least squares over all frames, report residual RMS | ✅ done |
| **4** | Patient reference frame (`T_ref_tool = T_cam_ref⁻¹ · T_cam_tool`), target point/axis, live distance and angular-deviation HUD with tolerance feedback | ✅ done |
| **5** | Accuracy characterisation: precision (jitter, repeatability) and trueness (distance, registration), with raw data and conditions recorded | ✅ done |

---

### 7. Characterise the accuracy (Phase 5)

```bash
python app.py accuracy jitter --lighting "office fluorescent"
python app.py accuracy repeatability --touches 10
python app.py accuracy distance --true-mm 100 --pairs 5
python app.py accuracy registration
python app.py accuracy headline
```

Four metrics, measuring **two different things**:

| | Ground truth? | Answers |
|---|---|---|
| **Precision** — static jitter, touch repeatability | not needed | do repeated measurements agree with *each other*? |
| **Trueness** — distance error, registration residual | required | do they agree with *reality*? |

The distinction is the point. A system can be beautifully precise and
completely untrue: if the printed geometry is 2% small, every measurement is
consistently 2% short and the repeatability looks *superb*. So the distance
metric reports the **mean signed error** separately from the mean absolute
error — a consistent sign is the signature of a scale error, and no amount of
averaging removes it. The report fits a scale factor and names the percentage.

Each run writes raw samples (`.csv`) and a summary (`.yaml`) to `results/`,
including the conditions: resolution, working distance measured during the run,
marker size, calibration RMS, pivot residual and a lighting note. A number
without its conditions is not a result.

`registration` is the closest analogue to a real navigation system's quoted
accuracy — digitise landmarks with known coordinates, fit one rigid transform
(Kabsch), and report what is left over. That residual cannot be reduced by
choosing a better transform; it is the genuine distortion of the measurement.

---

## Results

<!--
  Run `python app.py accuracy headline` and paste its paragraph here.
  Fill the table from the same output. Leave the method column intact --
  the conditions are what make the numbers mean anything.
-->

| Metric | Type | Result | Method |
|---|---|---|---|
| Calibration reprojection RMS | self-consistency | _x.xxx px_ | N ChArUco views, 1280×720 |
| Pivot (tip) residual | self-consistency | _x.xx mm_ | N poses, cone ≥30° |
| **Static jitter** | precision | _x.xx mm RMS_ | 500 frames, everything static |
| **Touch repeatability** | precision | _x.xx mm RMS_ | 10 touches of one divot, varied angles |
| **Distance error** | **trueness** | _x.xx mm RMS_ | N pairs at 50/100/150 mm known |
| **Registration residual** | **trueness** | _x.xx mm RMS_ | 9 landmarks, rigid Kabsch fit |

_Conditions: 1280×720 webcam, xxx–xxx mm working distance, 30 mm markers,
calibration x.xxx px RMS, <lighting note>._

**Quote the trueness figures as the accuracy.** The precision numbers are the
noise floor, not the error — they would be flattered by any systematic problem
in the chain. For scale: a commercial infrared tracker (NDI Polaris) quotes
~0.25 mm RMS volumetric accuracy, with a stereo IR camera and machined
retro-reflective spheres rather than one webcam and printed paper.

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

## Limitations & future work

Stated plainly, because knowing what a measurement *doesn't* cover is most of
what makes it trustworthy.

**Monocular.** Depth comes entirely from the known physical size of the
fiducials, so any error in the printed geometry scales the whole depth estimate
— and depth is consistently the worst axis. This is why the project is so
insistent about printing at 100% and re-measuring, and why the distance metric
separates signed from absolute error. *Future work:* a stereo pair would make
depth a triangulation rather than a scale inference.

**Coplanar marker bodies.** Both the tool and the reference are flat printed
sheets, so both retain the two-fold planar pose ambiguity — much better
conditioned than a single marker because the points span more image area, but
not eliminated. It bites at long range and near-fronto-parallel views, and the
HUD flags it. The reference is the more dangerous case: it sits still and looks
reassuringly stable while being what everything else is measured against.
*Future work:* mount markers on a folded bracket. `z` is already honoured in the
config, so this needs no code change.

**No temporal filtering, deliberately.** There is no Kalman filter or smoothing
anywhere. A filter would improve the jitter number and hide exactly the
behaviour the occlusion and camera-independence tests exist to demonstrate.
Every reported pose is an independent fit to that frame's data. *Future work:*
a filter is easy to add once you can measure what it costs — but measure first.

**Rolling shutter.** Most webcams skew fast motion, so poses during rapid
movement are less trustworthy than static ones. All accuracy characterisation
here is static, which is a real limitation of the numbers, not just of the
hardware. *Future work:* a global-shutter camera, and a dynamic-accuracy metric.

**Lighting and print quality.** Paper fiducials are sensitive to specular glare,
shadow and motion blur. Matte paper and diffuse light matter more than they
should. Results record a lighting note for this reason.

**Single tool, single reference.** No tool-swapping, no multi-instrument
tracking, no verification divot to detect a bent instrument mid-procedure — all
of which a real system has.

**Not a medical device.** This is a demonstrator of the mathematics and the
workflow. Paper fiducials are not sterile, not retro-reflective, and this code
has none of the verification, risk management or fault detection that a
clinical navigation system requires.

---

## Requirements

Python 3.10+, `opencv-contrib-python` 4.10–4.x (contrib, not plain
`opencv-python`), NumPy, SciPy, PyYAML. Exact tested versions are pinned in
[requirements.txt](requirements.txt).

Print-ready targets are committed in [assets/](assets/) so you can print and try
the project before running anything.
