# Testing & acceptance

Manual acceptance steps for each phase, plus the synthetic checks that run
without hardware. Every phase is only "done" when the physical test passes —
synthetic ground truth proves the maths, not the millimetres.

**Before anything:** the printed geometry is the measurement. If a check below
fails, suspect the print scale and the config millimetres before suspecting the
code.

---

## Phase 1 — calibration and single-marker pose

### Setup

```powershell
.venv\Scripts\python.exe app.py check --camera
.venv\Scripts\python.exe app.py generate-markers
```

Print `output/print/` at **100% scale**, measure a ChArUco square, and correct
`charuco.square_length_mm` in `config.yaml` if it differs.

### P1.1 — Calibration

```powershell
.venv\Scripts\python.exe app.py calibrate --auto
```

| Check | Expected |
|---|---|
| Coverage grid fills up | green cells reach all four corners of the grid |
| Views captured | 15–25 |
| Reprojection RMS | < 0.5 px good, < 1.0 px acceptable |
| Field of view | plausible for a webcam (~55–75° horizontal) |
| Per-view outliers | few or none flagged above 1.0 px |

A low RMS from views clustered in the middle of the frame is **not** a pass —
the distortion model is unconstrained where the board never went.

### P1.2 — Single-marker pose

```powershell
.venv\Scripts\python.exe app.py track
```

| Check | Expected |
|---|---|
| Axes drawn on marker | X red, Y green, Z blue |
| Marker held square-on | `rpy ≈ (0, 0, 0)`; blue is a dot at the centre |
| Tilt the marker | axes tilt with it; blue swings **away** from you |
| Move to a known distance | reported `z` matches a ruler within a few mm |
| `amb` value | high (> 3) when close and tilted; may drop near 1 when far or square-on |

If `z` is consistently off by a few percent, the printed marker is not the size
`config.yaml` claims.

---

## Phase 2 — rigid-body tool tracking

The point of this phase: **one** pose for the whole tool, which survives
individual markers being hidden.

### Setup

```powershell
.venv\Scripts\python.exe app.py generate-tool-sheet
```

Then, and this is the part that decides whether the numbers mean anything:

1. Print `output/print/tool_pointer_a.png` at **100% / "Actual size"**. Turn off
   "Fit to page".
2. **Measure the printed 100 mm scale bar.** If it is not 100.0 mm the page was
   rescaled — reprint, or scale every `tool.markers` value by the same factor.
3. **Measure marker centre-to-centre spacing** (nominally 60 mm horizontally and
   vertically). Correct `tool.markers` `x`/`y` in `config.yaml` to what you
   actually measured.
4. Mount the sheet **flat and rigid** — glue it to card or foam board. A sheet
   that bends is not a rigid body, and the whole method assumes it is.

### P2.1 — Tool axes lock on and move in 3D

```powershell
.venv\Scripts\python.exe app.py track-tool
```

| Check | Expected |
|---|---|
| Axes drawn once, at the tool origin | centre of the 4-marker square, not on any marker |
| Origin dot | sits at the printed crosshair on the sheet |
| Move/rotate the whole board | axes follow rigidly; origin dot stays on the crosshair |
| Readout | `markers used: 4 / 4`, status `TRACKING` |
| Reprojection RMS | **< 1–2 px** |

### P2.2 — Occlusion robustness (the win condition)

Cover markers with your hand, one at a time, while watching the axes and the
origin dot.

| Markers covered | Expected status | Axes/origin behaviour |
|---|---|---|
| 0 | `TRACKING`, 4 used | baseline |
| 1 | `TRACKING`, 3 used | **essentially no movement** |
| 2 | `TRACKING`, 2 used | slight movement at most |
| 3 | `LOW CONFIDENCE`, 1 used | still tracking; more visible wobble is expected |
| 4 | `TOOL LOST` | axes disappear |

**The key observation:** the axes must not *jump* at the moment a marker is
covered or uncovered. A small change is fine; a visible snap is not.

Covered markers are drawn as **amber predicted outlines** labelled `12?`. Watch
these: they should stay sitting exactly on top of the real (hidden) markers. If
a ghost outline drifts off its marker, the configured geometry is wrong.

For reference, the synthetic equivalent of this test measures:

| Markers used | Pose shift vs the 4-marker fit |
|---|---|
| 3 | 0.23 mm / 0.12° |
| 2 | 0.59 mm / 0.38° |
| 1 | 1.50 mm / 0.78° |

Real-world numbers will be larger — that run has no lighting, blur or focus
error — but the *shape* should hold: the shift grows only as markers are lost,
and stays sub-millimetre with 2+ visible.

### P2.3 — Marker count and RMS update live

| Check | Expected |
|---|---|
| `markers used: k / 4` | updates immediately as you cover/uncover |
| Reprojection RMS | stays < 2 px throughout; turns red above the configured limit |
| `per-marker px:` line | all members roughly similar |

### P2.4 — Per-marker residual finds a bad measurement

This is the diagnostic that tells you *which* number in `config.yaml` is wrong.

Deliberately mis-type one member's `x` by 5 mm, re-run `track-tool`, and check
that member shows the largest `per-marker px` value and the HUD names it.

Note that **every** member's residual rises when one is wrong — the fit is
pulled by the bad point — so read it as "the worst one is the suspect", not as
"only the bad one moves". Undo the edit afterwards.

### P2.5 — Ambiguity honesty

The default tool is **coplanar** (all markers on one flat sheet), so it still
has the two-fold planar ambiguity from Phase 1, just much better conditioned.
Expect the HUD to occasionally show `AMBIGUOUS` when the sheet is far away and
nearly square-on to the camera.

This is not a bug to fix in software. Give the markers different `z` values in
`config.yaml` (mount two on a raised step) and the ambiguity disappears — which
is exactly why real tracked instruments are not flat.

---

## Phase 3 — pivot (tip) calibration

Recovers where the tool's **tip** is, in the tool frame, by pivoting it in a
fixed divot. This is the number that turns a tracked body into a pointer.

### Setup

**Build a tip.** Tape or glue a rigid pointed object to the marker sheet — a
skewer, a pen, a screwdriver — so it cannot move relative to the markers. If
the tip flexes or shifts, the whole method is measuring something that isn't
there.

**Make a divot the tip cannot slide out of.** A countersunk screw hole, the
dimple in a door hinge, a V cut into stiff card, or a nut taped to the bench. A
shallow dent is not enough: the tip must stay at *one point*, not a small area.
Slippage shows up directly as residual.

**Do not move the camera.** `p_pivot` is solved in camera coordinates, so it is
only constant if the camera is. A bumped tripod invalidates every frame captured
before the bump. (Phase 4's reference frame is what removes this restriction.)

### P3.1 — Capture and solve

```powershell
.venv\Scripts\python.exe app.py pivot
```

Plant the tip, keep it planted, and **precess the tool around a wide cone** —
tilt it well over and go *all the way around* the azimuth, not just side to
side. Spin the tool about its own axis as you go. Press `c` to solve.

Watch the **orientation wheel** on the right, not the frame counter. Each dot is
a captured orientation: radius is tilt, angle is azimuth.

| Wheel pattern | Meaning |
|---|---|
| Full ring at/outside the dashed circle | good — this is what you want |
| Blob in the middle | never tilted enough — the tip offset is barely constrained |
| Arc on one side | only swept half the cone |

Frames are only kept once the tool has rotated 5° since the last one, so pausing
to steady your hand costs nothing.

### P3.2 — Reading the result

**Two independent checks, and both must pass.**

| Metric | Target |
|---|---|
| Residual RMS | < 1 mm excellent, 1–2 mm good, 2–3 mm marginal, > 3 mm redo |
| Conditioning | `well conditioned` (cond < 5) or `acceptable` (cond < 10) |
| Cone half-angle | ≥ 30° |
| Azimuth covered | ≥ 60% of sectors |
| `\|tip\|` from origin | matches a ruler measurement of your actual tool |

**The residual alone is not a pass.** It is close to blind to conditioning. On
synthetic captures at a fixed noise level the residual sat at ~0.8 mm whether
the tool swept a 45° cone or a 3° one — while the true tip error was **0.44 mm
vs 5.04 mm**, a 12× difference the residual could not see. A capture with almost
no orientation variety agrees with itself beautifully. That is why the condition
number is reported separately and why the tool refuses to call a narrow capture
good.

If conditioning is fine but the residual is high, the problem is physical: a
slipping tip, a divot that is not a point, a flexing tool, or wrong
`tool.markers` millimetres.

Sanity-check `|tip|` against the real object with a ruler. If the tool origin is
the centroid of the marker sheet and the tip sticks out 90 mm, the report should
say about 90 mm.

To re-solve from the saved poses without pivoting again:

```powershell
.venv\Scripts\python.exe app.py pivot --replay
```

### P3.3 — Tip stability (the Phase 3 win condition)

```powershell
.venv\Scripts\python.exe app.py track-tool
```

The tip is now drawn as a **cyan crosshair** on a stalk from the tool origin,
and `TIP in camera` appears in the readout.

**Replant the tip in the divot and rotate the tool around it.**

| Check | Expected |
|---|---|
| Cyan crosshair | stays pinned to the divot while the body swings around it |
| `TIP in camera` x, y, z | stay ~constant as the tool rotates |
| Tool origin x, y, z | change a lot — that's the point |

The crosshair's wander *is* your tip calibration error, shown at the scale you
care about. A crosshair that orbits the divot rather than sitting on it means
the tip offset is wrong — recapture with a wider cone.

For reference, the synthetic version of this test wanders **0.688 mm (1.05 px)**
across five orientations spanning 10–35° of tilt and the full azimuth. Real
numbers will be larger.

### P3.4 — Stale-calibration guard

The tip file records a fingerprint of the tool geometry it was measured against.
Change any `tool.markers` value in `config.yaml` and re-run `track-tool`: it must
warn that the tip no longer matches. This matters because the failure is
otherwise silent — the tool frame moves, the old tip now points somewhere else on
the instrument, and nothing looks wrong. Undo the edit afterwards.

---

## Phase 4 — reference frame and navigation

The point of this phase: pose is reported **relative to the patient**, not the
camera, so the camera becomes free to move.

### Setup

```powershell
.venv\Scripts\python.exe app.py generate-tool-sheet --body reference
```

Print at 100%, measure, correct `reference.markers` in `config.yaml` exactly as
you did for the tool. Then:

- **Fix the reference sheet to the "patient"** — tape it to the object or bench
  you are navigating on. It must not move relative to that object during use.
  The tool sheet is the thing you pick up.
- Keep the **tool** and **reference** marker IDs disjoint (10–13 vs 20–23). The
  app refuses to start if they overlap, because both trackers would claim the
  same detection and the relative pose would be quietly meaningless.
- Do **not** leave the ChArUco calibration board in frame: it uses IDs 0–17 of
  the same dictionary, which overlaps the tool's 10–13.
- Set `navigation.target.point` to somewhere you can physically touch with the
  tip, in **reference-frame** millimetres.

```powershell
.venv\Scripts\python.exe app.py navigate
```

### P4.1 — Camera independence (the key test)

**Put the tool down.** Both bodies now static relative to each other. Then
**pick the camera up and move it** — walk it around, change the angle, change
the distance.

| Quantity | Expected |
|---|---|
| `tip in CAM` | changes a lot — hundreds of mm |
| `tip in REF` | **stays essentially constant** |
| `T_ref_tool` t and r | stay essentially constant |
| Target bullseye in the video | stays glued to the same physical spot |

This is the whole phase in one observation. `T_ref_tool = T_cam_ref⁻¹ ·
T_cam_tool` — both measurements share the camera, so it cancels.

For reference, the synthetic version of this test across five very different
camera poses:

| | Tip spread |
|---|---|
| Camera frame | 120.6 mm |
| Reference frame | 5.2 mm |
| Reference frame, closer / more oblique views | **1.7 mm** |

Note the last row. The residual few millimetres is **not** the reference-frame
maths — it is the coplanar planar ambiguity at long range. In the far set, 4 of
5 views were flagged `AMBIGUOUS`; in the close set, 1 of 5, and the spread fell
by 3×. If your numbers drift as you move the camera, look at the ambiguity
warning before doubting the transform.

Expect reference-frame numbers to jitter somewhat **more** than camera-frame
ones when both bodies are well conditioned: the relative pose carries the error
of *two* rigid-body fits, not one. That is the price of camera independence.

### P4.2 — Fixed physical point

Touch the tip to a fixed mark and hold it there. Move the camera around.
`tip in REF` should stay put. Press `d` at several camera positions and the
digitised points should cluster.

### P4.3 — Reference lost

Cover the reference sheet. Expect a red **REFERENCE LOST**, the guidance
readout blank, and the distance banner showing `--`.

There is deliberately **no fallback to the last known reference pose**. Reusing
a stale `T_cam_ref` would produce confident, wrong numbers the instant the
camera moved — exactly the failure this phase exists to prevent.

### P4.4 — Target guidance

Bring the tip towards the target point.

| Check | Expected |
|---|---|
| Distance banner | counts down towards 0 as you approach |
| Banner colour | red → amber → **green** inside `tolerances.distance_mm` |
| Offset / angle rows | green inside their tolerances |
| Bullseye dot | moves towards the centre; inside the green ring in tolerance |
| Tool held backwards | angle reads ~180°, **not** 0° |

That last row is deliberate: the angular deviation uses the full 0–180° range
rather than the acute angle, so holding the instrument reversed is visible
rather than hidden.

### P4.5 — Digitising and the ruler check (first end-to-end accuracy number)

Touch the tip to two marks a **known** distance apart — ruler graduations,
the ends of a gauge block, two drilled holes. Press `d` at each. The console
prints the gap immediately, and the HUD shows `last gap`.

Compare against the ruler. This is the first number with the **whole chain** in
it: camera calibration, both rigid-body fits, the tip calibration and the
reference transform. Phase 5 turns this into a proper characterisation.

Do it from **different camera positions** for each point — if the answer changes
when the camera does, the reference frame is not doing its job. The synthetic
version of exactly that (two points 40 mm apart, each digitised from a different
camera pose) measures **39.889 mm**, an error of 0.111 mm.

Press `s` to save, or quit — points are written to
`navigation.captured_points_file` with all pairwise distances printed.

---

## Synthetic checks (no hardware)

These run against rendered ground truth and validate the maths independently of
any printed target:

- Calibration recovers known intrinsics to ~1 px from synthetic ChArUco views.
- `solvePnP` inverts the projection to < 0.001 mm / < 0.001° over 200 random
  poses, with and without lens distortion.
- A one-marker *tool* produces object points identical to the single-marker
  convention, and a pose agreeing with the Phase 1 tracker to < 0.2 mm.
- Occlusion shifts stay within the table in P2.2.
- A deliberately mis-typed member is correctly identified by per-marker RMS.
- Tool status transitions `tracking → low_confidence → lost` at the configured
  thresholds.
- The pivot least-squares solve is **exact** on noiseless input, and recovers a
  known tip to 0.10 mm through the full render → detect → fit → solve pipeline.
- The conditioning check flags a narrow-cone capture that the residual passes
  (0.79 mm residual, 5.04 mm tip error).
- The rotation gate reduces 40 poses 1° apart to 8 kept.
- A stale tip calibration is detected via the tool-geometry fingerprint.
- Across five camera poses, the reference-frame tip moves 23× less than the
  camera-frame tip (5.2 mm vs 120.6 mm), falling to 1.7 mm on well-conditioned
  views.
- A hidden reference produces `REFERENCE LOST` with no stale fallback.
- Guidance offset, depth and angle are exact to 1e-9 against analytic geometry;
  a reversed tool reads 180°, not 0°.
- Two points 40 mm apart, digitised from *different* camera poses, measure
  39.889 mm.
