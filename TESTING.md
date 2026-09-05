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
