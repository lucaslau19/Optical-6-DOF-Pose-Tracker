# Print-ready targets

Generated from [`config.yaml`](../config.yaml) so they match what the tracker
expects. Committed so you can print and try the project before running any code.

| File | What it is | Printed size |
|---|---|---|
| `charuco_7x5_25mm_DICT_5X5_100.png` | ChArUco board for camera calibration | 174.8 × 124.9 mm pattern, 24.98 mm squares |
| `tool_pointer_a.png` | The tracked tool: markers 10–13, 60 mm apart | 126 × 126 mm sheet, 29.97 mm markers |
| `reference_patient_reference.png` | The patient reference body: markers 20–23 | 126 × 126 mm sheet, 29.97 mm markers |

## Printing

**Print at 100% / "Actual size". Turn off "Fit to page".** Page scaling is the
single largest error source in this project — every millimetre it reports is
derived from these printed dimensions, and a 4% scaling becomes a 4% error in
every measurement with nothing in the software able to detect it.

Then **measure what you printed**:

- ChArUco board → measure one chessboard square, put the result in
  `charuco.square_length_mm`.
- Tool and reference sheets → measure the printed **100 mm scale bar** and the
  marker **centre-to-centre spacing**, and correct `tool.markers` /
  `reference.markers` accordingly.

Print on matte paper if you can (glossy produces specular highlights that wash
out marker cells), and mount everything **flat and rigid**. A sheet that bends
is not a rigid body, and the whole method assumes it is.

## Regenerating

If you change any dimension in `config.yaml`, regenerate rather than editing
these by hand:

```bash
python app.py generate-markers --board-only --out assets
python app.py generate-tool-sheet --out assets
```
