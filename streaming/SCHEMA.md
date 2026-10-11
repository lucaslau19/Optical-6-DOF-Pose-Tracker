# Tracking stream — schema v1

The contract between the optical tracker (producer, on Windows) and the ROS 2
driver (consumer, in WSL2). The tracker is the source of truth and nothing
flows back. If this document and [packet.py](packet.py) ever disagree, that is
a bug; [tests/test_packet.py](../tests/test_packet.py) and the
[golden files](#golden-files) exist to stop that happening.

## Transport

| | |
|---|---|
| Protocol | UDP. One datagram = one packet = one camera frame |
| Encoding | UTF-8 JSON object, compact (no whitespace) |
| Default destination | `127.0.0.1:5005` — `app.py stream --dest HOST:PORT` |
| Rate | one packet per camera frame (~30 Hz), or capped with `--max-rate HZ` |
| Size | under 1 kB; ~700 B in practice (705 B was the largest seen) |
| Delivery | fire-and-forget: no acknowledgement, no retransmission |

The sender gets no delivery receipt. Its `sent` counter counts datagrams handed
to the operating system, and stays error-free with nobody listening. Only the
receiver can confirm arrival ([scripts/udp_listen.py](../scripts/udp_listen.py)).

## Packet

```json
{
  "v": 1,
  "seq": 1200,
  "t_capture_ns": 1791651600123456789,
  "t_send_ns": 1791651600125806789,
  "units": "mm",
  "tool": BODY,
  "reference": BODY,
  "p_tip_tool_mm": [1.25, 98.6, -0.75],
  "inputs": {"clutch": false}
}
```

| Field | Type | Meaning |
|---|---|---|
| `v` | int | Schema version: `1`. Reject anything else. |
| `seq` | int ≥ 0 | Packet counter. 0 for the first packet after the tracker starts, +1 per packet. See [Sequence numbers](#sequence-numbers). |
| `t_capture_ns` | int | `time.time_ns()` on the tracker PC immediately after `cap.read()` returned the frame. |
| `t_send_ns` | int | `time.time_ns()` on the tracker PC immediately before the socket send. |
| `units` | `"mm"` | Every length in the packet is millimetres. |
| `tool` | BODY | The tool's rigid-body fit for this frame. |
| `reference` | BODY | The patient reference's fit for this frame. |
| `p_tip_tool_mm` | [x, y, z] or null | The calibrated tip offset **in the tool frame** (Phase 3 pivot calibration). Constant for a session — not a per-frame tip position. null when there is no tip calibration. |
| `inputs.clutch` | bool | Reserved for the next project. Always `false` in v1. |

## BODY

```json
{
  "status": "tracking",
  "T_cam": [r00, r01, r02, tx,  r10, r11, r12, ty,  r20, r21, r22, tz,  0, 0, 0, 1],
  "markers_used": [10, 11, 12, 13],
  "rms_px": 0.41,
  "ambiguity": 9.87,
  "ambiguous": false
}
```

| Field | Type | Meaning |
|---|---|---|
| `status` | string | `"tracking"`, `"low_confidence"` (no more than `low_confidence_markers` markers used — config.yaml, 1 by default), or `"lost"`. |
| `T_cam` | 16 numbers or null | `T_cam_body` as a row-major 4×4, in mm. null **if and only if** `status` is `"lost"`. |
| `markers_used` | [int] | Marker IDs that went into this frame's fit, ascending. `[]` when lost. |
| `rms_px` | number or null | Reprojection RMS of the fit, in pixels. null when lost. |
| `ambiguity` | number or null | Planar ambiguity ratio: second-best solution's reprojection error ÷ the best one's. **null does not mean lost** — see below. |
| `ambiguous` | bool | True when `ambiguity` < 3.0: the pose may be the mirrored planar solution. Always false when lost. |

### `ambiguity: null` is not "lost"

The tracker reports the ratio as infinite when there is no second planar
solution to compare against: a non-coplanar body, or a solver that returned
only one solution. Infinity has no JSON form, so it arrives as `null` **on a
body that is tracking normally**. Only `status` says whether a body is lost.

## Frames and units

`T_cam` follows the repo-wide `T_a_b` convention
([core/transforms.py](../core/transforms.py)): it maps points from the body's
frame into the camera's,

    p_cam = T_cam_body · p_body

so its translation column is the body origin in camera coordinates.

- **Camera frame**: OpenCV's — X right, Y down, Z out of the lens. This is the
  convention ROS uses for `*_optical_frame` frames (REP 103).
- **Body frames** (tool and reference): X right, Y down the printed sheet, Z
  into the sheet. The origin is wherever config.yaml's marker coordinates put
  it — the marker centroid by default.
- **Units**: millimetres, unconverted. ROS uses metres; converting is the
  consumer's job.

The consumer composes the relative pose itself:

    T_ref_tool = T_cam_ref⁻¹ · T_cam_tool
    p_tip_ref  = T_ref_tool · [p_tip_tool_mm, 1]

using the rigid inverse, `[R t; 0 1]⁻¹ = [Rᵀ  −Rᵀt; 0 1]`.

**If either body is lost there is no relative pose for that frame.** Never
reuse an earlier `T_cam_ref`: if the camera has moved since, the result is
confidently wrong. The tracker refuses to do this itself (README, Phase 4),
and a consumer that did it would undo that guarantee.

## Rules a v1 packet satisfies

`streaming.packet.validate()` enforces every rule here, and the tracker
validates each packet before sending it.

1. `v == 1`, checked before anything else.
2. Exactly the fields listed above: none missing, none extra. Adding or
   changing a field means a new `v`.
3. No NaN or Infinity anywhere. JSON has neither. A value the tracker cannot
   express as a finite number is sent as `null`.
4. `status == "lost"` ⇔ `T_cam == null`. A lost body also has
   `markers_used == []`, `rms_px == null`, `ambiguity == null` and
   `ambiguous == false`. It never carries an old measurement.
5. A tracked body (`"tracking"` or `"low_confidence"`) has a `T_cam` of 16
   finite numbers with a bottom row of `[0, 0, 0, 1]`, a rotation block with
   `max |RᵀR − I| ≤ 1e-6` and determinant +1 (a reflection is rejected), a
   non-empty `markers_used`, and a numeric `rms_px`.
6. `ambiguous == true` implies that `ambiguity` is a number.
7. Integers (`v`, `seq`, `t_*_ns`, marker IDs) are JSON integers — never
   `true`/`false`, and never `1.0`.

## Sequence numbers

- `seq` counts packets, not camera frames. Frames skipped by `--max-rate` do
  not use up a number, so **a gap always means a lost datagram**, never
  throttling.
- A send that fails still uses up its number, so it shows up as a gap — which,
  to the receiver, it is.
- `seq` restarts at 0 whenever the tracker restarts. Treat a decrease as a new
  session.

## Clocks

Both timestamps are `time.time_ns()` on the tracker PC: Windows wall-clock
nanoseconds since the Unix epoch.

- **They exceed 2⁵³.** Parse them as 64-bit integers. A parser that turns every
  JSON number into a double silently loses the last few hundred nanoseconds.
- **`t_send_ns − t_capture_ns` is reliable.** Both come from one clock, so the
  difference is the tracker's processing time for that frame.
- **`t_capture_ns` is when `cap.read()` returned**, not when the sensor exposed
  the frame. Driver and USB latency before that point are unknown and not
  included.
- **Never compare either stamp with a clock on the ROS side.** WSL2 runs its
  own clock, and on the development machine it was measurably off (2026-10-10,
  mirrored networking, 12–20 s runs of `tools/fake_stream.py`):

  | Compared with the tracker's clock | Measured |
  |---|---|
  | WSL monotonic clock rate | 3.0–3.8 % slow |
  | WSL wall clock − Windows wall clock | wandered from −0.4 s to +0.9 s within 20 s |

  "Latency" computed as WSL receive time − `t_send_ns` came out **negative**
  for some packets.

**Recommended `header.stamp` for ROS messages** — the capture instant,
expressed in the ROS clock:

    stamp = (ROS time at receipt) − (t_send_ns − t_capture_ns)

This uses only the receiver's own clock plus a duration measured on a single
clock, and ignores loopback transit, which is well under a millisecond. Use
the raw `t_*_ns` values for logging and ordering only, never as ROS times.
`scripts/udp_listen.py` measures the skew and warns about it after 10 s.

## Precision

Values are rounded before encoding: rotation entries to 9 decimal places,
translations and `p_tip_tool_mm` to 1e-6 mm, and `rms_px` and `ambiguity` to
4 decimal places. The tip composed from a rounded packet moves by under 1 µm
(0.4 µm on the golden packet), orders of magnitude below what the tracker can
resolve. In exchange the datagram shrinks from ~860 B to ~700 B, which leaves
a wide margin under 1 kB.

## Golden files

[golden/](golden/) holds three example packets, shared with the ROS repo for
its parser tests:

| File | Exercises |
|---|---|
| `tracking.json` | both bodies tracked (the reference with one marker covered); tip calibrated |
| `reference_lost.json` | `reference.T_cam == null` while the tool is still tracked |
| `no_tip.json` | `p_tip_tool_mm == null`; `seq == 0`; a `low_confidence`, `ambiguous` tool on one marker; a tracked reference with `ambiguity == null` |

They are built from fixed inputs by [examples.py](examples.py), through the
same `build_packet` the tracker uses, and `tests/test_packet.py` fails if the
committed files and the code drift apart. The files are indented to be
readable; whitespace means nothing to a JSON parser, and the wire format is
compact.

Regenerate them only for a deliberate schema change — which also means a new
`v` — then copy them to the ROS repo:

    .venv\Scripts\python.exe -m streaming.examples

## Reference implementations

- [packet.py](packet.py): `decode()` and `validate()`. Standard library only,
  so a bare `python3` in WSL can import it.
- [scripts/udp_listen.py](../scripts/udp_listen.py): a receiver that does the
  consumer-side maths above in plain Python.
