"""Tracking packets: the wire contract with the ROS 2 side.

Pure functions only -- no sockets, no clocks -- and nothing outside the
standard library is imported, so a bare python3 under WSL (or the ROS driver
itself) can import `decode` and `validate` straight from this file. The schema
is documented in streaming/SCHEMA.md; that file is the contract, this module
is the tracker's implementation of it.

WHAT IS SENT, AND WHAT IS NOT
-----------------------------
Each body goes out as T_cam_body -- the pose the rigid-body tracker fitted,
camera frame, millimetres, no conversion. T_ref_tool is deliberately NOT sent.
The consumer composes it the same way navigation/relative.py does:

        T_ref_tool = T_cam_ref^-1 @ T_cam_tool

Sending the two measurements rather than the derived pose keeps the tracker
the single source of truth for what was *measured*, and means a lost reference
arrives as a lost reference, not as an unexplained gap in a derived number.

`p_tip_tool_mm` is the CALIBRATED TIP OFFSET in the tool frame -- the Phase 3
result, constant for the whole session -- not a per-frame tip position. The
tip in the reference frame is T_ref_tool @ p_tip_tool_mm.

NUMBERS ON THE WIRE
-------------------
JSON has no NaN and no infinity, but the tracker uses both: a lost body's RMS
is NaN, and the ambiguity ratio is inf when there is no second planar
solution to compare against (a non-coplanar body, or IPPE returning only
one). Both become `null`. Note that `ambiguity: null` therefore does NOT mean
"lost" -- `status` is the only field that says that.

Values are rounded before they go on the wire (see the *_DECIMALS constants).
Full double precision would put a two-body packet close to 1 kB; the rounding
is many orders of magnitude below anything a webcam can resolve.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # type hints only: importing it would pull in OpenCV
    from tracking.rigid_body import ToolPose

SCHEMA_VERSION = 1
UNITS = "mm"
STATUSES = ("tracking", "low_confidence", "lost")

# Reserved for the next project. Always false in v1.
DEFAULT_INPUTS: dict[str, Any] = {"clutch": False}

# Rotation entries to 1e-9 is ~1 nm of position error at a metre's lever arm;
# translations to 1e-6 mm is 1 nm. Pixel residuals and the (unitless)
# ambiguity ratio to 1e-4.
ROTATION_DECIMALS = 9
MM_DECIMALS = 6
PX_DECIMALS = 4

# How far R^T R may sit from the identity. Rounding to ROTATION_DECIMALS
# leaves it ~1e-9 off; a matrix that is genuinely not a rotation (scaled,
# sheared, wrong element order) is off by far more than this.
ORTHONORMAL_TOL = 1e-6

TOP_LEVEL_FIELDS = (
    "v",
    "seq",
    "t_capture_ns",
    "t_send_ns",
    "units",
    "tool",
    "reference",
    "p_tip_tool_mm",
    "inputs",
)
BODY_FIELDS = ("status", "T_cam", "markers_used", "rms_px", "ambiguity", "ambiguous")
INPUT_FIELDS = ("clutch",)


class PacketError(ValueError):
    """A packet that does not meet the schema. The message names the field."""


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def _finite_or_none(value: float | None, decimals: int) -> float | None:
    """Round a finite number; turn NaN/inf (or None) into JSON null."""
    if value is None:
        return None
    value = float(value)
    return round(value, decimals) if math.isfinite(value) else None


def build_body(body: ToolPose) -> dict[str, Any]:
    """One rigid body's fit, as the BODY object of the schema."""
    if body.pose is None:
        T_cam = None
    else:
        M = body.pose.as_matrix()  # T_cam_body, row-major below
        T_cam = []
        for r in range(4):
            for c in range(4):
                v = float(M[r][c])
                if r == 3:
                    v = 1.0 if c == 3 else 0.0  # homogeneous row: exact, not rounded noise
                elif c == 3:
                    v = round(v, MM_DECIMALS)
                else:
                    v = round(v, ROTATION_DECIMALS)
                T_cam.append(v)

    return {
        "status": str(body.status),
        "T_cam": T_cam,
        "markers_used": [int(i) for i in body.marker_ids_used],
        "rms_px": _finite_or_none(body.reprojection_rms_px, PX_DECIMALS),
        "ambiguity": _finite_or_none(body.ambiguity_ratio, PX_DECIMALS),
        # A lost body has no pose to be ambiguous about.
        "ambiguous": bool(body.pose is not None and body.is_ambiguous),
    }


def build_packet(
    seq: int,
    t_capture_ns: int,
    tool: ToolPose,
    reference: ToolPose,
    p_tip: Iterable[float] | None,
    inputs: dict[str, Any] | None = None,
    *,
    t_send_ns: int | None = None,
) -> dict[str, Any]:
    """Assemble one frame's packet.

    `p_tip` is the calibrated tip offset in the TOOL frame (mm), or None when
    there is no tip calibration. `t_send_ns` is normally left as None:
    UdpSender.send stamps the real value immediately before the socket call,
    and until then the capture time stands in so that a built packet is
    schema-valid on its own.
    """
    tip = None
    if p_tip is not None:
        tip = [round(float(v), MM_DECIMALS) for v in p_tip]

    return {
        "v": SCHEMA_VERSION,
        "seq": int(seq),
        "t_capture_ns": int(t_capture_ns),
        "t_send_ns": int(t_capture_ns if t_send_ns is None else t_send_ns),
        "units": UNITS,
        "tool": build_body(tool),
        "reference": build_body(reference),
        "p_tip_tool_mm": tip,
        "inputs": dict(DEFAULT_INPUTS if inputs is None else inputs),
    }


# --------------------------------------------------------------------------
# encode / decode
# --------------------------------------------------------------------------


def encode(packet: dict[str, Any]) -> bytes:
    """Compact UTF-8 JSON. Refuses NaN/inf rather than emitting invalid JSON.

    Python's json module writes `NaN` and `Infinity` by default -- which are
    not JSON, and which a strict parser on the ROS side would choke on.
    `allow_nan=False` makes that an error here instead.
    """
    try:
        text = json.dumps(packet, separators=(",", ":"), allow_nan=False)
    except ValueError as exc:
        raise PacketError(
            f"packet: cannot encode a NaN or infinity ({exc}); validate() names the field"
        ) from exc
    except TypeError as exc:
        raise PacketError(f"packet: value is not JSON-serialisable ({exc})") from exc
    return text.encode("utf-8")


def _reject_constant(name: str) -> Any:
    raise PacketError(f"packet: contains {name}, which is not valid JSON")


def decode(data: bytes | bytearray | memoryview) -> dict[str, Any]:
    """Parse one datagram. Structural checks only -- call validate() for the schema."""
    try:
        text = bytes(data).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PacketError(f"packet: not UTF-8 ({exc})") from exc
    try:
        # json.loads accepts NaN/Infinity by default; refuse them so a
        # non-conforming sender is caught at the boundary.
        obj = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise PacketError(f"packet: not valid JSON ({exc})") from exc
    if not isinstance(obj, dict):
        raise PacketError(f"packet: top level must be a JSON object, got {type(obj).__name__}")
    return obj


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------


def _fail(path: str, problem: str) -> None:
    raise PacketError(f"{path}: {problem}")


def _is_int(x: Any) -> bool:
    # bool is a subclass of int in Python; `true` is not a sequence number.
    return isinstance(x, int) and not isinstance(x, bool)


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _check_fields(obj: Any, fields: tuple[str, ...], path: str) -> None:
    if not isinstance(obj, dict):
        _fail(path, f"expected an object, got {type(obj).__name__}")
    prefix = f"{path}." if path != "packet" else ""
    for name in fields:
        if name not in obj:
            _fail(prefix + name, "missing")
    for name in obj:
        if name not in fields:
            _fail(prefix + str(name), f"unexpected field (not in schema v{SCHEMA_VERSION})")


def _check_non_negative_int(value: Any, path: str) -> None:
    if not _is_int(value) or value < 0:
        _fail(path, f"expected a non-negative integer, got {value!r}")


def _check_optional_number(value: Any, path: str) -> None:
    if value is None:
        return
    if not _is_finite_number(value):
        _fail(path, f"expected a finite number or null, got {value!r}")
    if value < 0:
        _fail(path, f"must not be negative, got {value!r}")


def _check_T_cam(T: Any, path: str) -> None:
    if not isinstance(T, list):
        _fail(path, f"expected 16 numbers, got {type(T).__name__}")
    if len(T) != 16:
        _fail(path, f"expected 16 numbers (row-major 4x4), got {len(T)}")
    for i, v in enumerate(T):
        if not _is_finite_number(v):
            _fail(f"{path}[{i}]", f"expected a finite number, got {v!r}")

    for i, expected in zip(range(12, 16), (0.0, 0.0, 0.0, 1.0)):
        if abs(T[i] - expected) > 1e-9:
            _fail(f"{path}[{i}]", f"bottom row must be [0, 0, 0, 1], got {T[12:16]}")

    R = [[T[0], T[1], T[2]], [T[4], T[5], T[6]], [T[8], T[9], T[10]]]
    worst = 0.0
    for i in range(3):
        for j in range(3):
            dot = sum(R[k][i] * R[k][j] for k in range(3))  # (R^T R)[i][j]
            worst = max(worst, abs(dot - (1.0 if i == j else 0.0)))
    if worst > ORTHONORMAL_TOL:
        _fail(
            path,
            f"rotation block is not orthonormal (max |R^T R - I| = {worst:.2e}, "
            f"tolerance {ORTHONORMAL_TOL:.0e})",
        )
    det = (
        R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
        - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
        + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0])
    )
    if det < 0:
        _fail(path, "rotation block is a reflection (det = -1), not a rotation")


def _check_body(body: Any, path: str) -> None:
    _check_fields(body, BODY_FIELDS, path)

    status = body["status"]
    if status not in STATUSES:
        _fail(f"{path}.status", f"expected one of {list(STATUSES)}, got {status!r}")

    markers = body["markers_used"]
    if not isinstance(markers, list) or not all(_is_int(m) for m in markers):
        _fail(f"{path}.markers_used", f"expected a list of integer ids, got {markers!r}")
    if len(set(markers)) != len(markers):
        _fail(f"{path}.markers_used", f"duplicate ids in {markers}")

    _check_optional_number(body["rms_px"], f"{path}.rms_px")
    _check_optional_number(body["ambiguity"], f"{path}.ambiguity")
    if not isinstance(body["ambiguous"], bool):
        _fail(f"{path}.ambiguous", f"expected true or false, got {body['ambiguous']!r}")

    if status == "lost":
        # A lost body carries no measurement at all -- never a stale one.
        for name in ("T_cam", "rms_px", "ambiguity"):
            if body[name] is not None:
                _fail(f"{path}.{name}", f"must be null when status is 'lost', got {body[name]!r}")
        if markers:
            _fail(f"{path}.markers_used", f"must be empty when status is 'lost', got {markers}")
        if body["ambiguous"]:
            _fail(f"{path}.ambiguous", "must be false when status is 'lost'")
        return

    if body["T_cam"] is None:
        _fail(f"{path}.T_cam", f"must not be null when status is {status!r}")
    _check_T_cam(body["T_cam"], f"{path}.T_cam")
    if not markers:
        _fail(f"{path}.markers_used", f"must not be empty when status is {status!r}")
    if body["rms_px"] is None:
        _fail(f"{path}.rms_px", f"must not be null when status is {status!r}")
    if body["ambiguous"] and body["ambiguity"] is None:
        _fail(f"{path}.ambiguous", "is true but ambiguity is null")


def validate(packet: Any) -> None:
    """Raise PacketError, naming the offending field, unless `packet` is schema v1."""
    if not isinstance(packet, dict):
        _fail("packet", f"expected an object, got {type(packet).__name__}")

    # Version first: a packet from a different schema version would fail the
    # field checks below with a misleading message.
    if "v" not in packet:
        _fail("v", "missing")
    if not _is_int(packet["v"]) or packet["v"] != SCHEMA_VERSION:
        _fail("v", f"unsupported schema version {packet['v']!r} (expected {SCHEMA_VERSION})")

    _check_fields(packet, TOP_LEVEL_FIELDS, "packet")

    _check_non_negative_int(packet["seq"], "seq")
    _check_non_negative_int(packet["t_capture_ns"], "t_capture_ns")
    _check_non_negative_int(packet["t_send_ns"], "t_send_ns")

    if packet["units"] != UNITS:
        _fail("units", f"expected {UNITS!r}, got {packet['units']!r}")

    _check_body(packet["tool"], "tool")
    _check_body(packet["reference"], "reference")

    tip = packet["p_tip_tool_mm"]
    if tip is not None:
        if not isinstance(tip, list) or len(tip) != 3:
            _fail("p_tip_tool_mm", f"expected [x, y, z] or null, got {tip!r}")
        for i, v in enumerate(tip):
            if not _is_finite_number(v):
                _fail(f"p_tip_tool_mm[{i}]", f"expected a finite number, got {v!r}")

    _check_fields(packet["inputs"], INPUT_FIELDS, "inputs")
    if not isinstance(packet["inputs"]["clutch"], bool):
        _fail("inputs.clutch", f"expected true or false, got {packet['inputs']['clutch']!r}")
