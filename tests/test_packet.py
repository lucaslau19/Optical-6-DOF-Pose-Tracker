"""Schema v1 packets: the contract with the ROS 2 side.

Everything here runs without a camera or a network. The golden-file test is
the one that protects the ROS repo: it fails if a change here would alter what
the ROS side receives, before the ROS side finds out the hard way.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.transforms import SE3
from streaming.examples import GOLDEN_DIR, GOLDEN_NAMES, example_packets
from streaming.packet import (
    DEFAULT_INPUTS,
    PacketError,
    build_body,
    build_packet,
    decode,
    encode,
    validate,
)
from tracking.rigid_body import ToolPose

MAX_DATAGRAM_BYTES = 1024


@pytest.fixture
def packet() -> dict:
    """A known-good packet: both bodies tracked, tip calibrated."""
    return copy.deepcopy(example_packets()["tracking"])


def _set(packet: dict, path: tuple, value) -> None:
    target = packet
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _delete(packet: dict, path: tuple) -> None:
    target = packet
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]


def _rejects(packet: dict, field: str) -> str:
    """Assert validate() rejects the packet and names `field` first in the message."""
    with pytest.raises(PacketError) as info:
        validate(packet)
    message = str(info.value)
    assert message.startswith(f"{field}:"), message
    return message


def _tracked(t_mm=(12.5, -40.0, 455.0), rpy_deg=(-35.0, 4.0, 10.0), **kw) -> ToolPose:
    R = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    defaults = dict(
        status="tracking",
        marker_ids_used=[10, 11, 12, 13],
        reprojection_rms_px=0.4,
        ambiguity_ratio=8.0,
        n_points=16,
    )
    defaults.update(kw)
    return ToolPose(pose=SE3(R, np.array(t_mm)), **defaults)


# --------------------------------------------------------------------------
# round trip and golden files
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_encode_decode_round_trip(name):
    original = example_packets()[name]
    data = encode(original)
    assert isinstance(data, bytes)
    decoded = decode(data)
    assert decoded == original  # exact: Python floats survive JSON unchanged
    validate(decoded)


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_golden_file_is_valid_and_current(name):
    """The committed file validates AND still matches what the code builds.

    If this fails after a deliberate schema change, regenerate with
    `python -m streaming.examples` and copy the files to the ROS repo.
    """
    on_disk = json.loads((GOLDEN_DIR / f"{name}.json").read_text(encoding="utf-8"))
    validate(on_disk)
    assert on_disk == example_packets()[name]


def test_golden_files_cover_the_cases_they_are_named_for():
    g = example_packets()
    assert g["tracking"]["tool"]["status"] == "tracking"
    assert g["tracking"]["reference"]["status"] == "tracking"
    assert g["tracking"]["p_tip_tool_mm"] is not None

    assert g["reference_lost"]["reference"]["status"] == "lost"
    assert g["reference_lost"]["reference"]["T_cam"] is None
    assert g["reference_lost"]["tool"]["T_cam"] is not None

    assert g["no_tip"]["p_tip_tool_mm"] is None
    assert g["no_tip"]["seq"] == 0
    # A tracked body with no second planar solution: ambiguity null, NOT lost.
    assert g["no_tip"]["reference"]["status"] == "tracking"
    assert g["no_tip"]["reference"]["ambiguity"] is None


# --------------------------------------------------------------------------
# validate: what it must reject
# --------------------------------------------------------------------------


@pytest.mark.parametrize("version", [2, 0, "1", 1.0, True, None])
def test_rejects_wrong_version(packet, version):
    packet["v"] = version
    _rejects(packet, "v")


def test_rejects_missing_version(packet):
    del packet["v"]
    assert "missing" in _rejects(packet, "v")


@pytest.mark.parametrize(
    "path",
    [
        ("seq",),
        ("t_capture_ns",),
        ("t_send_ns",),
        ("units",),
        ("tool",),
        ("p_tip_tool_mm",),
        ("inputs",),
        ("tool", "rms_px"),
        ("reference", "T_cam"),
        ("reference", "ambiguous"),
        ("inputs", "clutch"),
    ],
)
def test_rejects_missing_field(packet, path):
    _delete(packet, path)
    assert "missing" in _rejects(packet, ".".join(path))


def test_rejects_unexpected_field(packet):
    packet["tool"]["quaternion"] = [0.0, 0.0, 0.0, 1.0]
    assert "unexpected" in _rejects(packet, "tool.quaternion")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "path, field",
    [
        (("tool", "T_cam", 3), "tool.T_cam[3]"),
        (("reference", "T_cam", 0), "reference.T_cam[0]"),
        (("tool", "rms_px"), "tool.rms_px"),
        (("reference", "ambiguity"), "reference.ambiguity"),
        (("p_tip_tool_mm", 1), "p_tip_tool_mm[1]"),
    ],
)
def test_rejects_non_finite_numbers(packet, path, field, bad):
    _set(packet, path, bad)
    _rejects(packet, field)


def test_encode_refuses_nan_instead_of_writing_invalid_json(packet):
    packet["tool"]["rms_px"] = math.nan
    with pytest.raises(PacketError):
        encode(packet)


def test_decode_refuses_nan_literals(packet):
    packet["tool"]["rms_px"] = math.nan
    text = json.dumps(packet)  # Python's default: writes the non-JSON token NaN
    assert "NaN" in text
    with pytest.raises(PacketError, match="NaN"):
        decode(text.encode("utf-8"))


@pytest.mark.parametrize("length", [0, 9, 12, 15, 17])
def test_rejects_T_cam_of_wrong_length(packet, length):
    T = packet["tool"]["T_cam"]
    packet["tool"]["T_cam"] = (T * 2)[:length]
    assert "16 numbers" in _rejects(packet, "tool.T_cam")


def test_rejects_scaled_rotation(packet):
    T = packet["tool"]["T_cam"]
    for i in (0, 1, 2, 4, 5, 6, 8, 9, 10):
        T[i] *= 1.01
    assert "orthonormal" in _rejects(packet, "tool.T_cam")


def test_rejects_sheared_rotation(packet):
    packet["reference"]["T_cam"][1] += 0.05
    assert "orthonormal" in _rejects(packet, "reference.T_cam")


def test_rejects_reflection(packet):
    T = packet["tool"]["T_cam"]
    for i in (0, 4, 8):  # negate the first column: still orthonormal, det = -1
        T[i] = -T[i]
    assert "reflection" in _rejects(packet, "tool.T_cam")


def test_rejects_bad_homogeneous_row(packet):
    packet["tool"]["T_cam"][15] = 2.0
    _rejects(packet, "tool.T_cam[15]")


def test_accepts_rounding_noise_in_rotation(packet):
    # Far below ORTHONORMAL_TOL: a consumer re-serialising the matrix must not
    # be able to make a valid packet invalid.
    packet["tool"]["T_cam"][0] += 1e-9
    validate(packet)


@pytest.mark.parametrize(
    "path, value, field",
    [
        (("seq",), -1, "seq"),
        (("seq",), True, "seq"),
        (("seq",), 3.0, "seq"),
        (("t_capture_ns",), "now", "t_capture_ns"),
        (("tool", "status"), "TRACKING", "tool.status"),
        (("tool", "markers_used"), ["10"], "tool.markers_used"),
        (("tool", "markers_used"), [10, 10], "tool.markers_used"),
        (("tool", "rms_px"), -0.1, "tool.rms_px"),
        (("tool", "ambiguous"), 0, "tool.ambiguous"),
        (("inputs", "clutch"), "false", "inputs.clutch"),
        (("p_tip_tool_mm",), [1.0, 2.0], "p_tip_tool_mm"),
    ],
)
def test_rejects_wrong_types(packet, path, value, field):
    _set(packet, path, value)
    _rejects(packet, field)


@pytest.mark.parametrize(
    "data, reason",
    [(b"\xff\xfe", "UTF-8"), (b"{", "JSON"), (b"[1, 2, 3]", "object")],
)
def test_decode_rejects_garbage(data, reason):
    with pytest.raises(PacketError, match=reason):
        decode(data)


# --------------------------------------------------------------------------
# lost bodies
# --------------------------------------------------------------------------


def test_lost_body_has_null_T_cam():
    lost = build_body(ToolPose(pose=None, status="lost"))
    assert lost == {
        "status": "lost",
        "T_cam": None,
        "markers_used": [],
        "rms_px": None,  # the tracker's NaN
        "ambiguity": None,  # the tracker's inf
        "ambiguous": False,
    }


def test_lost_body_with_a_pose_is_rejected(packet):
    # "lost" carrying a T_cam would be a stale pose dressed up as a lost one.
    packet["reference"].update(status="lost", markers_used=[], rms_px=None, ambiguity=None)
    assert "null" in _rejects(packet, "reference.T_cam")


def test_tracked_body_without_a_pose_is_rejected(packet):
    packet["tool"]["T_cam"] = None
    _rejects(packet, "tool.T_cam")


def test_reference_lost_packet_still_carries_the_tool():
    pkt = build_packet(7, 1_000, _tracked(), ToolPose(pose=None, status="lost"), [0.0, 99.0, 0.0])
    validate(pkt)
    assert pkt["reference"]["T_cam"] is None
    assert pkt["tool"]["T_cam"] is not None


# --------------------------------------------------------------------------
# the tracker's numbers, unchanged
# --------------------------------------------------------------------------


def test_units_stay_mm():
    t_mm = np.array([123.456789, -78.9, 456.0])
    body = _tracked(t_mm=tuple(t_mm))
    pkt = build_packet(0, 1_000, body, _tracked(), [1.5, 98.25, -0.75])
    validate(pkt)

    assert pkt["units"] == "mm"
    T = np.array(pkt["tool"]["T_cam"]).reshape(4, 4)
    # Translation is the tracker's millimetres as-is: not metres, not rescaled.
    np.testing.assert_allclose(T[:3, 3], t_mm, atol=1e-6)
    # And the frame is unchanged: this is T_cam_tool itself, row-major.
    np.testing.assert_allclose(T[:3, :3], body.pose.R, atol=1e-9)
    assert pkt["p_tip_tool_mm"] == [1.5, 98.25, -0.75]


def test_units_other_than_mm_are_rejected(packet):
    packet["units"] = "m"
    _rejects(packet, "units")


def test_T_cam_is_row_major():
    body = _tracked(t_mm=(1.0, 2.0, 3.0))
    T = build_body(body)["T_cam"]
    # Row-major puts the translation at indices 3, 7, 11.
    assert [T[3], T[7], T[11]] == [1.0, 2.0, 3.0]
    assert T[12:] == [0.0, 0.0, 0.0, 1.0]


def test_non_finite_tracker_values_become_null():
    # A tracked non-coplanar body reports ambiguity = inf: no planar twin.
    body = build_body(_tracked(ambiguity_ratio=math.inf))
    assert body["ambiguity"] is None
    assert body["ambiguous"] is False
    assert body["status"] == "tracking"  # null ambiguity is NOT "lost"


def test_ambiguous_flag_follows_the_tracker():
    assert build_body(_tracked(ambiguity_ratio=1.5))["ambiguous"] is True
    assert build_body(_tracked(ambiguity_ratio=8.0))["ambiguous"] is False


def test_inputs_clutch_is_false_and_not_shared():
    pkt = build_packet(0, 1_000, _tracked(), _tracked(), None)
    assert pkt["inputs"] == {"clutch": False}
    pkt["inputs"]["clutch"] = True  # mutating one packet must not leak into the next
    assert DEFAULT_INPUTS == {"clutch": False}


def test_t_send_defaults_to_capture_until_the_sender_stamps_it():
    pkt = build_packet(0, 5_000, _tracked(), _tracked(), None)
    assert pkt["t_send_ns"] == pkt["t_capture_ns"] == 5_000


# --------------------------------------------------------------------------
# size
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_golden_datagrams_are_under_1kB(name):
    assert len(encode(example_packets()[name])) < MAX_DATAGRAM_BYTES


def test_worst_case_datagram_is_under_1kB():
    """Every field at its longest plausible encoding."""
    # Rotation entries with as many digits as rounding allows, translations
    # beyond a metre, the longest status string, a huge seq.
    rpy = (-37.123456, 151.987654, -103.456789)
    body = _tracked(
        t_mm=(-1234.567891, -1234.567891, 1234.567891),
        rpy_deg=rpy,
        status="low_confidence",
        reprojection_rms_px=12.34567,
        ambiguity_ratio=123.45678,
    )
    pkt = build_packet(
        2**53, 1_999_999_999_999_999_999, body, body, [-123.456789, -123.456789, -123.456789]
    )
    validate(pkt)
    size = len(encode(pkt))
    assert size < MAX_DATAGRAM_BYTES, size
