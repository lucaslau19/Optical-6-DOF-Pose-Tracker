"""The sender, the per-frame streamer, the headless loop, and the fake stream.

The headless test drives the real loop -- real detector, real rigid-body
fits, real UDP over loopback -- with synthetic frames standing in for the
camera, then stops it the way Ctrl+C does.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core.config import ToolConfig, ToolMemberConfig
from core.cv_compat import get_dictionary
from core.hud import AMBER, GREEN
from core.intrinsics import CameraIntrinsics
from markers.tool import ToolGeometry
from navigation.live import NavigationSetup
from streaming import live as streaming_live
from streaming.examples import P_TIP_TOOL, T_CAM_REF, example_packets
from streaming.live import FrameStreamer, run_stream_headless
from streaming.packet import build_packet, decode, validate
from streaming.sender import UdpSender, parse_dest
from tracking.rigid_body import RigidBodyTracker, ToolPose

REPO = Path(__file__).resolve().parent.parent


def _load_script(relative: str):
    """Import a script that lives outside any package (tools/, scripts/)."""
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2.0)
    yield sock
    sock.close()


def _drain(sock: socket.socket, n: int) -> list[bytes]:
    return [sock.recvfrom(65535)[0] for _ in range(n)]


# --------------------------------------------------------------------------
# parse_dest
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("127.0.0.1:5005", ("127.0.0.1", 5005)),
        ("localhost:9000", ("localhost", 9000)),
        ("[::1]:5005", ("::1", 5005)),
    ],
)
def test_parse_dest(text, expected):
    assert parse_dest(text) == expected


@pytest.mark.parametrize("text", ["5005", ":5005", "host:", "host:abc", "host:0", "host:70000"])
def test_parse_dest_rejects(text):
    with pytest.raises(ValueError):
        parse_dest(text)


# --------------------------------------------------------------------------
# UdpSender
# --------------------------------------------------------------------------


def test_sender_delivers_a_valid_datagram_and_stamps_send_time(receiver):
    packet = example_packets()["tracking"]
    port = receiver.getsockname()[1]
    with UdpSender("127.0.0.1", port) as sender:
        assert sender.send(packet) is True
        (data,) = _drain(receiver, 1)

    got = decode(data)
    validate(got)
    assert got["seq"] == packet["seq"]
    assert got["t_send_ns"] >= got["t_capture_ns"]
    assert got["t_send_ns"] != packet["t_send_ns"]  # re-stamped at send time...
    assert packet == example_packets()["tracking"]  # ...without mutating the caller's packet
    assert (sender.sent, sender.errors) == (1, 0)
    assert sender.max_bytes == len(data) < 1024


def test_sender_counts_an_invalid_packet_instead_of_raising(receiver):
    packet = example_packets()["tracking"]
    packet["tool"]["T_cam"] = packet["tool"]["T_cam"][:12]
    with UdpSender("127.0.0.1", receiver.getsockname()[1]) as sender:
        assert sender.send(packet) is False
        assert (sender.sent, sender.errors) == (0, 1)
        assert "tool.T_cam" in sender.last_error

    receiver.settimeout(0.2)
    with pytest.raises(socket.timeout):  # nothing schema-violating went on the wire
        receiver.recvfrom(65535)


def test_sender_counts_a_socket_failure_instead_of_raising(receiver):
    sender = UdpSender("127.0.0.1", receiver.getsockname()[1])
    sender.close()  # every send now fails inside the OS
    assert sender.send(example_packets()["tracking"]) is False
    assert (sender.sent, sender.errors) == (0, 1)
    assert sender.last_error.startswith("OSError")


# --------------------------------------------------------------------------
# FrameStreamer: seq and rate limiting
# --------------------------------------------------------------------------


class RecordingSender:
    """Stands in for UdpSender; records what it was asked to send."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.packets: list[dict] = []
        self.sent = self.errors = self.max_bytes = 0
        self.last_error = None
        self.dest_label = "test:0"

    def send(self, packet: dict) -> bool:
        self.packets.append(packet)
        if self.ok:
            self.sent += 1
        else:
            self.errors += 1
        return self.ok


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _bodies() -> tuple[ToolPose, ToolPose]:
    """Any valid pair will do: a tracked tool and a lost reference."""
    tool = ToolPose(pose=T_CAM_REF, status="tracking", marker_ids_used=[10], reprojection_rms_px=0.5)
    return tool, ToolPose(pose=None, status="lost")


def test_seq_starts_at_zero_and_counts_packets():
    sender = RecordingSender()
    streamer = FrameStreamer(sender)
    assert streamer.last_seq is None
    for _ in range(3):
        streamer.publish(1_000, *_bodies(), P_TIP_TOOL)
    assert [p["seq"] for p in sender.packets] == [0, 1, 2]
    assert streamer.last_seq == 2
    for p in sender.packets:
        validate(p)


def test_seq_advances_even_when_a_send_fails():
    # So the receiver sees the failed packet as a gap -- which it is.
    sender = RecordingSender(ok=False)
    streamer = FrameStreamer(sender)
    for _ in range(3):
        streamer.publish(1_000, *_bodies(), None)
    assert streamer.seq == 3
    assert sender.errors == 3


def test_max_rate_caps_the_average_rate_without_seq_gaps():
    sender = RecordingSender()
    clock = FakeClock()
    streamer = FrameStreamer(sender, max_rate_hz=20.0, clock=clock)
    for i in range(90):  # 3 s of a 30 fps camera
        clock.t = i / 30.0
        streamer.publish(1_000, *_bodies(), None)
    assert 59 <= len(sender.packets) <= 61  # 20 Hz, not 15 Hz
    # Throttled frames do not consume seq numbers: a gap only ever means loss.
    assert [p["seq"] for p in sender.packets] == list(range(len(sender.packets)))


def test_max_rate_does_not_burst_after_a_stall():
    sender = RecordingSender()
    clock = FakeClock()
    streamer = FrameStreamer(sender, max_rate_hz=20.0, clock=clock)
    for t in (0.0, 5.0, 5.0 + 1 / 30, 5.0 + 2 / 30):
        clock.t = t
        streamer.publish(1_000, *_bodies(), None)
    assert len(sender.packets) == 3  # 0.0, 5.0, then 5.067 -- not a catch-up burst


def test_rejects_non_positive_max_rate():
    with pytest.raises(ValueError):
        FrameStreamer(RecordingSender(), max_rate_hz=0)


def test_hud_line_is_green_until_a_send_fails():
    sender = RecordingSender()
    streamer = FrameStreamer(sender)
    streamer.publish(1_000, *_bodies(), None)
    text, colour = streamer.hud_line()
    assert text.startswith("STREAM -> test:0 | seq 0 | ")
    assert text.endswith("pkt/s")
    assert colour == GREEN

    sender.ok = False
    streamer.publish(1_000, *_bodies(), None)
    text, colour = streamer.hud_line()
    assert text.endswith("| 1 send errors")
    assert colour == AMBER


# --------------------------------------------------------------------------
# the headless loop, end to end
# --------------------------------------------------------------------------

K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
PX_PER_MM = 2.0  # fronto-parallel at fx / PX_PER_MM = 450 mm
TOOL_OFFSET_PX, REF_OFFSET_PX = -200, 200  # -> tool at x = -100 mm, reference at +100 mm


def _body(name: str, ids: list[int]) -> ToolGeometry:
    corners = [(-30.0, -30.0), (30.0, -30.0), (-30.0, 30.0), (30.0, 30.0)]
    members = [ToolMemberConfig(i, x, y, 0.0) for i, (x, y) in zip(ids, corners)]
    return ToolGeometry(ToolConfig(name=name, marker_length_mm=30.0, members=members))


def _render(bodies: list[tuple[ToolGeometry, int]]) -> np.ndarray:
    """A white frame with each body's markers drawn square-on to the camera."""
    frame = np.full((720, 1280, 3), 255, np.uint8)
    dictionary = get_dictionary("DICT_5X5_100")
    for geom, offset_px in bodies:
        side = int(round(geom.marker_length_mm * PX_PER_MM))
        for marker in geom.markers.values():
            tile = cv2.aruco.generateImageMarker(dictionary, marker.marker_id, side)
            cx = K[0, 2] + offset_px + marker.center[0] * PX_PER_MM
            cy = K[1, 2] + marker.center[1] * PX_PER_MM
            x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
            frame[y0 : y0 + side, x0 : x0 + side] = tile[..., None]
    return frame


class FakeCapture:
    """Hands out the given frames, then behaves like Ctrl+C."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)

    def read(self):
        if not self._frames:
            raise KeyboardInterrupt
        return True, self._frames.pop(0)


def _headless_setup() -> NavigationSetup:
    intrinsics = CameraIntrinsics(K, np.zeros(5), (1280, 720))
    tool_geom, ref_geom = _body("tool", [10, 11, 12, 13]), _body("reference", [20, 21, 22, 23])
    return NavigationSetup(
        intrinsics=intrinsics,
        tool_geom=tool_geom,
        ref_geom=ref_geom,
        tool_tracker=RigidBodyTracker(tool_geom, intrinsics, "DICT_5X5_100", "CORNER_REFINE_SUBPIX"),
        ref_tracker=RigidBodyTracker(ref_geom, intrinsics, "DICT_5X5_100", "CORNER_REFINE_SUBPIX"),
        tip=None,
        tip_warning=None,
        p_tip=np.array([0.0, 100.0, 0.0]),
        tool_axis=None,
    )


def _fake_camera(frames: list[np.ndarray], released: list[bool]):
    @contextmanager
    def camera(*_args, **_kwargs):
        released.append(False)
        try:
            yield FakeCapture(frames)
        finally:
            released[-1] = True

    return camera


CFG = SimpleNamespace(
    camera=SimpleNamespace(device_index=0, frame_width=1280, frame_height=720, fps=30)
)


def test_headless_loop_streams_every_frame_and_stops_cleanly_on_ctrl_c(
    monkeypatch, receiver, capsys
):
    setup = _headless_setup()
    scene = _render([(setup.tool_geom, TOOL_OFFSET_PX), (setup.ref_geom, REF_OFFSET_PX)])
    tool_only = _render([(setup.tool_geom, TOOL_OFFSET_PX)])
    blank = np.full_like(scene, 255)
    frames = [scene, scene, tool_only, blank, scene]

    released: list[bool] = []
    monkeypatch.setattr(streaming_live, "camera", _fake_camera(frames, released))

    with UdpSender("127.0.0.1", receiver.getsockname()[1]) as sender:
        streamer = FrameStreamer(sender)
        run_stream_headless(CFG, setup, streamer)  # returns on the fake Ctrl+C

    assert released == [True]  # camera released despite the interrupt
    assert "Ctrl+C" in capsys.readouterr().out
    assert (sender.sent, sender.errors) == (5, 0)

    packets = [decode(d) for d in _drain(receiver, 5)]
    for p in packets:
        validate(p)
    assert [p["seq"] for p in packets] == [0, 1, 2, 3, 4]
    assert [(p["tool"]["status"] != "lost", p["reference"]["status"] != "lost") for p in packets] == [
        (True, True),
        (True, True),
        (True, False),
        (False, False),
        (True, True),
    ]

    # The ROS side's job, done here to show the numbers are what was drawn:
    # the tool sits 200 mm to the left of the reference, both 450 mm away.
    p = packets[0]
    T_cam_tool = np.array(p["tool"]["T_cam"]).reshape(4, 4)
    T_cam_ref = np.array(p["reference"]["T_cam"]).reshape(4, 4)
    np.testing.assert_allclose(T_cam_tool[:3, 3], [-100.0, 0.0, 450.0], atol=1.0)
    np.testing.assert_allclose(T_cam_ref[:3, 3], [100.0, 0.0, 450.0], atol=1.0)
    p_tip_ref = np.linalg.inv(T_cam_ref) @ T_cam_tool @ np.r_[p["p_tip_tool_mm"], 1.0]
    np.testing.assert_allclose(p_tip_ref[:3], [-200.0, 100.0, 0.0], atol=1.0)


def test_headless_loop_survives_every_send_failing(monkeypatch, receiver):
    setup = _headless_setup()
    scene = _render([(setup.tool_geom, TOOL_OFFSET_PX), (setup.ref_geom, REF_OFFSET_PX)])
    monkeypatch.setattr(streaming_live, "camera", _fake_camera([scene] * 4, []))

    sender = UdpSender("127.0.0.1", receiver.getsockname()[1])
    sender.close()  # nobody can send anything
    streamer = FrameStreamer(sender)
    run_stream_headless(CFG, setup, streamer)  # must not raise

    assert (sender.sent, sender.errors) == (0, 4)
    assert streamer.seq == 4  # tracking carried on for every frame


# --------------------------------------------------------------------------
# tools/fake_stream.py and scripts/udp_listen.py
# --------------------------------------------------------------------------


def _tip_ref_numpy(packet: dict) -> np.ndarray:
    T_cam_tool = np.array(packet["tool"]["T_cam"]).reshape(4, 4)
    T_cam_ref = np.array(packet["reference"]["T_cam"]).reshape(4, 4)
    return (np.linalg.inv(T_cam_ref) @ T_cam_tool @ np.r_[packet["p_tip_tool_mm"], 1.0])[:3]


@pytest.mark.parametrize("t", [0.0, 1.3, 2.9, 4.4, 7.7, 12.0])
def test_fake_stream_tip_runs_round_the_circle(t):
    fake = _load_script("tools/fake_stream.py")
    tool, reference = fake.fake_bodies(t)
    packet = build_packet(0, 1, tool, reference, P_TIP_TOOL)
    validate(packet)

    x, y, z = _tip_ref_numpy(packet)
    assert np.hypot(x, y) == pytest.approx(fake.CIRCLE_RADIUS_MM, abs=1e-3)
    assert z == pytest.approx(-fake.CIRCLE_HEIGHT_MM, abs=1e-3)


def test_fake_stream_loses_the_reference_for_half_a_second_every_ten():
    fake = _load_script("tools/fake_stream.py")
    lost = [t / 100 for t in range(0, 3000) if fake.reference_lost_at(t / 100)]
    assert lost[0] == pytest.approx(9.5)
    assert len(lost) == 150  # 3 windows x 0.5 s at 100 samples/s

    tool, reference = fake.fake_bodies(9.7)
    packet = build_packet(0, 1, tool, reference, P_TIP_TOOL)
    validate(packet)
    assert packet["reference"]["T_cam"] is None
    assert packet["tool"]["status"] == "tracking"


def test_listener_tip_math_matches_numpy():
    listen = _load_script("scripts/udp_listen.py")
    g = example_packets()
    np.testing.assert_allclose(
        listen.tip_in_reference(g["tracking"]), _tip_ref_numpy(g["tracking"]), atol=1e-9
    )
    assert listen.tip_in_reference(g["reference_lost"]) is None
    assert listen.tip_in_reference(g["no_tip"]) is None
