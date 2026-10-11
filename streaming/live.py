"""`app.py stream`: send every frame's tracking result to ROS 2 over UDP.

The tracker stays the source of truth. ROS only consumes what is sent here,
and nothing comes back: there is no channel by which the ROS side could change
what the tracker measures or how.

Two modes, one setup:

  * windowed -- exactly the `navigate` screen (same loop, same keys, same
    HUD), plus a STREAM status line. Implemented by handing a FrameStreamer
    to navigation.live.run_navigation, so the two cannot drift apart.
  * headless -- no window, no drawing, one status line per second. Ctrl+C
    stops it cleanly.

Both use navigation.live.prepare_navigation, so the intrinsics, the two
trackers, the disjoint-ID check and the tip are set up identically to
`navigate`.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from core.config import Config
from core.hud import AMBER, GREEN
from core.video import camera
from navigation.live import NavigationSetup, prepare_navigation, run_navigation
from streaming.packet import DEFAULT_INPUTS, SCHEMA_VERSION, build_packet
from streaming.sender import UdpSender, parse_dest
from tracking.rigid_body import ToolPose

STATUS_INTERVAL_S = 1.0


class RateCounter:
    """Events per second, counted over consecutive ~1 s windows.

    A plain count per window rather than a smoothed rate: "pkt/s" should
    mean what it says, and a one-second dropout should show as a dip rather
    than be averaged away.
    """

    def __init__(self, window_s: float = STATUS_INTERVAL_S) -> None:
        self.window_s = window_s
        self.rate = 0.0
        self._t0: float | None = None
        self._n0 = 0

    def update(self, total: int, now: float) -> bool:
        """Feed the running total. Returns True when a new rate was computed."""
        if self._t0 is None:
            self._t0, self._n0 = now, total
            return False
        dt = now - self._t0
        if dt < self.window_s:
            return False
        self.rate = (total - self._n0) / dt
        self._t0, self._n0 = now, total
        return True


class FrameStreamer:
    """Turns each frame's two ToolPoses into a numbered packet and sends it.

    `seq` counts packets handed to the sender, starting at 0. It advances
    even when a send fails, so a receiver sees a failed send as a gap -- which
    is what it is. Frames skipped by --max-rate do NOT consume a seq number:
    a gap always means a packet was lost, never that it was throttled.
    """

    def __init__(
        self,
        sender: UdpSender,
        max_rate_hz: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_rate_hz is not None and max_rate_hz <= 0:
            raise ValueError(f"--max-rate must be positive, got {max_rate_hz}")
        self.sender = sender
        self.seq = 0  # the seq the NEXT packet will carry
        self._period = 1.0 / max_rate_hz if max_rate_hz else 0.0
        self._next_due = 0.0
        self._clock = clock
        self._rate = RateCounter()

    @property
    def last_seq(self) -> int | None:
        return self.seq - 1 if self.seq > 0 else None

    @property
    def packet_rate(self) -> float:
        return self._rate.rate

    def update_rate(self, now: float | None = None) -> bool:
        return self._rate.update(self.sender.sent, self._clock() if now is None else now)

    def _due(self) -> bool:
        if self._period == 0.0:
            return True
        now = self._clock()
        if now < self._next_due:
            return False
        # Schedule from the previous due time, not from `now`: frames rarely
        # land exactly on the boundary, and stepping from `now` would make a
        # 20 Hz cap on a 30 fps camera send only 15 Hz. After a stall (more
        # than a period late) resync instead of bursting to catch up.
        late = now - self._next_due
        base = self._next_due if late < self._period else now
        self._next_due = base + self._period
        return True

    def publish(
        self,
        t_capture_ns: int,
        tool: ToolPose,
        reference: ToolPose,
        p_tip: np.ndarray | None,
    ) -> bool:
        """Send this frame's result, unless --max-rate says to skip it."""
        if not self._due():
            return False
        packet = build_packet(self.seq, t_capture_ns, tool, reference, p_tip, DEFAULT_INPUTS)
        self.seq += 1
        return self.sender.send(packet)

    def status_text(self) -> str:
        seq = "-" if self.last_seq is None else str(self.last_seq)
        text = f"STREAM -> {self.sender.dest_label} | seq {seq} | {self.packet_rate:.1f} pkt/s"
        if self.sender.errors:
            text += f" | {self.sender.errors} send errors"
        return text

    def hud_line(self) -> tuple[str, tuple[int, int, int]]:
        """The STREAM line for the navigation panel: green, amber once sends fail."""
        self.update_rate()
        return self.status_text(), (AMBER if self.sender.errors else GREEN)

    def summary(self) -> str:
        s = self.sender
        lines = [
            "",
            "Stream summary",
            "-" * 60,
            f"  destination      : {s.dest_label}",
            f"  packets sent     : {s.sent}",
            f"  send errors      : {s.errors}",
            f"  largest datagram : {s.max_bytes} bytes",
        ]
        if s.last_error:
            lines.append(f"  last error       : {s.last_error}")
        return "\n".join(lines)


def _body_text(body: ToolPose) -> str:
    if body.pose is None:
        return body.status
    return f"{body.status} ({body.n_markers_used} mk)"


def _header(streamer: FrameStreamer, max_rate_hz: float | None, headless: bool) -> str:
    rate = f"capped at {max_rate_hz:g} Hz" if max_rate_hz else "one packet per camera frame"
    lines = [
        "",
        "Streaming tracking results over UDP",
        "-" * 60,
        f"Destination : {streamer.sender.dest_label}  (schema v{SCHEMA_VERSION}, "
        "see streaming/SCHEMA.md)",
        f"Rate        : {rate}",
        "Sent per frame: T_cam_tool and T_cam_ref (camera frame, mm) plus the",
        "calibrated tip offset. The ROS side composes T_ref_tool itself.",
    ]
    if headless:
        lines.append("Headless: one status line per second. Ctrl+C to stop.")
    return "\n".join(lines)


def _headless_setup_text(setup: NavigationSetup) -> str:
    lines = [
        f"Tool        : {setup.tool_geom.name}, ids {setup.tool_geom.ids}",
        f"Reference   : {setup.ref_geom.name}, ids {setup.ref_geom.ids}",
    ]
    if setup.tip is not None:
        lines.append("Tip calibration (sent as p_tip_tool_mm):")
        lines.append(setup.tip.summary())
        if setup.tip_warning:
            lines.append(f"  WARNING: {setup.tip_warning}")
    else:
        lines.append("NO TIP CALIBRATION -- p_tip_tool_mm will be null. Run `app.py pivot`.")
    lines.append("")
    return "\n".join(lines)


def run_stream_headless(cfg: Config, setup: NavigationSetup, streamer: FrameStreamer) -> None:
    """Track and stream with no window. Returns when Ctrl+C is pressed."""
    frames = 0
    frame_rate = RateCounter()
    errors_reported = 0

    with camera(
        cfg.camera.device_index, cfg.camera.frame_width, cfg.camera.frame_height, cfg.camera.fps
    ) as cap:
        try:
            while True:
                ok, frame = cap.read()
                t_capture_ns = time.time_ns()
                if not ok or frame is None:
                    print("Camera read failed; stopping.")
                    break
                frames += 1

                tool = setup.tool_tracker.process(frame)
                reference = setup.ref_tracker.process(frame)
                streamer.publish(t_capture_ns, tool, reference, setup.p_tip)

                # Both counters see the same `now` from the first frame on, so
                # their windows line up and fps and pkt/s are directly
                # comparable on the same line.
                now = time.monotonic()
                frame_rate.update(frames, now)
                if not streamer.update_rate(now):
                    continue

                seq = "-" if streamer.last_seq is None else streamer.last_seq
                print(
                    f"{streamer.packet_rate:5.1f} pkt/s | {frame_rate.rate:5.1f} fps | "
                    f"tool {_body_text(tool)} | reference {_body_text(reference)} | seq {seq}"
                )
                if streamer.sender.errors > errors_reported:
                    errors_reported = streamer.sender.errors
                    print(f"  send errors: {errors_reported}  (last: {streamer.sender.last_error})")
        except KeyboardInterrupt:
            # Caught here rather than in app.py so the camera context manager
            # still releases the device and the summary still prints.
            print("\nCtrl+C -- stopping.")


def run_stream(
    cfg: Config,
    dest: str,
    *,
    headless: bool = False,
    max_rate_hz: float | None = None,
) -> int:
    host, port = parse_dest(dest)
    with UdpSender(host, port) as sender:
        streamer = FrameStreamer(sender, max_rate_hz)
        print(_header(streamer, max_rate_hz, headless))

        if headless:
            setup = prepare_navigation(cfg)
            print(_headless_setup_text(setup))
            run_stream_headless(cfg, setup, streamer)
        else:
            run_navigation(cfg, stream=streamer)

        print(streamer.summary())
    return 0
