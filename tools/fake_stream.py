#!/usr/bin/env python
r"""Send schema-valid tracking packets with no camera, to build the ROS side against.

    .venv\Scripts\python.exe tools\fake_stream.py                  # 127.0.0.1:5005, 30 Hz, until Ctrl+C
    .venv\Scripts\python.exe tools\fake_stream.py --duration 60 --rate 60

The scene uses the same reference pose and tip offset as the golden files
(streaming/examples.py):

  * the reference is fixed;
  * the tool tip runs round a 40 mm-radius circle centred 60 mm above the
    reference origin, one lap every 6 s. In the reference frame that is
    [40 cos wt, 40 sin wt, -60] mm -- reference +Z points INTO the sheet, so
    "above" is -Z, and the circle is centred on config.yaml's default target;
  * the tool tilts slowly as it goes (two periods that do not divide each
    other, so the motion never repeats exactly);
  * every 10 s the reference is lost for 0.5 s: status "lost", T_cam null --
    exactly what the real tracker sends when the reference sheet is covered.

Packets go through the same FrameStreamer / build_packet / UdpSender path as
`app.py stream`, so anything the ROS side parses from here it will parse from
the real tracker.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Make `python tools/fake_stream.py` work from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.spatial.transform import Rotation  # noqa: E402

from streaming.examples import (  # noqa: E402
    P_TIP_TOOL,
    REFERENCE_IDS,
    T_CAM_REF,
    TOOL_BASE_TILT_DEG,
    TOOL_IDS,
    tool_pose_for_tip,
)
from streaming.live import FrameStreamer  # noqa: E402
from streaming.sender import DEFAULT_DEST, UdpSender, parse_dest  # noqa: E402
from tracking.rigid_body import ToolPose  # noqa: E402

CIRCLE_RADIUS_MM = 40.0
CIRCLE_HEIGHT_MM = 60.0
CIRCLE_PERIOD_S = 6.0

# Tilt about the reference X and Y axes: amplitude (deg) and period (s).
TILT_X = (12.0, 8.0)
TILT_Y = (10.0, 11.0)

LOST_EVERY_S = 10.0
LOST_FOR_S = 0.5


def reference_lost_at(t: float) -> bool:
    """True during the last LOST_FOR_S of every LOST_EVERY_S (9.5-10 s, 19.5-20 s, ...)."""
    return (t % LOST_EVERY_S) >= LOST_EVERY_S - LOST_FOR_S


def tip_in_reference(t: float) -> list[float]:
    w = 2.0 * math.pi * t / CIRCLE_PERIOD_S
    return [CIRCLE_RADIUS_MM * math.cos(w), CIRCLE_RADIUS_MM * math.sin(w), -CIRCLE_HEIGHT_MM]


def fake_bodies(t: float) -> tuple[ToolPose, ToolPose]:
    """The tool and reference fits the tracker would report at time t (s)."""
    tilt_x = TOOL_BASE_TILT_DEG + TILT_X[0] * math.sin(2.0 * math.pi * t / TILT_X[1])
    tilt_y = TILT_Y[0] * math.sin(2.0 * math.pi * t / TILT_Y[1])
    R_ref_tool = Rotation.from_euler("xyz", [tilt_x, tilt_y, 0.0], degrees=True).as_matrix()
    T_cam_tool = T_CAM_REF @ tool_pose_for_tip(R_ref_tool, tip_in_reference(t))

    # Plausible quality numbers, gently varying so a consumer that displays
    # them can be seen to update.
    wobble = math.sin(2.0 * math.pi * t / 5.0)
    tool = ToolPose(
        pose=T_cam_tool,
        status="tracking",
        marker_ids_used=list(TOOL_IDS),
        reprojection_rms_px=0.40 + 0.05 * wobble,
        ambiguity_ratio=9.0 + 2.0 * wobble,
        n_points=4 * len(TOOL_IDS),
    )
    if reference_lost_at(t):
        reference = ToolPose(pose=None, status="lost")
    else:
        reference = ToolPose(
            pose=T_CAM_REF,
            status="tracking",
            marker_ids_used=list(REFERENCE_IDS),
            reprojection_rms_px=0.35,
            ambiguity_ratio=14.0,
            n_points=4 * len(REFERENCE_IDS),
        )
    return tool, reference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Send fake tracking packets (no camera) for developing the ROS 2 side."
    )
    parser.add_argument("--dest", default=DEFAULT_DEST, metavar="HOST:PORT",
                        help="where to send (default: %(default)s)")
    parser.add_argument("--rate", type=float, default=30.0, metavar="HZ",
                        help="packets per second (default: %(default)s)")
    parser.add_argument("--duration", type=float, default=None, metavar="S",
                        help="stop after this many seconds (default: run until Ctrl+C)")
    args = parser.parse_args(argv)
    if args.rate <= 0:
        parser.error("--rate must be positive")

    host, port = parse_dest(args.dest)
    period = 1.0 / args.rate

    with UdpSender(host, port) as sender:
        streamer = FrameStreamer(sender)
        print(f"Fake stream -> {sender.dest_label} at {args.rate:g} Hz (no camera). Ctrl+C to stop.")
        print(
            f"Tip circles r={CIRCLE_RADIUS_MM:g} mm at z={-CIRCLE_HEIGHT_MM:g} mm in the reference "
            f"frame; reference lost for {LOST_FOR_S:g} s every {LOST_EVERY_S:g} s.\n"
        )

        t0 = time.monotonic()
        next_tick = t0
        was_lost = False
        try:
            while args.duration is None or time.monotonic() - t0 < args.duration:
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

                t_capture_ns = time.time_ns()
                now = time.monotonic()
                t = now - t0
                tool, reference = fake_bodies(t)
                streamer.publish(t_capture_ns, tool, reference, P_TIP_TOOL)

                lost = reference.pose is None
                if lost != was_lost:
                    print(f"  t={t:6.2f} s  reference {'LOST' if lost else 'back'}")
                    was_lost = lost

                # Step the schedule, not `now`, so the average rate is exact;
                # after a stall (e.g. the console was paused) resync rather
                # than bursting to catch up.
                next_tick += period
                if now - next_tick > period:
                    next_tick = now + period

                if streamer.update_rate(now):
                    print(
                        f"{streamer.packet_rate:5.1f} pkt/s | t {t:7.1f} s | "
                        f"reference {'lost' if lost else 'tracking'} | seq {streamer.last_seq}"
                    )
        except KeyboardInterrupt:
            print("\nCtrl+C -- stopping.")

        print(streamer.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
