#!/usr/bin/env python3
"""Show what arrives on the tracking stream. Standard library only.

Runs under WSL's own python3 -- no ROS, no OpenCV, no venv:

    python3 "/mnt/c/Projects/Optical 6-DOF pose tracker/scripts/udp_listen.py"
    python3 ".../udp_listen.py" -v              # also print every packet
    python3 ".../udp_listen.py" --count 100     # exit after 100 packets

Every datagram is decoded and validated with the repo's own
streaming/packet.py (imported by path, so this checks the real contract rather
than a copy of it). Once a second it prints the packet rate (measured on the
tracker's clock -- WSL2's own clock can run several percent off), sequence
gaps, the largest datagram, both body statuses, and the tip in the REFERENCE
frame, composed here exactly as the ROS side must do it:

    p_tip_ref = T_cam_ref^-1 @ T_cam_tool @ p_tip_tool

That number should match `tip in REF` on the navigate HUD.

Exit status: 0 if every packet was valid, 1 if any was not, 2 if --timeout
expired with nothing received.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

# Do not leave __pycache__ behind in the Windows checkout.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streaming.packet import PacketError, decode, validate  # noqa: E402


def tip_in_reference(packet: dict) -> list[float] | None:
    """p_tip_ref = T_cam_ref^-1 @ T_cam_tool @ p_tip_tool, from the raw packet."""
    T_tool = packet["tool"]["T_cam"]
    T_ref = packet["reference"]["T_cam"]
    tip = packet["p_tip_tool_mm"]
    if T_tool is None or T_ref is None or tip is None:
        return None
    # Row-major: element (r, c) of a 4x4 is T[4*r + c].
    p_cam = [sum(T_tool[4 * r + c] * tip[c] for c in range(3)) + T_tool[4 * r + 3] for r in range(3)]
    # Inverse of a rigid transform: R^T (p - t). No general matrix inverse needed.
    d = [p_cam[r] - T_ref[4 * r + 3] for r in range(3)]
    return [sum(T_ref[4 * k + c] * d[k] for k in range(3)) for c in range(3)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Receive and check tracking packets.")
    parser.add_argument("--host", default="0.0.0.0", help="address to bind (default: %(default)s)")
    parser.add_argument("--port", type=int, default=5005, help="UDP port (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every packet")
    parser.add_argument("--count", type=int, default=0, help="exit after this many packets")
    parser.add_argument("--timeout", type=float, default=0.0, metavar="S",
                        help="exit if nothing arrives for S seconds (default: wait forever)")
    args = parser.parse_args(argv)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(0.2)
    print(f"Listening on udp {args.host}:{args.port}  (Ctrl+C to stop)")

    received = invalid = lost = 0
    max_bytes = 0
    last_seq: int | None = None
    last: dict | None = None
    t_window = t_last_rx = time.monotonic()

    # The packet rate is measured on the SENDER's clock (t_send_ns), not on
    # this machine's: WSL2's clock can run measurably fast or slow relative to
    # Windows, and a rate measured here would then be wrong by that much.
    anchor_send_ns: int | None = None
    n_since_anchor = 0
    rate = 0.0
    first_pair: tuple[int, int] | None = None  # (t_send_ns, local monotonic_ns)
    skew_reported = False

    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                data = None
            now = time.monotonic()

            if data is not None:
                t_last_rx = now
                received += 1
                max_bytes = max(max_bytes, len(data))
                try:
                    packet = decode(data)
                    validate(packet)
                except PacketError as exc:
                    invalid += 1
                    print(f"INVALID packet from {addr[0]} ({len(data)} B): {exc}")
                    packet = None

                if packet is not None:
                    seq = packet["seq"]
                    if last_seq is not None:
                        if seq > last_seq + 1:
                            lost += seq - last_seq - 1
                        elif seq <= last_seq:
                            print(f"seq went {last_seq} -> {seq}: sender restarted (or reordering)")
                    last_seq = seq
                    last = packet
                    if anchor_send_ns is None:
                        anchor_send_ns = packet["t_send_ns"]
                    else:
                        n_since_anchor += 1
                    if args.verbose:
                        print(json.dumps(packet))

                    # Clock check: after 10 s of sender time, compare how much
                    # time passed here. One warning, because it changes how the
                    # ROS side must timestamp (streaming/SCHEMA.md, "Clocks").
                    if first_pair is None:
                        first_pair = (packet["t_send_ns"], time.monotonic_ns())
                    elif not skew_reported:
                        sender_s = (packet["t_send_ns"] - first_pair[0]) / 1e9
                        local_s = (time.monotonic_ns() - first_pair[1]) / 1e9
                        if sender_s >= 10.0:
                            skew_reported = True
                            ratio = local_s / sender_s
                            if abs(ratio - 1.0) > 0.01:
                                print(
                                    f"WARNING: {sender_s:.1f} s of tracker time took {local_s:.2f} s "
                                    f"here -- this clock runs {abs(ratio - 1) * 100:.1f}% "
                                    f"{'slow' if ratio < 1 else 'fast'} relative to the tracker's.\n"
                                    "  Never compare t_capture_ns/t_send_ns with local time; see "
                                    "streaming/SCHEMA.md, 'Clocks'."
                                )

                if args.count and received >= args.count:
                    break

            if now - t_window >= 1.0:
                if last is not None:
                    if n_since_anchor and last["t_send_ns"] > anchor_send_ns:
                        rate = n_since_anchor / ((last["t_send_ns"] - anchor_send_ns) / 1e9)
                        anchor_send_ns, n_since_anchor = last["t_send_ns"], 0
                    tip = tip_in_reference(last)
                    tip_text = (
                        f"tip_ref [{tip[0]:7.2f} {tip[1]:7.2f} {tip[2]:7.2f}] mm" if tip else "tip_ref --"
                    )
                    trk_ms = (last["t_send_ns"] - last["t_capture_ns"]) / 1e6
                    print(
                        f"{rate:5.1f} pkt/s | seq {last_seq} | lost {lost} | "
                        f"invalid {invalid} | max {max_bytes} B | tool {last['tool']['status']} | "
                        f"ref {last['reference']['status']} | {tip_text} | capture->send {trk_ms:.1f} ms"
                    )
                t_window = now

            if args.timeout and now - t_last_rx > args.timeout:
                print(f"No packets for {args.timeout:g} s -- stopping.")
                if received == 0:
                    print("Nothing arrived at all: check --dest on the sender and the WSL "
                          "networking mode (see README, 'Streaming to ROS 2').")
                    return 2
                break
    except KeyboardInterrupt:
        print()
    finally:
        sock.close()

    print(f"Received {received} packets: {invalid} invalid, {lost} lost (seq gaps), "
          f"largest {max_bytes} bytes.")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
