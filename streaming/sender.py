"""UDP sender for tracking packets.

Fire-and-forget, on purpose. A navigation stream wants the NEWEST pose, not
every pose: a datagram that is late is worthless, so there is no
retransmission, no connection and no queue. The receiver detects loss from
gaps in `seq`.

The one hard rule: nothing that goes wrong here may stop the tracking loop.
A ROS driver that is not running, a WSL VM that is restarting, a full socket
buffer -- all of these are counted and reported, never raised. The camera
loop keeps tracking (and the navigation HUD keeps working) whether or not
anyone is listening.
"""

from __future__ import annotations

import socket
import time
from typing import Any

from streaming.packet import PacketError, encode, validate

DEFAULT_DEST = "127.0.0.1:5005"


def parse_dest(text: str) -> tuple[str, int]:
    """'HOST:PORT' -> (host, port). Accepts '[::1]:5005' for IPv6 literals."""
    host, sep, port_text = text.strip().rpartition(":")
    if not sep or not host:
        raise ValueError(f"--dest must be HOST:PORT (e.g. {DEFAULT_DEST}), got {text!r}")
    host = host.strip("[]")
    try:
        port = int(port_text)
    except ValueError:
        raise ValueError(f"--dest port must be a number, got {port_text!r}") from None
    if not 1 <= port <= 65535:
        raise ValueError(f"--dest port must be 1-65535, got {port}")
    return host, port


class UdpSender:
    """Validates, timestamps and sends packets; counts what happened."""

    def __init__(self, host: str, port: int) -> None:
        # Resolve once, up front. A name lookup per packet would put a DNS
        # round trip inside the frame loop, and a bad host should fail at
        # startup with a clear message, not as 30 errors a second.
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)
        except socket.gaierror as exc:
            raise ValueError(f"cannot resolve stream destination {host!r}: {exc}") from None
        family, socktype, proto, _canon, addr = infos[0]

        self.host = host
        self.port = port
        self._addr = addr
        self._sock = socket.socket(family, socktype, proto)
        # Non-blocking: if the OS send buffer is ever full, drop this packet
        # (counted as an error) rather than stall the camera loop waiting.
        self._sock.setblocking(False)

        self.sent = 0
        self.errors = 0
        self.last_error: str | None = None
        self.last_bytes = 0
        self.max_bytes = 0

    @property
    def dest_label(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{host}:{self.port}"

    def send(self, packet: dict[str, Any]) -> bool:
        """Send one packet. Returns False (and counts an error) instead of raising.

        The packet is validated first, so the tracker can never put a
        schema-violating datagram on the wire. `t_send_ns` is stamped after
        validation and immediately before encoding and the socket call -- the
        gap to the actual send is the JSON encode, tens of microseconds.
        """
        try:
            validate(packet)
            wire = dict(packet)  # do not mutate the caller's packet
            wire["t_send_ns"] = time.time_ns()
            data = encode(wire)
            self._sock.sendto(data, self._addr)
        except (PacketError, OSError) as exc:
            # OSError covers everything the socket can do to us: no route,
            # buffer full (BlockingIOError), ICMP port-unreachable surfacing
            # as a connection reset on Windows, a closed socket.
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

        self.sent += 1
        self.last_bytes = len(data)
        self.max_bytes = max(self.max_bytes, self.last_bytes)
        return True

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> "UdpSender":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
