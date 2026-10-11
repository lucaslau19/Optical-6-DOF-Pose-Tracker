"""Stream each frame's tracking result to ROS 2 as one UDP datagram.

`packet` is the wire contract (pure, standard library only), `sender` puts
packets on the network, `live` wires both into the camera loop. The schema is
documented in streaming/SCHEMA.md.

Deliberately empty of imports: `streaming.packet` must stay importable by a
bare python3 under WSL (see scripts/udp_listen.py), which has no OpenCV.
"""
