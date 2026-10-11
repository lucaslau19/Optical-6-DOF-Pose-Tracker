r"""The golden example packets in streaming/golden/, and the scene behind them.

The three golden files are shared with the ROS repo, which tests its parser
against exactly these files. They are built here from fixed inputs through the
same `build_packet` the tracker uses, and tests/test_packet.py fails if the
committed files stop matching -- so a code change that would alter what ROS
receives shows up as a failing test in this repo, not as a surprise on the
other side.

    tracking.json        both bodies tracked, tip calibrated: the normal case
    reference_lost.json  reference hidden: reference.T_cam is null, tool still tracked
    no_tip.json          no tip calibration (p_tip_tool_mm null), plus the edge
                         cases a parser must not trip on: seq 0, a
                         low-confidence ambiguous tool, and a TRACKED reference
                         whose ambiguity is null

Regenerate only when the schema deliberately changes, then copy the files to
the ROS repo:

    .venv\Scripts\python.exe -m streaming.examples
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from core.transforms import SE3
from streaming.packet import build_packet, encode, validate
from tracking.rigid_body import ToolPose

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
GOLDEN_NAMES = ("tracking", "reference_lost", "no_tip")

# --- the scene ---------------------------------------------------------------
#
# Shared with tools/fake_stream.py, so the golden files and the fake stream
# describe the same layout. The reference sheet lies on the bench ~46 cm in
# front of the camera, which looks down on it at ~40 degrees. The tool is held
# above it with its tip pointing down towards the bench and its marker sheet
# turned towards the camera.

T_CAM_REF = SE3(
    Rotation.from_euler("xyz", [-40.0, 3.0, 5.0], degrees=True).as_matrix(),
    [15.0, 10.0, 460.0],
)

# Calibrated tip offset in the TOOL frame, mm: ~99 mm down the sheet's +Y from
# the marker centroid, i.e. a pointer whose tip sticks out below the sheet.
P_TIP_TOOL = np.array([1.25, 98.6, -0.75])

TOOL_IDS = [10, 11, 12, 13]
REFERENCE_IDS = [20, 21, 22, 23]

# Tilting the tool frame +78 deg about the reference X axis points its +Y (the
# tip direction) down into the bench while its sheet still faces the camera.
TOOL_BASE_TILT_DEG = 78.0


def tool_pose_for_tip(R_ref_tool: np.ndarray, p_tip_ref: np.ndarray) -> SE3:
    """T_ref_tool with orientation R_ref_tool that puts the tip at p_tip_ref.

    From p_tip_ref = R_ref_tool @ P_TIP_TOOL + t_ref_tool.
    """
    R = np.asarray(R_ref_tool, dtype=float)
    return SE3(R, np.asarray(p_tip_ref, dtype=float) - R @ P_TIP_TOOL)


def _tool_in_camera(tilt_xyz_deg: list[float], p_tip_ref: list[float]) -> SE3:
    R_ref_tool = Rotation.from_euler("xyz", tilt_xyz_deg, degrees=True).as_matrix()
    return T_CAM_REF @ tool_pose_for_tip(R_ref_tool, np.array(p_tip_ref))


# --- the three examples --------------------------------------------------------

_T0_NS = 1_791_651_600_123_456_789  # an arbitrary wall-clock instant
_FRAME_NS = 33_333_333  # one frame at 30 fps
_LATENCY_NS = 2_350_000  # capture -> send, a typical tracking time


def example_packets() -> dict[str, dict[str, Any]]:
    """Fresh copies of the three golden packets, keyed by file stem."""
    tool = ToolPose(
        pose=_tool_in_camera([TOOL_BASE_TILT_DEG, 6.0, -4.0], [3.2, -1.5, -58.0]),
        status="tracking",
        marker_ids_used=list(TOOL_IDS),
        reprojection_rms_px=0.41,
        ambiguity_ratio=9.87,
        n_points=16,
    )
    reference = ToolPose(
        pose=T_CAM_REF,
        status="tracking",
        marker_ids_used=[20, 21, 23],  # one member covered: still tracking
        reprojection_rms_px=0.38,
        ambiguity_ratio=14.2,
        n_points=12,
    )
    reference_lost = ToolPose(pose=None, status="lost")

    tool_low_conf = ToolPose(
        pose=_tool_in_camera([70.0, -10.0, 8.0], [-20.0, 15.0, -75.0]),
        status="low_confidence",
        marker_ids_used=[12],
        reprojection_rms_px=0.73,
        ambiguity_ratio=1.82,  # < 3: the flag comes out true
        n_points=4,
    )
    reference_no_second_solution = ToolPose(
        pose=T_CAM_REF,
        status="tracking",
        marker_ids_used=list(REFERENCE_IDS),
        reprojection_rms_px=0.35,
        # What the tracker reports for a non-coplanar body: no planar twin to
        # compare against, so the ratio is inf -> null. NOT lost.
        ambiguity_ratio=float("inf"),
        n_points=16,
    )

    t1 = _T0_NS + _FRAME_NS
    return {
        "tracking": build_packet(
            1200, _T0_NS, tool, reference, P_TIP_TOOL, t_send_ns=_T0_NS + _LATENCY_NS
        ),
        "reference_lost": build_packet(
            1201, t1, tool, reference_lost, P_TIP_TOOL, t_send_ns=t1 + _LATENCY_NS
        ),
        "no_tip": build_packet(
            0,
            _T0_NS - 60 * 1_000_000_000,
            tool_low_conf,
            reference_no_second_solution,
            None,
            t_send_ns=_T0_NS - 60 * 1_000_000_000 + _LATENCY_NS,
        ),
    }


def write_golden(directory: Path = GOLDEN_DIR) -> list[Path]:
    """Validate and write the examples as indented JSON (whitespace is irrelevant to parsers)."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, packet in example_packets().items():
        validate(packet)
        path = directory / f"{name}.json"
        path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for path in write_golden():
        packet = json.loads(path.read_text(encoding="utf-8"))
        print(f"wrote {path}  ({len(encode(packet))} bytes on the wire)")
