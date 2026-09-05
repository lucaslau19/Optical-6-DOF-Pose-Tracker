"""Digitised points: record where the tip touched, in the reference frame.

Touching a point with a calibrated tip and recording its coordinates is what
"digitising" means in navigation -- it is how anatomical landmarks get
registered. Here it doubles as the project's first end-to-end accuracy check:
digitise two marks a known distance apart, and the distance between the
recorded points can be compared against a ruler. That number has the whole
chain in it -- camera calibration, both rigid-body fits, the tip calibration
and the reference transform -- which is exactly what makes it meaningful.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass
class DigitizedPoint:
    p_ref: np.ndarray  # (3,) in the REFERENCE frame, mm
    label: str
    tool_markers: int
    reference_markers: int

    def __post_init__(self) -> None:
        self.p_ref = np.asarray(self.p_ref, dtype=float).reshape(3)


@dataclass
class PointLog:
    """An ordered list of digitised points, with pairwise distances."""

    points: list[DigitizedPoint] = field(default_factory=list)

    def add(
        self, p_ref: np.ndarray, tool_markers: int, reference_markers: int
    ) -> DigitizedPoint:
        point = DigitizedPoint(
            p_ref=p_ref,
            label=f"P{len(self.points) + 1}",
            tool_markers=tool_markers,
            reference_markers=reference_markers,
        )
        self.points.append(point)
        return point

    def undo(self) -> DigitizedPoint | None:
        return self.points.pop() if self.points else None

    @property
    def last_gap_mm(self) -> float | None:
        """Distance between the last two points -- the ruler check, live."""
        if len(self.points) < 2:
            return None
        return float(np.linalg.norm(self.points[-1].p_ref - self.points[-2].p_ref))

    def pairwise_summary(self) -> str:
        if len(self.points) < 2:
            return "  (digitise at least two points to get a distance)"
        lines = []
        for i in range(len(self.points)):
            for j in range(i + 1, len(self.points)):
                d = float(np.linalg.norm(self.points[j].p_ref - self.points[i].p_ref))
                lines.append(
                    f"  {self.points[i].label} -> {self.points[j].label} : {d:8.2f} mm"
                )
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "frame": "reference",
            "units": "mm",
            "saved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "points": [
                {
                    "label": p.label,
                    "p_ref_mm": [round(float(v), 4) for v in p.p_ref],
                    "tool_markers": p.tool_markers,
                    "reference_markers": p.reference_markers,
                }
                for p in self.points
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)
        return path
