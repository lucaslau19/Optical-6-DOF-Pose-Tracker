"""Storage for a solved tip offset.

Written to its own YAML file rather than back into config.yaml, for two
reasons: config.yaml is hand-edited and a program that rewrites it will
eventually eat someone's comments, and a calibration is a *measurement*, so it
belongs in a file that carries its own provenance -- when it was taken, from
how many frames, with what residual, against which tool geometry.

That last one is load-bearing. See ToolGeometry.fingerprint.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class TipCalibration:
    """A tip offset in the tool frame, with the evidence behind it."""

    p_tip: np.ndarray  # (3,) mm, in the TOOL frame
    tool_name: str
    tool_fingerprint: str
    residual_rms_mm: float
    residual_max_mm: float
    n_frames: int
    condition_number: float
    cone_half_angle_deg: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.p_tip = np.asarray(self.p_tip, dtype=float).reshape(3)

    @property
    def distance_mm(self) -> float:
        return float(np.linalg.norm(self.p_tip))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "p_tip_mm": self.p_tip.tolist(),
            "tool_name": self.tool_name,
            "tool_fingerprint": self.tool_fingerprint,
            "residual_rms_mm": float(self.residual_rms_mm),
            "residual_max_mm": float(self.residual_max_mm),
            "n_frames": int(self.n_frames),
            "condition_number": float(self.condition_number),
            "cone_half_angle_deg": float(self.cone_half_angle_deg),
            "frame": "tool",
            "units": "mm",
            "metadata": {
                "saved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                **self.metadata,
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)
        return path

    @staticmethod
    def load(path: str | Path) -> "TipCalibration":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"No tip calibration at {path}.\nRun `python app.py pivot` first."
            )
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        return TipCalibration(
            p_tip=np.array(doc["p_tip_mm"], dtype=float),
            tool_name=str(doc.get("tool_name", "")),
            tool_fingerprint=str(doc.get("tool_fingerprint", "")),
            residual_rms_mm=float(doc.get("residual_rms_mm", float("nan"))),
            residual_max_mm=float(doc.get("residual_max_mm", float("nan"))),
            n_frames=int(doc.get("n_frames", 0)),
            condition_number=float(doc.get("condition_number", float("nan"))),
            cone_half_angle_deg=float(doc.get("cone_half_angle_deg", float("nan"))),
            metadata=doc.get("metadata") or {},
        )

    def check_matches(self, geometry) -> str | None:
        """Return a warning string if this tip does not belong to `geometry`."""
        if not self.tool_fingerprint:
            return None
        current = geometry.fingerprint()
        if current != self.tool_fingerprint:
            return (
                f"Tip calibration was measured against tool geometry "
                f"{self.tool_fingerprint}, but config.yaml now describes "
                f"{current}.\n"
                "  The tool frame has moved since, so this tip offset points at the\n"
                "  wrong place on the instrument. Re-run `python app.py pivot`."
            )
        return None

    def summary(self) -> str:
        return (
            f"  tip (tool frame): [{self.p_tip[0]:8.2f} {self.p_tip[1]:8.2f} "
            f"{self.p_tip[2]:8.2f}] mm  (|tip| = {self.distance_mm:.2f} mm)\n"
            f"  measured from   : {self.n_frames} frames, residual RMS "
            f"{self.residual_rms_mm:.3f} mm, cone {self.cone_half_angle_deg:.0f} deg, "
            f"cond {self.condition_number:.1f}"
        )


def load_tip_if_available(cfg, geometry) -> tuple["TipCalibration | None", str | None]:
    """Load the tip for this tool, if one has been measured.

    Returns (tip, warning). A missing file is not an error -- `track-tool`
    works fine without a tip, it just cannot draw one -- so this returns None
    rather than raising, and the caller decides what to say.
    """
    path = cfg.tool.tip_calibration_file if cfg.tool else None
    if path is None or not Path(path).is_file():
        return None, None
    tip = TipCalibration.load(path)
    return tip, tip.check_matches(geometry)
