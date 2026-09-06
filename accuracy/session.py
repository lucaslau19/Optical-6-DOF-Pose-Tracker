"""Results recording: raw samples, summaries, and the conditions behind them.

A number without its conditions is not a result. "0.8 mm RMS" means nothing on
its own -- at what range, what resolution, what marker size, what lighting,
against which camera calibration? So every run writes:

    results/<timestamp>_<metric>.csv    every raw sample, one per row
    results/<timestamp>_<metric>.yaml   summary statistics + full conditions

Raw samples are written, not just summaries, so a result can be re-analysed
later (or challenged) without repeating the physical measurement -- the same
reasoning that keeps the ChArUco frames in Phase 1 and the pivot poses in
Phase 3.
"""

from __future__ import annotations

import csv
import datetime as _dt
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import yaml


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


@dataclass
class Conditions:
    """Everything needed to interpret a measurement made under it."""

    metric: str
    timestamp_local: str
    camera_resolution: str
    camera_index: int
    intrinsics_file: str
    calibration_rms_px: float
    calibration_views: int | None
    aruco_dictionary: str
    corner_refinement: str
    tool_name: str
    tool_marker_length_mm: float
    tool_marker_ids: list[int]
    tool_coplanar: bool
    tool_fingerprint: str
    reference_name: str | None = None
    reference_marker_ids: list[int] | None = None
    tip_offset_mm: list[float] | None = None
    tip_residual_rms_mm: float | None = None
    working_distance_min_mm: float | None = None
    working_distance_max_mm: float | None = None
    lighting_note: str = "not recorded"
    opencv_version: str = field(default_factory=lambda: cv2.__version__)
    platform: str = field(default_factory=lambda: platform.platform())
    notes: str = ""

    def format(self) -> str:
        lines = [
            "  Conditions",
            f"    camera         : index {self.camera_index}, {self.camera_resolution}",
            f"    calibration    : {self.calibration_rms_px:.3f} px reprojection RMS"
            + (f" from {self.calibration_views} views" if self.calibration_views else ""),
            f"    markers        : {self.aruco_dictionary}, "
            f"{self.tool_marker_length_mm:.1f} mm, {self.corner_refinement}",
            f"    tool           : '{self.tool_name}' ids {self.tool_marker_ids}"
            f" ({'coplanar' if self.tool_coplanar else 'non-coplanar'})",
        ]
        if self.reference_name:
            lines.append(
                f"    reference      : '{self.reference_name}' ids {self.reference_marker_ids}"
            )
        if self.tip_offset_mm is not None:
            tip = np.asarray(self.tip_offset_mm)
            lines.append(
                f"    tip offset     : [{tip[0]:.2f} {tip[1]:.2f} {tip[2]:.2f}] mm"
                + (
                    f", pivot residual {self.tip_residual_rms_mm:.3f} mm"
                    if self.tip_residual_rms_mm is not None
                    else ""
                )
            )
        if self.working_distance_min_mm is not None:
            lines.append(
                f"    working range  : {self.working_distance_min_mm:.0f} - "
                f"{self.working_distance_max_mm:.0f} mm (measured during this run)"
            )
        lines.append(f"    lighting       : {self.lighting_note}")
        return "\n".join(lines)


def build_conditions(
    cfg,
    metric: str,
    intrinsics,
    tool_geometry,
    *,
    reference_geometry=None,
    tip=None,
    lighting_note: str | None = None,
    notes: str = "",
) -> Conditions:
    """Snapshot the setup at the moment a measurement is taken."""
    meta = intrinsics.metadata or {}
    return Conditions(
        metric=metric,
        timestamp_local=_dt.datetime.now().isoformat(timespec="seconds"),
        camera_resolution=f"{intrinsics.image_size[0]}x{intrinsics.image_size[1]}",
        camera_index=cfg.camera.device_index,
        intrinsics_file=str(cfg.camera.intrinsics_file),
        calibration_rms_px=float(intrinsics.reprojection_rms_px),
        calibration_views=meta.get("n_views"),
        aruco_dictionary=cfg.aruco.dictionary,
        corner_refinement=cfg.aruco.corner_refinement,
        tool_name=tool_geometry.name,
        tool_marker_length_mm=tool_geometry.marker_length_mm,
        tool_marker_ids=list(tool_geometry.ids),
        tool_coplanar=bool(tool_geometry.is_coplanar()),
        tool_fingerprint=tool_geometry.fingerprint(),
        reference_name=reference_geometry.name if reference_geometry else None,
        reference_marker_ids=list(reference_geometry.ids) if reference_geometry else None,
        tip_offset_mm=[float(v) for v in tip.p_tip] if tip is not None else None,
        tip_residual_rms_mm=float(tip.residual_rms_mm) if tip is not None else None,
        lighting_note=lighting_note or cfg.accuracy.lighting_note,
        notes=notes,
    )


class ResultsWriter:
    """Writes one metric run: a CSV of raw samples and a YAML summary."""

    def __init__(self, results_dir: Path, metric: str) -> None:
        self.dir = Path(results_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.stamp = _now_stamp()
        self.base = self.dir / f"{self.stamp}_{metric}"

    @property
    def csv_path(self) -> Path:
        return self.base.with_suffix(".csv")

    @property
    def summary_path(self) -> Path:
        return self.base.with_suffix(".yaml")

    def write_samples(self, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> Path:
        with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for row in rows:
                writer.writerow(
                    [f"{v:.6f}" if isinstance(v, (float, np.floating)) else v for v in row]
                )
        return self.csv_path

    def write_summary(
        self, conditions: Conditions, summary: dict[str, Any], verdict: str
    ) -> Path:
        doc = {
            "metric": self.metric,
            "units": "mm and degrees",
            "verdict": verdict,
            "summary": _plain(summary),
            "conditions": _plain(asdict(conditions)),
            "raw_samples_csv": self.csv_path.name,
        }
        with open(self.summary_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)
        return self.summary_path

    def report_paths(self) -> str:
        return f"  raw samples -> {self.csv_path}\n  summary     -> {self.summary_path}"


def _plain(obj: Any) -> Any:
    """Convert numpy types to plain Python so PyYAML emits readable scalars."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 6)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def load_latest_summaries(results_dir: Path) -> dict[str, dict]:
    """Newest summary for each metric, for the headline aggregation."""
    results_dir = Path(results_dir)
    latest: dict[str, dict] = {}
    if not results_dir.is_dir():
        return latest
    # Filenames start with a sortable timestamp, so lexical order is
    # chronological and the last one wins.
    for path in sorted(results_dir.glob("*.yaml")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
        except Exception:
            continue
        if not isinstance(doc, dict) or "metric" not in doc:
            continue
        doc["_path"] = path
        latest[doc["metric"]] = doc
    return latest
