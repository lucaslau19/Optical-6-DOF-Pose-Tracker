"""Typed access to `config.yaml`.

Every geometric constant in this project comes from one file so that a
mis-measured board is a one-line fix rather than a hunt through the source.
The dataclasses below exist so that a typo in the YAML fails loudly at load
time instead of silently producing `None` three modules later.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repository root = parent of the `core` package. All relative paths in
# config.yaml are resolved against this, so the CLI behaves the same no
# matter which directory it is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent


def _require(d: dict[str, Any], key: str, section: str) -> Any:
    if key not in d:
        raise KeyError(f"config.yaml: missing required key '{section}.{key}'")
    return d[key]


@dataclass
class CameraConfig:
    device_index: int
    frame_width: int
    frame_height: int
    fps: int
    intrinsics_file: Path

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CameraConfig":
        return CameraConfig(
            device_index=int(_require(d, "device_index", "camera")),
            frame_width=int(_require(d, "frame_width", "camera")),
            frame_height=int(_require(d, "frame_height", "camera")),
            fps=int(d.get("fps", 30)),
            intrinsics_file=resolve_path(_require(d, "intrinsics_file", "camera")),
        )


@dataclass
class ArucoConfig:
    dictionary: str
    corner_refinement: str

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ArucoConfig":
        return ArucoConfig(
            dictionary=str(_require(d, "dictionary", "aruco")),
            corner_refinement=str(d.get("corner_refinement", "CORNER_REFINE_SUBPIX")),
        )


@dataclass
class CharucoConfig:
    squares_x: int
    squares_y: int
    square_length_mm: float
    marker_length_mm: float
    render_dpi: int
    margin_mm: float

    def __post_init__(self) -> None:
        # The markers sit inside the white squares, so a marker at least as
        # large as a square is geometrically impossible and OpenCV will
        # produce an unusable board rather than complain.
        if self.marker_length_mm >= self.square_length_mm:
            raise ValueError(
                "config.yaml: charuco.marker_length_mm must be smaller than "
                f"charuco.square_length_mm (got {self.marker_length_mm} >= "
                f"{self.square_length_mm})"
            )

    @property
    def board_width_mm(self) -> float:
        return self.squares_x * self.square_length_mm

    @property
    def board_height_mm(self) -> float:
        return self.squares_y * self.square_length_mm

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CharucoConfig":
        return CharucoConfig(
            squares_x=int(_require(d, "squares_x", "charuco")),
            squares_y=int(_require(d, "squares_y", "charuco")),
            square_length_mm=float(_require(d, "square_length_mm", "charuco")),
            marker_length_mm=float(_require(d, "marker_length_mm", "charuco")),
            render_dpi=int(d.get("render_dpi", 300)),
            margin_mm=float(d.get("margin_mm", 10.0)),
        )


@dataclass
class MarkersConfig:
    marker_length_mm: float
    ids: list[int]
    render_dpi: int
    margin_mm: float

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "MarkersConfig":
        return MarkersConfig(
            marker_length_mm=float(_require(d, "marker_length_mm", "markers")),
            ids=[int(i) for i in d.get("ids", [0])],
            render_dpi=int(d.get("render_dpi", 300)),
            margin_mm=float(d.get("margin_mm", 10.0)),
        )


@dataclass
class CalibrationConfig:
    min_corners_per_view: int
    min_views: int
    images_dir: Path
    outlier_rms_px: float

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CalibrationConfig":
        return CalibrationConfig(
            min_corners_per_view=int(d.get("min_corners_per_view", 12)),
            min_views=int(d.get("min_views", 12)),
            images_dir=resolve_path(d.get("images_dir", "output/calib_frames")),
            outlier_rms_px=float(d.get("outlier_rms_px", 1.0)),
        )


@dataclass
class ToolMemberConfig:
    """One marker of a rigid tool, as written in config.yaml."""

    marker_id: int
    x: float
    y: float
    z: float
    rotation_deg: float = 0.0


@dataclass
class ToolSheetConfig:
    render_dpi: int = 300
    margin_mm: float = 18.0


@dataclass
class ToolConfig:
    name: str
    marker_length_mm: float
    members: list[ToolMemberConfig]
    min_markers: int = 1
    low_confidence_markers: int = 1
    max_reprojection_rms_px: float = 2.0
    sheet: ToolSheetConfig = field(default_factory=ToolSheetConfig)

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("config.yaml: tool.markers is empty -- a tool needs at least one marker")
        if self.marker_length_mm <= 0:
            raise ValueError("config.yaml: tool.marker_length_mm must be positive")

        # A repeated ID is not a harmless typo: the detector reports each ID
        # once, so a duplicate silently means one of the two positions is never
        # used and the fit is quietly biased by the wrong geometry.
        counts = Counter(m.marker_id for m in self.members)
        duplicates = sorted(marker_id for marker_id, n in counts.items() if n > 1)
        if duplicates:
            raise ValueError(f"config.yaml: tool.markers has duplicate id(s) {duplicates}")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ToolConfig | None":
        # An absent or empty `tool:` section is legitimate -- Phase 1 commands
        # do not need one. The tool commands raise their own clear error.
        if not d:
            return None

        members: list[ToolMemberConfig] = []
        for i, raw in enumerate(d.get("markers") or []):
            if "id" not in raw:
                raise KeyError(f"config.yaml: tool.markers[{i}] is missing 'id'")
            members.append(
                ToolMemberConfig(
                    marker_id=int(raw["id"]),
                    x=float(_require(raw, "x", f"tool.markers[{i}]")),
                    y=float(_require(raw, "y", f"tool.markers[{i}]")),
                    z=float(raw.get("z", 0.0)),
                    rotation_deg=float(raw.get("rotation_deg", 0.0)),
                )
            )

        sheet_raw = d.get("sheet") or {}
        return ToolConfig(
            name=str(d.get("name", "tool")),
            marker_length_mm=float(_require(d, "marker_length_mm", "tool")),
            members=members,
            min_markers=int(d.get("min_markers", 1)),
            low_confidence_markers=int(d.get("low_confidence_markers", 1)),
            max_reprojection_rms_px=float(d.get("max_reprojection_rms_px", 2.0)),
            sheet=ToolSheetConfig(
                render_dpi=int(sheet_raw.get("render_dpi", 300)),
                margin_mm=float(sheet_raw.get("margin_mm", 18.0)),
            ),
        )


@dataclass
class Config:
    camera: CameraConfig
    aruco: ArucoConfig
    charuco: CharucoConfig
    markers: MarkersConfig
    calibration: CalibrationConfig
    # None when config.yaml has no `tool:` section -- Phase 1 commands do not
    # need one, and the tool commands raise their own explanatory error.
    tool: ToolConfig | None = None
    # Phases 3-4 are not parsed into dataclasses yet; keep the raw dicts so
    # the file round-trips and nothing is silently dropped.
    navigation: dict[str, Any] = field(default_factory=dict)
    tolerances: dict[str, Any] = field(default_factory=dict)

    source_path: Path = REPO_ROOT / "config.yaml"


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the repo root (absolute paths pass through)."""
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_config(path: str | Path | None = None) -> Config:
    """Read and validate `config.yaml`."""
    cfg_path = Path(path) if path is not None else (REPO_ROOT / "config.yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return Config(
        camera=CameraConfig.from_dict(raw.get("camera", {})),
        aruco=ArucoConfig.from_dict(raw.get("aruco", {})),
        charuco=CharucoConfig.from_dict(raw.get("charuco", {})),
        markers=MarkersConfig.from_dict(raw.get("markers", {})),
        calibration=CalibrationConfig.from_dict(raw.get("calibration", {})),
        tool=ToolConfig.from_dict(raw.get("tool") or {}),
        navigation=raw.get("navigation") or {},
        tolerances=raw.get("tolerances") or {},
        source_path=cfg_path,
    )
