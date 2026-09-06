"""Aggregate the latest of each metric into one citable paragraph.

The paragraph is written to be pasted straight into the README's Results
section, which is why it names its conditions inline: a bare "0.8 mm" invites
the reader to assume the most flattering interpretation, and stating the range
and resolution alongside it removes that.
"""

from __future__ import annotations

from pathlib import Path

from accuracy.session import load_latest_summaries
from core.config import Config

_METRIC_ORDER = ["jitter", "repeatability", "distance", "registration"]

_METRIC_LABEL = {
    "jitter": ("Static jitter (precision)", "tip_ref_rms_3d_mm", "mm RMS"),
    "repeatability": ("Touch repeatability (precision)", "rms_3d_mm", "mm RMS"),
    "distance": ("Distance error (trueness)", "rms_absolute_error_mm", "mm RMS"),
    "registration": ("Registration residual (trueness)", "rms_mm", "mm RMS"),
}


def run_headline(cfg: Config) -> int:
    results_dir = Path(cfg.accuracy.results_dir)
    latest = load_latest_summaries(results_dir)

    print("\nAccuracy headline")
    print("=" * 62)
    if not latest:
        print(f"  No results found in {results_dir}.")
        print("  Run the measurements first:")
        for m in _METRIC_ORDER:
            print(f"    python app.py accuracy {m}")
        return 1

    print(f"  source: {results_dir}\n")
    print(f"  {'metric':<34} {'value':>12}   measured")
    print("  " + "-" * 60)

    values: dict[str, float] = {}
    all_conditions: list[dict] = []
    for metric in _METRIC_ORDER:
        doc = latest.get(metric)
        label, key, unit = _METRIC_LABEL[metric]
        if doc is None:
            print(f"  {label:<34} {'not measured':>12}")
            continue
        summary = doc.get("summary") or {}
        value = summary.get(key)
        when = (doc.get("conditions") or {}).get("timestamp_local", "?")
        if doc.get("conditions"):
            all_conditions.append(doc["conditions"])
        if value is None:
            print(f"  {label:<34} {'n/a':>12}   {when}")
            continue
        values[metric] = float(value)
        print(f"  {label:<34} {float(value):9.3f} {unit.split()[0]:<2}   {when}")

    conditions = _merge_conditions(all_conditions)

    missing = [m for m in _METRIC_ORDER if m not in values]
    print()
    if conditions:
        print(_conditions_sentence(conditions))
        print()
    print(_paragraph(values, conditions))
    if missing:
        print()
        print(f"  (missing: {', '.join(missing)} -- run those for a complete headline)")
    return 0


def _merge_conditions(all_conditions: list[dict]) -> dict | None:
    """One conditions record describing the whole characterisation.

    The metrics are separate runs, so the headline needs a summary of all of
    them rather than whichever happened to be processed last. Static fields
    (resolution, marker size) are taken from the most recent run; the working
    distance is widened to span every run, because the honest statement is the
    range over which the quoted numbers were actually measured.
    """
    if not all_conditions:
        return None
    merged = dict(all_conditions[-1])

    lows = [c["working_distance_min_mm"] for c in all_conditions
            if c.get("working_distance_min_mm") is not None]
    highs = [c["working_distance_max_mm"] for c in all_conditions
             if c.get("working_distance_max_mm") is not None]
    merged["working_distance_min_mm"] = min(lows) if lows else None
    merged["working_distance_max_mm"] = max(highs) if highs else None

    # Prefer a real lighting note over the placeholder, whichever run has one.
    notes = [c.get("lighting_note") for c in all_conditions
             if c.get("lighting_note") and c.get("lighting_note") != "not recorded"]
    if notes:
        merged["lighting_note"] = notes[-1]
    return merged


def _conditions_sentence(c: dict) -> str:
    bits = [f"  Conditions: {c.get('camera_resolution', '?')} webcam"]
    lo, hi = c.get("working_distance_min_mm"), c.get("working_distance_max_mm")
    if lo is not None and hi is not None:
        bits.append(f"at {lo:.0f}-{hi:.0f} mm working distance")
    if c.get("tool_marker_length_mm"):
        bits.append(f"{c['tool_marker_length_mm']:.0f} mm markers")
    if c.get("calibration_rms_px") is not None:
        bits.append(f"calibration {c['calibration_rms_px']:.3f} px RMS")
    if c.get("tip_residual_rms_mm") is not None:
        bits.append(f"pivot residual {c['tip_residual_rms_mm']:.2f} mm")
    lighting = c.get("lighting_note")
    if lighting and lighting != "not recorded":
        bits.append(lighting)
    return ", ".join(bits) + "."


def _paragraph(values: dict[str, float], conditions: dict | None) -> str:
    """The sentence to paste into the README."""
    if not values:
        return "  (no metrics available yet)"

    parts: list[str] = []
    if "jitter" in values:
        parts.append(f"static jitter {values['jitter']:.2f} mm RMS")
    if "repeatability" in values:
        parts.append(f"touch repeatability {values['repeatability']:.2f} mm RMS")
    if "distance" in values:
        parts.append(f"distance error {values['distance']:.2f} mm RMS over known separations")
    if "registration" in values:
        parts.append(f"registration residual {values['registration']:.2f} mm RMS")

    sentence = (
        "  Measured performance: " + "; ".join(parts) + "."
    )

    lines = [sentence]

    # The interpretive line. Precision and trueness are different claims and
    # conflating them is the most common way an accuracy number misleads.
    precision = [values[k] for k in ("jitter", "repeatability") if k in values]
    trueness = [values[k] for k in ("distance", "registration") if k in values]
    if precision and trueness:
        lines.append(
            f"  Precision (repeatability, no ground truth) is "
            f"{min(precision):.2f}-{max(precision):.2f} mm; trueness (against known"
        )
        lines.append(
            f"  geometry) is {min(trueness):.2f}-{max(trueness):.2f} mm. Quote the trueness"
            " figure as the accuracy --"
        )
        lines.append(
            "  precision alone would be flattered by any systematic error in the chain."
        )

    if "jitter" in values and "repeatability" in values:
        if values["repeatability"] <= values["jitter"]:
            lines.append(
                "  NOTE: touch repeatability is not worse than static jitter, which is"
            )
            lines.append(
                "  suspicious -- the jitter run may not have been genuinely static."
            )
    return "\n".join(lines)
