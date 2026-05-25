"""ELAN .eaf writer for the target TUG tier structure."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import os
import xml.etree.ElementTree as ET

from .config import CRT_TIER_ORDER, TIER_ORDER, TUG_PHASE_LABELS, TUG_TIER_ORDER
from .types import AnnotationBundle, FootAnnotation, Interval, PhaseAnnotation

XSI = "http://www.w3.org/2001/XMLSchema-instance"


def write_eaf(*args: Any) -> Path:
    """Write one ELAN .eaf file.

    Supports the strict 3DGait AnnotationBundle API and the legacy dict API used
    by the existing batch annotator.
    """

    if len(args) != 3:
        raise TypeError("write_eaf expects exactly three arguments")
    if isinstance(args[0], AnnotationBundle):
        return _write_bundle_eaf(args[0], Path(args[1]), Path(args[2]))
    return _write_legacy_eaf(str(args[0]), args[1], str(args[2]))


def _write_legacy_eaf(video_path: str, annotations: dict[str, Any], output_path: str) -> Path:
    """Write one ELAN .eaf file linked to the source video."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    video = Path(video_path)

    configured_tiers = annotations.get("tier_order")
    if configured_tiers:
        active_tiers = tuple(str(tier) for tier in configured_tiers)
    else:
        active_tiers = CRT_TIER_ORDER if annotations.get("is_crt") else TIER_ORDER
    duration_ms = _coerce_duration_ms(annotations.get("duration_ms"))
    tier_intervals = {
        tier: _coerce_intervals(annotations.get(tier, []), duration_ms)
        for tier in active_tiers
    }
    times = sorted(
        {
            time
            for intervals in tier_intervals.values()
            for iv in intervals
            for time in (iv.start_ms, iv.end_ms)
            if iv.end_ms > iv.start_ms
        }
    )
    time_ids = {time: f"ts{idx + 1}" for idx, time in enumerate(times)}

    ET.register_namespace("xsi", XSI)
    root = ET.Element(
        "ANNOTATION_DOCUMENT",
        {
            "AUTHOR": "auto_annotate",
            "DATE": datetime.now(timezone.utc).isoformat(),
            "FORMAT": "3.0",
            "VERSION": "3.0",
            f"{{{XSI}}}noNamespaceSchemaLocation": "http://www.mpi.nl/tools/elan/EAFv3.0.xsd",
        },
    )

    header = ET.SubElement(root, "HEADER", {"MEDIA_FILE": "", "TIME_UNITS": "milliseconds"})
    ET.SubElement(
        header,
        "MEDIA_DESCRIPTOR",
        {
            "MEDIA_URL": _media_url(video),
            "MIME_TYPE": "video/mp4",
            "RELATIVE_MEDIA_URL": _relative_media_url(video, out.parent),
        },
    )
    ET.SubElement(header, "PROPERTY", {"NAME": "lastUsedAnnotationId"}).text = str(
        sum(len(v) for v in tier_intervals.values())
    )

    time_order = ET.SubElement(root, "TIME_ORDER")
    for time in times:
        ET.SubElement(
            time_order,
            "TIME_SLOT",
            {"TIME_SLOT_ID": time_ids[time], "TIME_VALUE": str(int(time))},
        )

    ann_id = 1
    for tier_id in active_tiers:
        tier = ET.SubElement(
            root,
            "TIER",
            {
                "LINGUISTIC_TYPE_REF": "default-lt",
                "TIER_ID": tier_id,
            },
        )
        for interval in tier_intervals[tier_id]:
            if interval.end_ms <= interval.start_ms:
                continue
            ann = ET.SubElement(tier, "ANNOTATION")
            alignable = ET.SubElement(
                ann,
                "ALIGNABLE_ANNOTATION",
                {
                    "ANNOTATION_ID": f"a{ann_id}",
                    "TIME_SLOT_REF1": time_ids[interval.start_ms],
                    "TIME_SLOT_REF2": time_ids[interval.end_ms],
                },
            )
            ET.SubElement(alignable, "ANNOTATION_VALUE").text = interval.label
            ann_id += 1

    ET.SubElement(
        root,
        "LINGUISTIC_TYPE",
        {
            "GRAPHIC_REFERENCES": "false",
            "LINGUISTIC_TYPE_ID": "default-lt",
            "TIME_ALIGNABLE": "true",
        },
    )
    ET.SubElement(root, "LOCALE", {"COUNTRY_CODE": "US", "LANGUAGE_CODE": "en"})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="UTF-8", xml_declaration=True)
    return out


def _write_bundle_eaf(bundle: AnnotationBundle, video_path: Path, output_path: Path) -> Path:
    """Write the strict four-tier TUG AnnotationBundle EAF."""

    _validate_bundle_bounds(bundle)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tier_intervals = {
        "test": [Interval(bundle.tug.start_ms, bundle.tug.end_ms, "time")],
        "phase": [_phase_to_interval(phase) for phase in bundle.phases],
        "left_foot": [_foot_to_interval(foot) for foot in bundle.left],
        "right_foot": [_foot_to_interval(foot) for foot in bundle.right],
    }
    return _write_tier_intervals(video_path, out, TUG_TIER_ORDER, tier_intervals)


def validate_eaf(eaf_path: Path) -> list[str]:
    """Validate the strict four-tier TUG EAF shape."""

    errors: list[str] = []
    try:
        tree = ET.parse(eaf_path)
    except Exception as exc:
        return [f"XML parse failed: {exc}"]
    root = tree.getroot()
    time_slots: dict[str, int] = {}
    previous_time: int | None = None
    for slot in root.findall(".//TIME_SLOT"):
        slot_id = slot.get("TIME_SLOT_ID")
        raw_time = slot.get("TIME_VALUE")
        if not slot_id:
            errors.append("TIME_SLOT without TIME_SLOT_ID")
            continue
        try:
            time_value = int(raw_time or "0")
        except ValueError:
            errors.append(f"TIME_SLOT {slot_id} has non-integer TIME_VALUE")
            continue
        if previous_time is not None and time_value <= previous_time:
            errors.append("TIME_SLOT entries are not in strict ascending order")
        previous_time = time_value
        time_slots[slot_id] = time_value

    tiers = {tier.get("TIER_ID"): tier for tier in root.findall(".//TIER")}
    for tier_id in TUG_TIER_ORDER:
        if tier_id not in tiers:
            errors.append(f"missing tier: {tier_id}")
        elif tiers[tier_id].get("LINGUISTIC_TYPE_REF") != "default-lt":
            errors.append(f"tier {tier_id} missing LINGUISTIC_TYPE_REF=default-lt")

    test_intervals = _tier_intervals_from_xml(tiers.get("test"), time_slots)
    if len(test_intervals) != 1:
        errors.append("test tier must contain exactly one annotation")
        tug_start, tug_end = 0, 0
    else:
        tug_start, tug_end, label = test_intervals[0]
        if label != "time":
            errors.append("test tier annotation label must be time")

    for tier_id, tier in tiers.items():
        if tier_id not in TUG_TIER_ORDER:
            continue
        intervals = _tier_intervals_from_xml(tier, time_slots)
        previous_end: int | None = None
        for start, end, _label in intervals:
            if end <= start:
                errors.append(f"tier {tier_id} contains a non-positive interval")
            if previous_end is not None and start < previous_end:
                errors.append(f"tier {tier_id} contains overlapping annotations")
            previous_end = end
            if test_intervals and (start < tug_start or end > tug_end):
                errors.append(f"tier {tier_id} contains annotation outside TUG interval")
    return errors


def _write_tier_intervals(
    video: Path,
    out: Path,
    active_tiers: tuple[str, ...],
    tier_intervals: dict[str, list[Interval]],
) -> Path:
    times = sorted(
        {
            time
            for intervals in tier_intervals.values()
            for iv in intervals
            for time in (iv.start_ms, iv.end_ms)
            if iv.end_ms > iv.start_ms
        }
    )
    time_ids = {time: f"ts{idx + 1}" for idx, time in enumerate(times)}

    ET.register_namespace("xsi", XSI)
    root = ET.Element(
        "ANNOTATION_DOCUMENT",
        {
            "AUTHOR": "auto_annotate",
            "DATE": datetime.now(timezone.utc).isoformat(),
            "FORMAT": "3.0",
            "VERSION": "3.0",
            f"{{{XSI}}}noNamespaceSchemaLocation": "http://www.mpi.nl/tools/elan/EAFv3.0.xsd",
        },
    )
    header = ET.SubElement(root, "HEADER", {"MEDIA_FILE": "", "TIME_UNITS": "milliseconds"})
    ET.SubElement(
        header,
        "MEDIA_DESCRIPTOR",
        {
            "MEDIA_URL": _media_url(video),
            "MIME_TYPE": "video/mp4",
            "RELATIVE_MEDIA_URL": _relative_media_url(video, out.parent),
        },
    )
    annotation_count = sum(len(v) for v in tier_intervals.values())
    ET.SubElement(header, "PROPERTY", {"NAME": "lastUsedAnnotationId"}).text = str(annotation_count)
    time_order = ET.SubElement(root, "TIME_ORDER")
    for time in times:
        ET.SubElement(
            time_order,
            "TIME_SLOT",
            {"TIME_SLOT_ID": time_ids[time], "TIME_VALUE": str(int(time))},
        )

    ann_id = 1
    for tier_id in active_tiers:
        tier = ET.SubElement(root, "TIER", {"LINGUISTIC_TYPE_REF": "default-lt", "TIER_ID": tier_id})
        for interval in tier_intervals.get(tier_id, []):
            if interval.end_ms <= interval.start_ms:
                continue
            ann = ET.SubElement(tier, "ANNOTATION")
            alignable = ET.SubElement(
                ann,
                "ALIGNABLE_ANNOTATION",
                {
                    "ANNOTATION_ID": f"a{ann_id}",
                    "TIME_SLOT_REF1": time_ids[interval.start_ms],
                    "TIME_SLOT_REF2": time_ids[interval.end_ms],
                },
            )
            ET.SubElement(alignable, "ANNOTATION_VALUE").text = interval.label
            ann_id += 1
    ET.SubElement(
        root,
        "LINGUISTIC_TYPE",
        {
            "GRAPHIC_REFERENCES": "false",
            "LINGUISTIC_TYPE_ID": "default-lt",
            "TIME_ALIGNABLE": "true",
        },
    )
    ET.SubElement(root, "LOCALE", {"COUNTRY_CODE": "US", "LANGUAGE_CODE": "en"})
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="UTF-8", xml_declaration=True)
    return out


def _validate_bundle_bounds(bundle: AnnotationBundle) -> None:
    if bundle.tug.end_ms <= bundle.tug.start_ms:
        raise ValueError("TUG interval end_ms must be greater than start_ms")
    for tier_name, annotations in (
        ("phase", bundle.phases),
        ("left_foot", bundle.left),
        ("right_foot", bundle.right),
    ):
        previous_end: int | None = None
        for annotation in sorted(annotations, key=lambda item: (item.start_ms, item.end_ms)):
            if annotation.end_ms <= annotation.start_ms:
                raise ValueError(f"{tier_name} contains non-positive annotation")
            if annotation.start_ms < bundle.tug.start_ms or annotation.end_ms > bundle.tug.end_ms:
                raise ValueError(f"{tier_name} contains annotation outside TUG interval")
            if tier_name == "phase" and annotation.label not in TUG_PHASE_LABELS:
                raise ValueError(f"phase contains invalid label: {annotation.label}")
            if tier_name == "left_foot" and annotation.label not in {"left_stance", "left_swing", "unknown"}:
                raise ValueError(f"left_foot contains invalid label: {annotation.label}")
            if tier_name == "right_foot" and annotation.label not in {"right_stance", "right_swing", "unknown"}:
                raise ValueError(f"right_foot contains invalid label: {annotation.label}")
            if previous_end is not None and annotation.start_ms < previous_end:
                raise ValueError(f"{tier_name} contains overlapping annotations")
            previous_end = annotation.end_ms


def _phase_to_interval(phase: PhaseAnnotation) -> Interval:
    return Interval(phase.start_ms, phase.end_ms, phase.label)


def _foot_to_interval(foot: FootAnnotation) -> Interval:
    return Interval(foot.start_ms, foot.end_ms, foot.label)


def _tier_intervals_from_xml(
    tier: ET.Element | None,
    time_slots: dict[str, int],
) -> list[tuple[int, int, str]]:
    if tier is None:
        return []
    intervals: list[tuple[int, int, str]] = []
    for ann in tier.findall(".//ALIGNABLE_ANNOTATION"):
        start = time_slots.get(ann.get("TIME_SLOT_REF1", ""), 0)
        end = time_slots.get(ann.get("TIME_SLOT_REF2", ""), 0)
        value = ann.find("ANNOTATION_VALUE")
        label = value.text or "" if value is not None else ""
        intervals.append((start, end, label))
    return sorted(intervals, key=lambda item: (item[0], item[1], item[2]))


def _coerce_intervals(raw: Any, duration_ms: int | None = None) -> list[Interval]:
    intervals: list[Interval] = []
    for item in raw or []:
        if isinstance(item, Interval):
            interval = item
        elif isinstance(item, dict):
            interval = Interval(
                int(item["start_ms"]),
                int(item["end_ms"]),
                str(item["label"]),
                float(item.get("confidence", 1.0)),
                str(item.get("source", "")),
            )
        else:
            continue
        if duration_ms is not None:
            interval = Interval(
                max(0, min(duration_ms, interval.start_ms)),
                max(0, min(duration_ms, interval.end_ms)),
                interval.label,
                interval.confidence,
                interval.source,
            )
        if interval.end_ms > interval.start_ms:
            intervals.append(interval)
    return sorted(intervals, key=lambda iv: (iv.start_ms, iv.end_ms, iv.label))


def _coerce_duration_ms(value: Any) -> int | None:
    try:
        duration_ms = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return duration_ms if duration_ms > 0 else None


def _media_url(path: Path) -> str:
    absolute = path.expanduser().resolve()
    return absolute.as_uri()


def _relative_media_url(video: Path, eaf_dir: Path) -> str:
    try:
        rel = os.path.relpath(video.expanduser().resolve(), eaf_dir.expanduser().resolve())
    except ValueError:
        rel = video.name
    return "./" + rel.replace(os.sep, "/")
