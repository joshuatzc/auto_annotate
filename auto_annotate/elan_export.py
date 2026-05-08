"""ELAN .eaf writer for the target TUG tier structure."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import os
import xml.etree.ElementTree as ET

from .config import CRT_TIER_ORDER, TIER_ORDER
from .types import Interval

XSI = "http://www.w3.org/2001/XMLSchema-instance"


def write_eaf(video_path: str, annotations: dict[str, Any], output_path: str) -> Path:
    """Write one ELAN .eaf file linked to the source video."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    video = Path(video_path)

    active_tiers = CRT_TIER_ORDER if annotations.get("is_crt") else TIER_ORDER
    tier_intervals = {
        tier: _coerce_intervals(annotations.get(tier, []))
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


def _coerce_intervals(raw: Any) -> list[Interval]:
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
        if interval.end_ms > interval.start_ms:
            intervals.append(interval)
    return sorted(intervals, key=lambda iv: (iv.start_ms, iv.end_ms, iv.label))


def _media_url(path: Path) -> str:
    absolute = path.expanduser().resolve()
    return absolute.as_uri()


def _relative_media_url(video: Path, eaf_dir: Path) -> str:
    try:
        rel = os.path.relpath(video.expanduser().resolve(), eaf_dir.expanduser().resolve())
    except ValueError:
        rel = video.name
    return "./" + rel.replace(os.sep, "/")
