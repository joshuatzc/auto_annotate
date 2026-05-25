"""Adapter from current 3DGait bundle/output shapes to ELAN annotations."""

from __future__ import annotations

import ast
import csv
import glob
import importlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import (
    BALANCE_TEST_DURATION_MS,
    DEFAULT_FALLBACK_TEST_MS,
    DEFAULT_FPS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    OCCLUSION_GAP_MERGE_MS,
    OCCLUSION_MIN_DURATION_MS,
    OCCLUSION_NEAR_CENTER_RATIO,
)
from .elan_export import write_eaf
from .phase_rules import (
    crt_phase_intervals_from_detections,
    fallback_foot_intervals,
    fallback_phase_intervals,
    frame_to_ms,
    foot_intervals_from_gait_cycles,
    phase_intervals_from_crt_events,
    phase_intervals_from_tug_boundaries,
    smooth_phase_intervals,
    test_interval_from_phases,
    unknown_phase_interval,
    walking_intervals,
)
from .subject_selection import iou, load_detection_json, select_main_subject, track_detections
from .types import DetectionBox, GaitEvent, Interval, Pose3DResult, SubjectTrack, TUGInterval, VideoMetadata

PHASE_TEST_TIER_ORDER = ("phase", "test")
FULL_TIER_ORDER = ("phase", "left_foot", "right_foot", "test")
OCCLUSION_TIER = "occlusion"

TEST_TYPE_TUG = "tug"
TEST_TYPE_CRT = "crt"
TEST_TYPE_GAIT_SPEED = "gait_speed"
TEST_TYPE_SIDE_BY_SIDE = "side_by_side"
TEST_TYPE_SEMI_TANDEM = "semi_tandem"
TEST_TYPE_FULL_TANDEM = "full_tandem"
TEST_TYPE_BALANCE = "balance"
TEST_TYPE_OBSERVED_PHASE = "observed_phase"

BALANCE_TEST_TYPES = {
    TEST_TYPE_BALANCE,
    TEST_TYPE_SIDE_BY_SIDE,
    TEST_TYPE_SEMI_TANDEM,
    TEST_TYPE_FULL_TANDEM,
}
BALANCE_SPPB_KEYS = {
    TEST_TYPE_SIDE_BY_SIDE: "side_by_side",
    TEST_TYPE_SEMI_TANDEM: "semi_tandem",
    TEST_TYPE_FULL_TANDEM: "full_tandem",
}
BALANCE_PHASE_LABELS = {
    TEST_TYPE_BALANCE: "in_pos",
    TEST_TYPE_SIDE_BY_SIDE: "side-by-side",
    TEST_TYPE_SEMI_TANDEM: "semi-tandem",
    TEST_TYPE_FULL_TANDEM: "full-tandem",
}
BALANCE_IN_POSITION_LABEL = "in_pos"
BALANCE_OUT_OF_POSITION_LABEL = "out_of_pos"

_VISFRAILTY_PATIENT_OUTPUT_CACHE: dict[tuple[str, str], list[Path]] = {}


def load_tug_interval(biomarker_path: Path) -> TUGInterval:
    """Read TUG timing from 3DGait biomarker.json.

    This adapter is deliberately strict: TUG timing is the anchor for every ELAN
    tier, so missing or malformed values should stop the auto-annotation run.
    """

    if not biomarker_path.exists():
        raise FileNotFoundError(f"biomarker.json not found: {biomarker_path}")
    try:
        data = json.loads(biomarker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"biomarker.json is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("tug"), dict):
        raise ValueError("biomarker.json must contain object field tug")

    tug = data["tug"]
    missing = [key for key in ("start_ms", "end_ms", "duration_ms") if key not in tug]
    if missing:
        raise ValueError("biomarker.json tug missing field(s): " + ", ".join(missing))
    start_ms = _coerce_int(tug.get("start_ms"), -1)
    end_ms = _coerce_int(tug.get("end_ms"), -1)
    duration_ms = _coerce_int(tug.get("duration_ms"), -1)
    if start_ms < 0:
        raise ValueError("biomarker.json tug.start_ms must be >= 0")
    if end_ms <= start_ms:
        raise ValueError("biomarker.json tug.end_ms must be greater than tug.start_ms")
    if duration_ms <= 0:
        raise ValueError("biomarker.json tug.duration_ms must be > 0")
    return TUGInterval(start_ms, end_ms, duration_ms)


def load_gait_events(boundaries_path: Path, side: str) -> list[GaitEvent]:
    """Read stride-level 3DGait boundaries for one foot.

    The 3DGait boundary artifacts have appeared in a few nearby JSON shapes, so
    this reader accepts event lists, side-keyed objects, and simple stride rows.
    Missing or empty files return an empty list; callers decide whether that
    foot tier becomes unknown.
    """

    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    if not boundaries_path.exists():
        return []
    try:
        data = json.loads(boundaries_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"gait boundaries are not valid JSON: {boundaries_path}: {exc}") from exc

    events = _boundary_events_from_payload(data, side)
    deduped = {
        (event.time_ms, event.side, event.event_type): event
        for event in events
        if event.time_ms >= 0 and event.event_type in {"stance_start", "swing_start"}
    }
    return sorted(deduped.values(), key=lambda event: (event.time_ms, event.event_type))


def select_primary_subject(bboxes: list, frame_width: int) -> int | None:
    """Return the index of the most stable, centre-near person bbox."""

    if not bboxes or frame_width <= 0:
        return None
    parsed = [(_bbox_center_x(item), _bbox_track_id(item)) for item in bboxes]
    track_counts: dict[Any, int] = defaultdict(int)
    for _center_x, track_id in parsed:
        if track_id is not None:
            track_counts[track_id] += 1

    frame_center = frame_width / 2.0
    best_idx: int | None = None
    best_score: float | None = None
    for idx, (center_x, track_id) in enumerate(parsed):
        if center_x is None or center_x != center_x:
            continue
        center_distance = abs(center_x - frame_center)
        stability_bonus = min(track_counts.get(track_id, 1), 30) * 3.0 if track_id is not None else 0.0
        score = center_distance - stability_bonus
        if best_score is None or score < best_score:
            best_idx = idx
            best_score = score
    return best_idx


def analyze_and_export(input_folder: str, output_eaf: str) -> Path:
    """Analyze an input bundle and write exactly one .eaf file."""

    annotations = extract_annotations(input_folder)
    video_path = annotations.get("video_path")
    if not video_path:
        raise ValueError("No rgb_video_*.mp4 source video found to link in the EAF")
    return write_eaf(str(video_path), annotations, output_eaf)


def extract_annotations(input_folder: str) -> dict[str, Any]:
    """Extract target ELAN annotations from an input bundle."""

    started_at = time.monotonic()
    deadline = _annotation_deadline(started_at)
    folder = Path(input_folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"input_folder does not exist or is not a directory: {input_folder}")

    warnings: list[str] = []
    video_path = _first_match(folder, "rgb_video*.mp4")
    bbox_path = _select_bundle_file(
        folder,
        "bounding_box_data*.json",
        video_path,
        "body bounding boxes",
        warnings,
    )
    face_path = _select_bundle_file(
        folder,
        "bounding_box_face_data*.json",
        video_path,
        "face bounding boxes",
        warnings,
    )

    metadata = _read_video_metadata(video_path)
    body_boxes = _load_boxes_safely(bbox_path, metadata, warnings)
    if not body_boxes and video_path and video_path.exists() and video_path.stat().st_size > 0:
        body_boxes = _generate_bboxes_from_video(video_path, metadata, warnings)
    tracked_boxes = track_detections(body_boxes) if body_boxes else []
    subject = (
        select_main_subject(
            tracked_boxes,
            metadata.width,
            metadata.height,
            metadata.num_frames,
        )
        if tracked_boxes
        else SubjectTrack(None, [], 0.0, "no bbox", [])
    )
    if subject.boxes:
        subject = _fill_subject_track_gaps(subject)

    person_masks = None
    if video_path and video_path.exists() and video_path.stat().st_size > 0:
        from .person_masks import load_or_run_person_masks

        person_masks = load_or_run_person_masks(folder, metadata, warnings, deadline=deadline)
    name_test_type = detect_test_type(folder)
    test_type = _classify_test_type_from_motion(name_test_type, metadata, subject, warnings)
    occlusion = _occlusion_intervals_from_tracks(
        subject,
        tracked_boxes,
        metadata,
        warnings,
        face_path,
        person_masks,
        test_type,
    )

    # Extract or load normalized 3D pose once; shared by all pose-derived detectors.
    pose3d = None
    if video_path and video_path.exists() and video_path.stat().st_size > 0 and _has_budget(deadline, 1.0):
        from .pose3d import load_or_run_pose3d

        pose3d = load_or_run_pose3d(folder, metadata, warnings, deadline=deadline)
    elif video_path and video_path.exists() and video_path.stat().st_size > 0:
        warnings.append("pose3d skipped: annotation time budget exhausted")

    if test_type == TEST_TYPE_CRT:
        annotations = _extract_crt_annotations(folder, video_path, metadata, subject, pose3d, occlusion, warnings)
        annotations["person_masks"] = _person_mask_summary(person_masks)
        return annotations
    if test_type == TEST_TYPE_GAIT_SPEED:
        annotations = _extract_gait_speed_annotations(folder, video_path, metadata, subject, pose3d, occlusion, warnings)
        annotations["person_masks"] = _person_mask_summary(person_masks)
        return annotations
    if test_type in BALANCE_TEST_TYPES:
        annotations = _extract_balance_annotations(folder, video_path, metadata, subject, pose3d, occlusion, test_type, warnings)
        annotations["person_masks"] = _person_mask_summary(person_masks)
        return annotations

    phases = _phase_from_bundle_fallback(
        metadata=metadata,
        subject=subject,
        face_path=face_path,
        pose3d=pose3d,
        warnings=warnings,
    )
    if not phases:
        precomputed = _extract_from_precomputed_outputs(folder, metadata, warnings)
        if precomputed:
            precomputed["video_path"] = str(video_path) if video_path else None
            precomputed["subject"] = _subject_summary(subject)
            precomputed["warnings"] = warnings + precomputed.get("warnings", [])
            precomputed["test_type"] = test_type
            precomputed["occlusion"] = _intervals_to_dicts(occlusion)
            precomputed["tier_order"] = _tier_order(FULL_TIER_ORDER, occlusion)
            precomputed["pose3d"] = _pose3d_summary(pose3d)
            precomputed["person_masks"] = _person_mask_summary(person_masks)
            return precomputed
    test = _mobility_test_interval_from_phases(phases)
    occlusion = _clip_intervals_to_windows(occlusion, test)
    foot_spans = _foot_spans_for_test(test_type, phases, metadata)

    left_foot, right_foot = ([], [])
    if pose3d is not None and foot_spans:
        from .pose3d import foot_intervals_from_pose3d

        left_foot, right_foot = foot_intervals_from_pose3d(pose3d, metadata, foot_spans, warnings)
    if foot_spans and (not left_foot or not right_foot):
        fallback_left, fallback_right = fallback_foot_intervals(foot_spans)
        if not left_foot:
            left_foot = fallback_left
        if not right_foot:
            right_foot = fallback_right
    left_foot = _align_foot_to_walk_spans(left_foot, foot_spans)
    right_foot = _align_foot_to_walk_spans(right_foot, foot_spans)

    return {
        "video_path": str(video_path) if video_path else None,
        "fps": metadata.fps,
        "duration_ms": metadata.duration_ms,
        "test": _intervals_to_dicts(test),
        "phase": _intervals_to_dicts(phases),
        "left_foot": _intervals_to_dicts(left_foot),
        "right_foot": _intervals_to_dicts(right_foot),
        "occlusion": _intervals_to_dicts(occlusion),
        "test_type": test_type,
        "tier_order": _tier_order(FULL_TIER_ORDER, occlusion),
        "subject": _subject_summary(subject),
        "pose3d": _pose3d_summary(pose3d),
        "person_masks": _person_mask_summary(person_masks),
        "warnings": warnings,
    }


def _annotation_deadline(started_at: float) -> float | None:
    try:
        max_seconds = float(os.environ.get("AUTO_ANNOTATE_MAX_SECONDS", "30"))
    except ValueError:
        max_seconds = 30.0
    if max_seconds <= 0:
        return None
    return started_at + max_seconds


def _has_budget(deadline: float | None, min_remaining_seconds: float = 0.0) -> bool:
    if deadline is None:
        return True
    return deadline - time.monotonic() >= min_remaining_seconds


def detect_test_type(path: str | Path) -> str | None:
    """Infer the FrailScreen test type from a folder name.

    The T7 raw data uses names like ``10017 GS1``, ``10012st``, ``MC0005 TUG2``,
    and ``10079 FT F``.  This intentionally looks only at compact test tokens,
    not arbitrary substrings, so patient IDs and parent names do not dominate
    the result.
    """
    folder = Path(path)
    return _test_type_from_name(folder.name) or _test_type_from_name(folder.parent.name)


def _test_type_from_name(name: str) -> str | None:
    lower = name.lower()
    compact = re.sub(r"[^a-z0-9]+", "", lower)
    tokens = re.findall(r"[a-z]+", lower)
    token_set = set(tokens)

    if "tug" in token_set or "tug" in compact:
        return TEST_TYPE_TUG
    if "crt" in token_set or "crt" in compact or "chairrise" in compact:
        return TEST_TYPE_CRT
    if "sbs" in token_set or "sbs" in compact or "sidebyside" in compact:
        return TEST_TYPE_SIDE_BY_SIDE
    if "st" in token_set or "semitandem" in compact:
        return TEST_TYPE_SEMI_TANDEM
    if "ft" in token_set or "fulltandem" in compact:
        return TEST_TYPE_FULL_TANDEM
    if "gs" in token_set or "gaitspeed" in compact or re.search(r"gs[12]?", compact):
        return TEST_TYPE_GAIT_SPEED
    return None


def _classify_test_type_from_motion(
    name_test_type: str | None,
    metadata: VideoMetadata,
    subject: SubjectTrack,
    warnings: list[str],
) -> str:
    evidence = _subject_motion_evidence(metadata, subject)
    if evidence.get("static_standing"):
        if name_test_type in BALANCE_TEST_TYPES:
            return name_test_type
        if name_test_type in {None, TEST_TYPE_TUG, TEST_TYPE_OBSERVED_PHASE}:
            warnings.append(
                "test classifier: static standing posture detected; using balance instead of TUG-style phases"
            )
            return TEST_TYPE_BALANCE

    if name_test_type is not None:
        return name_test_type

    if evidence.get("crt_like"):
        warnings.append("test classifier: repeated vertical sit-stand motion detected; using CRT")
        return TEST_TYPE_CRT
    if evidence.get("tug_like"):
        warnings.append("test classifier: sit-stand plus return trajectory detected; using TUG")
        return TEST_TYPE_TUG
    if evidence.get("walk_like"):
        warnings.append("test classifier: walking trajectory detected; using gait speed")
        return TEST_TYPE_GAIT_SPEED

    return TEST_TYPE_OBSERVED_PHASE


def _subject_motion_evidence(
    metadata: VideoMetadata,
    subject: SubjectTrack,
) -> dict[str, bool | float]:
    boxes = sorted(subject.boxes, key=lambda box: box.frame)
    boxes = [
        box
        for box in boxes
        if box.frame >= 0 and (not metadata.num_frames or box.frame < metadata.num_frames)
    ]
    if len(boxes) < 10:
        return {}

    width = max(float(metadata.width), 1.0)
    height = max(float(metadata.height), 1.0)
    cx = [box.center_x for box in boxes]
    cy = [box.center_y for box in boxes]
    body_h = [box.height for box in boxes]
    x_range = (_percentile(cx, 0.90) - _percentile(cx, 0.10)) / width
    y_range = (_percentile(cy, 0.90) - _percentile(cy, 0.10)) / height
    median_h = max(_median(body_h), 1.0)
    h_range = (_percentile(body_h, 0.90) - _percentile(body_h, 0.10)) / median_h
    net_x = abs(cx[-1] - cx[0]) / width

    direction_changes = 0
    min_direction = width * 0.035
    prev_sign = 0
    anchor = cx[0]
    for value in cx[1:]:
        delta = value - anchor
        if abs(delta) < min_direction:
            continue
        sign = 1 if delta > 0 else -1
        if prev_sign and sign != prev_sign:
            direction_changes += 1
            anchor = value
        prev_sign = sign

    static_standing = x_range < 0.065 and y_range < 0.035 and h_range < 0.16
    walk_like = x_range >= 0.08 or net_x >= 0.10
    sit_stand_like = h_range >= 0.25 or y_range >= 0.055
    tug_like = sit_stand_like and x_range >= 0.045 and net_x < 0.08
    crt_like = sit_stand_like and x_range < 0.045 and net_x < 0.06
    if direction_changes >= 1 and sit_stand_like:
        tug_like = True

    return {
        "static_standing": static_standing,
        "walk_like": walk_like,
        "sit_stand_like": sit_stand_like,
        "tug_like": tug_like,
        "crt_like": crt_like,
        "x_range": x_range,
        "y_range": y_range,
        "h_range": h_range,
        "net_x": net_x,
        "direction_changes": float(direction_changes),
    }


def _foot_spans_for_test(
    test_type: str | None,
    phases: list[Interval],
    metadata: VideoMetadata,
) -> list[Interval]:
    if not phases:
        return []
    cleaned = sorted(phases, key=lambda item: (item.start_ms, item.end_ms))
    duration_ms = max(0, metadata.duration_ms)

    if test_type == TEST_TYPE_GAIT_SPEED:
        walks = [phase for phase in cleaned if phase.label == "walk"]
        if not walks:
            return []
        start_ms = 0
        end_ms = max(phase.end_ms for phase in walks)
        if duration_ms > 0:
            end_ms = min(end_ms, duration_ms)
        return [Interval(start_ms, end_ms, "walk", 0.40, "gait_speed_foot_span")] if end_ms > start_ms else []

    if test_type == TEST_TYPE_TUG:
        sit_to_stand = next((phase for phase in cleaned if phase.label == "sit-to-stand"), None)
        stand_to_sit = next((phase for phase in reversed(cleaned) if phase.label == "stand-to-sit"), None)
        if sit_to_stand and stand_to_sit and stand_to_sit.start_ms > sit_to_stand.end_ms:
            return [
                Interval(
                    sit_to_stand.end_ms,
                    stand_to_sit.start_ms,
                    "walk",
                    0.40,
                    "tug_locomotion_foot_span",
                )
            ]

    walks = walking_intervals(cleaned)
    return walks


def _extract_crt_annotations(
    folder: Path,
    video_path: "Path | None",
    metadata: "VideoMetadata",
    subject: "SubjectTrack",
    pose3d: "Pose3DResult | None",
    occlusion: list[Interval],
    warnings: list[str],
) -> "dict[str, Any]":
    phases = _extract_from_crt_precomputed(folder, metadata, warnings)
    if not phases:
        phases = _crt_phase_from_bundle_fallback(metadata, subject, pose3d, warnings)

    test = test_interval_from_phases(phases)
    occlusion = _clip_intervals_to_windows(occlusion, test)
    occlusion = _normalize_crt_occlusion_intervals(occlusion)
    return {
        "video_path": str(video_path) if video_path else None,
        "fps": metadata.fps,
        "duration_ms": metadata.duration_ms,
        "test": _intervals_to_dicts(test),
        "phase": _intervals_to_dicts(phases),
        "left_foot": [],
        "right_foot": [],
        "occlusion": _intervals_to_dicts(occlusion),
        "is_crt": True,
        "test_type": TEST_TYPE_CRT,
        "tier_order": _tier_order(PHASE_TEST_TIER_ORDER, occlusion),
        "subject": _subject_summary(subject),
        "pose3d": _pose3d_summary(pose3d),
        "warnings": warnings,
    }


def _extract_from_crt_precomputed(
    folder: Path,
    metadata: "VideoMetadata",
    warnings: list[str],
) -> "list[Interval]":
    sppb = _find_sppb_results(folder)
    if sppb is None:
        return []
    try:
        import json

        with open(sppb, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        item = _select_sppb_item(data, "chair_rise", folder)
        if item is None:
            return []
        events = (item.get("result") or {}).get("events") or []
        if not events:
            return []
        fps = _coerce_float(data.get("fps"), metadata.fps or DEFAULT_FPS)
        phases = phase_intervals_from_crt_events(events, fps)
        if phases:
            return phases
        warnings.append("sppb_results.json chair_rise events did not produce usable intervals")
    except Exception as exc:
        warnings.append(f"could not read sppb_results.json for CRT: {exc}")
    return []


def _find_sppb_results(folder: Path) -> "Path | None":
    for search in (folder, folder.parent, folder.parent.parent) + tuple(_visfrailty_patient_output_dirs(folder)):
        candidate = search / "sppb_results.json"
        if candidate.exists():
            return candidate
    return None


def _select_sppb_item(data: dict[str, Any], key: str, folder: Path) -> dict[str, Any] | None:
    items = (data.get("tests") or {}).get(key) or []
    if not items:
        return None

    folder_name = _normalized_name(folder.name)
    for item in items:
        subfolder = str(item.get("subfolder") or "")
        if _normalized_name(Path(subfolder).name) == folder_name:
            return item
    return items[0] if isinstance(items[0], dict) else None


def _extract_gait_speed_annotations(
    folder: Path,
    video_path: "Path | None",
    metadata: "VideoMetadata",
    subject: "SubjectTrack",
    pose3d: "Pose3DResult | None",
    occlusion: list[Interval],
    warnings: list[str],
) -> "dict[str, Any]":
    phases, left_foot, right_foot = _gait_speed_pose3d_fallback(pose3d, metadata, warnings)
    if not phases:
        phases, left_foot, right_foot = _extract_gait_speed_precomputed(folder, metadata, warnings)
    if not phases:
        phases = _gait_speed_phase_from_bundle_fallback(metadata, subject, warnings)
        foot_spans = _foot_spans_for_test(TEST_TYPE_GAIT_SPEED, phases, metadata)
        left_foot, right_foot = fallback_foot_intervals(foot_spans) if foot_spans else ([], [])
        left_foot = _align_foot_to_walk_spans(left_foot, foot_spans)
        right_foot = _align_foot_to_walk_spans(right_foot, foot_spans)

    test = test_interval_from_phases(phases)
    occlusion = _clip_intervals_to_windows(occlusion, test)
    return {
        "video_path": str(video_path) if video_path else None,
        "fps": metadata.fps,
        "duration_ms": metadata.duration_ms,
        "test": _intervals_to_dicts(test),
        "phase": _intervals_to_dicts(phases),
        "left_foot": _intervals_to_dicts(left_foot),
        "right_foot": _intervals_to_dicts(right_foot),
        "occlusion": _intervals_to_dicts(occlusion),
        "test_type": TEST_TYPE_GAIT_SPEED,
        "tier_order": _tier_order(FULL_TIER_ORDER, occlusion),
        "subject": _subject_summary(subject),
        "pose3d": _pose3d_summary(pose3d),
        "warnings": warnings,
    }


def _gait_speed_pose3d_fallback(
    pose3d: "Pose3DResult | None",
    metadata: "VideoMetadata",
    warnings: list[str],
) -> "tuple[list[Interval], list[Interval], list[Interval]]":
    if pose3d is None:
        return [], [], []
    from .pose3d import gait_speed_from_pose3d

    phases, left_foot, right_foot = gait_speed_from_pose3d(pose3d, metadata, warnings)
    if phases:
        if not left_foot or not right_foot:
            fallback_left, fallback_right = fallback_foot_intervals(
                _foot_spans_for_test(TEST_TYPE_GAIT_SPEED, phases, metadata)
            )
            if not left_foot:
                left_foot = fallback_left
            if not right_foot:
                right_foot = fallback_right
        warnings.append("gait-speed walk and foot phases detected from pose3d kinematics")
    return phases, left_foot, right_foot


def _extract_gait_speed_precomputed(
    folder: Path,
    metadata: "VideoMetadata",
    warnings: list[str],
) -> "tuple[list[Interval], list[Interval], list[Interval]]":
    meta = _load_meta_dataframe(folder, warnings)
    if not meta:
        return [], [], []

    gait = _parse_gait_breakdown(meta.get("Gait_breakdown_frames"))
    if not gait:
        warnings.append("precomputed gait-speed metadata did not include usable Gait_breakdown_frames")
        return [], [], []

    fps = _coerce_float(meta.get("fps"), metadata.fps or DEFAULT_FPS)
    left_foot, right_foot = foot_intervals_from_gait_cycles(
        gait[0],
        gait[1],
        fps,
        frame_base="zero",
    )
    all_foot = sorted(left_foot + right_foot, key=lambda iv: (iv.start_ms, iv.end_ms))
    if not all_foot:
        warnings.append("precomputed gait-speed cycles were empty")
        return [], [], []

    start_ms = min(iv.start_ms for iv in all_foot)
    end_ms = max(iv.end_ms for iv in all_foot)
    if end_ms <= start_ms:
        return [], [], []

    phases = [Interval(start_ms, end_ms, "walk", 0.95, "core_gait_cycles")]
    left_foot = _align_foot_to_walk_spans(left_foot, phases)
    right_foot = _align_foot_to_walk_spans(right_foot, phases)
    return phases, left_foot, right_foot


def _gait_speed_phase_from_bundle_fallback(
    metadata: "VideoMetadata",
    subject: "SubjectTrack",
    warnings: list[str],
) -> "list[Interval]":
    if subject.boxes:
        start_frame, end_frame = _infer_gait_speed_walk_span_frames(metadata, subject)
        start_ms = frame_to_ms(start_frame, metadata.fps, "zero")
        end_ms = frame_to_ms(end_frame, metadata.fps, "zero")
        if metadata.duration_ms > 0:
            end_ms = min(end_ms, metadata.duration_ms)
        if end_ms > start_ms:
            warnings.append("using motion-onset subject track as low-confidence gait-speed walk span")
            return [Interval(start_ms, end_ms, "walk", 0.45, "gait_bbox_fallback")]

    if metadata.duration_ms > 0:
        warnings.append("no gait-speed cycles or subject track found; writing unknown phase over video duration")
        return unknown_phase_interval(0, metadata.duration_ms)

    warnings.append("insufficient input data for gait-speed timing")
    return []


def _infer_gait_speed_walk_span_frames(
    metadata: VideoMetadata,
    subject: SubjectTrack,
) -> tuple[int, int]:
    boxes = sorted(subject.boxes, key=lambda box: box.frame)
    boxes = [
        box
        for box in boxes
        if box.frame >= 0 and (not metadata.num_frames or box.frame < metadata.num_frames)
    ]
    if not boxes:
        return 0, 0

    fps = metadata.fps or DEFAULT_FPS
    first_frame = boxes[0].frame
    last_frame = boxes[-1].frame
    end_frame = last_frame + 1
    if metadata.num_frames:
        end_frame = min(end_frame, metadata.num_frames)
    if len(boxes) < 10:
        return first_frame, max(first_frame + 1, end_frame)

    window = max(3, int(round(fps * 0.20)))
    centers_x = _moving_average([box.center_x for box in boxes], window)
    centers_y = _moving_average([box.center_y for box in boxes], window)
    heights = _moving_average([box.height for box in boxes], window)
    baseline_count = min(len(boxes), max(3, int(round(fps * 0.70))))
    base_x = _median(centers_x[:baseline_count])
    base_y = _median(centers_y[:baseline_count])
    base_h = max(_median(heights[:baseline_count]), 1.0)
    width = max(float(metadata.width), 1.0)
    height = max(float(metadata.height), 1.0)

    displacement = [
        abs(cx - base_x) / width
        + 0.50 * abs(cy - base_y) / height
        + 0.25 * abs(body_h - base_h) / base_h
        for cx, cy, body_h in zip(centers_x, centers_y, heights)
    ]
    sustain = max(2, int(round(fps * 0.25)))
    threshold = max(0.018, _median(displacement[:baseline_count]) + 0.015)
    onset_idx = 0
    search_start = min(len(boxes) - 1, max(1, int(round(fps * 0.45))))
    search_end = max(search_start, len(boxes) - sustain)
    for idx in range(search_start, search_end + 1):
        if all(value >= threshold for value in displacement[idx : idx + sustain]):
            onset_idx = max(0, idx - int(round(fps * 0.60)))
            break

    start_frame = boxes[onset_idx].frame
    if end_frame <= start_frame:
        end_frame = start_frame + 1
    return start_frame, end_frame


def _extract_balance_annotations(
    folder: Path,
    video_path: "Path | None",
    metadata: "VideoMetadata",
    subject: "SubjectTrack",
    pose3d: "Pose3DResult | None",
    occlusion: list[Interval],
    test_type: str,
    warnings: list[str],
) -> "dict[str, Any]":
    in_position = _extract_balance_precomputed(folder, metadata, test_type, warnings)
    if not in_position:
        in_position = _balance_pose3d_fallback(pose3d, metadata, test_type, warnings)
    if not in_position:
        in_position = _balance_phase_from_bundle_fallback(metadata, subject, test_type, warnings)
    phases = _balance_position_intervals(in_position, metadata, subject, test_type, warnings)

    test = _balance_test_interval_from_phases(phases, metadata)
    occlusion = _clip_intervals_to_windows(occlusion, test)
    return {
        "video_path": str(video_path) if video_path else None,
        "fps": metadata.fps,
        "duration_ms": metadata.duration_ms,
        "test": _intervals_to_dicts(test),
        "phase": _intervals_to_dicts(phases),
        "left_foot": [],
        "right_foot": [],
        "occlusion": _intervals_to_dicts(occlusion),
        "is_balance": True,
        "test_type": test_type,
        "tier_order": _tier_order(PHASE_TEST_TIER_ORDER, occlusion),
        "subject": _subject_summary(subject),
        "pose3d": _pose3d_summary(pose3d),
        "warnings": warnings,
    }


def _balance_pose3d_fallback(
    pose3d: "Pose3DResult | None",
    metadata: "VideoMetadata",
    test_type: str,
    warnings: list[str],
) -> "list[Interval]":
    if pose3d is None:
        return []
    from .pose3d import balance_phase_from_pose3d

    in_pos_label = BALANCE_PHASE_LABELS.get(test_type, BALANCE_IN_POSITION_LABEL)
    phases = balance_phase_from_pose3d(pose3d, metadata, in_pos_label, warnings)
    if phases:
        warnings.append(f"{test_type} hold detected from pose3d kinematics")
    return phases


def _extract_balance_precomputed(
    folder: Path,
    metadata: "VideoMetadata",
    test_type: str,
    warnings: list[str],
) -> "list[Interval]":
    sppb = _find_sppb_results(folder)
    if sppb is None:
        return []
    sppb_key = BALANCE_SPPB_KEYS.get(test_type)
    if sppb_key is None:
        return []
    try:
        import json

        with open(sppb, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        item = _select_sppb_item(data, sppb_key, folder)
        if item is None:
            return []
        result = item.get("result") or {}
        fsm = result.get("fsm") or {}
        start_frame = _coerce_int(fsm.get("hold_start_frame"), -1)
        end_frame = _coerce_int(fsm.get("hold_end_frame"), -1)
        if start_frame < 0 or end_frame <= start_frame:
            warnings.append(f"sppb_results.json did not contain a usable {test_type} hold interval")
            return []
        fps = _coerce_float(data.get("fps"), metadata.fps or DEFAULT_FPS)
        start_ms = frame_to_ms(start_frame, fps, "zero")
        end_ms = frame_to_ms(end_frame + 1, fps, "zero")
        in_pos_label = BALANCE_PHASE_LABELS.get(test_type, BALANCE_IN_POSITION_LABEL)
        return [Interval(start_ms, end_ms, in_pos_label, 0.95, "sppb_balance_fsm")]
    except Exception as exc:
        warnings.append(f"could not read sppb_results.json for {test_type}: {exc}")
    return []


def _balance_phase_from_bundle_fallback(
    metadata: "VideoMetadata",
    subject: "SubjectTrack",
    test_type: str,
    warnings: list[str],
) -> "list[Interval]":
    if subject.boxes:
        start_frame = _infer_balance_in_position_start_frame(metadata, subject)
        end_frame = (subject.last_frame or start_frame) + 1
        start_ms = frame_to_ms(start_frame, metadata.fps, "zero")
        end_ms = frame_to_ms(end_frame, metadata.fps, "zero")
        if end_ms > start_ms:
            warnings.append("using selected subject track as low-confidence in_pos balance span")
            in_pos_label = BALANCE_PHASE_LABELS.get(test_type, BALANCE_IN_POSITION_LABEL)
            return [Interval(start_ms, end_ms, in_pos_label, 0.45, "balance_bbox_fallback")]

    if metadata.duration_ms > 0:
        warnings.append("no balance FSM or subject track found; marking full video out_of_pos")
        return [Interval(0, metadata.duration_ms, BALANCE_OUT_OF_POSITION_LABEL, 0.20, "balance_duration_fallback")]

    warnings.append("insufficient input data for balance timing")
    return []


def _infer_balance_in_position_start_frame(
    metadata: VideoMetadata,
    subject: SubjectTrack,
) -> int:
    boxes = sorted(subject.boxes, key=lambda box: box.frame)
    boxes = [
        box
        for box in boxes
        if box.frame >= 0 and (not metadata.num_frames or box.frame < metadata.num_frames)
    ]
    if len(boxes) < 10:
        return subject.first_frame or 0

    fps = metadata.fps or DEFAULT_FPS
    window = max(3, int(round(fps * 0.30)))
    heights = _moving_average([box.height for box in boxes], window)
    centers_y = _moving_average([box.center_y for box in boxes], window)
    centers_x = _moving_average([box.center_x for box in boxes], window)
    median_height = max(_median(heights), 1.0)

    motion: list[float] = [0.0]
    for idx in range(1, len(boxes)):
        motion.append(
            abs(heights[idx] - heights[idx - 1]) / median_height
            + abs(centers_y[idx] - centers_y[idx - 1]) / max(float(metadata.height), 1.0)
            + abs(centers_x[idx] - centers_x[idx - 1]) / max(float(metadata.width), 1.0)
        )

    search_start = max(1, int(round(fps * 0.50)))
    search_end = min(len(boxes), max(search_start + 1, int(round(fps * 3.20))))
    if search_end <= search_start:
        return subject.first_frame or 0

    threshold = max(_percentile(motion, 0.90), _median(motion) * 3.0, 0.0025)
    peak_idx = max(range(search_start, search_end), key=lambda idx: motion[idx])
    if motion[peak_idx] < threshold:
        return subject.first_frame or 0

    start_idx = min(len(boxes) - 1, peak_idx + int(round(fps * 0.55)))
    return boxes[start_idx].frame


def _balance_test_interval_from_phases(
    phases: list[Interval],
    metadata: VideoMetadata,
) -> list[Interval]:
    in_position = [
        phase
        for phase in phases
        if phase.label not in {BALANCE_OUT_OF_POSITION_LABEL, "unknown"}
    ]
    if not in_position:
        return test_interval_from_phases(phases)
    start_ms = min(phase.start_ms for phase in in_position)
    end_ms = start_ms + BALANCE_TEST_DURATION_MS
    if metadata.duration_ms > 0:
        end_ms = min(end_ms, metadata.duration_ms)
    max_in_pos_end = max(phase.end_ms for phase in in_position)
    end_ms = min(end_ms, max_in_pos_end)
    if end_ms <= start_ms:
        return []
    return [Interval(start_ms, end_ms, "time", 1.0, "balance_in_pos_span")]


def _mobility_test_interval_from_phases(phases: list[Interval]) -> list[Interval]:
    if not phases:
        return []

    start = next((phase.start_ms for phase in phases if phase.label == "sit-to-stand"), None)
    if start is None:
        start = next(
            (
                phase.start_ms
                for phase in phases
                if phase.label not in {"sit", "unknown", BALANCE_OUT_OF_POSITION_LABEL}
            ),
            None,
        )
    end = next(
        (phase.end_ms for phase in reversed(phases) if phase.label == "stand-to-sit"),
        None,
    )
    if end is None:
        end = next(
            (
                phase.end_ms
                for phase in reversed(phases)
                if phase.label not in {"sit", "unknown", BALANCE_OUT_OF_POSITION_LABEL}
            ),
            None,
        )
    if start is None or end is None or end <= start:
        return test_interval_from_phases(phases)
    return [Interval(start, end, "time", 1.0, "mobility_phase_span")]


def _balance_position_intervals(
    in_position: list[Interval],
    metadata: VideoMetadata,
    subject: SubjectTrack,
    test_type: str,
    warnings: list[str],
) -> list[Interval]:
    span = _balance_annotation_span(metadata, subject, in_position)
    if span is None:
        return []
    start_ms, end_ms = span
    if end_ms <= start_ms:
        return []

    in_pos_label = BALANCE_PHASE_LABELS.get(test_type, BALANCE_IN_POSITION_LABEL)
    clipped_in_pos = [
        Interval(
            max(start_ms, iv.start_ms),
            min(end_ms, iv.end_ms),
            in_pos_label if iv.label != BALANCE_OUT_OF_POSITION_LABEL else iv.label,
            iv.confidence,
            iv.source,
        )
        for iv in sorted(in_position, key=lambda item: (item.start_ms, item.end_ms))
        if iv.end_ms > start_ms and iv.start_ms < end_ms
    ]
    clipped_in_pos = [iv for iv in clipped_in_pos if iv.end_ms > iv.start_ms]
    if not clipped_in_pos:
        warnings.append("no in_pos balance interval found; marking balance span out_of_pos")
        return [Interval(start_ms, end_ms, BALANCE_OUT_OF_POSITION_LABEL, 0.20, "balance_no_in_pos")]

    out: list[Interval] = []
    cursor = start_ms
    for interval in clipped_in_pos:
        if interval.start_ms > cursor:
            out.append(
                Interval(
                    cursor,
                    interval.start_ms,
                    BALANCE_OUT_OF_POSITION_LABEL,
                    min(0.60, interval.confidence),
                    "balance_position_gap",
                )
            )
        out.append(interval)
        cursor = max(cursor, interval.end_ms)
    if cursor < end_ms:
        out.append(
            Interval(
                cursor,
                end_ms,
                BALANCE_OUT_OF_POSITION_LABEL,
                min(0.60, clipped_in_pos[-1].confidence),
                "balance_position_gap",
            )
        )
    return _merge_adjacent_phase_labels(out, 1)


def _balance_annotation_span(
    metadata: VideoMetadata,
    subject: SubjectTrack,
    intervals: list[Interval],
) -> tuple[int, int] | None:
    if metadata.duration_ms > 0:
        return 0, metadata.duration_ms
    if subject.boxes:
        start_frame = subject.first_frame or 0
        end_frame = (subject.last_frame or start_frame) + 1
        start_ms = frame_to_ms(start_frame, metadata.fps, "zero")
        end_ms = frame_to_ms(end_frame, metadata.fps, "zero")
        if end_ms > start_ms:
            return start_ms, end_ms
    if intervals:
        return min(iv.start_ms for iv in intervals), max(iv.end_ms for iv in intervals)
    return None


def _crt_phase_from_bundle_fallback(
    metadata: "VideoMetadata",
    subject: "SubjectTrack",
    pose3d: "Pose3DResult | None",
    warnings: list[str],
) -> "list[Interval]":
    if pose3d is not None:
        from .pose3d import crt_phases_from_pose3d

        pose_phases = crt_phases_from_pose3d(pose3d, metadata, warnings)
        if pose_phases:
            warnings.append("CRT phases detected from pose3d kinematics")
            return pose_phases

    if not subject.boxes:
        if metadata.duration_ms > 0:
            warnings.append("no person bounding boxes found for CRT; writing unknown phase")
            return unknown_phase_interval(0, metadata.duration_ms)
        warnings.append("insufficient input data for CRT timing")
        return []

    phases = crt_phase_intervals_from_detections(
        subject.boxes, metadata.fps, metadata.num_frames
    )
    if phases:
        phases = _drop_late_crt_interaction_cycles(phases, metadata, warnings)
        warnings.append("CRT phases detected from bounding box height signal")
        return phases

    warnings.append("CRT bbox detection found no rises; using unknown phase")
    start_ms = int(round((subject.first_frame or 0) * 1000.0 / metadata.fps))
    end_ms = int(round(((subject.last_frame or 0) + 1) * 1000.0 / metadata.fps))
    if end_ms <= start_ms:
        end_ms = start_ms + DEFAULT_FALLBACK_TEST_MS
    return unknown_phase_interval(start_ms, end_ms)


def _drop_late_crt_interaction_cycles(
    phases: list[Interval],
    metadata: VideoMetadata,
    warnings: list[str],
) -> list[Interval]:
    sit_to_stand_indices = [
        idx for idx, phase in enumerate(phases) if phase.label == "sit-to-stand"
    ]
    if len(sit_to_stand_indices) < 4:
        return phases
    duration_ms = metadata.duration_ms or (phases[-1].end_ms if phases else 0)
    if duration_ms <= 0:
        return phases

    late_idx = sit_to_stand_indices[3]
    late_phase = phases[late_idx]
    previous = phases[late_idx - 1] if late_idx > 0 else None
    if (
        late_phase.start_ms >= int(round(duration_ms * 0.72))
        and previous is not None
        and previous.label == "sit"
        and previous.duration_ms >= 1200
    ):
        warnings.append("CRT late post-test interaction trimmed from phase/test scoring")
        return phases[:late_idx]
    return phases


def _normalize_crt_occlusion_intervals(intervals: list[Interval]) -> list[Interval]:
    if len(intervals) < 2:
        return intervals
    merged: list[Interval] = []
    for interval in sorted(intervals, key=lambda item: (item.start_ms, item.end_ms)):
        if (
            merged
            and interval.label == merged[-1].label
            and interval.start_ms - merged[-1].end_ms <= 400
        ):
            prev = merged[-1]
            merged[-1] = Interval(
                prev.start_ms,
                max(prev.end_ms, interval.end_ms),
                prev.label,
                max(prev.confidence, interval.confidence),
                prev.source or interval.source,
            )
        else:
            merged.append(interval)
    if len(merged) < 2:
        return merged
    longest = max(interval.duration_ms for interval in merged)
    keep_threshold = max(900, int(round(longest * 0.45)))
    return [interval for interval in merged if interval.duration_ms >= keep_threshold]


def _extract_from_precomputed_outputs(
    folder: Path,
    metadata: VideoMetadata,
    warnings: list[str],
) -> dict[str, Any] | None:
    meta = _load_meta_dataframe(folder, warnings)
    if not meta:
        return None

    fps = _coerce_float(meta.get("fps"), metadata.fps or DEFAULT_FPS)
    num_frames = _coerce_int(meta.get("num_frames"), metadata.num_frames)
    boundaries = _parse_literalish(meta.get("TUG_breakdown_frames"))
    phases: list[Interval] = []
    if isinstance(boundaries, (list, tuple)):
        phases = phase_intervals_from_tug_boundaries(list(boundaries), fps, num_frames)

    if not phases:
        warnings.append("precomputed metadata did not include usable TUG_breakdown_frames")
        return None

    walks = walking_intervals(phases)
    foot_spans = _foot_spans_for_test(TEST_TYPE_TUG, phases, metadata)
    gait = _parse_gait_breakdown(meta.get("Gait_breakdown_frames"))
    if gait:
        left_foot, right_foot = foot_intervals_from_gait_cycles(
            gait[0],
            gait[1],
            fps,
            walk_spans=walks,
            frame_base="zero",
        )
        if not left_foot and not right_foot:
            warnings.append("precomputed gait cycles were empty; using fallback gait intervals")
            left_foot, right_foot = fallback_foot_intervals(foot_spans)
    else:
        warnings.append("precomputed metadata did not include usable Gait_breakdown_frames")
        left_foot, right_foot = fallback_foot_intervals(foot_spans)

    return {
        "fps": fps,
        "duration_ms": int(round((num_frames * 1000.0) / fps)) if num_frames else metadata.duration_ms,
        "test": _intervals_to_dicts(test_interval_from_phases(phases)),
        "phase": _intervals_to_dicts(phases),
        "left_foot": _intervals_to_dicts(left_foot),
        "right_foot": _intervals_to_dicts(right_foot),
        "warnings": [],
    }


def _phase_from_bundle_fallback(
    metadata: VideoMetadata,
    subject: SubjectTrack,
    face_path: Path | None,
    pose3d: "Pose3DResult | None",
    warnings: list[str],
) -> list[Interval]:
    if subject.boxes:
        observed_phases = _observed_tug_phases_from_track(metadata, subject, face_path, warnings)
        if observed_phases:
            warnings.append("TUG phases detected from locked patient state tracking")
            return observed_phases

        motion_phases = _phase_from_body_motion(metadata, subject.boxes)
        if motion_phases:
            warnings.append("using bbox motion phase fallback from selected subject track")
            return motion_phases

        start_frame = subject.first_frame or 0
        end_frame = (subject.last_frame or start_frame) + 1
        if metadata.num_frames:
            start_frame = min(max(0, start_frame), max(0, metadata.num_frames - 1))
            end_frame = min(max(start_frame + 1, end_frame), metadata.num_frames)
        start_ms = int(round(start_frame * 1000.0 / metadata.fps))
        end_ms = int(round(end_frame * 1000.0 / metadata.fps))
        if end_ms <= start_ms:
            end_ms = start_ms + DEFAULT_FALLBACK_TEST_MS
        warnings.append("using low-confidence ordered phase fallback from selected subject track")
        return fallback_phase_intervals(start_ms, end_ms)

    if metadata.duration_ms > 0:
        warnings.append("no person bounding boxes found; writing unknown phase over video duration")
        return unknown_phase_interval(0, metadata.duration_ms)

    warnings.append("insufficient input data for TUG timing")
    return []


def _try_core_bbox_phase_segmentation(
    metadata: VideoMetadata,
    body_boxes: list[DetectionBox],
    face_path: Path,
    warnings: list[str],
) -> list[Interval]:
    try:
        face_boxes = load_detection_json(face_path, metadata.width, metadata.height)
    except Exception as exc:
        warnings.append(f"could not read face bounding boxes: {exc}")
        return []
    if not face_boxes:
        return []
    if not _has_core_turn_gap(face_boxes, metadata.fps):
        warnings.append("face bounding boxes do not contain the turn-away gap expected by core segmentation")
        return []

    do_phase_seg_bbox = _optional_core_tug_bbox()
    if do_phase_seg_bbox is None:
        warnings.append("core TUG bbox phase segmentation is unavailable")
        return []

    c_body = [
        [box.frame + 1, 1, box.x1, box.y1, box.x2, box.y2]
        for box in sorted(body_boxes, key=lambda b: b.frame)
    ]
    c_face = [
        [box.frame + 1, 1, box.x1, box.y1, box.x2, box.y2]
        for box in sorted(face_boxes, key=lambda b: b.frame)
    ]
    try:
        import numpy as np

        boundaries, _total = do_phase_seg_bbox(
            np.asarray(c_body, dtype=float),
            np.asarray(c_face, dtype=float),
            int(round(metadata.fps or DEFAULT_FPS)),
        )
        boundaries = [int(round(value)) for value in list(boundaries)[:7]]
        if not _valid_tug_boundaries(boundaries, metadata.num_frames):
            warnings.append(f"core bbox phase segmentation returned non-increasing boundaries: {boundaries}")
            return []
        phases = phase_intervals_from_tug_boundaries(boundaries, metadata.fps, metadata.num_frames)
        if len(phases) == 7:
            return phases
        warnings.append(f"core bbox phase segmentation returned incomplete phases: {len(phases)}")
    except Exception as exc:
        warnings.append(f"core bbox phase segmentation failed; using fallback: {exc}")
    return []


def _observed_tug_phases_from_track(
    metadata: VideoMetadata,
    subject: SubjectTrack,
    face_path: Path | None,
    warnings: list[str],
) -> list[Interval]:
    boxes = sorted(
        (box for box in subject.boxes if box.frame >= 0 and (not metadata.num_frames or box.frame < metadata.num_frames)),
        key=lambda box: box.frame,
    )
    if len(boxes) < max(10, int(round((metadata.fps or DEFAULT_FPS) * 1.0))):
        return []

    fps = metadata.fps or DEFAULT_FPS
    frames = [box.frame for box in boxes]
    heights = _moving_average([box.height for box in boxes], max(3, int(round(fps * 0.30))))
    centers_x = _moving_average([box.center_x for box in boxes], max(3, int(round(fps * 0.20))))
    centers_y = _moving_average([box.center_y for box in boxes], max(3, int(round(fps * 0.20))))
    ordered_heights = sorted(heights)
    n = len(ordered_heights)
    sit_level = _median(ordered_heights[: max(1, n // 4)])
    stand_level = _median(ordered_heights[max(1, 3 * n // 4):])
    height_range = stand_level - sit_level
    if height_range <= max(1.0, stand_level * 0.04):
        return []

    speed: list[float] = [0.0]
    height_delta: list[float] = [0.0]
    for idx in range(1, len(boxes)):
        frame_gap = max(1, frames[idx] - frames[idx - 1])
        dx = abs(centers_x[idx] - centers_x[idx - 1]) / max(float(metadata.width), 1.0)
        dy = abs(centers_y[idx] - centers_y[idx - 1]) / max(float(metadata.height), 1.0)
        speed.append((dx + 0.5 * dy) * fps / frame_gap)
        height_delta.append((heights[idx] - heights[idx - 1]) / frame_gap)

    nonzero_speed = [value for value in speed if value > 0]
    motion_threshold = max(0.006, _percentile(nonzero_speed, 0.35) if nonzero_speed else 0.0)
    rise_threshold = max(0.12, height_range * 0.012)
    face_missing_frames = _face_missing_frames(face_path, metadata, warnings)
    turn_frames = _turn_candidate_frames(
        frames,
        centers_x,
        heights,
        speed,
        face_missing_frames,
        motion_threshold,
        fps,
        metadata.width,
        height_range,
    )

    states: list[str] = []
    for idx, box in enumerate(boxes):
        h_norm = (heights[idx] - sit_level) / max(height_range, 1.0)
        moving = speed[idx] >= motion_threshold
        transition_zone = 0.16 < h_norm < 0.84 and speed[idx] < motion_threshold * 1.8
        rising = transition_zone and height_delta[idx] >= rise_threshold
        falling = transition_zone and height_delta[idx] <= -rise_threshold
        if rising:
            label = "sit-to-stand"
        elif falling:
            label = "stand-to-sit"
        elif h_norm <= 0.22 and not moving:
            label = "sit"
        elif box.frame in turn_frames:
            label = "turn"
        elif moving:
            label = "walk"
        else:
            label = "stand"
        states.append(label)

    intervals = _state_sequence_to_intervals(frames, states, fps, 0.74, "patient_state_tracking")
    return _smooth_observed_phase_intervals(intervals)


def _face_missing_frames(
    face_path: Path | None,
    metadata: VideoMetadata,
    warnings: list[str],
) -> set[int]:
    if face_path is None:
        return set()
    try:
        face_boxes = load_detection_json(face_path, metadata.width, metadata.height)
    except Exception as exc:
        warnings.append(f"could not read face bounding boxes for observed turn detection: {exc}")
        return set()
    if not face_boxes:
        return set()
    face_frames = {box.frame for box in face_boxes}
    min_frame = min(face_frames)
    max_frame = max(face_frames)
    return {frame for frame in range(min_frame, max_frame + 1) if frame not in face_frames}


def _turn_candidate_frames(
    frames: list[int],
    centers_x: list[float],
    heights: list[float],
    speed: list[float],
    face_missing_frames: set[int],
    motion_threshold: float,
    fps: float,
    frame_width: int,
    height_range: float,
) -> set[int]:
    turn_frames: set[int] = set()
    for gap_start, gap_end in _contiguous_ranges(sorted(face_missing_frames)):
        gap_ms = frame_to_ms(gap_end - gap_start + 1, fps, "zero")
        if 350 <= gap_ms <= 1200:
            for frame, cur_speed in zip(frames, speed):
                if gap_start <= frame <= gap_end and cur_speed >= motion_threshold * 0.55:
                    turn_frames.add(frame)
    if len(frames) < 5:
        return turn_frames

    window = max(3, int(round(fps * 0.45)))
    h_window = max(3, int(round(fps * 1.00)))
    min_direction_px = max(20.0, frame_width * 0.045)
    min_height_swing = max(8.0, height_range * 0.05)
    for idx in range(window, len(frames) - window):
        left_dx = centers_x[idx] - centers_x[idx - window]
        right_dx = centers_x[idx + window] - centers_x[idx]
        local_speed = max(speed[max(0, idx - window): min(len(speed), idx + window + 1)])
        x_change = (
            left_dx != 0
            and right_dx != 0
            and ((left_dx > 0 > right_dx) or (left_dx < 0 < right_dx))
            and abs(left_dx) >= min_direction_px
            and abs(right_dx) >= min_direction_px
        )
        h_change = False
        if h_window <= idx < len(frames) - h_window:
            left_dh = heights[idx] - heights[idx - h_window]
            right_dh = heights[idx + h_window] - heights[idx]
            # TUG far-turn signature: bbox height direction reverses (depth-out then depth-in, or vice versa).
            h_change = (
                left_dh != 0
                and right_dh != 0
                and ((left_dh < 0 < right_dh) or (left_dh > 0 > right_dh))
                and abs(left_dh) >= min_height_swing
                and abs(right_dh) >= min_height_swing
            )
        if (x_change or h_change) and local_speed >= motion_threshold * 0.55:
            for pos in range(max(0, idx - window), min(len(frames), idx + window + 1)):
                turn_frames.add(frames[pos])
    return turn_frames


def _state_sequence_to_intervals(
    frames: list[int],
    states: list[str],
    fps: float,
    confidence: float,
    source: str,
) -> list[Interval]:
    if not frames or not states:
        return []
    intervals: list[Interval] = []
    start_idx = 0
    cur_label = states[0]
    for idx in range(1, min(len(frames), len(states))):
        if states[idx] != cur_label:
            start_ms = frame_to_ms(frames[start_idx], fps, "zero")
            end_ms = frame_to_ms(frames[idx], fps, "zero")
            if end_ms > start_ms:
                intervals.append(Interval(start_ms, end_ms, cur_label, confidence, source))
            start_idx = idx
            cur_label = states[idx]
    start_ms = frame_to_ms(frames[start_idx], fps, "zero")
    end_ms = frame_to_ms(frames[-1] + 1, fps, "zero")
    if end_ms > start_ms:
        intervals.append(Interval(start_ms, end_ms, cur_label, confidence, source))
    return intervals


def _drop_tug_edge_stands(intervals: list[Interval]) -> list[Interval]:
    return [interval for interval in intervals if interval.label != "stand" or interval.duration_ms >= 180]


def _smooth_observed_phase_intervals(intervals: list[Interval]) -> list[Interval]:
    cleaned = [
        interval
        for interval in sorted(intervals, key=lambda item: (item.start_ms, item.end_ms))
        if interval.end_ms > interval.start_ms
    ]
    if not cleaned:
        return []

    min_ms = 220
    changed = True
    while changed and len(cleaned) > 1:
        changed = False
        out: list[Interval] = []
        idx = 0
        while idx < len(cleaned):
            cur = cleaned[idx]
            if cur.duration_ms >= min_ms:
                out.append(cur)
                idx += 1
                continue
            prev = out[-1] if out else None
            nxt = cleaned[idx + 1] if idx + 1 < len(cleaned) else None
            if prev and nxt and prev.label == nxt.label:
                out[-1] = Interval(
                    prev.start_ms,
                    nxt.end_ms,
                    prev.label,
                    min(prev.confidence, cur.confidence, nxt.confidence),
                    prev.source,
                )
                idx += 2
                changed = True
            elif prev:
                out[-1] = Interval(
                    prev.start_ms,
                    cur.end_ms,
                    prev.label,
                    min(prev.confidence, cur.confidence),
                    prev.source,
                )
                idx += 1
                changed = True
            elif nxt:
                out.append(
                    Interval(
                        cur.start_ms,
                        nxt.end_ms,
                        nxt.label,
                        min(cur.confidence, nxt.confidence),
                        nxt.source,
                    )
                )
                idx += 2
                changed = True
            else:
                out.append(cur)
                idx += 1
        cleaned = _merge_adjacent_phase_labels(out, 180)

    cleaned = _merge_short_stands(cleaned)
    cleaned = _limit_turn_flicker(cleaned)
    cleaned = _merge_adjacent_phase_labels(_drop_tug_edge_stands(cleaned), 180)
    return _merge_isolated_sit_blips(_insert_observed_sit_transitions(cleaned))


def _merge_adjacent_phase_labels(intervals: list[Interval], max_gap_ms: int) -> list[Interval]:
    merged: list[Interval] = []
    for interval in sorted(intervals, key=lambda item: (item.start_ms, item.end_ms)):
        if (
            merged
            and merged[-1].label == interval.label
            and interval.start_ms - merged[-1].end_ms <= max_gap_ms
        ):
            prev = merged[-1]
            merged[-1] = Interval(
                prev.start_ms,
                max(prev.end_ms, interval.end_ms),
                prev.label,
                min(prev.confidence, interval.confidence),
                prev.source,
            )
        else:
            merged.append(interval)
    return merged


def _merge_short_stands(intervals: list[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for idx, interval in enumerate(intervals):
        if interval.label != "stand" or interval.duration_ms >= 500:
            out.append(interval)
            continue
        prev = out[-1] if out else None
        nxt = intervals[idx + 1] if idx + 1 < len(intervals) else None
        if prev and prev.label in {"walk", "turn"}:
            out[-1] = Interval(prev.start_ms, interval.end_ms, prev.label, min(prev.confidence, interval.confidence), prev.source)
        elif nxt and nxt.label in {"walk", "turn"}:
            out.append(Interval(interval.start_ms, interval.end_ms, nxt.label, min(interval.confidence, nxt.confidence), nxt.source))
        else:
            out.append(interval)
    return _merge_adjacent_phase_labels(out, 180)


def _limit_turn_flicker(intervals: list[Interval]) -> list[Interval]:
    turn_intervals = [interval for interval in intervals if interval.label == "turn"]
    turn_count = len(turn_intervals)
    if turn_count <= 1:
        return intervals
    plausible = [iv for iv in turn_intervals if iv.duration_ms <= 2400]
    longest = max(plausible, key=lambda iv: iv.duration_ms) if plausible else None
    out: list[Interval] = []
    for idx, interval in enumerate(intervals):
        if interval.label != "turn":
            out.append(interval)
            continue
        if interval.duration_ms > 2400:
            out.append(Interval(interval.start_ms, interval.end_ms, "walk", interval.confidence, interval.source))
            continue
        if longest is not None and interval is longest and interval.duration_ms >= 350:
            out.append(interval)
            continue
        prev = out[-1] if out else None
        nxt = intervals[idx + 1] if idx + 1 < len(intervals) else None
        replacement = "walk"
        if prev and nxt and prev.label == nxt.label:
            replacement = prev.label
        elif prev and prev.label in {"walk", "stand"}:
            replacement = prev.label
        elif nxt and nxt.label in {"walk", "stand"}:
            replacement = nxt.label
        out.append(Interval(interval.start_ms, interval.end_ms, replacement, interval.confidence, interval.source))
    return _merge_adjacent_phase_labels(out, 180)


def _insert_observed_sit_transitions(intervals: list[Interval]) -> list[Interval]:
    if len(intervals) < 2:
        return intervals
    transition_ms = 650
    moving_labels = {"walk", "turn", "stand"}
    first_pass: list[Interval] = []
    idx = 0
    while idx < len(intervals):
        cur = intervals[idx]
        nxt = intervals[idx + 1] if idx + 1 < len(intervals) else None
        if cur.label == "sit" and nxt and nxt.label in moving_labels:
            first_pass.append(cur)
            trans_end = min(nxt.end_ms, nxt.start_ms + transition_ms)
            if trans_end > nxt.start_ms:
                first_pass.append(Interval(nxt.start_ms, trans_end, "sit-to-stand", min(cur.confidence, nxt.confidence), cur.source))
            if nxt.end_ms > trans_end:
                first_pass.append(Interval(trans_end, nxt.end_ms, nxt.label, nxt.confidence, nxt.source))
            idx += 2
        else:
            first_pass.append(cur)
            idx += 1

    second_pass: list[Interval] = []
    idx = 0
    while idx < len(first_pass):
        cur = first_pass[idx]
        nxt = first_pass[idx + 1] if idx + 1 < len(first_pass) else None
        if cur.label in moving_labels and nxt and nxt.label == "sit":
            trans_start = max(cur.start_ms, cur.end_ms - transition_ms)
            if trans_start > cur.start_ms:
                second_pass.append(Interval(cur.start_ms, trans_start, cur.label, cur.confidence, cur.source))
            if cur.end_ms > trans_start:
                second_pass.append(Interval(trans_start, cur.end_ms, "stand-to-sit", min(cur.confidence, nxt.confidence), cur.source))
            second_pass.append(nxt)
            idx += 2
        else:
            second_pass.append(cur)
            idx += 1
    return _merge_adjacent_phase_labels(second_pass, 1)


def _merge_isolated_sit_blips(intervals: list[Interval]) -> list[Interval]:
    if len(intervals) < 3:
        return intervals
    out: list[Interval] = []
    idx = 0
    while idx < len(intervals):
        if (
            idx + 2 < len(intervals)
            and intervals[idx].label == "sit"
            and intervals[idx + 2].label == "sit"
            and intervals[idx + 1].duration_ms <= 900
        ):
            merged = Interval(
                intervals[idx].start_ms,
                intervals[idx + 2].end_ms,
                "sit",
                min(intervals[idx].confidence, intervals[idx + 1].confidence, intervals[idx + 2].confidence),
                intervals[idx].source,
            )
            out.append(merged)
            idx += 3
        else:
            out.append(intervals[idx])
            idx += 1
    return _merge_adjacent_phase_labels(out, 1)


def _contiguous_ranges(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    return ranges


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round(max(0.0, min(1.0, fraction)) * (len(ordered) - 1)))
    return ordered[index]


def _phase_from_body_motion(
    metadata: VideoMetadata,
    body_boxes: list[DetectionBox],
) -> list[Interval]:
    boxes = sorted(body_boxes, key=lambda b: b.frame)
    if metadata.num_frames:
        boxes = [box for box in boxes if 0 <= box.frame < metadata.num_frames]
    else:
        boxes = [box for box in boxes if box.frame >= 0]
    if len(boxes) < 10:
        return []

    fps = metadata.fps or DEFAULT_FPS
    frames = [box.frame for box in boxes]
    height = _moving_average([box.height for box in boxes], max(3, int(round(fps * 0.75))))
    count = len(height)
    edge = max(3, min(int(round(fps)), int(round(count * 0.10))))
    start_low = _median(height[:edge])
    end_low = _median(height[-edge:])

    peak_index = _argmax(height, int(count * 0.20), int(count * 0.80))
    peak = height[peak_index]
    rise = max(1.0, peak - start_low)
    fall = max(1.0, peak - end_low)

    # Validity gate: the rise must represent a substantial fraction of the height
    # range seen across the track.  When start_low is already close to the peak
    # (e.g. the person was detected only while close to the camera, or a bystander
    # dominates the early track), the sit-to-stand baseline is missing and any
    # derived boundaries are meaningless.  Returning [] lets the caller fall back
    # to proportional phase estimates.
    height_range = max(height) - min(height)
    if rise < height_range * 0.40 or start_low > peak * 0.85:
        return []

    t2 = _first_index(height, 0, peak_index, start_low + 0.30 * rise, direction="ge")
    if t2 is None:
        t2 = max(1, peak_index - int(round(count * 0.45)))

    t1 = _first_index(height, 0, t2, start_low + 0.05 * rise, direction="ge")
    if t1 is None:
        t1 = max(0, t2 - int(round(1.5 * fps)))

    t3 = _first_index(height, t2 + 1, peak_index, peak - 0.16 * rise, direction="ge")
    if t3 is None:
        t3 = max(t2 + 1, peak_index - int(round(0.6 * fps)))

    t4 = _first_index(height, peak_index, count - 1, peak - 0.16 * fall, direction="le")
    if t4 is None:
        t4 = min(count - 1, peak_index + int(round(0.6 * fps)))

    t5 = _first_index(height, t4 + 1, count - 1, end_low + 0.32 * fall, direction="le")
    if t5 is None:
        t5 = min(count - 1, t4 + int(round(0.25 * count)))

    t6 = _first_index(height, t5 + 1, count - 1, end_low + 0.16 * fall, direction="le")
    if t6 is None:
        t6 = min(count - 1, t5 + int(round(fps)))

    t7 = _first_index(height, t6 + 1, count - 1, end_low + 0.06 * fall, direction="le")
    if t7 is None:
        t7 = min(count - 1, t6 + int(round(fps)))

    indices = _repair_boundary_indices([t1, t2, t3, t4, t5, t6, t7], count, fps)
    boundaries = [frames[index] + 1 for index in indices]
    if not _valid_tug_boundaries(boundaries, metadata.num_frames):
        return []

    phases = phase_intervals_from_tug_boundaries(boundaries, fps, metadata.num_frames)
    if len(phases) != 7:
        return []
    return [
        Interval(phase.start_ms, phase.end_ms, phase.label, 0.70, "bbox_motion_fallback")
        for phase in phases
    ]


def _repair_boundary_indices(indices: list[int], count: int, fps: float) -> list[int]:
    min_gap = max(2, int(round(0.25 * fps)))
    repaired = [max(0, min(count - 1, int(index))) for index in indices]
    for idx in range(1, len(repaired)):
        if repaired[idx] <= repaired[idx - 1] + min_gap:
            repaired[idx] = repaired[idx - 1] + min_gap
    if repaired[-1] >= count:
        for idx in range(len(repaired) - 1, -1, -1):
            max_allowed = count - 1 - (len(repaired) - 1 - idx) * min_gap
            repaired[idx] = min(repaired[idx], max_allowed)
        for idx in range(1, len(repaired)):
            repaired[idx] = max(repaired[idx], repaired[idx - 1] + min_gap)
    return [max(0, min(count - 1, index)) for index in repaired]


def _has_core_turn_gap(face_boxes: list[DetectionBox], fps: float) -> bool:
    frames = sorted({box.frame for box in face_boxes})
    gaps = [right - left for left, right in zip(frames, frames[1:])]
    if not gaps:
        return False
    gap_index, gap = max(enumerate(gaps), key=lambda item: item[1])
    return gap_index > 0 and gap >= int(round(fps or DEFAULT_FPS))


def _valid_tug_boundaries(boundaries: list[int], num_frames: int | None) -> bool:
    if len(boundaries) < 7:
        return False
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        return False
    if boundaries[0] < 1:
        return False
    if num_frames and boundaries[-1] > num_frames + 1:
        return False
    return True


def _optional_core_tug_bbox() -> Any | None:
    _ensure_repo_package_paths()
    try:
        module = importlib.import_module("core_tests.tug.phase_segmentation")
        return getattr(module, "do_phase_seg_bbox", None)
    except Exception:
        return None


def _ensure_repo_package_paths() -> None:
    root = Path(__file__).resolve().parents[3]
    for package_dir in ("core_tests", "core_utils"):
        path = root / "packages" / package_dir
        if path.exists():
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)


def _select_bundle_file(
    folder: Path,
    pattern: str,
    video_path: Path | None,
    description: str,
    warnings: list[str],
) -> Path | None:
    candidates = sorted(folder.glob(pattern))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if video_path is None:
        warnings.append(f"multiple {description} files found; using {candidates[0].name}")
        return candidates[0]

    scored = [(_filename_match_score(candidate, video_path), candidate) for candidate in candidates]
    score, selected = max(scored, key=lambda item: (item[0], -candidates.index(item[1])))
    if score <= 0:
        warnings.append(f"multiple {description} files found without a video filename match; using {selected.name}")
    return selected


def _filename_match_score(candidate: Path, video_path: Path) -> int:
    candidate_tokens = _meaningful_filename_tokens(candidate)
    video_tokens = _meaningful_filename_tokens(video_path)
    if not candidate_tokens or not video_tokens:
        return 0
    return len(candidate_tokens & video_tokens)


def _meaningful_filename_tokens(path: Path) -> set[str]:
    ignored = {
        "bbox",
        "blurred",
        "bounding",
        "box",
        "data",
        "face",
        "mp4",
        "rgb",
        "video",
        "json",
    }
    tokens = {
        token
        for token in re.findall(r"[A-Za-z0-9]+", path.stem.lower())
        if len(token) > 1 and token not in ignored
    }
    return tokens


def _moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(1, int(window))
    half = window // 2
    averaged: list[float] = []
    for index in range(len(values)):
        start = max(0, index - half)
        end = min(len(values), index + half + 1)
        averaged.append(sum(values[start:end]) / (end - start))
    return averaged


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _argmax(values: list[float], start: int, end: int) -> int:
    start = max(0, start)
    end = min(len(values) - 1, end)
    if end < start:
        return start
    return max(range(start, end + 1), key=lambda index: values[index])


def _first_index(
    values: list[float],
    start: int,
    end: int,
    threshold: float,
    direction: str,
) -> int | None:
    start = max(0, start)
    end = min(len(values) - 1, end)
    if end < start:
        return None
    for index in range(start, end + 1):
        if direction == "ge" and values[index] >= threshold:
            return index
        if direction == "le" and values[index] <= threshold:
            return index
    return None


def _fill_subject_track_gaps(subject: SubjectTrack, max_gap_frames: int = 60) -> SubjectTrack:
    """Linearly interpolate missing frames in the subject track for short gaps.

    Gaps longer than max_gap_frames are left as-is — they likely indicate the
    person genuinely left the frame or the tracker made an irrecoverable error.
    """
    if len(subject.boxes) < 2:
        return subject
    boxes = sorted(subject.boxes, key=lambda b: b.frame)
    filled: list[DetectionBox] = [boxes[0]]
    for i in range(1, len(boxes)):
        prev = boxes[i - 1]
        curr = boxes[i]
        gap = curr.frame - prev.frame
        if 1 < gap <= max_gap_frames:
            for g in range(1, gap):
                t = g / gap
                filled.append(DetectionBox(
                    prev.frame + g,
                    prev.x1 + t * (curr.x1 - prev.x1),
                    prev.y1 + t * (curr.y1 - prev.y1),
                    prev.x2 + t * (curr.x2 - prev.x2),
                    prev.y2 + t * (curr.y2 - prev.y2),
                    track_id=prev.track_id,
                    score=0.0,
                    label=prev.label,
                ))
        filled.append(curr)
    return SubjectTrack(
        subject.track_id,
        filled,
        subject.confidence,
        subject.reason,
        list(subject.locked_track_ids),
    )


def _occlusion_intervals_from_tracks(
    subject: SubjectTrack,
    tracked_boxes: list[DetectionBox],
    metadata: VideoMetadata,
    warnings: list[str],
    face_path: Path | None = None,
    person_masks: "Any | None" = None,
    named_test_type: str | None = None,
) -> list[Interval]:
    if not subject.boxes or not tracked_boxes:
        return []

    locked_ids = set(subject.locked_track_ids)
    if subject.track_id is not None:
        locked_ids.add(subject.track_id)

    others_by_frame: dict[int, list[DetectionBox]] = defaultdict(list)
    for box in tracked_boxes:
        if box.track_id in locked_ids:
            continue
        others_by_frame[box.frame].append(box)

    face_boxes_by_frame = _face_boxes_by_frame(face_path, metadata, warnings)
    subject_core_by_frame = _core_boxes_for_track(subject.boxes, face_boxes_by_frame)
    subject_face_by_frame = {
        box.frame: _matched_face_for_body(box, face_boxes_by_frame.get(box.frame, []))
        for box in subject.boxes
    }
    other_core_by_frame: dict[int, list[tuple[DetectionBox, DetectionBox, DetectionBox | None]]] = defaultdict(list)
    for frame, boxes in others_by_frame.items():
        other_core_by_frame[frame] = [
            (
                box,
                _body_core_box(box, _matched_face_for_body(box, face_boxes_by_frame.get(frame, []))),
                _matched_face_for_body(box, face_boxes_by_frame.get(frame, [])),
            )
            for box in boxes
        ]

    mask_intervals = _occlusion_intervals_from_person_masks(
        subject,
        metadata,
        person_masks,
        subject_core_by_frame,
        named_test_type,
    )

    real_subject_boxes = [box for box in subject.boxes if box.score != 0.0]
    median_area = _median([box.area for box in real_subject_boxes]) or _median([box.area for box in subject.boxes])
    median_core_area = _median([box.area for box in subject_core_by_frame.values()]) or median_area

    frame_events: dict[int, tuple[str, float]] = {}
    for box in sorted(subject.boxes, key=lambda item: item.frame):
        if metadata.num_frames and not (0 <= box.frame < metadata.num_frames):
            continue
        subject_core = subject_core_by_frame.get(box.frame) or box
        others = other_core_by_frame.get(box.frame, [])
        label, confidence = _occlusion_label_for_frame(
            box,
            subject_core,
            subject_face_by_frame.get(box.frame),
            others,
            metadata,
            median_area,
            median_core_area,
        )
        if label:
            frame_events[box.frame] = (label, confidence)

    intervals = _frame_events_to_intervals(frame_events, metadata.fps)
    if named_test_type == TEST_TYPE_GAIT_SPEED and not intervals:
        intervals = _relaxed_gait_occlusion_intervals(
            subject,
            metadata,
            subject_core_by_frame,
            other_core_by_frame,
        )
    if mask_intervals is not None:
        if mask_intervals:
            warnings.append(f"detected {len(mask_intervals)} patient occlusion interval(s) from person masks")
            return mask_intervals
        if named_test_type in BALANCE_TEST_TYPES:
            return []
    if intervals:
        warnings.append(f"detected {len(intervals)} patient occlusion interval(s) from nearby person tracks")
    return intervals


def _occlusion_intervals_from_person_masks(
    subject: SubjectTrack,
    metadata: VideoMetadata,
    person_masks: "Any | None",
    subject_core_by_frame: dict[int, DetectionBox],
    test_type: str | None = None,
) -> list[Interval] | None:
    if person_masks is None or not getattr(person_masks, "frames", None):
        return None
    # Balance/hold tests routinely have an RA standing close to the patient. That
    # proximity creates mask overlap without being an actual visibility loss, so
    # skip mask-based occlusion inference for these test types.
    if test_type in BALANCE_TEST_TYPES:
        return []
    try:
        import numpy as np
    except Exception:
        return None

    frames_by_index = person_masks.frames_by_index
    subject_boxes = sorted(subject.boxes, key=lambda box: box.frame)
    if not subject_boxes:
        return None
    patient_track_id = (
        _locked_patient_mask_track_id(person_masks, subject_boxes, metadata)
        if _should_use_patient_mask_track_ids(person_masks)
        else None
    )

    usable_frames = 0
    frame_events: dict[int, tuple[str, float]] = {}
    for subject_box in subject_boxes:
        frame = frames_by_index.get(subject_box.frame)
        if frame is None or not frame.persons:
            continue
        decoded: list[tuple[int, Any, Any]] = []
        for idx, person in enumerate(frame.persons):
            mask = person.to_array()
            if mask is not None:
                decoded.append((idx, person, mask))
        if not decoded:
            continue

        reference_shape = decoded[0][2].shape
        subject_core = subject_core_by_frame.get(subject_box.frame) or subject_box
        expected = _box_mask(reference_shape, subject_core, metadata)
        expected_area = int(expected.sum())
        if expected_area <= 0:
            continue

        usable_frames += 1
        patient_idx = _match_patient_mask_by_track_id(frame.persons, patient_track_id, subject_box, metadata)
        if patient_idx is None:
            patient_idx = _match_patient_mask(frame.persons, subject_box, metadata)
        if patient_idx is None:
            continue
        patient_bbox = frame.persons[patient_idx].bbox if patient_idx is not None else None
        patient_visible = 0.0
        for idx, _person, mask in decoded:
            if idx == patient_idx:
                patient_visible = float(np.logical_and(mask, expected).sum()) / expected_area
                break

        best_cover = 0.0
        best_confidence = 0.0
        best_foreground = False
        for idx, person, mask in decoded:
            if patient_idx is not None and idx == patient_idx:
                continue
            if patient_bbox is not None and _mask_person_is_behind_patient(patient_bbox, person.bbox, metadata):
                continue
            cover = float(np.logical_and(mask, expected).sum()) / expected_area
            if cover <= best_cover:
                continue
            foreground = _mask_person_is_foreground_occluder(
                subject_box,
                subject_core,
                person.bbox,
                metadata,
                cover,
                patient_bbox,
            )
            if not foreground and cover < 0.18:
                continue
            best_cover = cover
            best_confidence = float(getattr(person, "confidence", 0.0) or 0.0)
            best_foreground = foreground

        trigger = best_cover >= 0.15 and (
            patient_visible < 0.55
            or best_cover >= 0.28
            or (best_foreground and best_cover >= 0.20 and patient_visible < 0.80)
        )
        if trigger:
            confidence = max(0.55, min(0.95, 0.62 + best_cover * 0.60 + best_confidence * 0.10))
            frame_events[subject_box.frame] = ("occluded_by_person", round(confidence, 2))
    min_usable = max(10, int(round(len(subject_boxes) * 0.20)))
    if usable_frames < min_usable:
        return None
    return _frame_events_to_intervals(frame_events, metadata.fps)


def _should_use_patient_mask_track_ids(person_masks: "Any") -> bool:
    if os.environ.get("AUTO_ANNOTATE_USE_MASK_TRACK_IDS") == "1":
        return True
    metadata = getattr(person_masks, "metadata", {}) or {}
    return metadata.get("track_association") == "ultralytics_tracker" and bool(metadata.get("tracker"))


def _locked_patient_mask_track_id(
    person_masks: "Any",
    subject_boxes: list[DetectionBox],
    metadata: VideoMetadata,
) -> int | None:
    scores: dict[int, float] = defaultdict(float)
    hits: dict[int, int] = defaultdict(int)
    frames_by_index = person_masks.frames_by_index
    for subject_box in subject_boxes:
        frame = frames_by_index.get(subject_box.frame)
        if frame is None:
            continue
        for person in frame.persons:
            track_id = getattr(person, "track_id", None)
            if track_id is None:
                continue
            score = _patient_mask_match_score(person, subject_box, metadata)
            if score <= 0.03:
                continue
            scores[int(track_id)] += score
            hits[int(track_id)] += 1
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_track_id, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    min_hits = 2 if len(subject_boxes) < 30 else 3
    if hits[best_track_id] < min_hits:
        return None
    if best_score < 0.35:
        return None
    if second_score > 0 and best_score < second_score * 1.08 and hits[best_track_id] <= hits[ranked[1][0]] + 1:
        return None
    return best_track_id


def _match_patient_mask_by_track_id(
    persons: list[Any],
    track_id: int | None,
    subject_box: DetectionBox,
    metadata: VideoMetadata,
) -> int | None:
    if track_id is None:
        return None
    for idx, person in enumerate(persons):
        if (
            getattr(person, "track_id", None) == track_id
            and _patient_mask_match_score(person, subject_box, metadata) >= 0.02
        ):
            return idx
    return None


def _relaxed_gait_occlusion_intervals(
    subject: SubjectTrack,
    metadata: VideoMetadata,
    subject_core_by_frame: dict[int, DetectionBox],
    other_core_by_frame: dict[int, list[tuple[DetectionBox, DetectionBox, DetectionBox | None]]],
) -> list[Interval]:
    frame_events: dict[int, tuple[str, float]] = {}
    for subject_box in sorted(subject.boxes, key=lambda item: item.frame):
        if metadata.num_frames and not (0 <= subject_box.frame < metadata.num_frames):
            continue
        subject_core = subject_core_by_frame.get(subject_box.frame) or subject_box
        best_overlap = 0.0
        best_distance = 1.0
        for _other_body, other_core, _other_face in other_core_by_frame.get(subject_box.frame, []):
            overlap = _intersection_area(subject_core, other_core) / max(subject_core.area, 1.0)
            if overlap <= best_overlap:
                continue
            best_overlap = overlap
            best_distance = _center_distance_ratio(subject_core, other_core, metadata)
        if best_overlap >= 0.24 and best_distance <= 0.16:
            frame_events[subject_box.frame] = (
                "occluded_by_person",
                round(min(0.82, 0.58 + best_overlap * 0.35), 2),
            )
    return _frame_events_to_intervals(frame_events, metadata.fps, max_gap_ms=850, min_duration_ms=450)


def _match_patient_mask(
    persons: list[Any],
    subject_box: DetectionBox,
    metadata: VideoMetadata,
) -> int | None:
    best_idx: int | None = None
    best_score = 0.0
    for idx, person in enumerate(persons):
        score = _patient_mask_match_score(person, subject_box, metadata)
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx if best_score >= 0.04 else None


def _patient_mask_match_score(person: Any, subject_box: DetectionBox, metadata: VideoMetadata) -> float:
    bbox = getattr(person, "bbox", None)
    if bbox is None:
        return 0.0
    overlap = _bbox_iou_tuple(bbox, subject_box)
    distance = _bbox_center_distance_tuple(bbox, subject_box, metadata)
    area_ratio = _bbox_area_tuple(bbox) / max(subject_box.area, 1.0)
    small_penalty = max(0.0, 0.55 - min(area_ratio, 0.55)) * 0.45
    oversized_penalty = max(0.0, min(area_ratio - 2.2, 1.0)) * 0.10
    return overlap - min(0.30, distance * 0.45) - small_penalty - oversized_penalty


def _mask_person_is_foreground_occluder(
    subject_box: DetectionBox,
    subject_core: DetectionBox,
    person_bbox: tuple[float, float, float, float],
    metadata: VideoMetadata,
    cover: float,
    patient_bbox: tuple[float, float, float, float] | None = None,
) -> bool:
    if cover <= 0:
        return False
    if patient_bbox is not None and _mask_person_is_behind_patient(patient_bbox, person_bbox, metadata):
        return False
    height = max(float(metadata.height), 1.0)
    bottom_delta = (float(person_bbox[3]) - subject_box.y2) / height
    area_ratio = _bbox_area_tuple(person_bbox) / max(subject_box.area, 1.0)
    distance = _bbox_center_distance_tuple(person_bbox, subject_core, metadata)
    return (
        bottom_delta >= 0.035
        or area_ratio >= 1.35
        or (cover >= 0.18 and distance <= 0.14 and bottom_delta >= -0.03)
    )


def _mask_person_is_behind_patient(
    patient_bbox: tuple[float, float, float, float],
    person_bbox: tuple[float, float, float, float],
    metadata: VideoMetadata,
) -> bool:
    height = max(float(metadata.height), 1.0)
    patient_area = _bbox_area_tuple(patient_bbox)
    person_area = _bbox_area_tuple(person_bbox)
    bottom_gap = (float(patient_bbox[3]) - float(person_bbox[3])) / height
    return bottom_gap >= 0.10 and patient_area >= person_area * 1.15


def _box_mask(shape: tuple[int, int], box: DetectionBox, metadata: VideoMetadata) -> Any:
    import numpy as np

    height, width = int(shape[0]), int(shape[1])
    scale_x = width / max(float(metadata.width), 1.0)
    scale_y = height / max(float(metadata.height), 1.0)
    x1 = max(0, min(width, int(round(box.x1 * scale_x))))
    y1 = max(0, min(height, int(round(box.y1 * scale_y))))
    x2 = max(0, min(width, int(round(box.x2 * scale_x))))
    y2 = max(0, min(height, int(round(box.y2 * scale_y))))
    mask = np.zeros((height, width), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def _bbox_iou_tuple(bbox: tuple[float, float, float, float], box: DetectionBox) -> float:
    x1 = max(float(bbox[0]), box.x1)
    y1 = max(float(bbox[1]), box.y1)
    x2 = min(float(bbox[2]), box.x2)
    y2 = min(float(bbox[3]), box.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(_bbox_area_tuple(bbox) + box.area - inter, 1.0)


def _bbox_area_tuple(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_center_distance_tuple(
    bbox: tuple[float, float, float, float],
    box: DetectionBox,
    metadata: VideoMetadata,
) -> float:
    center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
    center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
    dx = abs(center_x - box.center_x) / max(float(metadata.width), 1.0)
    dy = abs(center_y - box.center_y) / max(float(metadata.height), 1.0)
    return (dx * dx + dy * dy) ** 0.5


def _occlusion_label_for_frame(
    subject_box: DetectionBox,
    subject_core: DetectionBox,
    subject_face: DetectionBox | None,
    other_boxes: list[tuple[DetectionBox, DetectionBox, DetectionBox | None]],
    metadata: VideoMetadata,
    median_subject_area: float,
    median_subject_core_area: float,
) -> tuple[str | None, float]:
    best_overlap = 0.0
    best_iou = 0.0
    best_front_overlap = 0.0
    for other_body, other_core, other_face in other_boxes:
        inter = _intersection_area(subject_core, other_core)
        subject_overlap = inter / max(subject_core.area, 1.0)
        best_overlap = max(best_overlap, subject_overlap)
        best_iou = max(best_iou, iou(subject_core, other_core))
        distance = _center_distance_ratio(subject_core, other_core, metadata)
        if _is_foreground_occluder(
            subject_box,
            subject_core,
            subject_face,
            other_body,
            other_core,
            other_face,
            metadata,
            subject_overlap,
            distance,
        ):
            best_front_overlap = max(best_front_overlap, subject_overlap)

    interpolated = subject_box.score == 0.0

    if best_front_overlap >= 0.04:
        confidence = 0.70 + min(0.20, max(best_front_overlap, best_iou) * 0.45)
        if interpolated:
            confidence = min(confidence, 0.78)
        return "occluded_by_person", round(confidence, 2)
    return None, 0.0


def _is_foreground_occluder(
    subject_box: DetectionBox,
    subject_core: DetectionBox,
    subject_face: DetectionBox | None,
    other_body: DetectionBox,
    other_core: DetectionBox,
    other_face: DetectionBox | None,
    metadata: VideoMetadata,
    subject_overlap: float,
    distance: float,
) -> bool:
    if subject_overlap <= 0:
        return False

    axis_overlap = _axis_overlap_ratio(subject_core, other_core)
    if axis_overlap < 0.04 or distance > 0.24:
        return False

    height = max(float(metadata.height), 1.0)
    width = max(float(metadata.width), 1.0)
    bottom_delta = (other_body.y2 - subject_box.y2) / height
    core_bottom_delta = (other_core.y2 - subject_core.y2) / height
    area_ratio = other_body.area / max(subject_box.area, 1.0)
    close_cover = subject_overlap >= 0.25 and distance <= 0.12 and bottom_delta >= -0.025
    foreground_depth = bottom_delta >= 0.035 or core_bottom_delta >= 0.035 or area_ratio >= 1.45

    if _same_face_detection(subject_face, other_face, width, height):
        # Broad duplicate boxes around the locked patient often share the same
        # face detection. Keep only cases where the other body has independent
        # foreground evidence and is not just a duplicate patient box.
        duplicate_like = area_ratio < 1.30 and bottom_delta < 0.045 and subject_overlap >= 0.50
        if duplicate_like:
            return False
        return foreground_depth and (subject_overlap >= 0.08 or close_cover)

    return foreground_depth or close_cover


def _same_face_detection(
    a: DetectionBox | None,
    b: DetectionBox | None,
    frame_width: float,
    frame_height: float,
) -> bool:
    if a is None or b is None:
        return False
    dx = abs(a.center_x - b.center_x) / max(frame_width, 1.0)
    dy = abs(a.center_y - b.center_y) / max(frame_height, 1.0)
    size = abs(a.area - b.area) / max(a.area, b.area, 1.0)
    return dx <= 0.01 and dy <= 0.01 and size <= 0.20


def _frame_events_to_intervals(
    frame_events: dict[int, tuple[str, float]],
    fps: float,
    max_gap_ms: int = 200,
    min_duration_ms: int = OCCLUSION_MIN_DURATION_MS,
) -> list[Interval]:
    if not frame_events:
        return []
    fps = fps or DEFAULT_FPS
    intervals: list[Interval] = []
    items = sorted(frame_events.items())
    start_frame = prev_frame = items[0][0]
    label, confidence = items[0][1]
    max_gap = max(1, int(round(fps * max_gap_ms / 1000.0)))

    for frame, (cur_label, cur_confidence) in items[1:]:
        if frame - prev_frame > max_gap or cur_label != label:
            start_ms = frame_to_ms(start_frame, fps, "zero")
            end_ms = frame_to_ms(prev_frame + 1, fps, "zero")
            if end_ms > start_ms:
                intervals.append(Interval(start_ms, end_ms, label, confidence, "patient_lock_occlusion"))
            start_frame = frame
            label = cur_label
            confidence = cur_confidence
        else:
            confidence = max(confidence, cur_confidence)
        prev_frame = frame

    start_ms = frame_to_ms(start_frame, fps, "zero")
    end_ms = frame_to_ms(prev_frame + 1, fps, "zero")
    if end_ms > start_ms:
        intervals.append(Interval(start_ms, end_ms, label, confidence, "patient_lock_occlusion"))
    return _smooth_occlusion_intervals(intervals, min_duration_ms=min_duration_ms)


def _smooth_occlusion_intervals(
    intervals: list[Interval],
    min_duration_ms: int = OCCLUSION_MIN_DURATION_MS,
) -> list[Interval]:
    cleaned = [
        interval
        for interval in sorted(intervals, key=lambda item: (item.start_ms, item.end_ms, item.label))
        if interval.duration_ms >= min_duration_ms
        and interval.label in {"occluded_by_person", "occlusion_inferred", "outside_frame"}
    ]
    if not cleaned:
        return []
    merged: list[Interval] = []
    for interval in cleaned:
        if (
            merged
            and merged[-1].label == interval.label
            and interval.start_ms - merged[-1].end_ms <= OCCLUSION_GAP_MERGE_MS
        ):
            prev = merged[-1]
            merged[-1] = Interval(
                prev.start_ms,
                max(prev.end_ms, interval.end_ms),
                prev.label,
                max(prev.confidence, interval.confidence),
                prev.source or interval.source,
            )
        else:
            merged.append(interval)
    return merged


def _face_boxes_by_frame(
    face_path: Path | None,
    metadata: VideoMetadata,
    warnings: list[str],
) -> dict[int, list[DetectionBox]]:
    if face_path is None:
        return {}
    try:
        face_boxes = load_detection_json(face_path, metadata.width, metadata.height)
    except Exception as exc:
        warnings.append(f"could not read face bounding boxes for occlusion body-core matching: {exc}")
        return {}
    by_frame: dict[int, list[DetectionBox]] = defaultdict(list)
    for box in face_boxes:
        by_frame[box.frame].append(box)
    return by_frame


def _core_boxes_for_track(
    boxes: list[DetectionBox],
    face_boxes_by_frame: dict[int, list[DetectionBox]],
) -> dict[int, DetectionBox]:
    return {
        box.frame: _body_core_box(box, _matched_face_for_body(box, face_boxes_by_frame.get(box.frame, [])))
        for box in boxes
    }


def _body_core_box(body: DetectionBox, face: DetectionBox | None = None) -> DetectionBox:
    """Return a tighter body core used only for occlusion decisions.

    The person detector boxes in the FrailScreen bundles are intentionally broad
    and often overlap adjacent standing people.  Face boxes give a better body
    center when available; otherwise we keep only the central part of the bbox.
    """
    if face is not None:
        face_center_x = face.center_x
        estimated_width = max(face.width * 4.0, body.width * 0.28)
        core_width = min(body.width * 0.58, estimated_width)
        core_width = max(core_width, body.width * 0.24)
        x1 = max(body.x1, face_center_x - core_width / 2.0)
        x2 = min(body.x2, face_center_x + core_width / 2.0)
        if x2 <= x1:
            x1, x2 = _center_crop_x(body, 0.42)
    else:
        x1, x2 = _center_crop_x(body, 0.42)
    return DetectionBox(
        body.frame,
        x1,
        body.y1 + body.height * 0.02,
        x2,
        body.y2,
        track_id=body.track_id,
        score=body.score,
        label=body.label,
    )


def _matched_face_for_body(body: DetectionBox, faces: list[DetectionBox]) -> DetectionBox | None:
    candidates: list[tuple[float, DetectionBox]] = []
    for face in faces:
        if face.center_x < body.x1 - body.width * 0.10 or face.center_x > body.x2 + body.width * 0.10:
            continue
        if face.center_y < body.y1 - body.height * 0.10 or face.center_y > body.y1 + body.height * 0.45:
            continue
        x_distance = abs(face.center_x - body.center_x) / max(body.width, 1.0)
        y_target = body.y1 + body.height * 0.18
        y_distance = abs(face.center_y - y_target) / max(body.height, 1.0)
        size_bonus = min(0.25, face.area / max(body.area, 1.0) * 20.0)
        score = x_distance + y_distance - size_bonus
        candidates.append((score, face))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _center_crop_x(box: DetectionBox, width_ratio: float) -> tuple[float, float]:
    width = max(1.0, box.width * width_ratio)
    return box.center_x - width / 2.0, box.center_x + width / 2.0


def _intersection_area(a: DetectionBox, b: DetectionBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _center_distance_ratio(a: DetectionBox, b: DetectionBox, metadata: VideoMetadata) -> float:
    dx = abs(a.center_x - b.center_x) / max(float(metadata.width), 1.0)
    dy = abs(a.center_y - b.center_y) / max(float(metadata.height), 1.0)
    return (dx * dx + dy * dy) ** 0.5


def _axis_overlap_ratio(a: DetectionBox, b: DetectionBox) -> float:
    x_overlap = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) / max(min(a.width, b.width), 1.0)
    y_overlap = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1)) / max(min(a.height, b.height), 1.0)
    return min(x_overlap, y_overlap)


def _box_touches_frame_edge(box: DetectionBox, metadata: VideoMetadata) -> bool:
    margin_x = max(4.0, metadata.width * 0.015)
    margin_y = max(4.0, metadata.height * 0.015)
    return (
        box.x1 <= margin_x
        or box.y1 <= margin_y
        or box.x2 >= metadata.width - margin_x
        or box.y2 >= metadata.height - margin_y
    )


def _generate_bboxes_from_video(
    video_path: Path,
    metadata: VideoMetadata,
    warnings: list[str],
) -> list[DetectionBox]:
    """Generate approximate person bboxes via background subtraction when no bbox JSON exists."""
    try:
        import cv2
    except ImportError:
        warnings.append("cv2 not available; cannot detect person from unblurred video")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warnings.append(f"could not open {video_path.name} for person detection")
        return []

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or metadata.width
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or metadata.height
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or metadata.num_frames

    scale = min(1.0, 640.0 / max(w, h, 1))
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))

    # Moderate closing kernel (sw//20 ≈ 24px in 480px-wide scaled image) merges
    # body-part fragments without expanding noise pixels into giant blobs.
    # sw//8 (60px) was too large — a single noise pixel became a 378px blob in original coords.
    kw_close = max(5, sw // 20)
    kh_close = max(5, sh // 20)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw_close, kh_close))

    # history=150 keeps seated subjects visible for ~5s; varThreshold=40 filters
    # camera noise that the smaller threshold (16) was amplifying via morphological closing.
    back_sub = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40, detectShadows=False)

    min_area_scaled = sw * sh * 0.015  # 1.5% of scaled frame area
    warmup = min(15, max(1, n // 4))
    boxes: list[DetectionBox] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (sw, sh)) if scale < 1.0 else frame
        blurred = cv2.GaussianBlur(small, (5, 5), 0)
        fg = back_sub.apply(blurred)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close)

        if frame_idx >= warmup:
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = sorted(
                (cnt for cnt in contours if cv2.contourArea(cnt) >= min_area_scaled),
                key=cv2.contourArea,
                reverse=True,
            )[:3]
            for cnt in valid:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                boxes.append(DetectionBox(
                    frame_idx,
                    cx / scale,
                    cy / scale,
                    (cx + cw) / scale,
                    (cy + ch) / scale,
                ))
        frame_idx += 1

    cap.release()
    if boxes:
        warnings.append(
            f"no bounding box JSON found; generated {len(boxes)} detections "
            f"from {video_path.name} via background subtraction"
        )
    else:
        warnings.append(f"background subtraction on {video_path.name} found no foreground regions")
    return boxes


def _load_boxes_safely(
    path: Path | None,
    metadata: VideoMetadata,
    warnings: list[str],
) -> list[DetectionBox]:
    if not path:
        warnings.append("missing bounding_box_data_*.json")
        return []
    try:
        return load_detection_json(path, metadata.width, metadata.height)
    except Exception as exc:
        warnings.append(f"could not read bounding box data: {exc}")
        return []


def _read_video_metadata(video_path: Path | None) -> VideoMetadata:
    if not video_path:
        return VideoMetadata(None, DEFAULT_FPS, 0, DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT)
    if video_path.exists() and video_path.stat().st_size == 0:
        return VideoMetadata(video_path, DEFAULT_FPS, 0, DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT)
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or DEFAULT_VIDEO_WIDTH)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or DEFAULT_VIDEO_HEIGHT)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return VideoMetadata(video_path, fps if fps > 0 else DEFAULT_FPS, frames, width, height)
    except Exception:
        return VideoMetadata(video_path, DEFAULT_FPS, 0, DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT)


def _load_meta_dataframe(folder: Path, warnings: list[str]) -> dict[str, Any] | None:
    search_dirs = [folder] + _sibling_out_dirs(folder) + _visfrailty_output_dirs(folder)
    pkl_patterns = ("out2.pkl", "meta*.pkl", "**/out2.pkl")
    csv_patterns = ("out2.csv", "meta*.csv", "**/out2.csv")
    seen: set[Path] = set()
    read_errors: list[str] = []
    for search_dir in search_dirs:
        if search_dir in seen:
            continue
        seen.add(search_dir)
        for pattern in pkl_patterns:
            for path_str in glob.glob(str(search_dir / pattern), recursive=True):
                path = Path(path_str)
                try:
                    import pandas as pd

                    df = pd.read_pickle(path)
                    if len(df) > 0:
                        return {col: df.loc[df.index[0], col] for col in df.columns}
                except Exception as exc:
                    read_errors.append(f"could not read {path.name}: {exc}")
        for pattern in csv_patterns:
            for path_str in glob.glob(str(search_dir / pattern), recursive=True):
                path = Path(path_str)
                row = _load_csv_first_row(path)
                if row is not None:
                    return row
                try:
                    import pandas as pd

                    df = pd.read_csv(path)
                    if len(df) > 0:
                        return {col: df.loc[df.index[0], col] for col in df.columns}
                except Exception as exc:
                    read_errors.append(f"could not read {path.name}: {exc}")
    warnings.extend(read_errors[:3])
    return None


def _load_csv_first_row(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                return dict(row)
    except Exception:
        return None
    return None


def _sibling_out_dirs(folder: Path) -> list[Path]:
    """Find output directories in sibling 'out/' folders matching this folder's name."""
    name = folder.name.lower()
    candidates: list[Path] = []
    for out_root in (folder.parent / "out", folder.parent.parent / "out"):
        if not out_root.is_dir():
            continue
        for child in out_root.iterdir():
            if child.is_dir() and name in child.name.lower():
                candidates.append(child)
    return candidates


def _visfrailty_output_dirs(folder: Path) -> list[Path]:
    """Find matching visfrailty output test folders for a raw FrailScreen bundle."""
    target_name = _normalized_name(folder.name)
    if not target_name:
        return []

    candidates: list[Path] = []
    for patient_dir in _visfrailty_patient_output_dirs(folder):
        direct = patient_dir / folder.name
        if direct.is_dir():
            candidates.append(direct)
        try:
            children = list(patient_dir.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_dir() and _normalized_name(child.name) == target_name:
                candidates.append(child)
    return sorted(set(candidates), key=lambda path: str(path).lower(), reverse=True)


def _visfrailty_patient_output_dirs(folder: Path) -> list[Path]:
    patient_id = _infer_patient_id(folder)
    volume_root = _infer_volume_root(folder)
    if not patient_id or volume_root is None:
        return []

    output_root = volume_root / "visfrailty_outputs"
    if not output_root.is_dir():
        return []

    cache_key = (str(output_root), patient_id.lower())
    cached = _VISFRAILTY_PATIENT_OUTPUT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    matches: list[Path] = []
    try:
        date_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    except OSError:
        date_dirs = []
    for date_dir in date_dirs:
        try:
            patient_dirs = [path for path in date_dir.iterdir() if path.is_dir()]
        except OSError:
            continue
        for patient_dir in patient_dirs:
            if patient_id.lower() in patient_dir.name.lower():
                matches.append(patient_dir)

    matches = sorted(set(matches), key=lambda path: str(path).lower(), reverse=True)
    _VISFRAILTY_PATIENT_OUTPUT_CACHE[cache_key] = matches
    return matches


def _infer_patient_id(folder: Path) -> str | None:
    for part in reversed(folder.parts):
        if re.search(r"[a-z]{2,}.*\d|\d{4,}", part.lower()):
            if _test_type_from_name(part) is None and part not in {"/", ""}:
                return part
    return folder.parent.name if folder.parent.name else None


def _infer_volume_root(folder: Path) -> Path | None:
    parts = folder.resolve().parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        return Path(*parts[:3])
    for parent in (folder, *folder.parents):
        if (parent / "visfrailty_outputs").is_dir():
            return parent
    return None


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _parse_literalish(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "nan":
            return None
        try:
            return ast.literal_eval(text)
        except Exception:
            return None
    return value


def _parse_gait_breakdown(value: Any) -> tuple[Any, Any] | None:
    parsed = _parse_literalish(value)
    if parsed is None:
        if isinstance(value, str) and "array(" in value:
            arrays = _extract_numpy_array_literals(value)
            if len(arrays) >= 2:
                return arrays[0], arrays[1]
        return None
    if isinstance(parsed, tuple) and len(parsed) == 2:
        return parsed
    if isinstance(parsed, list) and len(parsed) == 2:
        return parsed[0], parsed[1]
    return None


def _extract_numpy_array_literals(text: str) -> list[Any]:
    """Parse numpy array reprs from CSV fields without evaluating code."""
    arrays: list[Any] = []
    index = 0
    while True:
        array_pos = text.find("array(", index)
        if array_pos < 0:
            break
        start = text.find("[", array_pos)
        if start < 0:
            break

        depth = 0
        end = None
        for pos in range(start, len(text)):
            char = text[pos]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    end = pos + 1
                    break
        if end is None:
            break

        try:
            arrays.append(ast.literal_eval(text[start:end]))
        except Exception:
            pass
        index = end
    return arrays


def _align_foot_to_walk_spans(foot: list[Interval], walks: list[Interval]) -> list[Interval]:
    """Clip foot intervals so each walk span starts and ends exactly at its boundaries.

    Foot intervals that overlap a walk span have their leading edge snapped to the
    span start (first interval only) and trailing edge snapped to the span end
    (last interval only).  Intervals that fall entirely outside every walk span are
    dropped.
    """
    if not foot or not walks:
        return foot
    aligned: list[Interval] = []
    for walk in walks:
        overlapping = [
            iv for iv in foot
            if iv.end_ms > walk.start_ms and iv.start_ms < walk.end_ms
        ]
        if not overlapping:
            continue
        for i, iv in enumerate(overlapping):
            start = walk.start_ms if i == 0 else iv.start_ms
            end = walk.end_ms if i == len(overlapping) - 1 else iv.end_ms
            if end > start:
                aligned.append(Interval(start, end, iv.label, iv.confidence, iv.source))
    return aligned


def _first_match(folder: Path, pattern: str) -> Path | None:
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None


def _intervals_to_dicts(intervals: list[Interval]) -> list[dict[str, Any]]:
    return [iv.to_dict() for iv in intervals]


def _clip_intervals_to_windows(
    intervals: list[Interval],
    windows: list[Interval],
) -> list[Interval]:
    if not intervals or not windows:
        return intervals
    clipped: list[Interval] = []
    for interval in intervals:
        for window in windows:
            start = max(interval.start_ms, window.start_ms)
            end = min(interval.end_ms, window.end_ms)
            if end - start >= OCCLUSION_MIN_DURATION_MS:
                clipped.append(Interval(start, end, interval.label, interval.confidence, interval.source))
    return _smooth_occlusion_intervals(clipped)


def _tier_order(base: tuple[str, ...], occlusion: list[Interval]) -> list[str]:
    tiers = list(base)
    if occlusion and OCCLUSION_TIER not in tiers:
        insert_at = 1 if tiers and tiers[0] == "phase" else len(tiers)
        tiers.insert(insert_at, OCCLUSION_TIER)
    return tiers


def _pose3d_summary(pose3d: Pose3DResult | None) -> dict[str, Any] | None:
    if pose3d is None:
        return None
    from .pose3d import pose3d_summary

    return pose3d_summary(pose3d)


def _person_mask_summary(person_masks: Any | None) -> dict[str, Any] | None:
    if person_masks is None:
        return None
    from .person_masks import person_mask_summary

    return person_mask_summary(person_masks)


def _subject_summary(subject: SubjectTrack) -> dict[str, Any]:
    return {
        "track_id": subject.track_id,
        "frames": len(subject.boxes),
        "first_frame": subject.first_frame,
        "last_frame": subject.last_frame,
        "confidence": subject.confidence,
        "reason": subject.reason,
        "locked_track_ids": list(subject.locked_track_ids),
    }


def _boundary_events_from_payload(data: Any, side: str) -> list[GaitEvent]:
    if isinstance(data, dict):
        for key in (side, f"{side}_foot", f"{side}_boundaries"):
            if key in data:
                nested = _boundary_events_from_payload(data[key], side)
                if nested:
                    return nested
        for key in ("events", "boundaries", "gait_boundaries", "refined_boundaries"):
            if key in data:
                nested = _boundary_events_from_payload(data[key], side)
                if nested:
                    return nested
        return _events_from_mapping(data, side)

    if isinstance(data, list):
        events: list[GaitEvent] = []
        for item in data:
            if isinstance(item, dict):
                event = _event_from_mapping(item, side)
                if event is not None:
                    events.append(event)
                else:
                    events.extend(_events_from_mapping(item, side))
            elif isinstance(item, (list, tuple)):
                events.extend(_events_from_row(item, side))
        return events
    return []


def _events_from_mapping(data: dict[str, Any], side: str) -> list[GaitEvent]:
    events: list[GaitEvent] = []
    stance_keys = ("stance_start", "stance_starts", "stance_start_ms", "initial_contacts")
    swing_keys = ("swing_start", "swing_starts", "swing_start_ms", "toe_offs")
    for key in stance_keys:
        for value in _time_values(data.get(key)):
            events.append(GaitEvent(value, side, "stance_start"))
    for key in swing_keys:
        for value in _time_values(data.get(key)):
            events.append(GaitEvent(value, side, "swing_start"))
    return events


def _event_from_mapping(data: dict[str, Any], side: str) -> GaitEvent | None:
    raw_side = data.get("side")
    if raw_side is not None and str(raw_side).lower() not in {side, side[0]}:
        return None
    event_type = _coerce_event_type(data.get("event_type", data.get("type", data.get("label"))))
    if event_type is None:
        return None
    time_ms = _coerce_time_ms(data)
    if time_ms is None:
        return None
    return GaitEvent(time_ms, side, event_type)


def _events_from_row(row: tuple[Any, ...] | list[Any], side: str) -> list[GaitEvent]:
    if len(row) >= 2 and isinstance(row[1], str):
        event_type = _coerce_event_type(row[1])
        time_ms = _coerce_int(row[0], -1)
        return [GaitEvent(time_ms, side, event_type)] if event_type and time_ms >= 0 else []
    if len(row) >= 2:
        stance_ms = _coerce_int(row[0], -1)
        swing_ms = _coerce_int(row[1], -1)
        events: list[GaitEvent] = []
        if stance_ms >= 0:
            events.append(GaitEvent(stance_ms, side, "stance_start"))
        if swing_ms >= 0:
            events.append(GaitEvent(swing_ms, side, "swing_start"))
        return events
    return []


def _time_values(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, float, str)):
        parsed = _coerce_int(value, -1)
        return [parsed] if parsed >= 0 else []
    if isinstance(value, dict):
        time_ms = _coerce_time_ms(value)
        return [time_ms] if time_ms is not None and time_ms >= 0 else []
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            out.extend(_time_values(item))
        return out
    return []


def _coerce_event_type(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "stance": "stance_start",
        "stance_start": "stance_start",
        "heel_strike": "stance_start",
        "initial_contact": "stance_start",
        "ic": "stance_start",
        "swing": "swing_start",
        "swing_start": "swing_start",
        "toe_off": "swing_start",
        "to": "swing_start",
    }
    return aliases.get(text)


def _coerce_time_ms(data: dict[str, Any]) -> int | None:
    for key in ("time_ms", "timestamp_ms", "t_ms", "ms"):
        if key in data:
            value = _coerce_int(data.get(key), -1)
            return value if value >= 0 else None
    return None


def _bbox_center_x(item: Any) -> float | None:
    if isinstance(item, DetectionBox):
        return item.center_x
    if isinstance(item, dict):
        if "center_x" in item:
            return _coerce_float(item.get("center_x"), float("nan"))
        for key in ("bbox", "box", "boundingBox"):
            value = item.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 4:
                x1 = _coerce_float(value[0], 0.0)
                x2_or_w = _coerce_float(value[2], 0.0)
                x2 = x2_or_w if x2_or_w > x1 else x1 + x2_or_w
                return (x1 + x2) / 2.0
        rect = item.get("rect")
        if isinstance(rect, (list, tuple)) and len(rect) >= 2:
            try:
                return (_coerce_float(rect[0][0], 0.0) + _coerce_float(rect[1][0], 0.0)) / 2.0
            except (TypeError, IndexError):
                return None
    if isinstance(item, (list, tuple)) and len(item) >= 4:
        x1 = _coerce_float(item[0], 0.0)
        x2_or_w = _coerce_float(item[2], 0.0)
        x2 = x2_or_w if x2_or_w > x1 else x1 + x2_or_w
        return (x1 + x2) / 2.0
    return None


def _bbox_track_id(item: Any) -> Any:
    if isinstance(item, DetectionBox):
        return item.track_id
    if isinstance(item, dict):
        for key in ("track_id", "trackId", "trackerId", "id"):
            if key in item and item[key] not in {None, -1, "-1"}:
                return item[key]
    return None


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
