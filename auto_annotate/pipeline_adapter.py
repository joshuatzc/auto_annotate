"""Adapter from current 3DGait bundle/output shapes to ELAN annotations."""

from __future__ import annotations

import ast
import glob
import importlib
import re
import sys
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_FALLBACK_TEST_MS,
    DEFAULT_FPS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
)
from .elan_export import write_eaf
from .phase_rules import (
    crt_phase_intervals_from_detections,
    fallback_foot_intervals,
    fallback_phase_intervals,
    foot_intervals_from_gait_cycles,
    phase_intervals_from_crt_events,
    phase_intervals_from_tug_boundaries,
    test_interval_from_phases,
    unknown_phase_interval,
    walking_intervals,
)
from .subject_selection import load_detection_json, select_main_subject
from .types import DetectionBox, Interval, SubjectTrack, VideoMetadata


def analyze_and_export(input_folder: str, output_eaf: str) -> Path:
    """Analyze an input bundle and write exactly one .eaf file."""

    annotations = extract_annotations(input_folder)
    video_path = annotations.get("video_path")
    if not video_path:
        raise ValueError("No rgb_video_*.mp4 source video found to link in the EAF")
    return write_eaf(str(video_path), annotations, output_eaf)


def extract_annotations(input_folder: str) -> dict[str, Any]:
    """Extract target ELAN annotations from an input bundle."""

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
    subject = select_main_subject(body_boxes, metadata.width) if body_boxes else SubjectTrack(None, [], 0.0, "no bbox")
    if subject.boxes:
        subject = _fill_subject_track_gaps(subject)

    # Extract pose signal once; shared by phase detection and foot detection below
    pose_signal = None
    if video_path and video_path.exists() and video_path.stat().st_size > 0:
        from .pose_estimator import extract_pose_signal, pose_available
        if pose_available():
            pose_signal = extract_pose_signal(video_path, warnings)

    if _is_crt_folder(folder):
        return _extract_crt_annotations(folder, video_path, metadata, subject, pose_signal, warnings)

    precomputed = _extract_from_precomputed_outputs(folder, metadata, warnings)
    if precomputed:
        precomputed["video_path"] = str(video_path) if video_path else None
        precomputed["subject"] = _subject_summary(subject)
        precomputed["warnings"] = warnings + precomputed.get("warnings", [])
        return precomputed

    phases = _phase_from_bundle_fallback(
        metadata=metadata,
        subject=subject,
        face_path=face_path,
        pose_signal=pose_signal,
        warnings=warnings,
    )
    test = test_interval_from_phases(phases)
    walks = walking_intervals(phases)

    left_foot, right_foot = [], []
    if pose_signal and walks:
        from .pose_estimator import foot_intervals_from_pose_signal
        left_foot, right_foot = foot_intervals_from_pose_signal(pose_signal, metadata, walks, warnings)
    if not left_foot and not right_foot:
        left_foot, right_foot = fallback_foot_intervals(walks) if walks else ([], [])

    return {
        "video_path": str(video_path) if video_path else None,
        "fps": metadata.fps,
        "duration_ms": metadata.duration_ms,
        "test": _intervals_to_dicts(test),
        "phase": _intervals_to_dicts(phases),
        "left_foot": _intervals_to_dicts(left_foot),
        "right_foot": _intervals_to_dicts(right_foot),
        "subject": _subject_summary(subject),
        "warnings": warnings,
    }


def _is_crt_folder(folder: Path) -> bool:
    return "crt" in folder.name.lower()


def _extract_crt_annotations(
    folder: Path,
    video_path: "Path | None",
    metadata: "VideoMetadata",
    subject: "SubjectTrack",
    pose_signal: "dict | None",
    warnings: list[str],
) -> "dict[str, Any]":
    phases = _extract_from_crt_precomputed(folder, metadata, warnings)
    if not phases:
        phases = _crt_phase_from_bundle_fallback(metadata, subject, pose_signal, warnings)

    test = test_interval_from_phases(phases)
    return {
        "video_path": str(video_path) if video_path else None,
        "fps": metadata.fps,
        "duration_ms": metadata.duration_ms,
        "test": _intervals_to_dicts(test),
        "phase": _intervals_to_dicts(phases),
        "left_foot": [],
        "right_foot": [],
        "is_crt": True,
        "subject": _subject_summary(subject),
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
        tests = data.get("tests") or {}
        crt_results = tests.get("chair_rise") or []
        if not crt_results:
            return []
        events = (crt_results[0].get("result") or {}).get("events") or []
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
    for search in (folder, folder.parent, folder.parent.parent):
        candidate = search / "sppb_results.json"
        if candidate.exists():
            return candidate
    return None


def _crt_phase_from_bundle_fallback(
    metadata: "VideoMetadata",
    subject: "SubjectTrack",
    pose_signal: "dict | None",
    warnings: list[str],
) -> "list[Interval]":
    if pose_signal is not None:
        from .pose_estimator import crt_phases_from_pose_signal
        pose_phases = crt_phases_from_pose_signal(pose_signal, metadata, warnings)
        if pose_phases:
            warnings.append("CRT phases detected using MediaPipe pose estimation")
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
        warnings.append("CRT phases detected from bounding box height signal")
        return phases

    warnings.append("CRT bbox detection found no rises; using unknown phase")
    start_ms = int(round((subject.first_frame or 0) * 1000.0 / metadata.fps))
    end_ms = int(round(((subject.last_frame or 0) + 1) * 1000.0 / metadata.fps))
    if end_ms <= start_ms:
        end_ms = start_ms + DEFAULT_FALLBACK_TEST_MS
    return unknown_phase_interval(start_ms, end_ms)


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
    gait = _parse_gait_breakdown(meta.get("Gait_breakdown_frames"))
    if gait:
        left_foot, right_foot = foot_intervals_from_gait_cycles(
            gait[0],
            gait[1],
            fps,
            walk_spans=walks,
            frame_base="zero",
        )
    else:
        warnings.append("precomputed metadata did not include usable Gait_breakdown_frames")
        left_foot, right_foot = fallback_foot_intervals(walks)

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
    pose_signal: "dict | None",
    warnings: list[str],
) -> list[Interval]:
    if subject.boxes and face_path:
        core_phases = _try_core_bbox_phase_segmentation(metadata, subject.boxes, face_path, warnings)
        if core_phases:
            return core_phases

    if pose_signal is not None:
        from .pose_estimator import tug_phases_from_pose_signal
        pose_phases = tug_phases_from_pose_signal(pose_signal, metadata, warnings)
        if pose_phases:
            warnings.append("TUG phases detected using MediaPipe pose estimation")
            return pose_phases

    if subject.boxes:
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
    return SubjectTrack(subject.track_id, filled, subject.confidence, subject.reason)


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
    search_dirs = [folder] + _sibling_out_dirs(folder)
    pkl_patterns = ("out2.pkl", "meta*.pkl", "**/out2.pkl")
    csv_patterns = ("out2.csv", "meta*.csv", "**/out2.csv")
    for search_dir in search_dirs:
        for pattern in pkl_patterns:
            for path_str in glob.glob(str(search_dir / pattern), recursive=True):
                path = Path(path_str)
                try:
                    import pandas as pd

                    df = pd.read_pickle(path)
                    if len(df) > 0:
                        return {col: df.loc[df.index[0], col] for col in df.columns}
                except Exception as exc:
                    warnings.append(f"could not read {path.name}: {exc}")
        for pattern in csv_patterns:
            for path_str in glob.glob(str(search_dir / pattern), recursive=True):
                path = Path(path_str)
                try:
                    import pandas as pd

                    df = pd.read_csv(path)
                    if len(df) > 0:
                        return {col: df.loc[df.index[0], col] for col in df.columns}
                except Exception as exc:
                    warnings.append(f"could not read {path.name}: {exc}")
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
        return None
    if isinstance(parsed, tuple) and len(parsed) == 2:
        return parsed
    if isinstance(parsed, list) and len(parsed) == 2:
        return parsed[0], parsed[1]
    return None


def _first_match(folder: Path, pattern: str) -> Path | None:
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None


def _intervals_to_dicts(intervals: list[Interval]) -> list[dict[str, Any]]:
    return [iv.to_dict() for iv in intervals]


def _subject_summary(subject: SubjectTrack) -> dict[str, Any]:
    return {
        "track_id": subject.track_id,
        "frames": len(subject.boxes),
        "first_frame": subject.first_frame,
        "last_frame": subject.last_frame,
        "confidence": subject.confidence,
        "reason": subject.reason,
    }


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
