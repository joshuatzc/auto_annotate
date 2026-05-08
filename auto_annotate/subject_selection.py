"""Single-subject selection and lightweight tracking for bundle detections."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import (
    SUBJECT_MIN_COVERAGE,
    TRACK_IOU_THRESHOLD,
    TRACK_MAX_FRAME_GAP,
)
from .types import DetectionBox, SubjectTrack


def load_detection_json(
    path: str | Path,
    video_width: int | None = None,
    video_height: int | None = None,
) -> list[DetectionBox]:
    """Load current and near-future bounding-box JSON shapes.

    Supported entries include `{"boundingBox": [x, y, w, h]}` as used by the
    current bundle, xyxy-like dicts, raw `[x, y, w, h]`, and rows shaped like
    `[frame, track_id, x1, y1, x2, y2]`.
    """

    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    boxes: list[DetectionBox] = []
    if isinstance(data, dict):
        raw_frames = []
        for frame_key, detections in data.items():
            try:
                frame = int(frame_key)
            except (TypeError, ValueError):
                continue
            if detections is None:
                continue
            if isinstance(detections, dict):
                detections = [detections]
            for det in detections:
                box = _parse_detection(det, frame, video_width, video_height)
                if box is not None:
                    boxes.append(box)
                    raw_frames.append(frame)
        if raw_frames and min(raw_frames) >= 1:
            boxes = [box_with_frame(box, box.frame - 1) for box in boxes]
        return boxes

    if isinstance(data, list):
        for row in data:
            box = _parse_detection(row, None, video_width, video_height)
            if box is not None:
                boxes.append(box)
        if boxes and min(box.frame for box in boxes) >= 1:
            boxes = [box_with_frame(box, box.frame - 1) for box in boxes]
    return boxes


def box_with_frame(box: DetectionBox, frame: int) -> DetectionBox:
    return DetectionBox(
        frame=frame,
        x1=box.x1,
        y1=box.y1,
        x2=box.x2,
        y2=box.y2,
        track_id=box.track_id,
        score=box.score,
        label=box.label,
    )


def track_detections(
    boxes: list[DetectionBox],
    iou_threshold: float = TRACK_IOU_THRESHOLD,
    max_frame_gap: int = TRACK_MAX_FRAME_GAP,
) -> list[DetectionBox]:
    """Assign stable track IDs when the bundle did not already provide them."""

    if not boxes:
        return []
    if all(box.track_id is not None for box in boxes):
        return sorted(boxes, key=lambda b: (b.frame, b.track_id or -1))

    next_track_id = 1
    active: dict[int, DetectionBox] = {}
    tracked: list[DetectionBox] = []

    for frame in sorted({box.frame for box in boxes}):
        frame_boxes = [box for box in boxes if box.frame == frame]
        used_tracks: set[int] = set()
        for box in frame_boxes:
            best_track: int | None = None
            best_score = 0.0
            for track_id, prev in active.items():
                if track_id in used_tracks or frame - prev.frame > max_frame_gap:
                    continue
                score = iou(box, prev)
                if score > best_score:
                    best_score = score
                    best_track = track_id
            if best_track is None or best_score < iou_threshold:
                best_track = next_track_id
                next_track_id += 1
            used_tracks.add(best_track)
            tracked_box = box.with_track(best_track)
            active[best_track] = tracked_box
            tracked.append(tracked_box)

        stale = [
            track_id
            for track_id, prev in active.items()
            if frame - prev.frame > max_frame_gap
        ]
        for track_id in stale:
            del active[track_id]

    return sorted(tracked, key=lambda b: (b.frame, b.track_id or -1))


def select_main_subject(
    boxes: list[DetectionBox],
    frame_width: int | None = None,
    min_coverage: float = SUBJECT_MIN_COVERAGE,
) -> SubjectTrack:
    """Select the first stable person closest to frame centre."""

    tracked = track_detections(boxes)
    if not tracked:
        return SubjectTrack(None, [], 0.0, "no person detections")

    width = frame_width or _infer_frame_width(tracked)
    frame_min = min(box.frame for box in tracked)
    frame_max = max(box.frame for box in tracked)
    total_frames = max(1, frame_max - frame_min + 1)
    min_frames = max(3, int(round(total_frames * min_coverage)))

    by_track: dict[int, list[DetectionBox]] = defaultdict(list)
    for box in tracked:
        if box.track_id is not None:
            by_track[box.track_id].append(box)

    max_count = max(len(group) for group in by_track.values())
    stable_cutoff = max(min_frames, int(max_count * 0.8))
    candidates = [
        (track_id, group)
        for track_id, group in by_track.items()
        if len(group) >= stable_cutoff
    ]
    if not candidates:
        candidates = list(by_track.items())

    half_width = max(width / 2.0, 1.0)

    max_median_area = max(_median([box.area for box in group]) for _track_id, group in candidates)

    def rank(item: tuple[int, list[DetectionBox]]) -> tuple[float, int, int]:
        track_id, group = item
        mean_center_dist = sum(
            abs(box.center_x - width / 2.0) / half_width for box in group
        ) / len(group)
        area_bonus = _median([box.area for box in group]) / max(max_median_area, 1.0)
        motion_bonus = min(
            1.0,
            (max(box.center_x for box in group) - min(box.center_x for box in group))
            / max(width * 0.15, 1.0),
        )
        adjusted_center_dist = mean_center_dist - 0.20 * area_bonus - 0.10 * motion_bonus
        return (adjusted_center_dist, -len(group), min(box.frame for box in group))

    selected_id, selected_boxes = min(candidates, key=rank)
    mean_dist = rank((selected_id, selected_boxes))[0]
    confidence = max(0.0, min(1.0, (len(selected_boxes) / total_frames) * (1.0 - mean_dist)))
    return SubjectTrack(
        selected_id,
        sorted(selected_boxes, key=lambda b: b.frame),
        confidence,
        "stable centered track",
    )


def iou(a: DetectionBox, b: DetectionBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def _parse_detection(
    det: Any,
    frame: int | None,
    video_width: int | None,
    video_height: int | None,
) -> DetectionBox | None:
    if isinstance(det, dict):
        label = str(det.get("label") or det.get("class") or det.get("className") or "").lower()
        if label and "person" not in label and "face" not in label:
            return None
        track_id = _safe_int(det.get("track_id", det.get("trackId", det.get("id"))))
        score = _safe_float(det.get("score", det.get("confidence")))
        original_size = det.get("originalImageSize") or det.get("imageSize")
        if det.get("xyxy") is not None:
            coords = det.get("xyxy")
            coords_are_xywh = False
        else:
            coords = det.get("boundingBox") or det.get("bbox") or det.get("box")
            coords_are_xywh = True
        parsed = _coords_to_xyxy(coords, original_size, video_width, video_height, coords_are_xywh)
        if parsed is None:
            return None
        cur_frame = frame if frame is not None else _safe_int(det.get("frame", det.get("frame_id")))
        if cur_frame is None:
            return None
        return DetectionBox(cur_frame, *parsed, track_id=track_id, score=score, label=label)

    if isinstance(det, (list, tuple)):
        if len(det) >= 6:
            cur_frame = frame if frame is not None else _safe_int(det[0])
            if cur_frame is None:
                return None
            track_id = _safe_int(det[1])
            return DetectionBox(cur_frame, float(det[2]), float(det[3]), float(det[4]), float(det[5]), track_id)
        if len(det) == 4 and frame is not None:
            parsed = _coords_to_xyxy(det, None, video_width, video_height, True)
            if parsed is None:
                return None
            return DetectionBox(frame, *parsed)
    return None


def _coords_to_xyxy(
    coords: Any,
    original_size: Any,
    video_width: int | None,
    video_height: int | None,
    coords_are_xywh: bool,
) -> tuple[float, float, float, float] | None:
    if not isinstance(coords, (list, tuple)) or len(coords) < 4:
        return None
    x1, y1, a, b = [float(v) for v in coords[:4]]
    if coords_are_xywh:
        x2, y2 = x1 + a, y1 + b
    else:
        x2, y2 = a, b

    if (
        original_size
        and video_width
        and video_height
        and isinstance(original_size, (list, tuple))
        and len(original_size) >= 2
    ):
        orig_w = float(original_size[0]) or float(video_width)
        orig_h = float(original_size[1]) or float(video_height)
        sx = video_width / orig_w
        sy = video_height / orig_h
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
    return (x1, y1, x2, y2)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_frame_width(boxes: list[DetectionBox]) -> int:
    max_x = max((box.x2 for box in boxes), default=0.0)
    return max(1, int(round(max_x)))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
