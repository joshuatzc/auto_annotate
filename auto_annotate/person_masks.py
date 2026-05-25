"""Optional person-mask provider for patient occlusion detection."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import VideoMetadata

SIDECAR_NAME = "person_masks.json"
SCHEMA_VERSION = 1
MIN_MASK_FRAMES = 10


@dataclass(frozen=True)
class PersonMask:
    bbox: tuple[float, float, float, float]
    confidence: float = 0.0
    mask_size: tuple[int, int] = (0, 0)
    counts: tuple[int, ...] = ()
    track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "bbox": [float(value) for value in self.bbox],
            "confidence": float(self.confidence),
            "mask": {
                "size": [int(self.mask_size[0]), int(self.mask_size[1])],
                "counts": [int(value) for value in self.counts],
            },
        }
        if self.track_id is not None:
            out["track_id"] = int(self.track_id)
        return out

    def to_array(self) -> Any | None:
        if not self.counts or self.mask_size[0] <= 0 or self.mask_size[1] <= 0:
            return None
        try:
            import numpy as np
        except Exception:
            return None
        total = int(self.mask_size[0]) * int(self.mask_size[1])
        arr = np.zeros(total, dtype=bool)
        value = False
        offset = 0
        for count in self.counts:
            end = min(total, offset + max(0, int(count)))
            if value and end > offset:
                arr[offset:end] = True
            offset = end
            value = not value
            if offset >= total:
                break
        return arr.reshape((int(self.mask_size[0]), int(self.mask_size[1])))

    @classmethod
    def from_dict(cls, data: Any) -> "PersonMask | None":
        if not isinstance(data, dict):
            return None
        bbox = data.get("bbox")
        mask = data.get("mask") or {}
        size = mask.get("size") or data.get("mask_size") or []
        counts = mask.get("counts") or data.get("counts") or []
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        if not isinstance(size, list) or len(size) != 2:
            return None
        if not isinstance(counts, list):
            return None
        raw_track_id = data.get("track_id", data.get("id"))
        track_id = None if raw_track_id is None else int(raw_track_id)
        return cls(
            bbox=tuple(float(value) for value in bbox),
            confidence=float(data.get("confidence") or data.get("score") or 0.0),
            mask_size=(int(size[0]), int(size[1])),
            counts=tuple(int(value) for value in counts),
            track_id=track_id,
        )


@dataclass(frozen=True)
class PersonMaskFrame:
    frame_index: int
    persons: list[PersonMask] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "persons": [person.to_dict() for person in self.persons],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "PersonMaskFrame | None":
        if not isinstance(data, dict):
            return None
        persons = [
            mask
            for item in (data.get("persons") or data.get("instances") or [])
            if (mask := PersonMask.from_dict(item)) is not None
        ]
        return cls(int(data.get("frame_index", data.get("frame", 0))), persons)


@dataclass
class PersonMaskResult:
    backend: str
    fps: float
    num_frames: int
    width: int
    height: int
    frames: list[PersonMaskFrame]
    video_signature: dict[str, Any] = field(default_factory=dict)
    model_config_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    cache_hit: bool = False

    @property
    def frames_by_index(self) -> dict[int, PersonMaskFrame]:
        return {frame.frame_index: frame for frame in self.frames}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "backend": self.backend,
            "fps": float(self.fps),
            "num_frames": int(self.num_frames),
            "width": int(self.width),
            "height": int(self.height),
            "video_signature": self.video_signature,
            "model_config_hash": self.model_config_hash,
            "metadata": dict(self.metadata),
            "frames": [frame.to_dict() for frame in self.frames],
        }


def load_or_run_person_masks(
    test_folder: str | Path,
    metadata: VideoMetadata | None = None,
    warnings: list[str] | None = None,
    deadline: float | None = None,
) -> PersonMaskResult | None:
    warn = warnings if warnings is not None else []
    folder = Path(test_folder)
    video_path = _first_match(folder, "rgb_video*.mp4")
    sidecar = folder / SIDECAR_NAME
    if video_path is None or not video_path.exists() or video_path.stat().st_size == 0:
        return _load_person_mask_sidecar(sidecar)

    meta = metadata or _read_video_metadata(video_path)
    video_sig = _video_signature(video_path, meta)
    config_hash = _model_config_hash()
    cached = _load_person_mask_sidecar(sidecar)
    if cached is not None and _cache_matches(cached, video_sig, config_hash):
        cached = _ensure_person_mask_track_ids(cached)
        cached.cache_hit = True
        cached.source_path = sidecar
        return cached

    if not _mask_backend_configured():
        return cached
    if not _has_time(deadline, _min_backend_seconds()):
        warn.append("person masks skipped: annotation time budget nearly exhausted")
        return _ensure_person_mask_track_ids(cached)

    result = _run_person_mask_backend(folder, video_path, meta, video_sig, config_hash, warn, deadline)
    if result is None:
        return _ensure_person_mask_track_ids(cached)
    result = _ensure_person_mask_track_ids(result)
    result.source_path = sidecar
    try:
        sidecar.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    except OSError as exc:
        warn.append(f"person mask cache could not be written: {exc}")
        result.source_path = None
    return result


def _ensure_person_mask_track_ids(result: PersonMaskResult | None) -> PersonMaskResult | None:
    if result is None or not result.frames:
        return result
    missing_track_ids = any(
        person.track_id is None
        for frame in result.frames
        for person in frame.persons
    )
    if not missing_track_ids:
        return result
    result.frames = _associate_person_mask_tracks(result.frames, result.width, result.height, result.fps)
    metadata = dict(result.metadata)
    if not metadata.get("track_association"):
        metadata["track_association"] = "bbox_temporal_iou_v1"
    result.metadata = metadata
    return result


def _associate_person_mask_tracks(
    frames: list[PersonMaskFrame],
    width: int,
    height: int,
    fps: float,
) -> list[PersonMaskFrame]:
    next_track_id = 1
    active: dict[int, tuple[int, PersonMask]] = {}
    max_gap = max(8, int(round((fps or 30.0) * 1.25)))
    tracked_frames: list[PersonMaskFrame] = []

    for frame in sorted(frames, key=lambda item: item.frame_index):
        assignments: dict[int, int] = {}
        candidates: list[tuple[float, int, int]] = []
        for person_index, person in enumerate(frame.persons):
            for track_id, (last_frame, last_person) in active.items():
                gap = frame.frame_index - last_frame
                if gap <= 0 or gap > max_gap:
                    continue
                score = _person_track_score(person, last_person, width, height, gap, fps)
                if score >= 0.10:
                    candidates.append((score, track_id, person_index))

        used_tracks: set[int] = set()
        used_persons: set[int] = set()
        for _score, track_id, person_index in sorted(candidates, reverse=True):
            if track_id in used_tracks or person_index in used_persons:
                continue
            assignments[person_index] = track_id
            used_tracks.add(track_id)
            used_persons.add(person_index)

        for person_index in range(len(frame.persons)):
            if person_index not in assignments:
                assignments[person_index] = next_track_id
                next_track_id += 1

        persons = [
            replace(person, track_id=assignments[index])
            for index, person in enumerate(frame.persons)
        ]
        for person in persons:
            if person.track_id is not None:
                active[person.track_id] = (frame.frame_index, person)
        active = {
            track_id: state
            for track_id, state in active.items()
            if frame.frame_index - state[0] <= max_gap
        }
        tracked_frames.append(PersonMaskFrame(frame.frame_index, persons))
    return tracked_frames


def _person_track_score(
    current: PersonMask,
    previous: PersonMask,
    width: int,
    height: int,
    gap: int,
    fps: float,
) -> float:
    iou = _bbox_iou(current.bbox, previous.bbox)
    distance = _bbox_center_distance(current.bbox, previous.bbox, width, height)
    area_similarity = _bbox_area_similarity(current.bbox, previous.bbox)
    if iou < 0.03 and distance > 0.16:
        return 0.0
    if area_similarity < 0.22 and iou < 0.18:
        return 0.0
    time_penalty = min(0.12, gap / max((fps or 30.0) * 4.0, 1.0))
    return iou + area_similarity * 0.24 - distance * 0.70 - time_penalty


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(_bbox_area(a) + _bbox_area(b) - inter, 1.0)


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_center_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    width: int,
    height: int,
) -> float:
    ax = (float(a[0]) + float(a[2])) / 2.0
    ay = (float(a[1]) + float(a[3])) / 2.0
    bx = (float(b[0]) + float(b[2])) / 2.0
    by = (float(b[1]) + float(b[3])) / 2.0
    dx = abs(ax - bx) / max(float(width), 1.0)
    dy = abs(ay - by) / max(float(height), 1.0)
    return (dx * dx + dy * dy) ** 0.5


def _bbox_area_similarity(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return min(area_a, area_b) / max(area_a, area_b)


def person_mask_summary(result: PersonMaskResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "backend": result.backend,
        "frames": len(result.frames),
        "cache_hit": result.cache_hit,
        "source_path": str(result.source_path) if result.source_path else None,
        "metadata": dict(result.metadata),
    }


def _run_person_mask_backend(
    folder: Path,
    video_path: Path,
    metadata: VideoMetadata,
    video_signature: dict[str, Any],
    config_hash: str,
    warnings: list[str],
    deadline: float | None,
) -> PersonMaskResult | None:
    command = os.environ.get("AUTO_ANNOTATE_PERSON_MASK_CMD")
    if command:
        result = _run_command_backend(
            folder,
            video_path,
            metadata,
            video_signature,
            config_hash,
            command,
            warnings,
            deadline,
        )
        if result is not None:
            return result

    if os.environ.get("AUTO_ANNOTATE_ENABLE_ULTRALYTICS_MASKS") == "1":
        return _run_ultralytics_backend(
            folder,
            video_path,
            metadata,
            video_signature,
            config_hash,
            warnings,
            deadline,
        )
    return None


def _run_command_backend(
    folder: Path,
    video_path: Path,
    metadata: VideoMetadata,
    video_signature: dict[str, Any],
    config_hash: str,
    command_text: str,
    warnings: list[str],
    deadline: float | None,
) -> PersonMaskResult | None:
    output_path = Path(tempfile.gettempdir()) / f"auto_annotate_person_masks_{os.getpid()}.json"
    command = [
        part.format(video=str(video_path), output=str(output_path), folder=str(folder))
        for part in shlex.split(command_text)
    ]
    try:
        timeout = _timeout_seconds("AUTO_ANNOTATE_PERSON_MASK_TIMEOUT_SEC", 1800, deadline)
        if timeout <= 0:
            warnings.append("person mask command skipped: annotation time budget exhausted")
            return None
        completed = subprocess.run(
            command,
            cwd=str(folder),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        warnings.append(f"person mask command failed to start: {exc}")
        return None
    if completed.returncode != 0:
        lines = (completed.stderr or completed.stdout or "").strip().splitlines()
        detail = lines[-1] if lines else f"exit code {completed.returncode}"
        warnings.append(f"person mask command failed: {detail}")
        return None
    result = _load_person_mask_sidecar(output_path)
    try:
        output_path.unlink()
    except OSError:
        pass
    if result is None or len(result.frames) < MIN_MASK_FRAMES:
        warnings.append("person mask command produced too few usable frames")
        return None
    result.backend = result.backend or "external_command"
    result.fps = result.fps or metadata.fps
    result.num_frames = result.num_frames or metadata.num_frames
    result.width = result.width or metadata.width
    result.height = result.height or metadata.height
    result.video_signature = video_signature
    result.model_config_hash = config_hash
    return result


def _run_ultralytics_backend(
    folder: Path,
    video_path: Path,
    metadata: VideoMetadata,
    video_signature: dict[str, Any],
    config_hash: str,
    warnings: list[str],
    deadline: float | None,
) -> PersonMaskResult | None:
    if not _has_time(deadline, _min_backend_seconds()):
        warnings.append("person mask ultralytics backend skipped: annotation time budget nearly exhausted")
        return None
    model_path = os.environ.get("AUTO_ANNOTATE_ULTRALYTICS_SEG_MODEL")
    if not model_path:
        warnings.append("person masks unavailable: set AUTO_ANNOTATE_ULTRALYTICS_SEG_MODEL to a local segmentation model")
        return None
    if not Path(model_path).exists():
        warnings.append(f"person masks unavailable: local segmentation model not found: {model_path}")
        return None
    try:
        from ultralytics import YOLO
    except Exception as exc:
        warnings.append(f"person masks unavailable: ultralytics is not installed ({exc})")
        return None

    model = YOLO(model_path)
    stride = max(1, int(os.environ.get("AUTO_ANNOTATE_PERSON_MASK_STRIDE", "4")))
    tracker = os.environ.get("AUTO_ANNOTATE_ULTRALYTICS_TRACKER", "bytetrack.yaml").strip()
    tracking_enabled = os.environ.get("AUTO_ANNOTATE_ENABLE_ULTRALYTICS_TRACKING", "0") == "1" and bool(tracker)
    batch_size = 1 if tracking_enabled else max(1, int(os.environ.get("AUTO_ANNOTATE_PERSON_MASK_BATCH", "4")))
    predict_kwargs: dict[str, Any] = {
        "classes": [0],
        "verbose": False,
        "retina_masks": True,
        "save": False,
    }
    if imgsz := os.environ.get("AUTO_ANNOTATE_PERSON_MASK_IMGSZ"):
        predict_kwargs["imgsz"] = int(imgsz)
    if conf := os.environ.get("AUTO_ANNOTATE_PERSON_MASK_CONF"):
        predict_kwargs["conf"] = float(conf)
    if device := os.environ.get("AUTO_ANNOTATE_ULTRALYTICS_DEVICE"):
        predict_kwargs["device"] = device
    max_dim = max(0, int(os.environ.get("AUTO_ANNOTATE_PERSON_MASK_MAX_DIM", "960")))

    frames: list[PersonMaskFrame] = []
    batch_images: list[Any] = []
    batch_indices: list[int] = []
    batch_scales: list[tuple[float, float]] = []
    try:
        stopped_for_budget = False
        tracking_failed = False
        for frame_index, image in _iter_sampled_video_frames(video_path, stride):
            if not _has_time(deadline, _min_batch_seconds()):
                stopped_for_budget = True
                break
            image, scale = _resize_for_mask_backend(image, max_dim)
            if tracking_enabled and not tracking_failed:
                try:
                    _append_ultralytics_tracked_frame(frames, model, image, frame_index, scale, predict_kwargs, tracker)
                    continue
                except Exception as exc:
                    tracking_failed = True
                    warnings.append(f"person mask ultralytics tracker unavailable; falling back to prediction: {exc}")
            batch_indices.append(frame_index)
            batch_images.append(image)
            batch_scales.append(scale)
            if len(batch_images) >= batch_size:
                _append_ultralytics_batch(frames, model, batch_images, batch_indices, batch_scales, predict_kwargs)
                batch_images = []
                batch_indices = []
                batch_scales = []
        if batch_images and _has_time(deadline, _min_batch_seconds()):
            _append_ultralytics_batch(frames, model, batch_images, batch_indices, batch_scales, predict_kwargs)
        elif batch_images:
            stopped_for_budget = True
        if stopped_for_budget:
            warnings.append("person mask ultralytics backend stopped early at annotation time budget")
    except Exception as exc:
        warnings.append(f"person mask ultralytics backend failed: {exc}")
        return None

    usable = [frame for frame in frames if frame.persons]
    if len(usable) < MIN_MASK_FRAMES:
        warnings.append("person mask ultralytics backend produced too few person masks")
        return None
    return PersonMaskResult(
        backend="ultralytics",
        fps=metadata.fps,
        num_frames=metadata.num_frames,
        width=metadata.width,
        height=metadata.height,
        frames=frames,
        video_signature=video_signature,
        model_config_hash=config_hash,
        metadata={
            "model": str(model_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frame_stride": stride,
            "batch_size": batch_size,
            "max_dim": max_dim,
            "tracker": tracker if tracking_enabled else "",
            "track_association": "ultralytics_tracker" if tracking_enabled else "",
        },
    )


def _min_backend_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("AUTO_ANNOTATE_PERSON_MASK_MIN_SECONDS", "4")))
    except ValueError:
        return 4.0


def _min_batch_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("AUTO_ANNOTATE_PERSON_MASK_BATCH_MIN_SECONDS", "2")))
    except ValueError:
        return 2.0


def _has_time(deadline: float | None, min_remaining_seconds: float = 0.0) -> bool:
    if deadline is None:
        return True
    return deadline - time.monotonic() >= min_remaining_seconds


def _timeout_seconds(env_name: str, default_seconds: int, deadline: float | None) -> int:
    try:
        timeout = int(os.environ.get(env_name, str(default_seconds)))
    except ValueError:
        timeout = default_seconds
    if deadline is None:
        return timeout
    remaining = int(deadline - time.monotonic())
    if remaining <= 0:
        return 0
    return max(1, min(timeout, remaining))


def _iter_sampled_video_frames(video_path: Path, stride: int) -> Any:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        frame_index = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if frame_index % stride == 0:
                yield frame_index, image
            frame_index += 1
    finally:
        cap.release()


def _append_ultralytics_batch(
    frames: list[PersonMaskFrame],
    model: Any,
    images: list[Any],
    frame_indices: list[int],
    scales: list[tuple[float, float]],
    predict_kwargs: dict[str, Any],
) -> None:
    results = model.predict(source=images, stream=False, **predict_kwargs)
    for frame_index, scale, result in zip(frame_indices, scales, results):
        frames.append(PersonMaskFrame(frame_index, _ultralytics_result_to_person_masks(result, scale)))


def _append_ultralytics_tracked_frame(
    frames: list[PersonMaskFrame],
    model: Any,
    image: Any,
    frame_index: int,
    scale: tuple[float, float],
    predict_kwargs: dict[str, Any],
    tracker: str,
) -> None:
    kwargs = dict(predict_kwargs)
    kwargs["tracker"] = tracker
    kwargs["persist"] = True
    results = model.track(source=image, stream=False, **kwargs)
    result = results[0] if results else None
    if result is None:
        frames.append(PersonMaskFrame(frame_index, []))
    else:
        frames.append(PersonMaskFrame(frame_index, _ultralytics_result_to_person_masks(result, scale)))


def _resize_for_mask_backend(image: Any, max_dim: int) -> tuple[Any, tuple[float, float]]:
    if max_dim <= 0:
        return image, (1.0, 1.0)
    height, width = image.shape[:2]
    largest = max(int(height), int(width))
    if largest <= max_dim:
        return image, (1.0, 1.0)
    import cv2

    scale = max_dim / float(largest)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return resized, (width / float(resized_width), height / float(resized_height))


def _ultralytics_result_to_person_masks(
    result: Any,
    bbox_scale: tuple[float, float] = (1.0, 1.0),
) -> list[PersonMask]:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or getattr(masks, "data", None) is None:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else boxes.xyxy
    conf = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else boxes.conf
    ids = None
    raw_ids = getattr(boxes, "id", None)
    if raw_ids is not None:
        ids = raw_ids.detach().cpu().numpy() if hasattr(raw_ids, "detach") else raw_ids
    data = masks.data.detach().cpu().numpy() if hasattr(masks.data, "detach") else masks.data
    out: list[PersonMask] = []
    scale_x, scale_y = bbox_scale
    for index, mask in enumerate(data):
        if index >= len(xyxy):
            break
        encoded = _encode_binary_mask(mask > 0.5)
        bbox = xyxy[index]
        out.append(
            PersonMask(
                bbox=(
                    float(bbox[0]) * scale_x,
                    float(bbox[1]) * scale_y,
                    float(bbox[2]) * scale_x,
                    float(bbox[3]) * scale_y,
                ),
                confidence=float(conf[index]) if index < len(conf) else 0.0,
                mask_size=encoded[0],
                counts=encoded[1],
                track_id=int(ids[index]) if ids is not None and index < len(ids) else None,
            )
        )
    return out


def _encode_binary_mask(mask: Any) -> tuple[tuple[int, int], tuple[int, ...]]:
    import numpy as np

    arr = np.asarray(mask, dtype=bool)
    flat = arr.reshape(-1)
    counts: list[int] = []
    current = False
    run = 0
    for value in flat:
        value_bool = bool(value)
        if value_bool == current:
            run += 1
        else:
            counts.append(run)
            run = 1
            current = value_bool
    counts.append(run)
    return (int(arr.shape[0]), int(arr.shape[1])), tuple(counts)


def _load_person_mask_sidecar(path: Path) -> PersonMaskResult | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return _coerce_person_mask_result(data, path)


def _coerce_person_mask_result(data: Any, source_path: Path | None = None) -> PersonMaskResult | None:
    if not isinstance(data, dict):
        return None
    frames = [
        frame
        for item in data.get("frames", [])
        if (frame := PersonMaskFrame.from_dict(item)) is not None
    ]
    if not frames:
        return None
    result = PersonMaskResult(
        backend=str(data.get("backend") or ""),
        fps=float(data.get("fps") or 0.0),
        num_frames=int(data.get("num_frames") or 0),
        width=int(data.get("width") or 0),
        height=int(data.get("height") or 0),
        frames=sorted(frames, key=lambda item: item.frame_index),
        video_signature=dict(data.get("video_signature") or {}),
        model_config_hash=str(data.get("model_config_hash") or ""),
        metadata=dict(data.get("metadata") or {}),
        source_path=source_path,
    )
    return result


def _cache_matches(result: PersonMaskResult, video_signature: dict[str, Any], config_hash: str) -> bool:
    return result.video_signature == video_signature and result.model_config_hash == config_hash


def _mask_backend_configured() -> bool:
    return bool(os.environ.get("AUTO_ANNOTATE_PERSON_MASK_CMD")) or (
        os.environ.get("AUTO_ANNOTATE_ENABLE_ULTRALYTICS_MASKS") == "1"
    )


def _model_config_hash() -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "command": os.environ.get("AUTO_ANNOTATE_PERSON_MASK_CMD", ""),
        "ultralytics": os.environ.get("AUTO_ANNOTATE_ENABLE_ULTRALYTICS_MASKS", ""),
        "model": os.environ.get("AUTO_ANNOTATE_ULTRALYTICS_SEG_MODEL", ""),
        "stride": os.environ.get("AUTO_ANNOTATE_PERSON_MASK_STRIDE", "4"),
        "batch": os.environ.get("AUTO_ANNOTATE_PERSON_MASK_BATCH", "4"),
        "imgsz": os.environ.get("AUTO_ANNOTATE_PERSON_MASK_IMGSZ", ""),
        "conf": os.environ.get("AUTO_ANNOTATE_PERSON_MASK_CONF", ""),
        "device": os.environ.get("AUTO_ANNOTATE_ULTRALYTICS_DEVICE", ""),
        "max_dim": os.environ.get("AUTO_ANNOTATE_PERSON_MASK_MAX_DIM", "960"),
    }
    if os.environ.get("AUTO_ANNOTATE_ENABLE_ULTRALYTICS_TRACKING", "0") == "1":
        payload["tracking"] = "1"
        payload["tracker"] = os.environ.get("AUTO_ANNOTATE_ULTRALYTICS_TRACKER", "bytetrack.yaml")
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _video_signature(video_path: Path, metadata: VideoMetadata) -> dict[str, Any]:
    stat = video_path.stat()
    return {
        "path": str(video_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "fps": metadata.fps,
        "num_frames": metadata.num_frames,
        "width": metadata.width,
        "height": metadata.height,
    }


def _read_video_metadata(video_path: Path) -> VideoMetadata:
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        return VideoMetadata(video_path, fps, frames, width, height)
    except Exception:
        return VideoMetadata(video_path, 30.0, 0, 0, 0)


def _first_match(folder: Path, pattern: str) -> Path | None:
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None
