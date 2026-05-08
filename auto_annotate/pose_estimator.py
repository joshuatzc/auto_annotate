"""MediaPipe pose estimation for TUG and CRT phase detection.

Provides a higher-confidence (0.85) fallback that runs directly on the video
when no precomputed pipeline output or face-detection data is available.

Processes every 3rd frame on CPU; typically finishes a 2-minute video in
under 60 seconds on a modern laptop.

Uses the mediapipe Tasks API (mediapipe >= 0.10).  The pose landmarker model
(~5.7 MB) is downloaded once to ~/.cache/auto_annotate/ on first use.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import Interval, VideoMetadata

# MediaPipe PoseLandmark indices (same numbering in both old and new API)
_LM_HIP_L = 23
_LM_HIP_R = 24
_LM_ANKLE_L = 27
_LM_ANKLE_R = 28

# Process every (_SKIP + 1)-th frame; balances speed vs temporal resolution
_SKIP = 2

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
_MODEL_CACHE = Path.home() / ".cache" / "auto_annotate" / "pose_landmarker_lite.task"


def pose_available() -> bool:
    """Return True if mediapipe and cv2 are both importable."""
    try:
        import mediapipe  # noqa: F401
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def _ensure_model() -> Path:
    """Download the pose landmarker model on first use and return its path."""
    if not _MODEL_CACHE.exists():
        _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_CACHE)
    return _MODEL_CACHE


def extract_pose_signal(
    video_path: Path,
    warnings: list[str],
) -> "dict[str, Any] | None":
    """Run MediaPipe Pose Lite on the video and return a signal dict.

    Keys:
      frames        – list[int]   original frame indices that were processed
      hip_elevation – list[float] 1.0 - mean(hip_L.y, hip_R.y)
                                  higher = more upright / standing
      hip_x         – list[float] lateral hip centre (0 = left edge, 1 = right)
      ankle_l_y     – list[float] left ankle Y in normalised coords
      ankle_r_y     – list[float] right ankle Y in normalised coords

    Returns None when mediapipe is unavailable, the video cannot be opened,
    the model cannot be downloaded, or too few landmarks are detected.
    """
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        return None

    try:
        model_path = _ensure_model()
    except Exception as exc:
        warnings.append(f"pose estimation: could not download model — {exc}")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warnings.append(f"pose estimation: could not open {video_path.name}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    hip_elev: list[float] = []
    hip_x_list: list[float] = []
    ankle_l: list[float] = []
    ankle_r: list[float] = []
    frame_indices: list[int] = []
    frame_idx = 0

    try:
        with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % (_SKIP + 1) == 0:
                    timestamp_ms = int(frame_idx * 1000.0 / fps)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    if result.pose_landmarks:
                        lm = result.pose_landmarks[0]
                        hip_y   = (lm[_LM_HIP_L].y   + lm[_LM_HIP_R].y)   / 2.0
                        ankle_y = (lm[_LM_ANKLE_L].y  + lm[_LM_ANKLE_R].y) / 2.0
                        # leg_signal: ankle_y - hip_y in image coords
                        #   small  → legs bent  → seated
                        #   large  → legs straight → standing
                        # Scale-invariant because both landmarks shift together
                        # as the person moves toward / away from the camera.
                        hip_elev.append(ankle_y - hip_y)
                        hip_x_list.append((lm[_LM_HIP_L].x + lm[_LM_HIP_R].x) / 2.0)
                        ankle_l.append(lm[_LM_ANKLE_L].y)
                        ankle_r.append(lm[_LM_ANKLE_R].y)
                        frame_indices.append(frame_idx)
                frame_idx += 1
    except Exception as exc:
        warnings.append(f"pose estimation: error during landmark extraction — {exc}")
        return None
    finally:
        cap.release()

    if len(frame_indices) < 20:
        warnings.append("pose estimation: too few landmarks detected — skipping")
        return None

    return {
        "frames": frame_indices,
        "hip_elevation": hip_elev,
        "hip_x": hip_x_list,
        "ankle_l_y": ankle_l,
        "ankle_r_y": ankle_r,
    }


# ─── TUG ──────────────────────────────────────────────────────────────────────

def tug_phases_from_pose_signal(
    signal: "dict[str, Any]",
    metadata: "VideoMetadata",
    warnings: list[str],
) -> "list[Interval] | None":
    """Derive TUG phase intervals from a pose signal (see extract_pose_signal).

    Uses the same boundary-finding approach as the bbox motion fallback but
    operates on the cleaner hip-elevation signal from MediaPipe landmarks.
    Returns 7 intervals at confidence 0.85, or None if detection fails.
    """
    from .phase_rules import phase_intervals_from_tug_boundaries
    from .types import Interval

    fps = metadata.fps or 30.0
    effective_fps = fps / (_SKIP + 1)
    frames = signal["frames"]

    smoothed = _ma(signal["hip_elevation"], max(3, int(round(effective_fps * 0.75))))
    count = len(smoothed)
    if count < 20:
        return None

    edge = max(3, min(int(round(effective_fps)), int(round(count * 0.10))))
    start_low = _med(smoothed[:edge])
    end_low = _med(smoothed[-edge:])
    peak_idx = _argmax(smoothed, int(count * 0.20), int(count * 0.80))
    peak = smoothed[peak_idx]
    rise = peak - start_low
    fall = peak - end_low

    # Use 10th-90th percentile range to avoid outliers from tracking dropouts
    # (e.g. landmark errors during turn when person faces away from camera)
    ordered_s = sorted(smoothed)
    n_s = len(ordered_s)
    p10 = ordered_s[max(0, n_s // 10)]
    p90 = ordered_s[min(n_s - 1, 9 * n_s // 10)]
    height_range = max(p90 - p10, 1e-6)

    if height_range < 0.01 or rise / height_range < 0.15:
        warnings.append("pose estimation: hip elevation range too small for TUG detection")
        return None

    rise = max(1e-6, rise)
    fall = max(1e-6, fall)

    t2 = _first_idx(smoothed, 0, peak_idx, start_low + 0.30 * rise, "ge")
    if t2 is None:
        t2 = max(1, peak_idx - int(round(count * 0.45)))
    t1 = _first_idx(smoothed, 0, t2, start_low + 0.05 * rise, "ge")
    if t1 is None:
        t1 = max(0, t2 - int(round(1.5 * effective_fps)))
    t3 = _first_idx(smoothed, t2 + 1, peak_idx, peak - 0.16 * rise, "ge")
    if t3 is None:
        t3 = max(t2 + 1, peak_idx - int(round(0.6 * effective_fps)))
    t4 = _first_idx(smoothed, peak_idx, count - 1, peak - 0.16 * fall, "le")
    if t4 is None:
        t4 = min(count - 1, peak_idx + int(round(0.6 * effective_fps)))
    t5 = _first_idx(smoothed, t4 + 1, count - 1, end_low + 0.32 * fall, "le")
    if t5 is None:
        t5 = min(count - 1, t4 + int(round(0.25 * count)))
    t6 = _first_idx(smoothed, t5 + 1, count - 1, end_low + 0.16 * fall, "le")
    if t6 is None:
        t6 = min(count - 1, t5 + int(round(effective_fps)))
    t7 = _first_idx(smoothed, t6 + 1, count - 1, end_low + 0.06 * fall, "le")
    if t7 is None:
        t7 = min(count - 1, t6 + int(round(effective_fps)))

    indices = _repair_indices([t1, t2, t3, t4, t5, t6, t7], count, effective_fps)
    boundaries = [frames[i] + 1 for i in indices]  # one-based frame numbers

    if not _valid_tug_boundaries(boundaries, metadata.num_frames):
        warnings.append("pose estimation: TUG boundaries invalid — skipping")
        return None

    phases = phase_intervals_from_tug_boundaries(boundaries, fps, metadata.num_frames)
    if len(phases) != 7:
        return None

    return [
        Interval(p.start_ms, p.end_ms, p.label, 0.85, "pose_estimation")
        for p in phases
    ]


# ─── CRT ──────────────────────────────────────────────────────────────────────

def crt_phases_from_pose_signal(
    signal: "dict[str, Any]",
    metadata: "VideoMetadata",
    warnings: list[str],
) -> "list[Interval] | None":
    """Detect CRT sit/stand cycles from hip elevation signal.

    Same threshold-crossing approach as crt_phase_intervals_from_detections
    but uses the cleaner MediaPipe hip position rather than bbox height.
    Returns intervals at confidence 0.85, or None on failure.
    """
    from .config import (
        CRT_MIN_RISE_FRAMES,
        CRT_REPS,
        CRT_SIT_STAND_THRESHOLD_RATIO,
        CRT_SMOOTH_WINDOW_SEC,
    )
    from .phase_rules import frame_to_ms, smooth_phase_intervals
    from .types import Interval

    fps = metadata.fps or 30.0
    effective_fps = fps / (_SKIP + 1)
    frames = signal["frames"]
    elevation = signal["hip_elevation"]

    if len(elevation) < int(effective_fps * 2):
        return None

    smoothed = _ma(elevation, max(3, int(round(effective_fps * CRT_SMOOTH_WINDOW_SEC))))
    ordered = sorted(smoothed)
    n = len(ordered)
    sit_level = _med(ordered[: max(1, n // 4)])
    stand_level = _med(ordered[max(1, 3 * n // 4):])

    if stand_level - sit_level < 0.02:
        warnings.append("pose estimation: CRT hip elevation range too small to detect cycles")
        return None

    threshold = sit_level + CRT_SIT_STAND_THRESHOLD_RATIO * (stand_level - sit_level)
    up_frames: list[int] = []
    down_frames: list[int] = []
    for i in range(1, len(smoothed)):
        prev, cur = smoothed[i - 1], smoothed[i]
        if prev < threshold <= cur:
            up_frames.append(frames[i])
        elif prev >= threshold > cur:
            down_frames.append(frames[i])

    min_rise = max(2, CRT_MIN_RISE_FRAMES // (_SKIP + 1))
    pairs: list[tuple[int, int]] = []
    di = 0
    for uf in up_frames:
        while di < len(down_frames) and down_frames[di] <= uf:
            di += 1
        if di < len(down_frames) and down_frames[di] - uf >= min_rise:
            pairs.append((uf, down_frames[di]))
            di += 1
    pairs = pairs[:CRT_REPS]

    if not pairs:
        warnings.append("pose estimation: no CRT rise cycles detected from hip signal")
        return None

    frame_to_idx_map = {f: i for i, f in enumerate(frames)}

    def _fi(fr: int) -> int:
        idx = frame_to_idx_map.get(fr)
        if idx is not None:
            return idx
        return min(range(len(frames)), key=lambda i: abs(frames[i] - fr))

    intervals: list[Interval] = []
    prev_end_ms = frame_to_ms(frames[0], fps, "zero")

    for up_frame, down_frame in pairs:
        rise_ms = frame_to_ms(up_frame, fps, "zero")
        sit_ms = frame_to_ms(down_frame, fps, "zero")
        if rise_ms > prev_end_ms:
            intervals.append(Interval(prev_end_ms, rise_ms, "sit", 0.85, "pose_estimation"))
        i_start = _fi(up_frame)
        i_end = _fi(down_frame)
        peak_local = max(range(i_start, max(i_end, i_start) + 1), key=lambda i: smoothed[i])
        peak_ms = max(rise_ms + 1, min(frame_to_ms(frames[peak_local], fps, "zero"), sit_ms - 1))
        intervals.append(Interval(rise_ms, peak_ms, "sit-to-stand", 0.85, "pose_estimation"))
        intervals.append(Interval(peak_ms, sit_ms, "stand-to-sit", 0.85, "pose_estimation"))
        prev_end_ms = sit_ms

    last_ms = frame_to_ms(frames[-1], fps, "zero")
    if last_ms > prev_end_ms:
        intervals.append(Interval(prev_end_ms, last_ms, "sit", 0.85, "pose_estimation"))

    return smooth_phase_intervals(intervals) or None


# ─── Foot intervals ───────────────────────────────────────────────────────────

def foot_intervals_from_pose_signal(
    signal: "dict[str, Any]",
    metadata: "VideoMetadata",
    walk_spans: "list[Interval]",
    warnings: list[str],
) -> "tuple[list[Interval], list[Interval]]":
    """Derive left/right stance/swing intervals from ankle Y positions.

    In MediaPipe normalised image coordinates (Y = 0 at top of frame):
      Higher ankle Y → ankle lower in image → foot on ground (stance)
      Lower ankle Y  → ankle higher in image → foot lifted (swing)

    ankle_diff = ankle_l.y − ankle_r.y
      > dead-band  → left foot lower  → left stance / right swing
      < −dead-band → right foot lower → right stance / left swing
    """
    from .phase_rules import clip_to_spans, frame_to_ms, smooth_foot_intervals
    from .types import Interval

    frames = signal["frames"]
    fps = metadata.fps or 30.0
    effective_fps = fps / (_SKIP + 1)

    if not frames or not walk_spans:
        return [], []

    window = max(3, int(round(effective_fps * 0.15)))
    smooth_l = _ma(signal["ankle_l_y"], window)
    smooth_r = _ma(signal["ankle_r_y"], window)
    diff = [l - r for l, r in zip(smooth_l, smooth_r)]
    dead = 0.02

    left_ivs: list[Interval] = []
    right_ivs: list[Interval] = []
    i = 0
    while i < len(frames):
        if diff[i] > dead:
            j = i + 1
            while j < len(frames) and diff[j] > -dead:
                j += 1
            s_ms = frame_to_ms(frames[i], fps, "zero")
            e_ms = frame_to_ms(frames[j - 1], fps, "zero")
            if e_ms > s_ms:
                left_ivs.append(Interval(s_ms, e_ms, "left_stance", 0.80, "pose_gait"))
                right_ivs.append(Interval(s_ms, e_ms, "right_swing", 0.80, "pose_gait"))
            i = j
        elif diff[i] < -dead:
            j = i + 1
            while j < len(frames) and diff[j] < dead:
                j += 1
            s_ms = frame_to_ms(frames[i], fps, "zero")
            e_ms = frame_to_ms(frames[j - 1], fps, "zero")
            if e_ms > s_ms:
                right_ivs.append(Interval(s_ms, e_ms, "right_stance", 0.80, "pose_gait"))
                left_ivs.append(Interval(s_ms, e_ms, "left_swing", 0.80, "pose_gait"))
            i = j
        else:
            i += 1

    return (
        smooth_foot_intervals(clip_to_spans(left_ivs, walk_spans)),
        smooth_foot_intervals(clip_to_spans(right_ivs, walk_spans)),
    )


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _ma(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(1, int(window))
    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        out.append(sum(values[start:end]) / (end - start))
    return out


def _med(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


def _argmax(values: list[float], start: int, end: int) -> int:
    start = max(0, start)
    end = min(len(values) - 1, end)
    if end < start:
        return start
    return max(range(start, end + 1), key=lambda i: values[i])


def _first_idx(
    values: list[float], start: int, end: int, threshold: float, direction: str
) -> "int | None":
    start, end = max(0, start), min(len(values) - 1, end)
    for i in range(start, end + 1):
        if direction == "ge" and values[i] >= threshold:
            return i
        if direction == "le" and values[i] <= threshold:
            return i
    return None


def _repair_indices(indices: list[int], count: int, fps: float) -> list[int]:
    min_gap = max(2, int(round(0.25 * fps)))
    rep = [max(0, min(count - 1, int(i))) for i in indices]
    for k in range(1, len(rep)):
        if rep[k] <= rep[k - 1] + min_gap:
            rep[k] = rep[k - 1] + min_gap
    if rep[-1] >= count:
        for k in range(len(rep) - 1, -1, -1):
            rep[k] = min(rep[k], count - 1 - (len(rep) - 1 - k) * min_gap)
        for k in range(1, len(rep)):
            rep[k] = max(rep[k], rep[k - 1] + min_gap)
    return [max(0, min(count - 1, i)) for i in rep]


def _valid_tug_boundaries(boundaries: list[int], num_frames: "int | None") -> bool:
    if len(boundaries) < 7:
        return False
    if any(r <= l for l, r in zip(boundaries, boundaries[1:])):
        return False
    if boundaries[0] < 1:
        return False
    if num_frames and boundaries[-1] > num_frames + 1:
        return False
    return True
