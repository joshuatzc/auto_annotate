"""Offline normalized 3D pose provider and pose-derived event detectors."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import (
    POSE_JOINT_NAMES,
    Interval,
    Pose3DResult,
    PoseFrame,
    PoseJoint,
    VideoMetadata,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

SIDECAR_NAME = "pose3d.json"
SCHEMA_VERSION = 1
POSE_FRAME_STRIDE = 3
MIN_POSE_FRAMES = 20
LOW_CONFIDENCE = 0.35
OCCLUDED_CONFIDENCE = 0.60

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
_MODEL_CACHE = Path.home() / ".cache" / "auto_annotate" / "pose_landmarker_lite.task"

_BACKEND_COMMAND_ENV = {
    "wham": "AUTO_ANNOTATE_WHAM_POSE3D_CMD",
    "comotion": "AUTO_ANNOTATE_COMOTION_POSE3D_CMD",
    "whole_body": "AUTO_ANNOTATE_WHOLEBODY_POSE3D_CMD",
}
_MEDIAPIPE_RUNTIME_DISABLED_REASON: str | None = None

_MP_JOINTS = {
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_toe": 31,
    "right_toe": 32,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}


def load_or_run_pose3d(
    test_folder: str | Path,
    metadata: VideoMetadata | None = None,
    warnings: list[str] | None = None,
    deadline: float | None = None,
) -> Pose3DResult | None:
    """Load a valid cached pose3d.json or run a local pose backend.

    The cache is reused only when the source video signature and backend config
    hash match. Backends are local-only: WHAM/CoMotion/whole-body commands can
    be configured with environment variables, and MediaPipe is used only when
    its model already exists locally unless download is explicitly enabled.
    """

    warn = warnings if warnings is not None else []
    folder = Path(test_folder)
    video_path = _first_match(folder, "rgb_video*.mp4")
    if video_path is None or not video_path.exists() or video_path.stat().st_size == 0:
        return None

    meta = metadata or _read_video_metadata(video_path)
    video_sig = _video_signature(video_path, meta)
    config_hash = _model_config_hash()
    sidecar = folder / SIDECAR_NAME

    cached = _load_pose3d_sidecar(sidecar)
    if cached is not None:
        if _cache_matches(cached, video_sig, config_hash):
            cached.cache_hit = True
            cached.source_path = sidecar
            return cached
        warn.append("pose3d cache is stale; regenerating pose3d.json")

    if not _has_time(deadline, _min_backend_seconds()):
        warn.append("pose3d skipped: annotation time budget nearly exhausted")
        return None

    provider = PoseProvider(config_hash)
    result = provider.run(folder, video_path, meta, video_sig, warn, deadline)
    if result is None:
        return None

    result.source_path = sidecar
    try:
        _write_pose3d_sidecar(sidecar, result)
    except OSError as exc:
        result.source_path = None
        warn.append(f"pose3d cache could not be written: {exc}")
    return result


class PoseProvider:
    """Backend orchestrator for normalized local 3D pose extraction."""

    def __init__(self, model_config_hash: str) -> None:
        self.model_config_hash = model_config_hash

    def run(
        self,
        folder: Path,
        video_path: Path,
        metadata: VideoMetadata,
        video_signature: dict[str, Any],
        warnings: list[str],
        deadline: float | None = None,
    ) -> Pose3DResult | None:
        notes: list[str] = []
        for backend in ("wham", "comotion", "whole_body"):
            if not _has_time(deadline, _min_backend_seconds()):
                notes.append("time budget exhausted before remaining pose3d backends")
                break
            result = self._run_command_backend(
                backend, folder, video_path, metadata, video_signature, warnings, notes, deadline
            )
            if result is not None:
                return result

        result = self._run_mediapipe(folder, video_path, metadata, video_signature, notes, deadline)
        if result is not None:
            return result

        if notes:
            warnings.append("pose3d unavailable: " + "; ".join(notes[:3]))
        else:
            warnings.append("pose3d unavailable: no local backend produced usable pose")
        return None

    def _run_command_backend(
        self,
        backend: str,
        folder: Path,
        video_path: Path,
        metadata: VideoMetadata,
        video_signature: dict[str, Any],
        warnings: list[str],
        notes: list[str],
        deadline: float | None,
    ) -> Pose3DResult | None:
        env_name = _BACKEND_COMMAND_ENV[backend]
        command_text = os.environ.get(env_name)
        if not command_text:
            return None

        video_hash = hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()[:8]
        output_path = (
            Path(tempfile.gettempdir())
            / f"auto_annotate_{backend}_pose3d_{os.getpid()}_{video_hash}.json"
        )
        cmd = _build_backend_command(command_text, video_path, output_path, folder)
        try:
            timeout = _timeout_seconds("AUTO_ANNOTATE_POSE3D_TIMEOUT_SEC", 1800, deadline)
            if timeout <= 0:
                warnings.append(f"pose3d {backend} backend skipped: annotation time budget exhausted")
                return None
            completed = subprocess.run(
                cmd,
                cwd=str(folder),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except Exception as exc:
            warnings.append(f"pose3d {backend} backend failed to start: {exc}")
            return None

        if completed.returncode != 0:
            msg = (completed.stderr or completed.stdout or "").strip().splitlines()
            detail = msg[-1] if msg else f"exit code {completed.returncode}"
            warnings.append(f"pose3d {backend} backend failed: {detail}")
            return None
        if not output_path.exists():
            warnings.append(f"pose3d {backend} backend did not write {output_path.name}")
            return None

        result = _load_pose3d_sidecar(output_path)
        try:
            output_path.unlink()
        except OSError:
            pass
        if result is None or len(result.frames) < MIN_POSE_FRAMES:
            warnings.append(f"pose3d {backend} backend produced too few usable frames")
            return None

        result.backend = backend
        result.fps = result.fps or metadata.fps
        result.num_frames = result.num_frames or metadata.num_frames
        result.width = result.width or metadata.width
        result.height = result.height or metadata.height
        result.model_config_hash = self.model_config_hash
        result.video_signature = video_signature
        result.metadata.update(_backend_metadata(backend, "external_command"))
        result.qc_flags = _result_qc_flags(result)
        return result

    def _run_mediapipe(
        self,
        folder: Path,
        video_path: Path,
        metadata: VideoMetadata,
        video_signature: dict[str, Any],
        notes: list[str],
        deadline: float | None,
    ) -> Pose3DResult | None:
        if not _has_time(deadline, _min_backend_seconds()):
            notes.append("mediapipe skipped: annotation time budget nearly exhausted")
            return None
        if os.environ.get("AUTO_ANNOTATE_ENABLE_MEDIAPIPE_POSE3D") != "1":
            notes.append(
                "mediapipe fallback disabled; set AUTO_ANNOTATE_ENABLE_MEDIAPIPE_POSE3D=1 to enable it"
            )
            return None
        if _MEDIAPIPE_RUNTIME_DISABLED_REASON:
            notes.append(_MEDIAPIPE_RUNTIME_DISABLED_REASON)
            return None
        if os.environ.get("AUTO_ANNOTATE_POSE3D_IN_PROCESS") == "1":
            return self._run_mediapipe_in_process(
                folder, video_path, metadata, video_signature, notes
            )
        return self._run_mediapipe_subprocess(
            folder, video_path, metadata, video_signature, notes, deadline
        )

    def _run_mediapipe_subprocess(
        self,
        folder: Path,
        video_path: Path,
        metadata: VideoMetadata,
        video_signature: dict[str, Any],
        notes: list[str],
        deadline: float | None,
    ) -> Pose3DResult | None:
        video_hash = hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()[:8]
        output_path = (
            Path(tempfile.gettempdir())
            / f"auto_annotate_mediapipe_pose3d_{os.getpid()}_{video_hash}.json"
        )
        package_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(package_root)
            if not existing_pythonpath
            else str(package_root) + os.pathsep + existing_pythonpath
        )
        cmd = [
            sys.executable,
            "-m",
            "auto_annotate.pose3d",
            "--mediapipe-worker",
            "--video",
            str(video_path),
            "--output",
            str(output_path),
            "--fps",
            str(metadata.fps or 30.0),
            "--num-frames",
            str(metadata.num_frames or 0),
            "--width",
            str(metadata.width or 0),
            "--height",
            str(metadata.height or 0),
            "--config-hash",
            self.model_config_hash,
            "--video-signature",
            json.dumps(video_signature),
        ]
        try:
            timeout = _timeout_seconds("AUTO_ANNOTATE_POSE3D_TIMEOUT_SEC", 1800, deadline)
            if timeout <= 0:
                notes.append("mediapipe worker skipped: annotation time budget exhausted")
                return None
            completed = subprocess.run(
                cmd,
                cwd=str(package_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except Exception as exc:
            notes.append(f"mediapipe worker failed to start: {exc}")
            return None

        if completed.returncode != 0:
            global _MEDIAPIPE_RUNTIME_DISABLED_REASON
            if completed.returncode < 0:
                _MEDIAPIPE_RUNTIME_DISABLED_REASON = (
                    f"mediapipe worker aborted by native runtime signal {-completed.returncode}"
                )
                notes.append(_MEDIAPIPE_RUNTIME_DISABLED_REASON)
            else:
                detail = (completed.stderr or completed.stdout or "").strip().splitlines()
                notes.append(
                    "mediapipe worker failed"
                    + (f": {detail[-1]}" if detail else f": exit code {completed.returncode}")
                )
            return None
        result = _load_pose3d_sidecar(output_path)
        try:
            output_path.unlink()
        except OSError:
            pass
        if result is None or len(result.frames) < MIN_POSE_FRAMES:
            notes.append("mediapipe worker produced too few usable pose frames")
            return None
        return result

    def _run_mediapipe_in_process(
        self,
        folder: Path,
        video_path: Path,
        metadata: VideoMetadata,
        video_signature: dict[str, Any],
        notes: list[str],
    ) -> Pose3DResult | None:
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError:
            notes.append("mediapipe/cv2 not installed")
            return None

        model_path = _ensure_mediapipe_model(notes)
        if model_path is None:
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            notes.append(f"could not open {video_path.name}")
            return None

        fps = float(cap.get(cv2.CAP_PROP_FPS) or metadata.fps or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or metadata.width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or metadata.height)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or metadata.num_frames)

        try:
            base_options = mp_python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp_python.BaseOptions.Delegate.CPU,
            )
        except Exception:
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )

        frames: list[PoseFrame] = []
        frame_idx = 0
        try:
            with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_idx % POSE_FRAME_STRIDE == 0:
                        timestamp_ms = int(round(frame_idx * 1000.0 / max(fps, 1e-6)))
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        result = landmarker.detect_for_video(mp_image, timestamp_ms)
                        if result.pose_landmarks:
                            landmarks = result.pose_landmarks[0]
                            world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
                            joints = _mediapipe_joints(landmarks, world)
                            if joints:
                                frames.append(
                                    PoseFrame(
                                        frame_idx,
                                        timestamp_ms,
                                        subject_id=1,
                                        joints=joints,
                                        qc_flags=_frame_qc_flags(joints),
                                    )
                                )
                    frame_idx += 1
        except Exception as exc:
            notes.append(f"mediapipe failed during inference: {exc}")
            return None
        finally:
            cap.release()

        frames = _interpolate_short_gaps(frames, fps)
        if len(frames) < MIN_POSE_FRAMES:
            notes.append("mediapipe produced too few usable pose frames")
            return None

        result = Pose3DResult(
            frames=frames,
            fps=fps,
            num_frames=num_frames,
            width=width,
            height=height,
            backend="mediapipe_world",
            model_config_hash=self.model_config_hash,
            video_signature=video_signature,
            metadata=_backend_metadata("mediapipe_world", "mediapipe_pose_landmarker_lite"),
        )
        result.metadata["source_folder"] = str(folder)
        result.qc_flags = _result_qc_flags(result)
        return result


def pose_signal_from_pose3d(result: Pose3DResult | None) -> dict[str, Any] | None:
    """Convert normalized pose3d frames to the legacy kinematic signal shape."""

    if result is None:
        return None

    frames: list[int] = []
    hip_elevation: list[float] = []
    hip_x: list[float] = []
    ankle_l_y: list[float] = []
    ankle_r_y: list[float] = []

    for pose_frame in result.frames:
        pelvis = _joint_or_pelvis(pose_frame)
        left_ankle = _best_foot_joint(pose_frame, "left")
        right_ankle = _best_foot_joint(pose_frame, "right")
        if not (
            _usable_joint(pelvis)
            and _usable_joint(left_ankle)
            and _usable_joint(right_ankle)
            and pelvis.x is not None
            and pelvis.y is not None
            and left_ankle.y is not None
            and right_ankle.y is not None
        ):
            continue
        frames.append(pose_frame.frame_index)
        hip_elevation.append(((left_ankle.y + right_ankle.y) / 2.0) - pelvis.y)
        hip_x.append(pelvis.x)
        ankle_l_y.append(left_ankle.y)
        ankle_r_y.append(right_ankle.y)

    if len(frames) < MIN_POSE_FRAMES:
        return None
    return {
        "frames": frames,
        "hip_elevation": hip_elevation,
        "hip_x": hip_x,
        "ankle_l_y": ankle_l_y,
        "ankle_r_y": ankle_r_y,
    }


def tug_phases_from_pose3d(
    result: Pose3DResult | None,
    metadata: VideoMetadata,
    warnings: list[str],
) -> list[Interval] | None:
    signal = pose_signal_from_pose3d(result)
    if signal is None:
        return None
    from .pose_estimator import tug_phases_from_pose_signal

    phases = tug_phases_from_pose_signal(signal, metadata, warnings)
    if not phases:
        return None
    confidence = _pose_interval_confidence(result, 0.88)
    return [
        Interval(iv.start_ms, iv.end_ms, iv.label, confidence, "pose3d_kinematics")
        for iv in phases
    ]


def crt_phases_from_pose3d(
    result: Pose3DResult | None,
    metadata: VideoMetadata,
    warnings: list[str],
) -> list[Interval] | None:
    signal = pose_signal_from_pose3d(result)
    if signal is None:
        return None
    from .pose_estimator import crt_phases_from_pose_signal

    phases = crt_phases_from_pose_signal(signal, metadata, warnings)
    if not phases:
        return None
    confidence = _pose_interval_confidence(result, 0.88)
    return [
        Interval(iv.start_ms, iv.end_ms, iv.label, confidence, "pose3d_kinematics")
        for iv in phases
    ]


def gait_speed_from_pose3d(
    result: Pose3DResult | None,
    metadata: VideoMetadata,
    warnings: list[str],
) -> tuple[list[Interval], list[Interval], list[Interval]]:
    span = _valid_pose_span(result, metadata)
    if span is None:
        return [], [], []
    confidence = _pose_interval_confidence(result, 0.84)
    phases = [Interval(span[0], span[1], "walk", confidence, "pose3d_walk_span")]
    left, right = foot_intervals_from_pose3d(result, metadata, phases, warnings)
    return phases, left, right


def balance_phase_from_pose3d(
    result: Pose3DResult | None,
    metadata: VideoMetadata,
    label: str,
    warnings: list[str],
) -> list[Interval]:
    usable = _usable_balance_frames(result)
    if len(usable) < MIN_POSE_FRAMES:
        return []

    fps = metadata.fps or (result.fps if result else 30.0)
    start_frame, end_frame = _longest_stable_run(usable, fps)
    start_ms = _frame_to_ms(start_frame, fps)
    end_ms = _frame_to_ms(end_frame + POSE_FRAME_STRIDE, fps)
    if metadata.duration_ms:
        end_ms = min(end_ms, metadata.duration_ms)
    if end_ms <= start_ms:
        return []

    confidence = _pose_interval_confidence(result, 0.82)
    if _balance_motion_level(usable) > 0.025:
        confidence = min(confidence, 0.68)
        warnings.append("pose3d balance hold has elevated motion; review interval")
    return [Interval(start_ms, end_ms, label, confidence, "pose3d_balance_hold")]


def foot_intervals_from_pose3d(
    result: Pose3DResult | None,
    metadata: VideoMetadata,
    walk_spans: list[Interval],
    warnings: list[str],
) -> tuple[list[Interval], list[Interval]]:
    signal = pose_signal_from_pose3d(result)
    if signal is None or not walk_spans:
        return [], []

    from .phase_rules import clip_to_spans, smooth_foot_intervals

    frames = signal["frames"]
    left_y = _moving_average(signal["ankle_l_y"], 3)
    right_y = _moving_average(signal["ankle_r_y"], 3)
    diff = [left - right for left, right in zip(left_y, right_y)]
    if len(diff) < 3:
        return [], []

    spread = _percentile(diff, 0.90) - _percentile(diff, 0.10)
    dead = max(0.012, min(0.04, spread * 0.12))
    left_states: list[str] = []
    right_states: list[str] = []
    current_left = "left_stance" if diff[0] >= 0 else "left_swing"
    current_right = "right_swing" if current_left == "left_stance" else "right_stance"

    for value in diff:
        if value > dead:
            current_left, current_right = "left_stance", "right_swing"
        elif value < -dead:
            current_left, current_right = "left_swing", "right_stance"
        left_states.append(current_left)
        right_states.append(current_right)

    confidence = _pose_interval_confidence(result, 0.80)
    end_ms = max(span.end_ms for span in walk_spans)
    left = _states_to_intervals(frames, left_states, metadata.fps, end_ms, confidence, "pose3d_foot_contact")
    right = _states_to_intervals(frames, right_states, metadata.fps, end_ms, confidence, "pose3d_foot_contact")
    left = smooth_foot_intervals(clip_to_spans(left, walk_spans))
    right = smooth_foot_intervals(clip_to_spans(right, walk_spans))
    left = _cover_foot_walk_spans(left, walk_spans, "left", confidence)
    right = _cover_foot_walk_spans(right, walk_spans, "right", confidence)
    if not left or not right:
        warnings.append("pose3d foot contact did not produce both foot tiers")
    return left, right


def pose3d_summary(result: Pose3DResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "path": str(result.source_path) if result.source_path else None,
        "backend": result.backend,
        "frames": len(result.frames),
        "fps": result.fps,
        "cache_hit": result.cache_hit,
        "qc_flags": list(result.qc_flags),
        "metadata": {
            key: value
            for key, value in result.metadata.items()
            if key in {"backend_family", "model", "created_at", "backend_note"}
        },
    }


def _first_match(folder: Path, pattern: str) -> Path | None:
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None


def _load_pose3d_sidecar(path: Path) -> Pose3DResult | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    return _coerce_pose3d_result(data, path)


def _write_pose3d_sidecar(path: Path, result: Pose3DResult) -> None:
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def _coerce_pose3d_result(data: Any, source_path: Path | None) -> Pose3DResult | None:
    result = Pose3DResult.from_dict(data, source_path)
    if result is not None and result.frames:
        return result
    if not isinstance(data, dict):
        return None

    raw_frames = data.get("frames") or data.get("poses") or []
    joint_names = data.get("joint_names") or list(POSE_JOINT_NAMES)
    frames: list[PoseFrame] = []
    if isinstance(raw_frames, list):
        for raw_frame in raw_frames:
            frame = _external_pose_frame(raw_frame, joint_names, data.get("fps"))
            if frame is not None:
                frames.append(frame)
    if not frames:
        return None

    return Pose3DResult(
        frames=sorted(frames, key=lambda frame: frame.frame_index),
        fps=_float_or(data.get("fps"), 30.0),
        num_frames=int(_float_or(data.get("num_frames"), 0.0)),
        width=int(_float_or(data.get("width"), 0.0)),
        height=int(_float_or(data.get("height"), 0.0)),
        backend=str(data.get("backend") or ""),
        model_config_hash=str(data.get("model_config_hash") or ""),
        video_signature=dict(data.get("video_signature") or {}),
        metadata=dict(data.get("metadata") or {}),
        qc_flags=[str(flag) for flag in (data.get("qc_flags") or [])],
        source_path=source_path,
    )


def _external_pose_frame(raw_frame: Any, joint_names: Any, fps: Any) -> PoseFrame | None:
    if not isinstance(raw_frame, dict):
        return None
    frame_index = int(_float_or(raw_frame.get("frame_index", raw_frame.get("frame")), 0.0))
    time_ms = int(_float_or(raw_frame.get("time_ms"), frame_index * 1000.0 / max(_float_or(fps, 30.0), 1e-6)))
    joints: dict[str, PoseJoint] = {}

    joints_2d = raw_frame.get("joints_2d") or raw_frame.get("keypoints_2d")
    joints_3d = raw_frame.get("joints_3d") or raw_frame.get("keypoints_3d")
    confidences = raw_frame.get("joint_confidence") or raw_frame.get("confidence")
    if not isinstance(joint_names, list):
        joint_names = list(POSE_JOINT_NAMES)

    if isinstance(joints_2d, list):
        for idx, name in enumerate(joint_names):
            if name not in POSE_JOINT_NAMES or idx >= len(joints_2d):
                continue
            xy = joints_2d[idx]
            xyz = joints_3d[idx] if isinstance(joints_3d, list) and idx < len(joints_3d) else None
            conf = _indexed_confidence(confidences, idx)
            joints[name] = _array_pose_joint(xy, xyz, conf)

    if not joints:
        return None
    return PoseFrame(
        frame_index=frame_index,
        time_ms=time_ms,
        subject_id=int(_float_or(raw_frame.get("subject_id", 1), 1.0)),
        joints=joints,
        qc_flags=_frame_qc_flags(joints),
    )


def _array_pose_joint(xy: Any, xyz: Any, confidence: float) -> PoseJoint:
    x = xy[0] if isinstance(xy, list) and len(xy) > 0 else None
    y = xy[1] if isinstance(xy, list) and len(xy) > 1 else None
    z = xy[2] if isinstance(xy, list) and len(xy) > 2 else None
    wx = xyz[0] if isinstance(xyz, list) and len(xyz) > 0 else None
    wy = xyz[1] if isinstance(xyz, list) and len(xyz) > 1 else None
    wz = xyz[2] if isinstance(xyz, list) and len(xyz) > 2 else None
    return PoseJoint(
        x=_optional_float(x),
        y=_optional_float(y),
        z=_optional_float(z),
        world_x=_optional_float(wx),
        world_y=_optional_float(wy),
        world_z=_optional_float(wz),
        confidence=confidence,
        visibility=_visibility_state(_optional_float(x), _optional_float(y), confidence),
    )


def _cache_matches(
    result: Pose3DResult,
    video_signature: dict[str, Any],
    model_config_hash: str,
) -> bool:
    return (
        result.video_signature == video_signature
        and result.model_config_hash == model_config_hash
        and len(result.frames) >= MIN_POSE_FRAMES
    )


def _video_signature(video_path: Path, metadata: VideoMetadata) -> dict[str, Any]:
    stat = video_path.stat()
    return {
        "name": video_path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "fps": round(float(metadata.fps or 0.0), 4),
        "num_frames": int(metadata.num_frames or 0),
        "width": int(metadata.width or 0),
        "height": int(metadata.height or 0),
    }


def _model_config_hash() -> str:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pose_frame_stride": POSE_FRAME_STRIDE,
        "low_confidence": LOW_CONFIDENCE,
        "occluded_confidence": OCCLUDED_CONFIDENCE,
        "commands": {
            name: os.environ.get(env_name, "")
            for name, env_name in sorted(_BACKEND_COMMAND_ENV.items())
        },
        "mediapipe_model": _file_signature(_MODEL_CACHE),
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _file_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {"exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _build_backend_command(command_text: str, video_path: Path, output_path: Path, folder: Path) -> list[str]:
    replacements = {
        "{video}": str(video_path),
        "{output}": str(output_path),
        "{folder}": str(folder),
    }
    expanded = command_text
    used_placeholder = False
    for key, value in replacements.items():
        if key in expanded:
            expanded = expanded.replace(key, value)
            used_placeholder = True
    cmd = shlex.split(expanded)
    if not used_placeholder:
        cmd.extend(["--video", str(video_path), "--output", str(output_path)])
    return cmd


def _ensure_mediapipe_model(notes: list[str]) -> Path | None:
    if _MODEL_CACHE.exists():
        return _MODEL_CACHE
    if os.environ.get("AUTO_ANNOTATE_ALLOW_MODEL_DOWNLOAD") == "1":
        try:
            _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_CACHE)
            return _MODEL_CACHE
        except Exception as exc:
            notes.append(f"could not download mediapipe model: {exc}")
            return None
    notes.append(
        "mediapipe model is not cached locally "
        f"({_MODEL_CACHE}); set AUTO_ANNOTATE_ALLOW_MODEL_DOWNLOAD=1 once to install it"
    )
    return None


def _mediapipe_joints(landmarks: Any, world_landmarks: Any) -> dict[str, PoseJoint]:
    joints: dict[str, PoseJoint] = {}
    for name, index in _MP_JOINTS.items():
        if index >= len(landmarks):
            continue
        lm = landmarks[index]
        world = world_landmarks[index] if world_landmarks is not None and index < len(world_landmarks) else None
        confidence = _landmark_confidence(lm)
        joints[name] = PoseJoint(
            x=_round_float(getattr(lm, "x", None)),
            y=_round_float(getattr(lm, "y", None)),
            z=_round_float(getattr(lm, "z", None)),
            world_x=_round_float(getattr(world, "x", None) if world is not None else None),
            world_y=_round_float(getattr(world, "y", None) if world is not None else None),
            world_z=_round_float(getattr(world, "z", None) if world is not None else None),
            confidence=confidence,
            visibility=_visibility_state(getattr(lm, "x", None), getattr(lm, "y", None), confidence),
        )

    pelvis = _average_joint(joints.get("left_hip"), joints.get("right_hip"))
    if pelvis is not None:
        joints["pelvis"] = pelvis
    return joints


def _landmark_confidence(landmark: Any) -> float:
    values: list[float] = []
    for attr in ("visibility", "presence"):
        value = getattr(landmark, attr, None)
        if value is not None:
            values.append(max(0.0, min(1.0, float(value))))
    if not values:
        return 1.0
    return round(min(values), 4)


def _visibility_state(x: float | None, y: float | None, confidence: float) -> str:
    if x is not None and y is not None and (x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0):
        return "outside_frame"
    if confidence < LOW_CONFIDENCE:
        return "low_confidence"
    if confidence < OCCLUDED_CONFIDENCE:
        return "occluded_inferred"
    return "visible"


def _average_joint(left: PoseJoint | None, right: PoseJoint | None) -> PoseJoint | None:
    if left is None or right is None:
        return None
    confidence = min(left.confidence, right.confidence)
    visibility = _combine_visibility(left.visibility, right.visibility, confidence)
    return PoseJoint(
        x=_avg_optional(left.x, right.x),
        y=_avg_optional(left.y, right.y),
        z=_avg_optional(left.z, right.z),
        world_x=_avg_optional(left.world_x, right.world_x),
        world_y=_avg_optional(left.world_y, right.world_y),
        world_z=_avg_optional(left.world_z, right.world_z),
        confidence=confidence,
        visibility=visibility,
    )


def _combine_visibility(left: str, right: str, confidence: float) -> str:
    if "outside_frame" in (left, right):
        return "outside_frame"
    if "low_confidence" in (left, right) or confidence < LOW_CONFIDENCE:
        return "low_confidence"
    if "interpolated" in (left, right):
        return "interpolated"
    if "occluded_inferred" in (left, right):
        return "occluded_inferred"
    return "visible"


def _frame_qc_flags(joints: dict[str, PoseJoint]) -> list[str]:
    flags: set[str] = set()
    required = ("pelvis", "left_knee", "right_knee", "left_ankle", "right_ankle")
    unavailable = [name for name in required if not _usable_joint(joints.get(name), min_conf=LOW_CONFIDENCE)]
    if len(unavailable) >= 2:
        flags.add("partial_body")
    if any(joint.visibility == "outside_frame" for joint in joints.values()):
        flags.add("outside_frame")
    if any(joint.visibility == "low_confidence" for joint in joints.values()):
        flags.add("low_confidence")
    if any(joint.visibility in {"occluded_inferred", "interpolated"} for joint in joints.values()):
        flags.add("occlusion_inferred")
    return sorted(flags)


def _result_qc_flags(result: Pose3DResult) -> list[str]:
    flags: set[str] = set()
    frames = result.frames
    if not frames:
        return ["failed_model"]
    fps = result.fps or 30.0
    gaps = [
        right.frame_index - left.frame_index
        for left, right in zip(frames, frames[1:])
        if right.frame_index > left.frame_index
    ]
    if gaps and max(gaps) > max(POSE_FRAME_STRIDE * 3, int(round(fps))):
        flags.add("long_tracking_gap")
    partial_count = sum(1 for frame in frames if "partial_body" in frame.qc_flags)
    if partial_count / max(len(frames), 1) > 0.35:
        flags.add("partial_body")
    low_count = sum(1 for frame in frames if "low_confidence" in frame.qc_flags)
    if low_count / max(len(frames), 1) > 0.30:
        flags.add("low_confidence")
    inferred_count = sum(1 for frame in frames if "occlusion_inferred" in frame.qc_flags)
    if inferred_count:
        flags.add("occlusion_inferred")
    return sorted(flags)


def _interpolate_short_gaps(frames: list[PoseFrame], fps: float) -> list[PoseFrame]:
    if len(frames) < 2:
        return frames
    max_gap = max(POSE_FRAME_STRIDE * 2, int(round((fps or 30.0) * 0.5)))
    out: list[PoseFrame] = [frames[0]]
    for prev, cur in zip(frames, frames[1:]):
        gap = cur.frame_index - prev.frame_index
        if POSE_FRAME_STRIDE < gap <= max_gap:
            for frame_index in range(prev.frame_index + POSE_FRAME_STRIDE, cur.frame_index, POSE_FRAME_STRIDE):
                ratio = (frame_index - prev.frame_index) / gap
                joints = _interpolate_joints(prev.joints, cur.joints, ratio)
                out.append(
                    PoseFrame(
                        frame_index,
                        _frame_to_ms(frame_index, fps),
                        subject_id=prev.subject_id,
                        joints=joints,
                        qc_flags=sorted(set(_frame_qc_flags(joints) + ["interpolated"])),
                    )
                )
        out.append(cur)
    return sorted(out, key=lambda frame: frame.frame_index)


def _interpolate_joints(
    left: dict[str, PoseJoint],
    right: dict[str, PoseJoint],
    ratio: float,
) -> dict[str, PoseJoint]:
    joints: dict[str, PoseJoint] = {}
    for name in POSE_JOINT_NAMES:
        a = left.get(name)
        b = right.get(name)
        if a is None or b is None:
            continue
        confidence = min(a.confidence, b.confidence) * 0.75
        joints[name] = PoseJoint(
            x=_lerp_optional(a.x, b.x, ratio),
            y=_lerp_optional(a.y, b.y, ratio),
            z=_lerp_optional(a.z, b.z, ratio),
            world_x=_lerp_optional(a.world_x, b.world_x, ratio),
            world_y=_lerp_optional(a.world_y, b.world_y, ratio),
            world_z=_lerp_optional(a.world_z, b.world_z, ratio),
            confidence=round(confidence, 4),
            visibility="interpolated",
        )
    return joints


def _joint_or_pelvis(frame: PoseFrame) -> PoseJoint:
    pelvis = frame.joints.get("pelvis")
    if pelvis is not None:
        return pelvis
    averaged = _average_joint(frame.joints.get("left_hip"), frame.joints.get("right_hip"))
    return averaged or PoseJoint()


def _best_foot_joint(frame: PoseFrame, side: str) -> PoseJoint:
    for suffix in ("ankle", "heel", "toe"):
        joint = frame.joints.get(f"{side}_{suffix}")
        if _usable_joint(joint):
            return joint  # type: ignore[return-value]
    return frame.joints.get(f"{side}_ankle") or PoseJoint()


def _usable_joint(
    joint: PoseJoint | None,
    min_conf: float = 0.15,
    require_xy: bool = True,
) -> bool:
    if joint is None:
        return False
    if require_xy and (joint.x is None or joint.y is None):
        return False
    if joint.visibility == "outside_frame":
        return False
    if joint.confidence < min_conf:
        return False
    if joint.visibility == "low_confidence" and joint.confidence < LOW_CONFIDENCE:
        return False
    return True


def _valid_pose_span(result: Pose3DResult | None, metadata: VideoMetadata) -> tuple[int, int] | None:
    if result is None:
        return None
    usable_frames: list[int] = []
    for frame in result.frames:
        if _usable_joint(_joint_or_pelvis(frame)) and _usable_joint(_best_foot_joint(frame, "left")) and _usable_joint(_best_foot_joint(frame, "right")):
            usable_frames.append(frame.frame_index)
    if len(usable_frames) < MIN_POSE_FRAMES:
        return None
    fps = metadata.fps or result.fps or 30.0
    start_ms = _frame_to_ms(min(usable_frames), fps)
    end_ms = _frame_to_ms(max(usable_frames) + POSE_FRAME_STRIDE, fps)
    if metadata.duration_ms:
        end_ms = min(end_ms, metadata.duration_ms)
    return (start_ms, end_ms) if end_ms > start_ms else None


def _states_to_intervals(
    frames: list[int],
    states: list[str],
    fps: float,
    final_end_ms: int,
    confidence: float,
    source: str,
) -> list[Interval]:
    if not frames or not states:
        return []
    intervals: list[Interval] = []
    start_index = 0
    current = states[0]
    for index in range(1, min(len(frames), len(states))):
        if states[index] != current:
            start_ms = _frame_to_ms(frames[start_index], fps)
            end_ms = _frame_to_ms(frames[index], fps)
            if end_ms > start_ms:
                intervals.append(Interval(start_ms, end_ms, current, confidence, source))
            start_index = index
            current = states[index]
    start_ms = _frame_to_ms(frames[start_index], fps)
    end_ms = max(start_ms + 1, min(final_end_ms, _frame_to_ms(frames[-1] + POSE_FRAME_STRIDE, fps)))
    if end_ms > start_ms:
        intervals.append(Interval(start_ms, end_ms, current, confidence, source))
    return intervals


def _cover_foot_walk_spans(
    intervals: list[Interval],
    walk_spans: list[Interval],
    side: str,
    confidence: float,
) -> list[Interval]:
    from .smoothing import clean_intervals, merge_short_gaps

    out: list[Interval] = []
    for span in clean_intervals(walk_spans):
        overlapping = [
            Interval(
                max(span.start_ms, iv.start_ms),
                min(span.end_ms, iv.end_ms),
                iv.label,
                iv.confidence,
                iv.source,
            )
            for iv in clean_intervals(intervals)
            if iv.end_ms > span.start_ms and iv.start_ms < span.end_ms
        ]
        overlapping = [iv for iv in overlapping if iv.end_ms > iv.start_ms]
        if not overlapping:
            out.append(Interval(span.start_ms, span.end_ms, f"{side}_stance", 0.35, "pose3d_foot_gap_fill"))
            continue

        cursor = span.start_ms
        previous_label = overlapping[0].label
        for iv in overlapping:
            if iv.start_ms > cursor:
                out.append(
                    Interval(
                        cursor,
                        iv.start_ms,
                        previous_label,
                        min(confidence, 0.55),
                        "pose3d_foot_gap_fill",
                    )
                )
            out.append(iv)
            cursor = max(cursor, iv.end_ms)
            previous_label = iv.label
        if cursor < span.end_ms:
            out.append(
                Interval(
                    cursor,
                    span.end_ms,
                    previous_label,
                    min(confidence, 0.55),
                    "pose3d_foot_gap_fill",
                )
            )
    return merge_short_gaps(clean_intervals(out), 1)


def _usable_balance_frames(result: Pose3DResult | None) -> list[PoseFrame]:
    if result is None:
        return []
    usable: list[PoseFrame] = []
    for frame in result.frames:
        pelvis = _joint_or_pelvis(frame)
        left = _best_foot_joint(frame, "left")
        right = _best_foot_joint(frame, "right")
        if _usable_joint(pelvis) and _usable_joint(left) and _usable_joint(right):
            usable.append(frame)
    return usable


def _longest_stable_run(frames: list[PoseFrame], fps: float) -> tuple[int, int]:
    if len(frames) < 3:
        return frames[0].frame_index, frames[-1].frame_index
    motions = _frame_motions(frames)
    threshold = max(0.006, _percentile(motions, 0.75) * 1.5)
    best_start = 0
    best_end = len(frames) - 1
    cur_start = 0
    for idx, motion in enumerate(motions, start=1):
        gap = frames[idx].frame_index - frames[idx - 1].frame_index
        if motion > threshold or gap > max(POSE_FRAME_STRIDE * 3, int(round(fps))):
            if idx - 1 - cur_start > best_end - best_start:
                best_start, best_end = cur_start, idx - 1
            cur_start = idx
    if len(frames) - 1 - cur_start > best_end - best_start:
        best_start, best_end = cur_start, len(frames) - 1
    total_duration = frames[-1].frame_index - frames[0].frame_index
    best_duration = frames[best_end].frame_index - frames[best_start].frame_index
    if total_duration > 0 and best_duration / total_duration < 0.40:
        return frames[0].frame_index, frames[-1].frame_index
    return frames[best_start].frame_index, frames[best_end].frame_index


def _balance_motion_level(frames: list[PoseFrame]) -> float:
    motions = _frame_motions(frames)
    return _percentile(motions, 0.50) if motions else 0.0


def _frame_motions(frames: list[PoseFrame]) -> list[float]:
    motions: list[float] = []
    for prev, cur in zip(frames, frames[1:]):
        values: list[float] = []
        for name in ("pelvis", "left_ankle", "right_ankle"):
            a = _joint_or_pelvis(prev) if name == "pelvis" else prev.joints.get(name)
            b = _joint_or_pelvis(cur) if name == "pelvis" else cur.joints.get(name)
            if a is None or b is None or a.x is None or a.y is None or b.x is None or b.y is None:
                continue
            values.append(abs(a.x - b.x) + abs(a.y - b.y))
        if values:
            motions.append(sum(values) / len(values))
    return motions


def _pose_interval_confidence(result: Pose3DResult | None, base: float) -> float:
    if result is None:
        return base
    confidence = base
    flags = set(result.qc_flags)
    if "long_tracking_gap" in flags:
        confidence -= 0.10
    if "partial_body" in flags:
        confidence -= 0.08
    if "low_confidence" in flags:
        confidence -= 0.08
    if result.backend == "mediapipe_world":
        confidence -= 0.03
    return round(max(0.45, min(base, confidence)), 2)


def _backend_metadata(backend: str, model: str) -> dict[str, Any]:
    return {
        "backend_family": backend,
        "model": model,
        "backend_note": "local/offline; no source video leaves the machine",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _min_backend_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("AUTO_ANNOTATE_POSE3D_MIN_SECONDS", "4")))
    except ValueError:
        return 4.0


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


def _read_video_metadata(video_path: Path) -> VideoMetadata:
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return VideoMetadata(video_path, fps if fps > 0 else 30.0, frames, width, height)
    except Exception:
        return VideoMetadata(video_path, 30.0, 0, 1920, 1080)


def _moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    half = max(0, window // 2)
    out: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        out.append(sum(values[start:end]) / (end - start))
    return out


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round(max(0.0, min(1.0, fraction)) * (len(ordered) - 1)))
    return ordered[index]


def _frame_to_ms(frame: int | float, fps: float) -> int:
    fps = fps if fps > 0 else 30.0
    return int(round(float(frame) * 1000.0 / fps))


def _avg_optional(left: float | None, right: float | None) -> float | None:
    if left is None and right is None:
        return None
    if left is None:
        return right
    if right is None:
        return left
    return _round_float((left + right) / 2.0)


def _lerp_optional(left: float | None, right: float | None, ratio: float) -> float | None:
    if left is None or right is None:
        return None
    return _round_float(left + ratio * (right - left))


def _round_float(value: Any) -> float | None:
    parsed = _optional_float(value)
    return None if parsed is None else round(parsed, 6)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or(value: Any, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _indexed_confidence(value: Any, index: int) -> float:
    if isinstance(value, dict):
        return max(0.0, min(1.0, _float_or(value.get(str(index), value.get(index)), 1.0)))
    if isinstance(value, list) and index < len(value):
        return max(0.0, min(1.0, _float_or(value[index], 1.0)))
    return 1.0


def _worker_main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="auto_annotate.pose3d")
    parser.add_argument("--mediapipe-worker", action="store_true")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--num-frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--video-signature", required=True)
    args = parser.parse_args(argv)

    if not args.mediapipe_worker:
        parser.error("only --mediapipe-worker is supported")

    video_path = Path(args.video)
    metadata = VideoMetadata(
        video_path,
        args.fps,
        args.num_frames,
        args.width,
        args.height,
    )
    try:
        video_signature = json.loads(args.video_signature)
    except Exception:
        video_signature = _video_signature(video_path, metadata)

    notes: list[str] = []
    provider = PoseProvider(args.config_hash)
    result = provider._run_mediapipe_in_process(
        video_path.parent,
        video_path,
        metadata,
        video_signature,
        notes,
    )
    if result is None:
        for note in notes[:3]:
            print(note, file=sys.stderr)
        return 1
    _write_pose3d_sidecar(Path(args.output), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main(sys.argv[1:]))
