"""Small shared types for FrailScreen auto-annotation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GaitEvent:
    """One stride event emitted by the 3DGait boundary pipeline."""

    time_ms: int
    side: str
    event_type: str


@dataclass(frozen=True)
class TUGInterval:
    """Timed Up and Go interval in milliseconds from video start."""

    start_ms: int
    end_ms: int
    duration_ms: int


@dataclass(frozen=True)
class PhaseAnnotation:
    """One ELAN phase annotation for the TUG phase tier."""

    start_ms: int
    end_ms: int
    label: str


@dataclass(frozen=True)
class FootAnnotation:
    """One ELAN foot-state annotation for a single foot tier."""

    start_ms: int
    end_ms: int
    label: str
    side: str


@dataclass(frozen=True)
class AnnotationBundle:
    """Complete four-tier TUG annotation payload."""

    tug: TUGInterval
    phases: list[PhaseAnnotation]
    left: list[FootAnnotation]
    right: list[FootAnnotation]


POSE_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_toe",
    "right_toe",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

POSE_VISIBILITY_STATES = (
    "visible",
    "occluded_inferred",
    "outside_frame",
    "low_confidence",
    "interpolated",
)


@dataclass(frozen=True)
class Interval:
    """A labelled time interval in milliseconds."""

    start_ms: int
    end_ms: int
    label: str
    confidence: float = 1.0
    source: str = ""

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "start_ms": int(self.start_ms),
            "end_ms": int(self.end_ms),
            "label": self.label,
        }
        if self.confidence != 1.0:
            out["confidence"] = float(self.confidence)
        if self.source:
            out["source"] = self.source
        return out


@dataclass(frozen=True)
class DetectionBox:
    """A person or face bounding box in one video frame."""

    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int | None = None
    score: float | None = None
    label: str | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def with_track(self, track_id: int) -> "DetectionBox":
        return DetectionBox(
            frame=self.frame,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            track_id=track_id,
            score=self.score,
            label=self.label,
        )


@dataclass
class SubjectTrack:
    """Locked single-subject track."""

    track_id: int | None
    boxes: list[DetectionBox] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    locked_track_ids: list[int] = field(default_factory=list)

    @property
    def frames(self) -> list[int]:
        return [box.frame for box in self.boxes]

    @property
    def first_frame(self) -> int | None:
        return min(self.frames) if self.boxes else None

    @property
    def last_frame(self) -> int | None:
        return max(self.frames) if self.boxes else None


@dataclass(frozen=True)
class VideoMetadata:
    """Best-effort source video metadata."""

    path: Path | None
    fps: float
    num_frames: int
    width: int
    height: int

    @property
    def duration_ms(self) -> int:
        if self.num_frames <= 0 or self.fps <= 0:
            return 0
        return int(round((self.num_frames * 1000.0) / self.fps))


@dataclass(frozen=True)
class PoseJoint:
    """One normalized 2D landmark with optional world-space 3D coordinates."""

    x: float | None = None
    y: float | None = None
    z: float | None = None
    world_x: float | None = None
    world_y: float | None = None
    world_z: float | None = None
    confidence: float = 0.0
    visibility: str = "low_confidence"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "confidence": float(self.confidence),
            "visibility": self.visibility,
        }
        for key in ("x", "y", "z", "world_x", "world_y", "world_z"):
            value = getattr(self, key)
            if value is not None:
                out[key] = float(value)
        return out

    @classmethod
    def from_dict(cls, data: Any) -> "PoseJoint | None":
        if not isinstance(data, dict):
            return None
        return cls(
            x=_optional_float(data.get("x")),
            y=_optional_float(data.get("y")),
            z=_optional_float(data.get("z")),
            world_x=_optional_float(data.get("world_x", data.get("x3d"))),
            world_y=_optional_float(data.get("world_y", data.get("y3d"))),
            world_z=_optional_float(data.get("world_z", data.get("z3d"))),
            confidence=_safe_float(data.get("confidence"), 0.0),
            visibility=str(data.get("visibility") or "low_confidence"),
        )


@dataclass
class PoseFrame:
    """Normalized pose for one processed video frame."""

    frame_index: int
    time_ms: int
    subject_id: int | None = None
    joints: dict[str, PoseJoint] = field(default_factory=dict)
    qc_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "time_ms": int(self.time_ms),
            "subject_id": self.subject_id,
            "joints": {
                name: joint.to_dict()
                for name, joint in self.joints.items()
                if name in POSE_JOINT_NAMES
            },
            "qc_flags": list(self.qc_flags),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "PoseFrame | None":
        if not isinstance(data, dict):
            return None
        joints: dict[str, PoseJoint] = {}
        raw_joints = data.get("joints") or {}
        if isinstance(raw_joints, dict):
            for name, raw_joint in raw_joints.items():
                joint = PoseJoint.from_dict(raw_joint)
                if joint is not None and name in POSE_JOINT_NAMES:
                    joints[name] = joint
        return cls(
            frame_index=int(_safe_float(data.get("frame_index", data.get("frame")), 0.0)),
            time_ms=int(_safe_float(data.get("time_ms"), 0.0)),
            subject_id=_optional_int(data.get("subject_id", data.get("track_id"))),
            joints=joints,
            qc_flags=[str(flag) for flag in (data.get("qc_flags") or [])],
        )


@dataclass
class Pose3DResult:
    """Cached normalized 3D pose track for one FrailScreen test video."""

    frames: list[PoseFrame] = field(default_factory=list)
    fps: float = 30.0
    num_frames: int = 0
    width: int = 0
    height: int = 0
    backend: str = ""
    model_config_hash: str = ""
    video_signature: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    qc_flags: list[str] = field(default_factory=list)
    source_path: Path | None = None
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "auto_annotate.pose3d",
            "schema_version": 1,
            "fps": float(self.fps),
            "num_frames": int(self.num_frames),
            "width": int(self.width),
            "height": int(self.height),
            "backend": self.backend,
            "model_config_hash": self.model_config_hash,
            "video_signature": dict(self.video_signature),
            "metadata": dict(self.metadata),
            "qc_flags": list(self.qc_flags),
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, data: Any, source_path: Path | None = None) -> "Pose3DResult | None":
        if not isinstance(data, dict):
            return None
        raw_frames = data.get("frames") or []
        frames: list[PoseFrame] = []
        if isinstance(raw_frames, list):
            for raw_frame in raw_frames:
                frame = PoseFrame.from_dict(raw_frame)
                if frame is not None and frame.joints:
                    frames.append(frame)
        return cls(
            frames=sorted(frames, key=lambda frame: frame.frame_index),
            fps=_safe_float(data.get("fps"), 30.0),
            num_frames=int(_safe_float(data.get("num_frames"), 0.0)),
            width=int(_safe_float(data.get("width"), 0.0)),
            height=int(_safe_float(data.get("height"), 0.0)),
            backend=str(data.get("backend") or ""),
            model_config_hash=str(data.get("model_config_hash") or ""),
            video_signature=dict(data.get("video_signature") or {}),
            metadata=dict(data.get("metadata") or {}),
            qc_flags=[str(flag) for flag in (data.get("qc_flags") or [])],
            source_path=source_path,
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
