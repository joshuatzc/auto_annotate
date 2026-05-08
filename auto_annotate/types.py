"""Small shared types for TUG auto-annotation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
