"""Configuration for the isolated TUG ELAN pre-labeller."""

from __future__ import annotations

PHASE_TIER = "phase"
LEFT_FOOT_TIER = "left_foot"
RIGHT_FOOT_TIER = "right_foot"
TEST_TIER = "test"

PHASE_LABELS = (
    "sit-to-stand",
    "walk",
    "turn",
    "stand-to-sit",
    "sit",
)
LEFT_FOOT_LABELS = ("left_stance", "left_swing")
RIGHT_FOOT_LABELS = ("right_stance", "right_swing")
TEST_LABELS = ("time",)
UNKNOWN_LABEL = "unknown"

TIER_ORDER = (PHASE_TIER, LEFT_FOOT_TIER, RIGHT_FOOT_TIER, TEST_TIER)
CRT_TIER_ORDER = (PHASE_TIER, TEST_TIER)

TUG_PHASE_SEQUENCE = (
    "sit-to-stand",
    "walk",
    "turn",
    "walk",
    "turn",
    "stand-to-sit",
    "sit",
)

DEFAULT_FPS = 30.0
DEFAULT_VIDEO_WIDTH = 1920
DEFAULT_VIDEO_HEIGHT = 1080
DEFAULT_FALLBACK_TEST_MS = 10_000

PHASE_MIN_DURATION_MS = 60
PHASE_GAP_MERGE_MS = 120
FOOT_MIN_DURATION_MS = 70
FOOT_GAP_MERGE_MS = 60

FALLBACK_PHASE_RATIOS = (0.10, 0.27, 0.08, 0.27, 0.08, 0.12, 0.08)
FALLBACK_STRIDE_MS = 1_100
FALLBACK_STANCE_RATIO = 0.62

SUBJECT_MIN_COVERAGE = 0.25
TRACK_IOU_THRESHOLD = 0.25
TRACK_MAX_FRAME_GAP = 5

# CRT-specific
CRT_REPS = 5
CRT_PHASE_LABELS = ("sit", "sit-to-stand", "stand-to-sit")
CRT_SIT_STAND_THRESHOLD_RATIO = 0.45
CRT_MIN_RISE_FRAMES = 8
CRT_SMOOTH_WINDOW_SEC = 0.35
