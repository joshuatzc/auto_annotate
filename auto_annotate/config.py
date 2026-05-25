"""Configuration for the FrailScreen ELAN pre-labeller."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PHASE_TIER = "phase"
LEFT_FOOT_TIER = "left_foot"
RIGHT_FOOT_TIER = "right_foot"
OCCLUSION_TIER = "occlusion"
TEST_TIER = "test"

PHASE_LABELS = (
    "sit-to-stand",
    "stand",
    "walk",
    "turn",
    "stand-to-sit",
    "sit",
    "in_pos",
    "out_of_pos",
)
LEFT_FOOT_LABELS = ("left_stance", "left_swing")
RIGHT_FOOT_LABELS = ("right_stance", "right_swing")
OCCLUSION_LABELS = ("occluded_by_person", "occlusion_inferred", "outside_frame")
TEST_LABELS = ("time",)
UNKNOWN_LABEL = "unknown"

TIER_ORDER = (PHASE_TIER, LEFT_FOOT_TIER, RIGHT_FOOT_TIER, TEST_TIER)
CRT_TIER_ORDER = (PHASE_TIER, TEST_TIER)
TUG_TIER_ORDER = (TEST_TIER, PHASE_TIER, LEFT_FOOT_TIER, RIGHT_FOOT_TIER)

TUG_PHASE_LABELS = (
    "sit",
    "sit-to-stand",
    "walk",
    "turn",
    "stand-to-sit",
    UNKNOWN_LABEL,
)

TUG_PHASE_STATE_SEQUENCE = (
    "sit",
    "sit-to-stand",
    "walk",
    "turn",
    "walk",
    "stand-to-sit",
    "sit",
)

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
BALANCE_TEST_DURATION_MS = 10_000

PHASE_MIN_DURATION_MS = 60
PHASE_GAP_MERGE_MS = 120
FOOT_MIN_DURATION_MS = 70
FOOT_GAP_MERGE_MS = 60

FALLBACK_PHASE_RATIOS = (0.10, 0.27, 0.08, 0.27, 0.08, 0.12, 0.08)
FALLBACK_STRIDE_MS = 1_325
FALLBACK_STANCE_RATIO = 0.51

SUBJECT_MIN_COVERAGE = 0.25
TRACK_IOU_THRESHOLD = 0.25
TRACK_MAX_FRAME_GAP = 5
PATIENT_LOCK_MAX_FRAGMENT_GAP = 90
OCCLUSION_MIN_DURATION_MS = 600
OCCLUSION_GAP_MERGE_MS = 160
OCCLUSION_MIN_IOU = 0.10
OCCLUSION_MIN_SUBJECT_OVERLAP = 0.22
OCCLUSION_NEAR_CENTER_RATIO = 0.18

# CRT-specific
CRT_REPS = 5
CRT_PHASE_LABELS = ("sit", "sit-to-stand", "stand-to-sit")
CRT_SIT_STAND_THRESHOLD_RATIO = 0.45
CRT_MIN_RISE_FRAMES = 8
CRT_SMOOTH_WINDOW_SEC = 0.35

# TUG-only 3DGait adapter defaults.
MIN_WALK_DURATION_MS = 500
MIN_PHASE_DURATION_MS = 200
TURN_VELOCITY_THRESHOLD = 0.3
SMOOTHING_WINDOW_MS = 150
BYSTANDER_CENTER_MARGIN_PX = 200
UNKNOWN_CONFIDENCE_THRESH = 0.6
OUTPUT_DIR = "./output"

CONFIGURABLE_PARAMETERS = {
    "MIN_WALK_DURATION_MS",
    "MIN_PHASE_DURATION_MS",
    "TURN_VELOCITY_THRESHOLD",
    "SMOOTHING_WINDOW_MS",
    "BYSTANDER_CENTER_MARGIN_PX",
    "UNKNOWN_CONFIDENCE_THRESH",
    "OUTPUT_DIR",
}


def load_config_overrides(path: str | Path | None) -> dict[str, Any]:
    """Load YAML-style config overrides for known tuning parameters."""

    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    data = _read_yaml_mapping(text)
    unknown = sorted(set(data) - CONFIGURABLE_PARAMETERS)
    if unknown:
        raise ValueError("unknown config key(s): " + ", ".join(unknown))
    return data


def apply_config_overrides(path: str | Path | None) -> dict[str, Any]:
    """Apply YAML-style overrides to this module's tuning constants."""

    overrides = load_config_overrides(path)
    for key, value in overrides.items():
        globals()[key] = value
    return overrides


def _read_yaml_mapping(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError("config YAML must be a key/value mapping")
        return dict(loaded)
    except ModuleNotFoundError:
        return _read_simple_yaml_mapping(text)


def _read_simple_yaml_mapping(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"invalid config line {line_number}: {raw_line}")
        key, raw_value = line.split(":", 1)
        data[key.strip()] = _coerce_config_value(raw_value.strip())
    return data


def _coerce_config_value(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
