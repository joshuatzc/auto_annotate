"""Rules for converting TUG and gait boundaries to ELAN intervals."""

from __future__ import annotations

from typing import Any

from . import config as config_values
from .config import (
    CRT_MIN_RISE_FRAMES,
    CRT_SIT_STAND_THRESHOLD_RATIO,
    CRT_SMOOTH_WINDOW_SEC,
    FALLBACK_PHASE_RATIOS,
    FALLBACK_STANCE_RATIO,
    FALLBACK_STRIDE_MS,
    FOOT_GAP_MERGE_MS,
    FOOT_MIN_DURATION_MS,
    PHASE_GAP_MERGE_MS,
    PHASE_MIN_DURATION_MS,
    TEST_LABELS,
    TUG_PHASE_SEQUENCE,
)
from .smoothing import (
    clean_intervals,
    fill_internal_gaps,
    smooth_foot_annotations,
    smooth_foot_event_times,
    smooth_intervals,
    smooth_phase_annotations,
)
from .types import AnnotationBundle, FootAnnotation, GaitEvent, Interval, PhaseAnnotation, TUGInterval


def build_tug_annotation_bundle(
    tug: TUGInterval,
    left_events: list[GaitEvent],
    right_events: list[GaitEvent],
    pose_keypoints: list[dict[str, Any]] | None = None,
    debug: dict[str, Any] | None = None,
) -> AnnotationBundle:
    """Create the four ELAN tiers for a TUG recording from 3DGait events."""

    all_events = _clip_gait_events(left_events + right_events, tug)
    phases = phase_annotations_from_gait_events(tug, all_events, pose_keypoints, debug)
    walk_phases = [phase for phase in phases if phase.label == "walk"]

    left = foot_annotations_from_events(tug, left_events, "left", walk_phases)
    right = foot_annotations_from_events(tug, right_events, "right", walk_phases)
    if debug is not None:
        debug["raw_events"] = [event.__dict__ for event in all_events]
        debug["post_smoothing_phases"] = [phase.__dict__ for phase in phases]
        debug["left_foot"] = [annotation.__dict__ for annotation in left]
        debug["right_foot"] = [annotation.__dict__ for annotation in right]
    return AnnotationBundle(tug, phases, left, right)


def phase_annotations_from_gait_events(
    tug: TUGInterval,
    events: list[GaitEvent],
    pose_keypoints: list[dict[str, Any]] | None = None,
    debug: dict[str, Any] | None = None,
) -> list[PhaseAnnotation]:
    """Transparent TUG state machine driven by gait cadence and optional pose."""

    cadence_events = _cadence_events(_clip_gait_events(events, tug))
    if not cadence_events:
        phases = [PhaseAnnotation(tug.start_ms, tug.end_ms, "unknown")]
        if debug is not None:
            debug["transition_notes"] = ["no usable gait events inside TUG interval"]
            debug["pre_smoothing_phases"] = [phase.__dict__ for phase in phases]
        return phases

    notes: list[str] = []
    first_event = cadence_events[0]
    alternation_time = _first_bilateral_alternation(cadence_events)
    if alternation_time is None:
        notes.append("no bilateral alternation; phase after first gait event is unknown")
        phases = [
            PhaseAnnotation(tug.start_ms, first_event.time_ms, "sit"),
            PhaseAnnotation(first_event.time_ms, tug.end_ms, "unknown"),
        ]
        phases = _clip_phase_annotations(phases, tug)
        if debug is not None:
            debug["transition_notes"] = notes
            debug["pre_smoothing_phases"] = [phase.__dict__ for phase in phases]
        return smooth_phase_annotations(phases, cadence_events)

    turn = _detect_turn_segment(tug, cadence_events, pose_keypoints, notes)
    final_sit_start = _final_sit_start(tug, cadence_events)
    stand_to_sit_start = _stand_to_sit_start(tug, cadence_events, final_sit_start)

    phases: list[PhaseAnnotation] = []
    _append_phase(phases, tug.start_ms, first_event.time_ms, "sit")
    _append_phase(phases, first_event.time_ms, alternation_time, "sit-to-stand")

    if turn is None:
        unknown_start, unknown_end = _fallback_unknown_turn_window(alternation_time, stand_to_sit_start)
        _append_phase(phases, alternation_time, unknown_start, "walk")
        _append_phase(phases, unknown_start, unknown_end, "unknown")
        _append_phase(phases, unknown_end, stand_to_sit_start, "walk")
        notes.append("turn confidence below threshold; middle turn segment labelled unknown")
    else:
        turn_start, turn_end, turn_label = turn
        _append_phase(phases, alternation_time, turn_start, "walk")
        _append_phase(phases, turn_start, turn_end, turn_label)
        if stand_to_sit_start - turn_end >= int(config_values.MIN_WALK_DURATION_MS):
            _append_phase(phases, turn_end, stand_to_sit_start, "walk")
        else:
            notes.append("second walk after turn is too short; pivot turner path kept without second walk")

    _append_phase(phases, stand_to_sit_start, final_sit_start or tug.end_ms, "stand-to-sit")
    if final_sit_start is not None:
        _append_phase(phases, final_sit_start, tug.end_ms, "sit")

    phases = _clip_phase_annotations(phases, tug)
    phases = _cover_tug_phase_span(phases, tug)
    if debug is not None:
        debug["transition_notes"] = notes
        debug["pre_smoothing_phases"] = [phase.__dict__ for phase in phases]
    return smooth_phase_annotations(phases, cadence_events)


def foot_annotations_from_events(
    tug: TUGInterval,
    events: list[GaitEvent],
    side: str,
    walk_phases: list[PhaseAnnotation],
) -> list[FootAnnotation]:
    """Convert one side's gait events to contiguous stance/swing intervals during walks."""

    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    clipped = [
        event
        for event in smooth_foot_event_times(_clip_gait_events(events, tug))
        if event.side == side and event.event_type in {"stance_start", "swing_start"}
    ]
    label_prefix = f"{side}_"
    if len(clipped) < 2:
        return [
            FootAnnotation(walk.start_ms, walk.end_ms, "unknown", side)
            for walk in walk_phases
            if walk.end_ms > walk.start_ms
        ]

    annotations: list[FootAnnotation] = []
    for walk in walk_phases:
        span_events = [event for event in clipped if walk.start_ms <= event.time_ms <= walk.end_ms]
        if len(span_events) < 2:
            annotations.append(FootAnnotation(walk.start_ms, walk.end_ms, "unknown", side))
            continue
        cursor = walk.start_ms
        state = "unknown"
        for event in span_events:
            if event.time_ms > cursor:
                annotations.append(FootAnnotation(cursor, event.time_ms, state, side))
            state = label_prefix + ("stance" if event.event_type == "stance_start" else "swing")
            cursor = max(cursor, event.time_ms)
        if cursor < walk.end_ms:
            annotations.append(FootAnnotation(cursor, walk.end_ms, state, side))
    return smooth_foot_annotations(annotations)


def _clip_gait_events(events: list[GaitEvent], tug: TUGInterval) -> list[GaitEvent]:
    return sorted(
        (
            event
            for event in events
            if tug.start_ms <= event.time_ms <= tug.end_ms
            and event.event_type in {"stance_start", "swing_start"}
            and event.side in {"left", "right"}
        ),
        key=lambda event: (event.time_ms, event.side, event.event_type),
    )


def _cadence_events(events: list[GaitEvent]) -> list[GaitEvent]:
    stance = [event for event in events if event.event_type == "stance_start"]
    return stance if len(stance) >= 2 else events


def _first_bilateral_alternation(events: list[GaitEvent]) -> int | None:
    previous: GaitEvent | None = None
    for event in events:
        if previous is not None and event.side != previous.side:
            return event.time_ms
        previous = event
    return None


def _detect_turn_segment(
    tug: TUGInterval,
    events: list[GaitEvent],
    pose_keypoints: list[dict[str, Any]] | None,
    notes: list[str],
) -> tuple[int, int, str] | None:
    pose_turn = _turn_from_pose_keypoints(tug, pose_keypoints)
    if pose_turn is not None:
        notes.append("turn detected from optional pose horizontal reversal")
        return pose_turn[0], pose_turn[1], "turn"

    if len(events) < 4:
        return None
    gaps = [
        (right.time_ms - left.time_ms, left.time_ms, right.time_ms)
        for left, right in zip(events, events[1:])
        if right.time_ms > left.time_ms
    ]
    if not gaps:
        return None
    gap_ms, start_ms, end_ms = max(gaps, key=lambda item: item[0])
    min_gap = max(500, int(config_values.MIN_WALK_DURATION_MS))
    if gap_ms >= min_gap:
        notes.append(f"turn detected from cadence gap of {gap_ms} ms")
        return start_ms, end_ms, "turn"
    return None


def _turn_from_pose_keypoints(
    tug: TUGInterval,
    pose_keypoints: list[dict[str, Any]] | None,
) -> tuple[int, int] | None:
    if not pose_keypoints or len(pose_keypoints) < 5:
        return None
    points: list[tuple[int, float]] = []
    for item in pose_keypoints:
        try:
            time_ms = int(round(float(item.get("time_ms", item.get("timestamp_ms")))))
            x_value = item.get("hip_x", item.get("pelvis_x", item.get("head_x")))
            x = float(x_value)
        except (TypeError, ValueError):
            continue
        if tug.start_ms <= time_ms <= tug.end_ms:
            points.append((time_ms, x))
    points.sort()
    if len(points) < 5:
        return None
    velocities: list[tuple[int, float]] = []
    for (left_t, left_x), (right_t, right_x) in zip(points, points[1:]):
        dt_s = max((right_t - left_t) / 1000.0, 1e-6)
        velocities.append((right_t, (right_x - left_x) / dt_s))
    threshold = float(config_values.TURN_VELOCITY_THRESHOLD)
    candidates = [time_ms for time_ms, velocity in velocities if abs(velocity) < threshold]
    if not candidates:
        return None
    start = candidates[0]
    end = candidates[-1]
    if end - start < int(config_values.MIN_PHASE_DURATION_MS):
        pad = int(config_values.MIN_PHASE_DURATION_MS) // 2
        start -= pad
        end += pad
    start = max(tug.start_ms, start)
    end = min(tug.end_ms, end)
    return (start, end) if end > start else None


def _final_sit_start(tug: TUGInterval, events: list[GaitEvent]) -> int | None:
    if not events:
        return None
    candidate = events[-1].time_ms + 500
    if tug.end_ms - candidate >= int(config_values.MIN_PHASE_DURATION_MS):
        return candidate
    return None


def _stand_to_sit_start(
    tug: TUGInterval,
    events: list[GaitEvent],
    final_sit_start: int | None,
) -> int:
    min_phase = int(config_values.MIN_PHASE_DURATION_MS)
    end_limit = (final_sit_start or tug.end_ms) - min_phase
    if events:
        candidate = events[-1].time_ms
    else:
        candidate = tug.end_ms - 700
    candidate = min(candidate, end_limit)
    return max(tug.start_ms + min_phase, candidate)


def _fallback_unknown_turn_window(start_ms: int, end_ms: int) -> tuple[int, int]:
    if end_ms <= start_ms:
        return start_ms, end_ms
    duration = end_ms - start_ms
    width = min(max(int(config_values.MIN_WALK_DURATION_MS), int(duration * 0.20)), duration)
    center = start_ms + duration // 2
    turn_start = max(start_ms, center - width // 2)
    turn_end = min(end_ms, turn_start + width)
    return turn_start, turn_end


def _append_phase(phases: list[PhaseAnnotation], start_ms: int, end_ms: int, label: str) -> None:
    if end_ms > start_ms:
        phases.append(PhaseAnnotation(start_ms, end_ms, label))


def _clip_phase_annotations(phases: list[PhaseAnnotation], tug: TUGInterval) -> list[PhaseAnnotation]:
    out: list[PhaseAnnotation] = []
    for phase in phases:
        start = max(tug.start_ms, phase.start_ms)
        end = min(tug.end_ms, phase.end_ms)
        if end > start:
            out.append(PhaseAnnotation(start, end, phase.label))
    return out


def _cover_tug_phase_span(phases: list[PhaseAnnotation], tug: TUGInterval) -> list[PhaseAnnotation]:
    out: list[PhaseAnnotation] = []
    cursor = tug.start_ms
    for phase in sorted(phases, key=lambda item: (item.start_ms, item.end_ms)):
        if phase.start_ms > cursor:
            out.append(PhaseAnnotation(cursor, phase.start_ms, "unknown"))
        if phase.end_ms > cursor:
            out.append(PhaseAnnotation(max(cursor, phase.start_ms), phase.end_ms, phase.label))
            cursor = phase.end_ms
    if cursor < tug.end_ms:
        out.append(PhaseAnnotation(cursor, tug.end_ms, "unknown"))
    return out


def frame_to_ms(frame: int | float, fps: float, frame_base: str = "zero") -> int:
    """Convert a frame boundary to milliseconds."""

    if fps <= 0:
        fps = 30.0
    value = float(frame)
    if frame_base == "one":
        value = max(0.0, value - 1.0)
    return int(round((value * 1000.0) / fps))


def phase_intervals_from_tug_boundaries(
    boundaries: list[int | float],
    fps: float,
    num_frames: int | None = None,
) -> list[Interval]:
    """Convert core pipeline TUG boundaries into the target phase sequence.

    Core TUG phase boundaries are one-based frame numbers. The first boundary is
    the start of sit-to-stand, and the final appended boundary is the video end.
    """

    if len(boundaries) < 7:
        return []
    normalized = [int(round(v)) for v in boundaries[:8]]
    if len(normalized) == 7:
        if num_frames and num_frames > normalized[-1]:
            normalized.append(num_frames + 1)
        else:
            normalized.append(normalized[-1] + max(1, int(round(fps * 0.5))))

    intervals: list[Interval] = []
    for idx, label in enumerate(TUG_PHASE_SEQUENCE):
        start = frame_to_ms(normalized[idx], fps, "one")
        if idx + 1 == len(TUG_PHASE_SEQUENCE) and num_frames:
            end = max(frame_to_ms(normalized[idx + 1], fps, "one"), frame_to_ms(num_frames + 1, fps, "one"))
        else:
            end = frame_to_ms(normalized[idx + 1], fps, "one")
        if end > start:
            intervals.append(Interval(start, end, label, 0.95, "core_tug_boundaries"))
    return smooth_phase_intervals(intervals)


def fallback_phase_intervals(start_ms: int, end_ms: int) -> list[Interval]:
    """Create an ordered low-confidence TUG phase skeleton."""

    if end_ms <= start_ms:
        return []
    duration = end_ms - start_ms
    cursor = start_ms
    intervals: list[Interval] = []
    for idx, label in enumerate(TUG_PHASE_SEQUENCE):
        if idx == len(TUG_PHASE_SEQUENCE) - 1:
            next_cursor = end_ms
        else:
            next_cursor = cursor + int(round(duration * FALLBACK_PHASE_RATIOS[idx]))
        next_cursor = min(end_ms, max(cursor + 1, next_cursor))
        intervals.append(Interval(cursor, next_cursor, label, 0.45, "fallback_phase_split"))
        cursor = next_cursor
    return smooth_phase_intervals(intervals)


def unknown_phase_interval(start_ms: int, end_ms: int) -> list[Interval]:
    if end_ms <= start_ms:
        return []
    return [Interval(start_ms, end_ms, "unknown", 0.0, "insufficient_data")]


def test_interval_from_phases(phases: list[Interval]) -> list[Interval]:
    cleaned = clean_intervals(phases)
    if not cleaned:
        return []
    return [Interval(cleaned[0].start_ms, cleaned[-1].end_ms, TEST_LABELS[0], 1.0, "phase_span")]


def walking_intervals(phases: list[Interval]) -> list[Interval]:
    return [iv for iv in clean_intervals(phases) if iv.label == "walk"]


def foot_intervals_from_gait_cycles(
    t_left: object,
    t_right: object,
    fps: float,
    walk_spans: list[Interval] | None = None,
    frame_base: str = "zero",
) -> tuple[list[Interval], list[Interval]]:
    """Convert core gait cycle arrays into left/right stance-swing intervals."""

    left = _cycles_to_intervals(t_left, "left", fps, frame_base)
    right = _cycles_to_intervals(t_right, "right", fps, frame_base)
    if walk_spans:
        left = clip_to_spans(left, walk_spans)
        right = clip_to_spans(right, walk_spans)
    return smooth_foot_intervals(left), smooth_foot_intervals(right)


def fallback_foot_intervals(walk_spans: list[Interval]) -> tuple[list[Interval], list[Interval]]:
    """Generate conservative alternating stance/swing labels during walk spans."""

    left: list[Interval] = []
    right: list[Interval] = []
    for span in clean_intervals(walk_spans):
        left.extend(_fallback_for_side(span, "left", 0))
        right.extend(_fallback_for_side(span, "right", FALLBACK_STRIDE_MS // 2))
    return smooth_foot_intervals(left), smooth_foot_intervals(right)


def smooth_phase_intervals(intervals: list[Interval]) -> list[Interval]:
    cleaned = smooth_intervals(intervals, PHASE_MIN_DURATION_MS, PHASE_GAP_MERGE_MS)
    if not cleaned:
        return []
    return fill_internal_gaps(cleaned, cleaned[0].start_ms, cleaned[-1].end_ms)


def smooth_foot_intervals(intervals: list[Interval]) -> list[Interval]:
    return smooth_intervals(intervals, FOOT_MIN_DURATION_MS, FOOT_GAP_MERGE_MS)


def clip_to_spans(intervals: list[Interval], spans: list[Interval]) -> list[Interval]:
    clipped: list[Interval] = []
    for interval in clean_intervals(intervals):
        for span in clean_intervals(spans):
            start = max(interval.start_ms, span.start_ms)
            end = min(interval.end_ms, span.end_ms)
            if end > start:
                clipped.append(
                    Interval(start, end, interval.label, interval.confidence, interval.source)
                )
    return clipped


def _cycles_to_intervals(
    cycles: object,
    side: str,
    fps: float,
    frame_base: str,
) -> list[Interval]:
    rows = _as_rows(cycles)
    intervals: list[Interval] = []
    for row in rows:
        if len(row) < 5:
            continue
        stance_start = frame_to_ms(row[0], fps, frame_base)
        swing_start = frame_to_ms(row[3], fps, frame_base)
        swing_end = frame_to_ms(row[4], fps, frame_base)
        if swing_start > stance_start:
            intervals.append(
                Interval(stance_start, swing_start, f"{side}_stance", 0.95, "core_gait_cycles")
            )
        if swing_end > swing_start:
            intervals.append(
                Interval(swing_start, swing_end, f"{side}_swing", 0.95, "core_gait_cycles")
            )
    return intervals


def _fallback_for_side(span: Interval, side: str, offset_ms: int) -> list[Interval]:
    intervals: list[Interval] = []
    stance_ms = int(round(FALLBACK_STRIDE_MS * FALLBACK_STANCE_RATIO))
    cursor = span.start_ms + offset_ms
    if cursor > span.start_ms:
        intervals.append(Interval(span.start_ms, min(cursor, span.end_ms), f"{side}_stance", 0.35, "fallback_gait"))
    while cursor < span.end_ms:
        stance_end = min(span.end_ms, cursor + stance_ms)
        swing_end = min(span.end_ms, cursor + FALLBACK_STRIDE_MS)
        if stance_end > cursor:
            intervals.append(Interval(cursor, stance_end, f"{side}_stance", 0.4, "fallback_gait"))
        if swing_end > stance_end:
            intervals.append(Interval(stance_end, swing_end, f"{side}_swing", 0.4, "fallback_gait"))
        cursor += FALLBACK_STRIDE_MS
    return intervals


def phase_intervals_from_crt_events(
    events: list[dict],
    fps: float,
) -> list[Interval]:
    """Convert precomputed chair_rise scorer events to sit/sit-to-stand/stand-to-sit intervals.

    Expects events with types: butt_off, sit_to_stand_start, stand, stand_to_sit_start,
    butt_on.  Only butt_off (rise start) and butt_on (sit return) are required; the
    stand event is used to split sit-to-stand from stand-to-sit.
    """
    if fps <= 0:
        fps = 30.0

    butt_offs = sorted(
        (int(e["frame"]) for e in events if e.get("type") == "butt_off"),
    )
    butt_ons = sorted(
        (int(e["frame"]) for e in events if e.get("type") == "butt_on"),
    )
    stands = sorted(
        (int(e["frame"]) for e in events if e.get("type") == "stand"),
    )

    if not butt_offs or not butt_ons:
        return []

    pairs = list(zip(butt_offs, butt_ons))
    if not pairs:
        return []

    intervals: list[Interval] = []
    prev_end_ms = frame_to_ms(pairs[0][0], fps, "zero")

    for idx, (off_frame, on_frame) in enumerate(pairs):
        rise_start_ms = frame_to_ms(off_frame, fps, "zero")
        sit_end_ms = frame_to_ms(on_frame, fps, "zero")

        # sit gap before this rise (first rep gets initial sit from test start)
        if rise_start_ms > prev_end_ms:
            intervals.append(Interval(prev_end_ms, rise_start_ms, "sit", 0.95, "crt_events"))

        # find the stand frame inside this rep to split sit-to-stand from stand-to-sit
        rep_stands = [f for f in stands if off_frame <= f <= on_frame]
        if rep_stands:
            peak_ms = frame_to_ms(rep_stands[0], fps, "zero")
        else:
            peak_ms = rise_start_ms + (sit_end_ms - rise_start_ms) // 2

        if peak_ms > rise_start_ms:
            intervals.append(
                Interval(rise_start_ms, peak_ms, "sit-to-stand", 0.95, "crt_events")
            )
        if sit_end_ms > peak_ms:
            intervals.append(
                Interval(peak_ms, sit_end_ms, "stand-to-sit", 0.95, "crt_events")
            )
        prev_end_ms = sit_end_ms

    return smooth_phase_intervals(intervals)


def crt_phase_intervals_from_detections(
    boxes: list,
    fps: float,
    num_frames: int | None = None,
) -> list[Interval]:
    """Detect CRT rise cycles from per-frame subject bounding box heights.

    Works entirely from bbox data without pose estimation.  Returns
    sit/sit-to-stand/stand-to-sit intervals at confidence 0.70.
    """
    if fps <= 0:
        fps = 30.0
    sorted_boxes = sorted(boxes, key=lambda b: b.frame)
    if num_frames:
        sorted_boxes = [b for b in sorted_boxes if 0 <= b.frame < num_frames]
    if len(sorted_boxes) < int(fps * 2):
        return []

    frames = [b.frame for b in sorted_boxes]
    raw_heights = [b.height for b in sorted_boxes]
    window = max(3, int(round(fps * CRT_SMOOTH_WINDOW_SEC)))
    smoothed = _ma(raw_heights, window)

    ordered = sorted(smoothed)
    n = len(ordered)
    sit_level = _med(ordered[: max(1, n // 4)])
    stand_level = _med(ordered[max(1, 3 * n // 4) :])
    if stand_level - sit_level < 1.0:
        return []
    threshold = sit_level + CRT_SIT_STAND_THRESHOLD_RATIO * (stand_level - sit_level)

    # Threshold crossings
    up_frames: list[int] = []
    down_frames: list[int] = []
    for i in range(1, len(smoothed)):
        prev, cur = smoothed[i - 1], smoothed[i]
        if prev < threshold <= cur:
            up_frames.append(frames[i])
        elif prev >= threshold > cur:
            down_frames.append(frames[i])

    # Pair upward with the next downward; require minimum rise width
    pairs: list[tuple[int, int]] = []
    di = 0
    for uf in up_frames:
        while di < len(down_frames) and down_frames[di] <= uf:
            di += 1
        if di < len(down_frames):
            df = down_frames[di]
            if df - uf >= CRT_MIN_RISE_FRAMES:
                pairs.append((uf, df))
            di += 1
    if not pairs:
        return []

    frame_to_idx = {f: i for i, f in enumerate(frames)}

    def _frame_idx(frame: int) -> int:
        if frame in frame_to_idx:
            return frame_to_idx[frame]
        return min(range(len(frames)), key=lambda i: abs(frames[i] - frame))

    intervals: list[Interval] = []
    prev_end_ms = frame_to_ms(frames[0], fps, "zero")

    for up_frame, down_frame in pairs:
        rise_ms = frame_to_ms(up_frame, fps, "zero")
        sit_ms = frame_to_ms(down_frame, fps, "zero")

        if rise_ms > prev_end_ms:
            intervals.append(Interval(prev_end_ms, rise_ms, "sit", 0.70, "crt_bbox"))

        # Find peak between crossings
        i_start = _frame_idx(up_frame)
        i_end = _frame_idx(down_frame)
        if i_end > i_start:
            peak_local = max(range(i_start, i_end + 1), key=lambda i: smoothed[i])
            peak_ms = frame_to_ms(frames[peak_local], fps, "zero")
        else:
            peak_ms = rise_ms + (sit_ms - rise_ms) // 2

        peak_ms = max(rise_ms + 1, min(peak_ms, sit_ms - 1))
        intervals.append(Interval(rise_ms, peak_ms, "sit-to-stand", 0.70, "crt_bbox"))
        intervals.append(Interval(peak_ms, sit_ms, "stand-to-sit", 0.70, "crt_bbox"))
        prev_end_ms = sit_ms

    last_frame_ms = frame_to_ms(frames[-1], fps, "zero")
    if last_frame_ms > prev_end_ms:
        intervals.append(Interval(prev_end_ms, last_frame_ms, "sit", 0.70, "crt_bbox"))

    return smooth_phase_intervals(intervals)


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
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _as_rows(value: object) -> list[list[int]]:
    if value is None:
        return []
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.astype(int).tolist()
    except Exception:
        pass
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return []
    rows: list[list[int]] = []
    for row in value:
        if isinstance(row, tuple):
            row = list(row)
        if isinstance(row, list):
            try:
                rows.append([int(round(float(v))) for v in row])
            except (TypeError, ValueError):
                continue
    return rows
