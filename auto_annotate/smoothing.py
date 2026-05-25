"""Interval smoothing helpers."""

from __future__ import annotations

from . import config as config_values
from .config import UNKNOWN_LABEL
from .types import FootAnnotation, GaitEvent, Interval, PhaseAnnotation


def clean_intervals(intervals: list[Interval]) -> list[Interval]:
    """Sort intervals and discard non-positive ranges."""

    return sorted(
        (iv for iv in intervals if iv.end_ms > iv.start_ms and iv.label),
        key=lambda iv: (iv.start_ms, iv.end_ms, iv.label),
    )


def merge_short_gaps(intervals: list[Interval], max_gap_ms: int) -> list[Interval]:
    """Merge adjacent intervals with the same label across a short gap."""

    merged: list[Interval] = []
    for iv in clean_intervals(intervals):
        if (
            merged
            and merged[-1].label == iv.label
            and iv.start_ms - merged[-1].end_ms <= max_gap_ms
        ):
            prev = merged[-1]
            merged[-1] = Interval(
                prev.start_ms,
                max(prev.end_ms, iv.end_ms),
                prev.label,
                min(prev.confidence, iv.confidence),
                prev.source or iv.source,
            )
        else:
            merged.append(iv)
    return merged


def suppress_short_flicker(
    intervals: list[Interval],
    min_duration_ms: int,
    unknown_label: str = UNKNOWN_LABEL,
) -> list[Interval]:
    """Replace isolated very short labels with neighbours or `unknown`.

    This is intentionally conservative. Foot swing segments can be short, so callers
    should pass a small threshold for foot tiers.
    """

    cleaned = clean_intervals(intervals)
    out: list[Interval] = []
    idx = 0
    while idx < len(cleaned):
        iv = cleaned[idx]
        if iv.duration_ms >= min_duration_ms or iv.label == unknown_label:
            out.append(iv)
            idx += 1
            continue

        prev_iv = out[-1] if out else None
        next_iv = cleaned[idx + 1] if idx + 1 < len(cleaned) else None
        if prev_iv and next_iv and prev_iv.label == next_iv.label:
            out[-1] = Interval(
                prev_iv.start_ms,
                next_iv.end_ms,
                prev_iv.label,
                min(prev_iv.confidence, iv.confidence, next_iv.confidence),
                prev_iv.source or next_iv.source,
            )
            idx += 2
            continue

        out.append(Interval(iv.start_ms, iv.end_ms, unknown_label, 0.0, iv.source))
        idx += 1
    return clean_intervals(out)


def fill_internal_gaps(
    intervals: list[Interval],
    start_ms: int,
    end_ms: int,
    unknown_label: str = UNKNOWN_LABEL,
) -> list[Interval]:
    """Fill gaps inside a required tier span with `unknown` intervals."""

    if end_ms <= start_ms:
        return []
    out: list[Interval] = []
    cursor = start_ms
    for iv in clean_intervals(intervals):
        clipped_start = max(start_ms, iv.start_ms)
        clipped_end = min(end_ms, iv.end_ms)
        if clipped_end <= clipped_start:
            continue
        if clipped_start > cursor:
            out.append(Interval(cursor, clipped_start, unknown_label, 0.0, "gap"))
        out.append(
            Interval(clipped_start, clipped_end, iv.label, iv.confidence, iv.source)
        )
        cursor = max(cursor, clipped_end)
    if cursor < end_ms:
        out.append(Interval(cursor, end_ms, unknown_label, 0.0, "gap"))
    return out


def smooth_intervals(
    intervals: list[Interval],
    min_duration_ms: int,
    merge_gap_ms: int,
) -> list[Interval]:
    """Apply the standard flicker and short-gap cleanup pass."""

    return merge_short_gaps(
        suppress_short_flicker(intervals, min_duration_ms),
        merge_gap_ms,
    )


def smooth_phase_annotations(
    phases: list[PhaseAnnotation],
    gait_events: list[GaitEvent] | None = None,
) -> list[PhaseAnnotation]:
    """Conservatively remove TUG phase jitter without inventing transitions."""

    cleaned = [
        phase
        for phase in sorted(phases, key=lambda item: (item.start_ms, item.end_ms))
        if phase.end_ms > phase.start_ms
    ]
    if not cleaned:
        return []
    cleaned = _absorb_short_phases(cleaned, int(config_values.MIN_PHASE_DURATION_MS))
    cleaned = _merge_walk_micro_turns(cleaned, int(config_values.MIN_WALK_DURATION_MS))
    if gait_events:
        cleaned = _snap_phase_boundaries(cleaned, gait_events, max_delta_ms=50)
    return _merge_adjacent_phase_annotations(cleaned)


def smooth_foot_event_times(events: list[GaitEvent]) -> list[GaitEvent]:
    """Median-smooth repeated foot event timings in a small local window."""

    if len(events) < 3:
        return sorted(events, key=lambda event: event.time_ms)
    window_ms = int(config_values.SMOOTHING_WINDOW_MS)
    smoothed: list[GaitEvent] = []
    ordered = sorted(events, key=lambda event: event.time_ms)
    for idx, event in enumerate(ordered):
        neighbours = [
            other.time_ms
            for other in ordered[max(0, idx - 1): min(len(ordered), idx + 2)]
            if other.event_type == event.event_type and abs(other.time_ms - event.time_ms) <= window_ms
        ]
        if neighbours:
            neighbours = sorted(neighbours)
            time_ms = neighbours[len(neighbours) // 2]
        else:
            time_ms = event.time_ms
        smoothed.append(GaitEvent(time_ms, event.side, event.event_type))
    return sorted(smoothed, key=lambda event: (event.time_ms, event.event_type))


def smooth_foot_annotations(annotations: list[FootAnnotation]) -> list[FootAnnotation]:
    """Merge adjacent same-label foot annotations and drop zero-length intervals."""

    cleaned = [
        annotation
        for annotation in sorted(annotations, key=lambda item: (item.start_ms, item.end_ms))
        if annotation.end_ms > annotation.start_ms
    ]
    if not cleaned:
        return []
    merged: list[FootAnnotation] = []
    for annotation in cleaned:
        if (
            merged
            and merged[-1].label == annotation.label
            and merged[-1].side == annotation.side
            and annotation.start_ms - merged[-1].end_ms <= 1
        ):
            prev = merged[-1]
            merged[-1] = FootAnnotation(prev.start_ms, annotation.end_ms, prev.label, prev.side)
        else:
            merged.append(annotation)
    return merged


def _absorb_short_phases(
    phases: list[PhaseAnnotation],
    min_duration_ms: int,
) -> list[PhaseAnnotation]:
    out: list[PhaseAnnotation] = []
    idx = 0
    while idx < len(phases):
        phase = phases[idx]
        if phase.end_ms - phase.start_ms >= min_duration_ms:
            out.append(phase)
            idx += 1
            continue
        prev = out[-1] if out else None
        nxt = phases[idx + 1] if idx + 1 < len(phases) else None
        if prev and nxt and prev.label == nxt.label:
            out[-1] = PhaseAnnotation(prev.start_ms, nxt.end_ms, prev.label)
            idx += 2
        elif prev:
            out[-1] = PhaseAnnotation(prev.start_ms, phase.end_ms, prev.label)
            idx += 1
        elif nxt:
            out.append(PhaseAnnotation(phase.start_ms, nxt.end_ms, nxt.label))
            idx += 2
        else:
            out.append(PhaseAnnotation(phase.start_ms, phase.end_ms, UNKNOWN_LABEL))
            idx += 1
    return out


def _merge_walk_micro_turns(
    phases: list[PhaseAnnotation],
    min_walk_duration_ms: int,
) -> list[PhaseAnnotation]:
    out: list[PhaseAnnotation] = []
    idx = 0
    while idx < len(phases):
        if (
            idx + 2 < len(phases)
            and phases[idx].label == "walk"
            and phases[idx + 1].label == "turn"
            and phases[idx + 2].label == "walk"
            and phases[idx + 1].end_ms - phases[idx + 1].start_ms < min_walk_duration_ms
            and (
                phases[idx].end_ms - phases[idx].start_ms < min_walk_duration_ms
                or phases[idx + 2].end_ms - phases[idx + 2].start_ms < min_walk_duration_ms
            )
        ):
            out.append(PhaseAnnotation(phases[idx].start_ms, phases[idx + 2].end_ms, "walk"))
            idx += 3
            continue
        out.append(phases[idx])
        idx += 1
    return out


def _snap_phase_boundaries(
    phases: list[PhaseAnnotation],
    gait_events: list[GaitEvent],
    max_delta_ms: int,
) -> list[PhaseAnnotation]:
    event_times = sorted({event.time_ms for event in gait_events})
    if not event_times or len(phases) < 2:
        return phases
    boundaries = [phases[0].start_ms]
    for phase in phases[:-1]:
        boundary = phase.end_ms
        nearest = min(event_times, key=lambda value: abs(value - boundary))
        boundaries.append(nearest if abs(nearest - boundary) <= max_delta_ms else boundary)
    boundaries.append(phases[-1].end_ms)
    snapped: list[PhaseAnnotation] = []
    for idx, phase in enumerate(phases):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        if end > start:
            snapped.append(PhaseAnnotation(start, end, phase.label))
    return snapped


def _merge_adjacent_phase_annotations(phases: list[PhaseAnnotation]) -> list[PhaseAnnotation]:
    merged: list[PhaseAnnotation] = []
    for phase in phases:
        if merged and merged[-1].label == phase.label and phase.start_ms <= merged[-1].end_ms + 1:
            prev = merged[-1]
            merged[-1] = PhaseAnnotation(prev.start_ms, max(prev.end_ms, phase.end_ms), prev.label)
        else:
            merged.append(phase)
    return merged
