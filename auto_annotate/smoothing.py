"""Interval smoothing helpers."""

from __future__ import annotations

from .config import UNKNOWN_LABEL
from .types import Interval


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
