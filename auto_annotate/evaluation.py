"""Ground-truth feedback utilities for FrailScreen auto-annotations."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pipeline_adapter import extract_annotations

EVALUATED_TIERS = ("phase", "occlusion", "left_foot", "right_foot", "test")
OCCLUSION_SCORE_HISTORY = "occlusion_score_history.jsonl"
BALANCE_IN_POSITION_ALIASES = {
    "full-tandem",
    "semi-tandem",
    "side-by-side",
    "balance",
    "in_pos",
}


def evaluate_ground_truth(
    ground_truth_root: str | Path = "ground_truth",
    input_root: str | Path = "input",
) -> dict[str, Any]:
    """Regenerate auto annotations and compare them against ground-truth EAFs."""

    gt_root = Path(ground_truth_root)
    in_root = Path(input_root)
    files: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for gt_path in sorted(gt_root.glob("*.eaf")):
        input_folder = find_input_for_ground_truth(gt_path, in_root)
        if input_folder is None:
            skipped.append({"ground_truth": str(gt_path), "reason": "matching input folder not found"})
            continue
        gt = read_eaf_annotations(gt_path)
        try:
            auto = extract_annotations(str(input_folder))
        except Exception as exc:
            skipped.append({"ground_truth": str(gt_path), "reason": f"auto annotation failed: {exc}"})
            continue
        files.append(_compare_file(gt_path, input_folder, gt, auto))

    tier_totals: dict[str, dict[str, float]] = {
        tier: {"matched_ms": 0.0, "denominator_ms": 0.0, "gt_count": 0.0, "auto_count": 0.0}
        for tier in EVALUATED_TIERS
    }
    for item in files:
        for tier, metrics in item["tiers"].items():
            total = tier_totals[tier]
            total["matched_ms"] += metrics["matched_ms"]
            total["denominator_ms"] += metrics["denominator_ms"]
            total["gt_count"] += metrics["gt_count"]
            total["auto_count"] += metrics["auto_count"]

    tiers = {
        tier: {
            **values,
            "overlap_score": (
                values["matched_ms"] / values["denominator_ms"]
                if values["denominator_ms"] > 0
                else None
            ),
        }
        for tier, values in tier_totals.items()
    }
    return {
        "ground_truth_root": str(gt_root),
        "input_root": str(in_root),
        "file_count": len(files),
        "skipped": skipped,
        "tiers": tiers,
        "files": files,
        "recommendations": _recommendations(files),
    }


def print_ground_truth_report(report: dict[str, Any]) -> None:
    print(f"Ground truth files evaluated: {report['file_count']}")
    if report.get("skipped"):
        print(f"Skipped: {len(report['skipped'])}")
        for item in report["skipped"]:
            print(f"  {item['ground_truth']}: {item['reason']}")
    print("\nTier overlap:")
    for tier, metrics in report["tiers"].items():
        score = metrics["overlap_score"]
        score_text = "n/a" if score is None else f"{score:.3f}"
        print(
            f"  {tier}: score={score_text} "
            f"gt={int(metrics['gt_count'])} auto={int(metrics['auto_count'])}"
        )
    print("\nLargest misses:")
    misses: list[tuple[float, str]] = []
    for item in report["files"]:
        for tier, metrics in item["tiers"].items():
            score = metrics["overlap_score"]
            if score is not None:
                misses.append((score, f"{Path(item['ground_truth']).name} {tier}: {score:.3f}"))
    for _score, text in sorted(misses)[:10]:
        print(f"  {text}")
    if report.get("recommendations"):
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")


def track_occlusion_score(
    report: dict[str, Any],
    history_path: str | Path | None = None,
) -> dict[str, Any]:
    """Append the current occlusion score to a JSONL history file."""

    path = Path(history_path) if history_path is not None else Path(report["ground_truth_root"]) / OCCLUSION_SCORE_HISTORY
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = _last_score_history_entry(path)
    entry = _occlusion_score_entry(report, previous)
    entry["history_path"] = str(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def print_occlusion_score_tracking(entry: dict[str, Any]) -> None:
    score = entry.get("score")
    score_text = "n/a" if score is None else f"{float(score):.3f}"
    delta = entry.get("delta")
    if delta is None:
        delta_text = "new baseline"
    else:
        delta_text = f"{float(delta):+.3f} vs previous"
    print(f"\nOcclusion score tracked: {score_text} ({delta_text})")
    print(f"  history: {entry['history_path']}")


def _occlusion_score_entry(
    report: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = dict((report.get("tiers") or {}).get("occlusion") or {})
    score = metrics.get("overlap_score")
    previous_score = previous.get("score") if previous else None
    delta = (
        float(score) - float(previous_score)
        if score is not None and previous_score is not None
        else None
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "score": score,
        "previous_score": previous_score,
        "delta": delta,
        "file_count": int(report.get("file_count") or 0),
        "ground_truth_root": str(report.get("ground_truth_root") or ""),
        "input_root": str(report.get("input_root") or ""),
        "history_path": "",
        "metrics": {
            "matched_ms": metrics.get("matched_ms"),
            "denominator_ms": metrics.get("denominator_ms"),
            "gt_count": metrics.get("gt_count"),
            "auto_count": metrics.get("auto_count"),
            "gt_duration_ms": metrics.get("gt_duration_ms"),
            "auto_duration_ms": metrics.get("auto_duration_ms"),
        },
        "largest_occlusion_misses": _largest_tier_misses(report, "occlusion", limit=5),
    }


def _last_score_history_entry(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _largest_tier_misses(report: dict[str, Any], tier: str, limit: int) -> list[dict[str, Any]]:
    misses: list[dict[str, Any]] = []
    for item in report.get("files") or []:
        metrics = (item.get("tiers") or {}).get(tier)
        if not metrics or metrics.get("overlap_score") is None:
            continue
        misses.append({
            "ground_truth": Path(str(item.get("ground_truth") or "")).name,
            "input_folder": Path(str(item.get("input_folder") or "")).name,
            "score": metrics["overlap_score"],
            "gt_count": metrics.get("gt_count"),
            "auto_count": metrics.get("auto_count"),
        })
    return sorted(misses, key=lambda item: float(item["score"]))[:limit]


def read_eaf_annotations(eaf_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    path = Path(eaf_path)
    try:
        tree = ET.parse(path)
    except Exception:
        return {}
    root = tree.getroot()
    time_slots = {
        ts.get("TIME_SLOT_ID"): int(ts.get("TIME_VALUE", 0))
        for ts in root.findall(".//TIME_SLOT")
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for tier in root.findall(".//TIER"):
        tier_id = tier.get("TIER_ID", "")
        intervals: list[dict[str, Any]] = []
        for ann in tier.findall(".//ALIGNABLE_ANNOTATION"):
            start = time_slots.get(ann.get("TIME_SLOT_REF1", ""), 0)
            end = time_slots.get(ann.get("TIME_SLOT_REF2", ""), 0)
            value = ann.find("ANNOTATION_VALUE")
            label = (value.text or "").strip() if value is not None else ""
            if end > start and label:
                intervals.append({"start_ms": start, "end_ms": end, "label": label})
        if intervals:
            result[tier_id] = sorted(intervals, key=lambda item: (item["start_ms"], item["end_ms"]))
    return result


def find_input_for_ground_truth(gt_path: str | Path, input_root: str | Path) -> Path | None:
    target = Path(gt_path).stem
    target = re.sub(r"(_ground_truth|_gt)$", "", target, flags=re.IGNORECASE)
    target_norm = _normalized_name(target)
    candidates = [
        path for path in Path(input_root).rglob("*")
        if path.is_dir() and any(path.glob("rgb_video*.mp4"))
    ]
    for path in candidates:
        if _normalized_name(path.name) == target_norm:
            return path
    best: Path | None = None
    best_score = 0
    target_tokens = set(_tokens(target))
    for path in candidates:
        score = len(target_tokens & set(_tokens(path.name)))
        if score > best_score:
            best = path
            best_score = score
    return best if best_score >= max(1, min(2, len(target_tokens))) else None


def _compare_file(
    gt_path: Path,
    input_folder: Path,
    gt: dict[str, list[dict[str, Any]]],
    auto: dict[str, Any],
) -> dict[str, Any]:
    tiers: dict[str, Any] = {}
    duration_ms = int(auto.get("duration_ms") or 0)
    for tier in EVALUATED_TIERS:
        gt_intervals = _clip_intervals_to_duration(
            _canonical_intervals(tier, gt.get(tier, [])),
            duration_ms,
        )
        auto_intervals = _clip_intervals_to_duration(
            _canonical_intervals(tier, auto.get(tier, [])),
            duration_ms,
        )
        tiers[tier] = _tier_metrics(tier, gt_intervals, auto_intervals)
    return {
        "ground_truth": str(gt_path),
        "input_folder": str(input_folder),
        "auto_test_type": auto.get("test_type"),
        "warnings": list(auto.get("warnings") or []),
        "tiers": tiers,
    }


def _tier_metrics(tier: str, gt: list[dict[str, Any]], auto: list[dict[str, Any]]) -> dict[str, Any]:
    gt_duration = _total_duration(gt)
    auto_duration = _total_duration(auto)
    matched = _matched_overlap(gt, auto)
    denominator = _comparison_denominator(tier, gt, auto, gt_duration, auto_duration)
    start_error = _mean_start_error(gt, auto)
    return {
        "gt_count": len(gt),
        "auto_count": len(auto),
        "gt_duration_ms": gt_duration,
        "auto_duration_ms": auto_duration,
        "matched_ms": matched,
        "denominator_ms": denominator,
        "overlap_score": matched / denominator if denominator > 0 else None,
        "mean_start_error_ms": start_error,
    }


def _canonical_intervals(tier: str, intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in intervals:
        label = _canonical_label(tier, str(item.get("label", "")))
        start = int(item.get("start_ms", 0))
        end = int(item.get("end_ms", 0))
        if end > start and label:
            out.append({"start_ms": start, "end_ms": end, "label": label})
    return sorted(out, key=lambda item: (item["start_ms"], item["end_ms"], item["label"]))


def _clip_intervals_to_duration(
    intervals: list[dict[str, Any]],
    duration_ms: int,
) -> list[dict[str, Any]]:
    if duration_ms <= 0:
        return intervals
    clipped: list[dict[str, Any]] = []
    for item in intervals:
        start = max(0, min(int(item["start_ms"]), duration_ms))
        end = max(0, min(int(item["end_ms"]), duration_ms))
        if end > start:
            clipped.append({**item, "start_ms": start, "end_ms": end})
    return clipped


def _canonical_label(tier: str, label: str) -> str:
    text = label.strip()
    if tier == "phase" and text in BALANCE_IN_POSITION_ALIASES:
        return "in_pos"
    if tier == "occlusion" and text in {
        "occlusion",
        "occluded_by_person",
        "occlusion_inferred",
        "outside_frame",
    }:
        return "occlusion"
    return text


def _comparison_denominator(
    tier: str,
    gt: list[dict[str, Any]],
    auto: list[dict[str, Any]],
    gt_duration: int,
    auto_duration: int,
) -> int:
    if _is_sparse_balance_phase_ground_truth(tier, gt, auto):
        return gt_duration
    return max(gt_duration, auto_duration)


def _is_sparse_balance_phase_ground_truth(
    tier: str,
    gt: list[dict[str, Any]],
    auto: list[dict[str, Any]],
) -> bool:
    if tier != "phase" or not gt or not auto:
        return False
    gt_labels = {item["label"] for item in gt}
    auto_labels = {item["label"] for item in auto}
    return gt_labels == {"in_pos"} and "out_of_pos" in auto_labels


def _matched_overlap(gt: list[dict[str, Any]], auto: list[dict[str, Any]]) -> int:
    matched = 0
    for expected in gt:
        for actual in auto:
            if expected["label"] != actual["label"]:
                continue
            start = max(expected["start_ms"], actual["start_ms"])
            end = min(expected["end_ms"], actual["end_ms"])
            if end > start:
                matched += end - start
    return matched


def _total_duration(intervals: list[dict[str, Any]]) -> int:
    return sum(max(0, int(item["end_ms"]) - int(item["start_ms"])) for item in intervals)


def _mean_start_error(gt: list[dict[str, Any]], auto: list[dict[str, Any]]) -> float | None:
    errors: list[int] = []
    used: set[int] = set()
    for expected in gt:
        best_idx = None
        best_error = None
        for idx, actual in enumerate(auto):
            if idx in used or actual["label"] != expected["label"]:
                continue
            error = abs(int(actual["start_ms"]) - int(expected["start_ms"]))
            if best_error is None or error < best_error:
                best_idx = idx
                best_error = error
        if best_idx is not None and best_error is not None:
            used.add(best_idx)
            errors.append(best_error)
    return (sum(errors) / len(errors)) if errors else None


def _recommendations(files: list[dict[str, Any]]) -> list[str]:
    recs: list[str] = []
    balance_errors = [
        metrics["mean_start_error_ms"]
        for item in files
        for tier, metrics in item["tiers"].items()
        if tier == "phase"
        and metrics["mean_start_error_ms"] is not None
        and item.get("auto_test_type") in {"full_tandem", "semi_tandem", "side_by_side", "balance"}
    ]
    if balance_errors:
        mean_error = sum(balance_errors) / len(balance_errors)
        recs.append(f"balance in-position start mean error: {mean_error:.0f} ms; tune setup-motion threshold if this drifts")
    for tier in ("left_foot", "right_foot"):
        scores = [
            item["tiers"][tier]["overlap_score"]
            for item in files
            if item["tiers"][tier]["overlap_score"] is not None
        ]
        if scores and sum(scores) / len(scores) < 0.70:
            recs.append(f"{tier} overlap is low; prioritize pose/foot-contact backend over fallback stride synthesis")
    return recs


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", value.lower()) if token]


def report_to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
