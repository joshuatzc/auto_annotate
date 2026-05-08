"""CLI for the TUG ELAN auto-annotation pre-labeller."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import FALLBACK_PHASE_RATIOS, FALLBACK_STANCE_RATIO, FALLBACK_STRIDE_MS, TUG_PHASE_SEQUENCE
from .elan_export import write_eaf
from .pipeline_adapter import extract_annotations

DEFAULT_BATCH_OUTPUT_FOLDER = "batch_annotations"
DEFAULT_INPUT_FOLDER = "input"
DEFAULT_OUTPUT_FOLDER = "output"
REVIEW_STATE_FILE = ".review_state.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create one TUG auto-annotation .eaf file.",
        epilog=(
            "Batch mode: python -m auto_annotate.cli batch "
            "INPUT_ROOT [OUTPUT_FOLDER]"
        ),
    )
    parser.add_argument("input_folder", nargs="?", help="Folder containing the TUG input bundle.")
    parser.add_argument("output_eaf", nargs="?", help="Output .eaf path.")
    parser.add_argument(
        "-i",
        "--input-folder",
        "--input_folder",
        dest="input_folder_option",
        help="Folder containing the TUG input bundle.",
    )
    parser.add_argument(
        "-o",
        "--output-eaf",
        "--output_eaf",
        dest="output_eaf_option",
        help="Output .eaf path.",
    )
    return parser


def build_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto_annotate.cli batch",
        description="Create TUG auto-annotation .eaf files for every TUG folder under an input root.",
    )
    parser.add_argument("input_root", nargs="?", help="Folder to scan recursively for TUG folders.")
    parser.add_argument(
        "output_folder",
        nargs="?",
        help=f"Folder for generated .eaf files. Defaults to ./{DEFAULT_BATCH_OUTPUT_FOLDER}.",
    )
    parser.add_argument(
        "-i",
        "--input-root",
        "--input-folder",
        "--input_folder",
        dest="input_root_option",
        help="Folder to scan recursively for TUG folders.",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        "--output-dir",
        "--output_folder",
        "--output_dir",
        dest="output_folder_option",
        help=f"Folder for generated .eaf files. Defaults to ./{DEFAULT_BATCH_OUTPUT_FOLDER}.",
    )
    return parser


def build_calibrate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto_annotate.cli calibrate",
        description=(
            "Read your ELAN corrections and suggest (or apply) improved annotation parameters."
        ),
    )
    parser.add_argument(
        "output_folder",
        nargs="?",
        help=f"Folder containing reviewed .eaf files. Defaults to ./{DEFAULT_OUTPUT_FOLDER}.",
    )
    parser.add_argument(
        "-i",
        "--input-folder",
        "--input_folder",
        dest="input_folder",
        help=f"Input folder root. Defaults to ./{DEFAULT_INPUT_FOLDER}.",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        "--output_folder",
        dest="output_folder_option",
        help=f"Folder containing reviewed .eaf files. Defaults to ./{DEFAULT_OUTPUT_FOLDER}.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the suggested changes to config.py immediately.",
    )
    return parser


def build_review_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto_annotate.cli review",
        description="Step through every .eaf in the output folder, opening each in ELAN.",
    )
    parser.add_argument(
        "output_folder",
        nargs="?",
        help=f"Folder containing .eaf files. Defaults to ./{DEFAULT_OUTPUT_FOLDER}.",
    )
    parser.add_argument(
        "-i",
        "--input-folder",
        "--input_folder",
        dest="input_folder",
        help="Input folder root — used to repair broken video links in .eaf files.",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        "--output_folder",
        dest="output_folder_option",
        help=f"Folder containing .eaf files. Defaults to ./{DEFAULT_OUTPUT_FOLDER}.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear all review progress and start from scratch.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "batch":
        return _run_batch(argv[1:])
    if argv and argv[0] == "review":
        return _run_review(argv[1:])
    if argv and argv[0] == "calibrate":
        return _run_calibrate(argv[1:])
    if not argv:
        return _run_default_batch()
    return _run_single(argv)


def _run_default_batch() -> int:
    input_root = Path(DEFAULT_INPUT_FOLDER)
    output_root = Path(DEFAULT_OUTPUT_FOLDER)
    if not input_root.exists() or not input_root.is_dir():
        print(f"No arguments given and no '{DEFAULT_INPUT_FOLDER}/' folder found.")
        print(f"  Drop TUG/CRT bundles into a folder named '{DEFAULT_INPUT_FOLDER}/' next to this script,")
        print(f"  then run again — or pass arguments directly:")
        print(f"    python3 -m auto_annotate.cli -i INPUT_FOLDER -o OUTPUT.eaf")
        print(f"    python3 -m auto_annotate.cli batch INPUT_ROOT [OUTPUT_FOLDER]")
        return 1
    print(f"Processing ./{DEFAULT_INPUT_FOLDER}/ → ./{DEFAULT_OUTPUT_FOLDER}/")
    annotated, skipped, failures = _annotate_batch(input_root, output_root)
    _print_batch_summary(annotated, skipped, failures)
    if failures or not annotated:
        return 1
    return 0


def _run_review(argv: list[str]) -> int:
    parser = build_review_parser()
    args = parser.parse_args(argv)

    output_folder_arg = args.output_folder_option or args.output_folder or DEFAULT_OUTPUT_FOLDER
    output_root = Path(output_folder_arg)
    input_root = Path(args.input_folder) if args.input_folder else Path(DEFAULT_INPUT_FOLDER)

    if not output_root.exists() or not output_root.is_dir():
        print(f"Output folder not found: {output_root}")
        print("  Run the annotator first to generate .eaf files.")
        return 1

    eaf_files = sorted(output_root.rglob("*.eaf"))
    if not eaf_files:
        print(f"No .eaf files found in {output_root}")
        return 1

    # Repair broken video links
    if input_root.is_dir():
        fixed = sum(1 for eaf in eaf_files if _fix_eaf_video_path(eaf, input_root, output_root))
        if fixed:
            print(f"Repaired video links in {fixed} .eaf file(s).")

    if args.reset:
        state_file = output_root / REVIEW_STATE_FILE
        if state_file.exists():
            state_file.unlink()
        print(f"Review progress cleared — {len(eaf_files)} file(s) marked as pending.")
        state: dict[str, str] = {}
    else:
        state = _load_review_state(output_root)

    # Start from the first pending file
    index = next(
        (i for i, e in enumerate(eaf_files) if _get_status(state, e, output_root) == "pending"),
        0,
    )

    while True:
        eaf = eaf_files[index]
        key = str(eaf.relative_to(output_root))
        _print_review_screen(index, eaf_files, output_root, state)
        ch = _getch()

        if ch in ("\r", "\n", "d"):
            state[key] = "done"
            _save_review_state(state, output_root)
            if index < len(eaf_files) - 1:
                index += 1
            else:
                _print_review_screen(index, eaf_files, output_root, state)
                print("\n  All files reviewed. Press any key to exit.")
                _getch()
                break
        elif ch == "o":
            _open_in_elan(eaf)
        elif ch == "s":
            state[key] = "skipped"
            _save_review_state(state, output_root)
            if index < len(eaf_files) - 1:
                index += 1
        elif ch == "c":
            state[key] = "corrected"
            _save_review_state(state, output_root)
            if index < len(eaf_files) - 1:
                index += 1
            else:
                _print_review_screen(index, eaf_files, output_root, state)
                print("\n  All files reviewed. Press any key to exit.")
                _getch()
                break
        elif ch == "n":
            if index < len(eaf_files) - 1:
                index += 1
        elif ch == "p":
            if index > 0:
                index -= 1
        elif ch == "g":
            _print_review_screen(index, eaf_files, output_root, state)
            print(f"\n  Jump to (1-{len(eaf_files)}): ", end="", flush=True)
            try:
                num_str = ""
                while True:
                    c = _getch()
                    if c in ("\r", "\n"):
                        break
                    if c in ("\x03", "\x04", "q"):
                        num_str = ""
                        break
                    if c == "\x7f" and num_str:
                        num_str = num_str[:-1]
                        print("\b \b", end="", flush=True)
                    elif c.isdigit():
                        num_str += c
                        print(c, end="", flush=True)
                if num_str:
                    num = int(num_str)
                    if 1 <= num <= len(eaf_files):
                        index = num - 1
            except (ValueError, EOFError):
                pass
        elif ch in ("q", "\x03", "\x04"):
            print("\n  Progress saved. Exiting.")
            break

    return 0


def _print_review_screen(
    index: int,
    eaf_files: list[Path],
    output_root: Path,
    state: dict[str, str],
) -> None:
    total = len(eaf_files)
    done_count = sum(1 for e in eaf_files if _get_status(state, e, output_root) == "done")
    corrected_count = sum(1 for e in eaf_files if _get_status(state, e, output_root) == "corrected")
    skipped_count = sum(1 for e in eaf_files if _get_status(state, e, output_root) == "skipped")
    reviewed = done_count + corrected_count
    pending = total - reviewed - skipped_count

    bar_width = 28
    filled = int(bar_width * reviewed / total) if total else 0
    bar = "#" * filled + "." * (bar_width - filled)

    eaf = eaf_files[index]
    rel = str(eaf.relative_to(output_root))
    status = _get_status(state, eaf, output_root)
    status_tag = {"done": "DONE", "corrected": " FIX", "skipped": "SKIP", "pending": "----"}.get(status, "----")

    meta = _read_annotation_metadata(eaf)
    if meta is not None:
        conf = meta.get("phase_confidence")
        conf_str = f"{_confidence_label(conf)} ({conf:.2f})" if conf is not None else "UNKN"
        src_str = _source_label(meta.get("phase_source"))
        warn_count = meta.get("warning_count", 0)
        warn_str = f"  ·  {warn_count} warning(s)" if warn_count else ""
        meta_line = f"  confidence: {conf_str}  ·  {src_str}{warn_str}"
    else:
        meta_line = "  confidence: not available (re-annotate to populate)"

    os.system("clear" if sys.platform != "win32" else "cls")
    print("=" * 62)
    print(f"  REVIEW  {index + 1}/{total}   [{bar}]")
    print(f"  reviewed: {reviewed}   corrected: {corrected_count}   skipped: {skipped_count}   pending: {pending}")
    print("=" * 62)
    print(f"  [{status_tag}]  {rel}")
    print(meta_line)
    print("-" * 62)
    print("  o        Open in ELAN")
    print("  Enter/d  Mark done + next     s  Skip + next")
    print("  c        Mark corrected + next (save in ELAN first)")
    print("  n        Next (keep status)   p  Previous")
    print("  g        Jump to #            q  Quit (auto-saved)")
    print("=" * 62)
    print("  > ", end="", flush=True)


def _get_status(state: dict[str, str], eaf: Path, output_root: Path) -> str:
    return state.get(str(eaf.relative_to(output_root)), "pending")


def _load_review_state(output_root: Path) -> dict[str, str]:
    state_file = output_root / REVIEW_STATE_FILE
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_review_state(state: dict[str, str], output_root: Path) -> None:
    state_file = output_root / REVIEW_STATE_FILE
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _write_annotation_metadata(eaf_path: Path, annotations: dict[str, Any]) -> None:
    """Write a sidecar .json with confidence/source/warning info for the review UI."""
    phases = annotations.get("phase") or []
    confidences = [float(p.get("confidence", 1.0)) for p in phases if isinstance(p, dict)]
    sources = [str(p.get("source", "")) for p in phases if isinstance(p, dict)]
    warnings = [str(w) for w in (annotations.get("warnings") or [])]
    meta = {
        "phase_confidence": min(confidences) if confidences else None,
        "phase_source": max(set(sources), key=sources.count) if sources else None,
        "warning_count": len(warnings),
        "warnings": warnings[:5],
    }
    eaf_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _read_annotation_metadata(eaf_path: Path) -> dict[str, Any] | None:
    meta_path = eaf_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _confidence_label(confidence: float | None) -> str:
    if confidence is None:
        return "UNKN"
    if confidence >= 0.90:
        return "HIGH"
    if confidence >= 0.65:
        return "MED "
    if confidence >= 0.35:
        return "LOW "
    return "NONE"


def _source_label(source: str | None) -> str:
    if not source:
        return "unknown"
    if any(k in source for k in ("core_tug", "crt_events", "core_gait", "precomputed")):
        return "precomputed pipeline"
    if "core_bbox" in source or ("bbox" in source and "fallback" not in source and "motion" not in source):
        return "bbox+face detection"
    if "motion_fallback" in source or "bbox_motion" in source:
        return "motion fallback"
    if "fallback" in source:
        return "fallback (low data)"
    if "insufficient" in source:
        return "no data"
    return source


def _getch() -> str:
    if sys.platform == "win32":
        import msvcrt
        return msvcrt.getwch()  # type: ignore[attr-defined]
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _fix_eaf_video_path(eaf_path: Path, input_root: Path, output_root: Path | None = None) -> bool:
    """Update MEDIA_URL/RELATIVE_MEDIA_URL so ELAN can find the source video.

    Resolution order:
      1. Absolute path in the .eaf already works → nothing to do.
      2. Relative path resolves from the .eaf's location → update absolute URL only.
      3. Find the input bundle folder that produced this .eaf and use whatever
         rgb_video*.mp4 is inside it (robust to video renaming / blurred variants).
      4. Search input_root for the exact original filename as a last resort.

    Returns True if the file was modified.
    """
    import xml.etree.ElementTree as ET
    from urllib.parse import quote, unquote, urlparse

    try:
        tree = ET.parse(eaf_path)
    except Exception:
        return False

    root_el = tree.getroot()
    descriptor = root_el.find(".//MEDIA_DESCRIPTOR")
    if descriptor is None:
        return False

    media_url = descriptor.get("MEDIA_URL", "")

    # 1. Absolute path resolves — re-write the URL with proper encoding (%20 etc.)
    #    even if the path is correct, because ELAN rejects file:// URLs with literal spaces.
    try:
        resolved = Path(unquote(urlparse(media_url).path)).resolve()
        if resolved.exists():
            clean_url = resolved.as_uri()  # always produces properly-encoded URL
            if clean_url == media_url:
                return False  # already correct
            descriptor.set("MEDIA_URL", clean_url)
            ET.indent(tree, space="  ")
            tree.write(str(eaf_path), encoding="UTF-8", xml_declaration=True)
            return True
    except Exception:
        pass

    # 2. Relative path works — just fix the stale absolute URL
    relative_url = descriptor.get("RELATIVE_MEDIA_URL", "")
    if relative_url:
        try:
            candidate = (eaf_path.parent / unquote(relative_url.lstrip("./"))).resolve()
            if candidate.exists():
                descriptor.set("MEDIA_URL", candidate.as_uri())
                ET.indent(tree, space="  ")
                tree.write(str(eaf_path), encoding="UTF-8", xml_declaration=True)
                return True
        except Exception:
            pass

    def _write_video(video_path: Path) -> bool:
        descriptor.set("MEDIA_URL", video_path.as_uri())
        try:
            rel = os.path.relpath(video_path, eaf_path.parent.resolve())
            descriptor.set("RELATIVE_MEDIA_URL", "./" + rel.replace(os.sep, "/"))
        except ValueError:
            descriptor.set("RELATIVE_MEDIA_URL", "./" + video_path.name)
        ET.indent(tree, space="  ")
        tree.write(str(eaf_path), encoding="UTF-8", xml_declaration=True)
        return True

    # 3. Find the matching input bundle and use its rgb_video*.mp4 — works even
    #    when the video was renamed (e.g. blurred UUID name → plain name)
    _output_root = output_root or eaf_path.parent
    input_folder = _find_input_for_eaf(eaf_path, _output_root, input_root)
    if input_folder is not None:
        videos = sorted(input_folder.glob("rgb_video*.mp4"))
        if videos:
            return _write_video(videos[0].resolve())

    # 4. Last resort: search for the exact original filename anywhere in input_root
    try:
        original_name = Path(unquote(urlparse(media_url).path)).name
    except Exception:
        return False
    if original_name:
        matches = list(input_root.rglob(original_name))
        if matches:
            return _write_video(matches[0].resolve())

    return False


def _open_in_elan(eaf_path: Path) -> None:
    if sys.platform == "darwin":
        result = subprocess.run(
            ["open", "-a", "ELAN", str(eaf_path)],
            capture_output=True,
        )
        if result.returncode != 0:
            subprocess.run(["open", str(eaf_path)])
    elif sys.platform == "win32":
        subprocess.run(["start", "", str(eaf_path)], shell=True)
    else:
        subprocess.run(["xdg-open", str(eaf_path)])


def _run_calibrate(argv: list[str]) -> int:
    parser = build_calibrate_parser()
    args = parser.parse_args(argv)

    output_root = Path(args.output_folder_option or args.output_folder or DEFAULT_OUTPUT_FOLDER)
    input_root = Path(args.input_folder or DEFAULT_INPUT_FOLDER)

    if not output_root.is_dir():
        print(f"Output folder not found: {output_root}")
        return 1
    if not input_root.is_dir():
        print(f"Input folder not found: {input_root}")
        return 1

    state = _load_review_state(output_root)
    corrected_keys = [k for k, v in state.items() if v == "corrected"]

    if not corrected_keys:
        print("No corrected files found.")
        print("  During review, press 'c' after saving your changes in ELAN to mark a file as corrected.")
        return 1

    print(f"Analysing {len(corrected_keys)} corrected file(s)...\n")

    phase_ratio_samples: list[list[float]] = []
    stride_samples: list[int] = []
    stance_ratio_samples: list[float] = []
    skipped_files: list[tuple[str, str]] = []
    used = 0

    for key in corrected_keys:
        eaf_path = output_root / key
        if not eaf_path.exists():
            skipped_files.append((key, "file not found"))
            continue

        gt = _read_eaf_annotations(eaf_path)
        gt_phases = gt.get("phase", [])

        input_folder = _find_input_for_eaf(eaf_path, output_root, input_root)
        if input_folder is None:
            skipped_files.append((key, "matching input folder not found"))
            continue

        try:
            auto = extract_annotations(str(input_folder))
        except Exception as exc:
            skipped_files.append((key, f"re-annotation failed: {exc}"))
            continue

        sources = {p.get("source", "") for p in (auto.get("phase") or [])}
        if not any("fallback" in s for s in sources):
            skipped_files.append((key, "high-confidence annotation — corrections noted but no config knobs apply"))
            continue

        used += 1
        is_crt = "crt" in eaf_path.stem.lower()

        if not is_crt:
            ratios = _phases_to_ratios(gt_phases)
            if ratios:
                phase_ratio_samples.append(ratios)

        gt_left = gt.get("left_foot", [])
        gt_right = gt.get("right_foot", [])
        stride_samples.extend(
            _estimate_strides_from_foot_intervals(gt_left) +
            _estimate_strides_from_foot_intervals(gt_right)
        )
        all_foot = gt_left + gt_right
        if all_foot:
            stance_ms = sum(iv["end_ms"] - iv["start_ms"] for iv in all_foot if "stance" in iv.get("label", ""))
            swing_ms = sum(iv["end_ms"] - iv["start_ms"] for iv in all_foot if "swing" in iv.get("label", ""))
            total_foot = stance_ms + swing_ms
            if total_foot > 0:
                stance_ratio_samples.append(stance_ms / total_foot)

    if skipped_files:
        print(f"Skipped {len(skipped_files)} file(s):")
        for key, reason in skipped_files:
            print(f"  {key}: {reason}")
        print()

    if used == 0:
        print("No usable calibration data.")
        print("  Calibration only updates fallback-path annotations (confidence < 0.95).")
        return 1

    if used < 5:
        print(f"Note: only {used} usable sample(s) — more corrections will give more reliable results.\n")

    new_ratios: tuple | None = None
    new_stride: int | None = None
    new_stance: float | None = None

    if phase_ratio_samples:
        n = len(phase_ratio_samples)
        width = len(phase_ratio_samples[0])
        avg = [sum(s[i] for s in phase_ratio_samples) / n for i in range(width)]
        total_avg = sum(avg)
        new_ratios = tuple(round(r / total_avg, 3) for r in avg)
        print(f"FALLBACK_PHASE_RATIOS  ({n} TUG sample(s))")
        print(f"  current:   {FALLBACK_PHASE_RATIOS}")
        print(f"  suggested: {new_ratios}")
        print()

    if stride_samples:
        new_stride = int(round(sum(stride_samples) / len(stride_samples)))
        print(f"FALLBACK_STRIDE_MS  ({len(stride_samples)} stride measurement(s))")
        print(f"  current:   {FALLBACK_STRIDE_MS}")
        print(f"  suggested: {new_stride}")
        print()

    if stance_ratio_samples:
        new_stance = round(sum(stance_ratio_samples) / len(stance_ratio_samples), 3)
        print(f"FALLBACK_STANCE_RATIO  ({len(stance_ratio_samples)} sample(s))")
        print(f"  current:   {FALLBACK_STANCE_RATIO}")
        print(f"  suggested: {new_stance}")
        print()

    if new_ratios is None and new_stride is None and new_stance is None:
        print("No parameter updates could be derived from these corrections.")
        return 0

    if args.apply:
        config_path = Path(__file__).parent / "config.py"
        _apply_calibration_to_config(config_path, new_ratios, new_stride, new_stance)
        print("config.py updated. Re-run the annotator to apply the new settings.")
    else:
        print("Run with --apply to update config.py automatically:")
        print("  python3 -m auto_annotate.cli calibrate --apply")

    return 0


def _read_eaf_annotations(eaf_path: Path) -> dict[str, list[dict]]:
    """Parse an ELAN .eaf and return {tier_id: [{start_ms, end_ms, label}]}."""
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(eaf_path)
    except Exception:
        return {}

    root_el = tree.getroot()
    time_slots = {
        ts.get("TIME_SLOT_ID"): int(ts.get("TIME_VALUE", 0))
        for ts in root_el.findall(".//TIME_SLOT")
    }
    result: dict[str, list[dict]] = {}
    for tier in root_el.findall(".//TIER"):
        tier_id = tier.get("TIER_ID", "")
        intervals = []
        for ann in tier.findall(".//ALIGNABLE_ANNOTATION"):
            start = time_slots.get(ann.get("TIME_SLOT_REF1", ""), 0)
            end = time_slots.get(ann.get("TIME_SLOT_REF2", ""), 0)
            val = ann.find("ANNOTATION_VALUE")
            label = (val.text or "") if val is not None else ""
            if end > start:
                intervals.append({"start_ms": start, "end_ms": end, "label": label})
        if intervals:
            result[tier_id] = sorted(intervals, key=lambda x: x["start_ms"])
    return result


def _find_input_for_eaf(eaf_path: Path, output_root: Path, input_root: Path) -> Path | None:
    """Find the input bundle folder that produced a given .eaf output file."""
    try:
        rel = eaf_path.relative_to(output_root)
    except ValueError:
        return None

    parent_parts = rel.parts[:-1]
    stem = rel.stem
    if stem.endswith("_auto"):
        stem = stem[:-5]

    # Walk the same sub-directory structure inside input_root
    search_root = input_root
    for part in parent_parts:
        candidate = search_root / part
        if candidate.is_dir():
            search_root = candidate

    # Exact match: folder whose safe_stem equals the eaf stem
    for folder in [search_root] + sorted(search_root.rglob("*")):
        if isinstance(folder, Path) and folder.is_dir():
            if _safe_stem(folder.name) == stem and any(folder.glob("rgb_video*.mp4")):
                return folder

    # Fuzzy fallback: match on all meaningful keywords from the stem
    keywords = [k.lower() for k in stem.split("_") if k]
    best: Path | None = None
    best_score = 0
    for folder in input_root.rglob("*"):
        if not (folder.is_dir() and any(folder.glob("rgb_video*.mp4"))):
            continue
        name_norm = folder.name.lower().replace(" ", "_").replace("-", "_")
        score = sum(1 for kw in keywords if kw in name_norm)
        if score > best_score:
            best_score = score
            best = folder
    return best if best_score >= min(2, len(keywords)) else None


def _phases_to_ratios(phases: list[dict]) -> list[float] | None:
    """Map ground truth phases to the 7-phase TUG sequence and return duration ratios."""
    expected = list(TUG_PHASE_SEQUENCE)
    matched: list[dict] = []
    ei = 0
    for phase in phases:
        if ei < len(expected) and phase.get("label") == expected[ei]:
            matched.append(phase)
            ei += 1
    if len(matched) != 7:
        return None
    total_ms = sum(p["end_ms"] - p["start_ms"] for p in matched)
    if total_ms <= 0:
        return None
    return [(p["end_ms"] - p["start_ms"]) / total_ms for p in matched]


def _estimate_strides_from_foot_intervals(intervals: list[dict]) -> list[int]:
    """Estimate stride durations (ms) from consecutive stance start times."""
    stances = sorted(
        [iv for iv in intervals if iv.get("label", "").endswith("_stance")],
        key=lambda x: x["start_ms"],
    )
    strides = []
    for i in range(1, len(stances)):
        stride = stances[i]["start_ms"] - stances[i - 1]["start_ms"]
        if 300 <= stride <= 3000:
            strides.append(stride)
    return strides


def _apply_calibration_to_config(
    config_path: Path,
    new_ratios: tuple | None,
    new_stride: int | None,
    new_stance: float | None,
) -> None:
    import re

    text = config_path.read_text(encoding="utf-8")

    if new_ratios is not None:
        ratios_str = "(" + ", ".join(f"{r:.3f}" for r in new_ratios) + ")"
        text = re.sub(
            r"FALLBACK_PHASE_RATIOS\s*=\s*\([^)]+\)",
            f"FALLBACK_PHASE_RATIOS = {ratios_str}",
            text,
        )

    if new_stride is not None:
        text = re.sub(
            r"FALLBACK_STRIDE_MS\s*=\s*[\d_]+",
            f"FALLBACK_STRIDE_MS = {new_stride}",
            text,
        )

    if new_stance is not None:
        text = re.sub(
            r"FALLBACK_STANCE_RATIO\s*=\s*[\d.]+",
            f"FALLBACK_STANCE_RATIO = {new_stance:.3f}",
            text,
        )

    config_path.write_text(text, encoding="utf-8")


def _run_single(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_folder_arg = args.input_folder_option or args.input_folder
    output_eaf_arg = args.output_eaf_option or args.output_eaf
    if input_folder_arg is None:
        parser.error("input folder is required: pass INPUT_FOLDER or -i/--input-folder")
    if output_eaf_arg is None:
        parser.error("output .eaf path is required: pass OUTPUT_EAF or -o/--output-eaf")

    input_folder = Path(input_folder_arg)
    output_eaf = Path(output_eaf_arg)
    if not input_folder.exists() or not input_folder.is_dir():
        parser.error(f"--input_folder is not a directory: {input_folder}")
    if output_eaf.suffix.lower() != ".eaf":
        parser.error("--output_eaf must end with .eaf")

    annotations = extract_annotations(str(input_folder))
    video_path = annotations.get("video_path")
    if not video_path:
        parser.error("input bundle does not contain rgb_video_*.mp4")

    write_eaf(str(video_path), annotations, str(output_eaf))
    _write_annotation_metadata(output_eaf, annotations)
    _print_summary(output_eaf, annotations)
    return 0


def _run_batch(argv: list[str]) -> int:
    parser = build_batch_parser()
    args = parser.parse_args(argv)

    input_root_arg = args.input_root_option or args.input_root
    output_folder_arg = (
        args.output_folder_option
        or args.output_folder
        or DEFAULT_BATCH_OUTPUT_FOLDER
    )
    if input_root_arg is None:
        parser.error("input root is required: pass INPUT_ROOT or -i/--input-root")

    input_root = Path(input_root_arg)
    output_root = Path(output_folder_arg)
    if not input_root.exists() or not input_root.is_dir():
        parser.error(f"--input_root is not a directory: {input_root}")

    annotated, skipped, failures = _annotate_batch(input_root, output_root)
    _print_batch_summary(annotated, skipped, failures)
    if failures or not annotated:
        return 1
    return 0


def _annotate_batch(
    input_root: Path,
    output_root: Path,
) -> tuple[list[Path], list[Path], list[tuple[Path, str]]]:
    annotated: list[Path] = []
    skipped: list[Path] = []
    failures: list[tuple[Path, str]] = []
    used_outputs: set[Path] = set()

    for tug_folder in _find_tug_folders(input_root):
        if not _has_source_video(tug_folder):
            skipped.append(tug_folder)
            continue

        output_eaf = _unique_output_path(
            _batch_output_path(input_root, output_root, tug_folder),
            used_outputs,
        )
        try:
            annotations = extract_annotations(str(tug_folder))
            video_path = annotations.get("video_path")
            if not video_path:
                skipped.append(tug_folder)
                continue
            write_eaf(str(video_path), annotations, str(output_eaf))
            _write_annotation_metadata(output_eaf, annotations)
            annotated.append(output_eaf)
        except Exception as exc:
            failures.append((tug_folder, str(exc)))

    return annotated, skipped, failures


def _find_tug_folders(input_root: Path) -> list[Path]:
    candidates = [input_root] + [path for path in input_root.rglob("*") if path.is_dir()]
    return sorted(
        (path for path in candidates if _is_test_folder(path, input_root)),
        key=lambda path: str(path).lower(),
    )


def _is_test_folder(path: Path, root: Path) -> bool:
    """Return True if this folder is identifiable as a TUG or CRT test folder.

    Checks the folder name and its immediate parent (up to the scan root) so that
    structures like root/tug/P45 are picked up even when the leaf name is neutral.
    """
    keywords = ("tug", "crt")
    if any(kw in path.name.lower() for kw in keywords):
        return True
    if path != root and any(kw in path.parent.name.lower() for kw in keywords):
        return True
    return False


def _has_source_video(folder: Path) -> bool:
    return any(folder.glob("rgb_video*.mp4"))


def _batch_output_path(input_root: Path, output_root: Path, tug_folder: Path) -> Path:
    try:
        relative = tug_folder.relative_to(input_root)
    except ValueError:
        relative = Path(tug_folder.name)
    parent = relative.parent if str(relative.parent) != "." else Path()
    return output_root / parent / f"{_safe_stem(tug_folder.name)}_auto.eaf"


def _safe_stem(name: str) -> str:
    chars: list[str] = []
    last_was_separator = False
    for char in name.strip():
        if char.isascii() and char.isalnum():
            chars.append(char)
            last_was_separator = False
        elif not last_was_separator:
            chars.append("_")
            last_was_separator = True
    stem = "".join(chars).strip("_")
    return stem or "TUG"


def _unique_output_path(path: Path, used_outputs: set[Path]) -> Path:
    candidate = path
    index = 2
    while candidate in used_outputs:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        index += 1
    used_outputs.add(candidate)
    return candidate


def _print_summary(output_eaf: Path, annotations: dict[str, Any]) -> None:
    test = annotations.get("test") or []
    phases = annotations.get("phase") or []
    left = annotations.get("left_foot") or []
    right = annotations.get("right_foot") or []
    print(f"Wrote {output_eaf}")
    if test:
        print(f"Test interval: {test[0]['start_ms']} ms - {test[0]['end_ms']} ms")
    else:
        print("Test interval: not detected")
    print(f"Phase intervals: {len(phases)}")
    if left or right:
        print(f"Left foot intervals: {len(left)}")
        print(f"Right foot intervals: {len(right)}")
    warnings = annotations.get("warnings") or []
    if warnings:
        print("Warnings: " + "; ".join(str(w) for w in warnings[:3]))


def _print_batch_summary(
    annotated: list[Path],
    skipped: list[Path],
    failures: list[tuple[Path, str]],
) -> None:
    if skipped:
        print(f"Skipped TUG folders without rgb_video_*.mp4: {len(skipped)}")
    if failures:
        print(f"Failed TUG folders: {len(failures)}")
        for folder, reason in failures[:5]:
            print(f"  {folder}: {reason}")
    print("Annotated files:")
    if annotated:
        for output_eaf in annotated:
            print(f"  {output_eaf}")
    else:
        print("  none")


if __name__ == "__main__":
    raise SystemExit(main())
