# Coding Agent Handoff

Last updated: 2026-05-25.

## Project Intent

This folder is the active coding project. The outer `Annotation/` directory is a data/workspace container; the git repo and Python package live in `auto_annotate_dist/`.

The goal is to reduce manual FrailScreen ELAN annotation work. The tool should scan FrailScreen test bundles, generate reviewable `.eaf` files, preserve links to source videos, expose confidence/warning metadata, and support a feedback loop where human corrections and ground truth improve future annotations.

The intended workflow is:

1. Put test bundles under `input/`.
2. Run auto annotation into `output/` or a chosen batch output folder.
3. Review generated `.eaf` files in ELAN.
4. Mark corrected files and compare against `ground_truth/`.
5. Use evaluation, calibration, and workbook tracking to identify what needs more human review.

Accuracy matters, but the product should degrade gracefully. If high-quality pipeline outputs are missing, generate a conservative fallback annotation with warnings rather than failing silently.

## Claude Context

The only `.claude` file found is `.claude/settings.local.json`. It is a local permission allowlist for Claude commands and reads; it does not contain project direction or coding standards. Important allowed patterns there include `python3 *`, `conda run *`, `conda env *`, git init/add/config/commit/remote/push, GitHub checks, web search, and reads under `/Users/joshuateh/Professional/**`.

Do not infer product requirements from `.claude/settings.local.json`; use this handoff, the code, tests, and user direction.

## Run Commands

Run commands from:

```bash
cd "/Users/joshuateh/Professional/Carecam/Annotation/auto_annotate_dist"
```

Use the conda environment when possible:

```bash
conda run -n auto_anno python -m auto_annotate.cli
```

Common commands:

```bash
# Annotate everything in input/ into output/
conda run -n auto_anno python -m auto_annotate.cli

# Annotate one selected bundle
conda run -n auto_anno python -m auto_annotate.cli -i "input/multiple ra TUG" -o "output/multiple_ra_TUG_auto.eaf"

# Batch custom input/output roots
conda run -n auto_anno python -m auto_annotate.cli batch INPUT_ROOT OUTPUT_ROOT

# Review generated files in ELAN
conda run -n auto_anno python -m auto_annotate.cli review -i input

# Compare current logic with hand annotations
conda run -n auto_anno python -m auto_annotate.cli evaluate-ground-truth ground_truth -i input

# Track occlusion score history
conda run -n auto_anno python -m auto_annotate.cli evaluate-ground-truth ground_truth -i input --track-occlusion-score

# Workbook flywheel
conda run -n auto_anno python -m auto_annotate.cli workbook-status --excel visfrailty_screening_v7.xlsx
conda run -n auto_anno python -m auto_annotate.cli workbook-scan --excel visfrailty_screening_v7.xlsx
```

If `ModuleNotFoundError: No module named 'auto_annotate'` appears, the command is being run from the wrong directory or with the wrong Python. Use the `cd` and `conda run` form above.

## Data Contract

Each test bundle is a directory with at least:

```text
rgb_video*.mp4
```

Helpful optional inputs:

```text
bounding_box_data*.json
bounding_box_face_data*.json
pose3d.json
person_masks.json
sppb_results.json
out2.pkl
out2.csv
biomarker.json plus gait boundary JSON files for strict 3DGait TUG mode
```

Supported test names/types include TUG, CRT, GS1/GS2, SBS, ST, and FT. Balance tests use `in_pos` and `out_of_pos` phase labels and generally do not emit foot tiers.

Generated outputs include:

```text
*_auto.eaf
*_auto.json
*_auto.pfsx
run_log.csv for strict 3DGait TUG mode
```

Sidecars such as `pose3d.json` and `person_masks.json` are caches tied to video signatures and backend config hashes. Reuse valid caches; regenerate only when stale or explicitly required by logic.

## Architecture

Main package: `auto_annotate/`.

- `cli.py`: command routing, batch/single annotation, ELAN review UI, calibration, ground-truth evaluation, strict 3DGait TUG mode, and workbook commands.
- `pipeline_adapter.py`: core extraction path. It finds video and sidecars, loads/generates detections, locks onto the patient subject, classifies test type, computes occlusion, invokes pose/mask providers, and returns annotation dictionaries.
- `phase_rules.py`: time interval construction for TUG/CRT/gait/balance, fallback phases, foot-state intervals, smoothing, and strict `AnnotationBundle` construction.
- `elan_export.py`: `.eaf` writer and strict EAF validation. Preserve valid XML, tier order, media URLs, and time slot ordering.
- `subject_selection.py`: detection parsing, tracking, patient subject selection, and IoU helpers.
- `pose3d.py`: local normalized pose cache/provider and pose-derived event detectors. Prefer local/cache-first behavior.
- `person_masks.py`: optional segmentation mask cache/provider for better person occlusion detection.
- `evaluation.py`: ground-truth EAF parsing, matching input bundles, overlap metrics, recommendations, and score history.
- `workbook_tracker.py`: Excel annotation flywheel tracking.
- `types.py`: shared dataclasses and interval/pose schemas.
- `config.py`: label sets, tier order, thresholds, and fallback ratios.

The public API in `auto_annotate/__init__.py` exposes `analyze_and_export`, `extract_annotations`, and `write_eaf`.

## Coding Direction

Keep behavior practical for annotators:

- Generate usable `.eaf` files even when inputs are incomplete.
- Surface confidence, source, and warning metadata in sidecar JSON so review can be prioritized.
- Preserve compatibility with existing input folder names and legacy sidecars.
- Prefer deterministic, testable rules around pipeline artifacts before using expensive pose or mask backends.
- Keep pose and mask providers cache-first and local-first. Do not make normal annotation depend on network access.
- Treat ground-truth comparison as the regression signal. Improve overlap scores without overfitting one sample.
- Avoid broad rewrites unless the data flow demands it. Most changes should be local to a detector, parser, rule, or CLI command.
- Do not remove or rename user data under `input/`, `output/`, `ground_truth/`, `batch_annotations/`, `merge/`, or the Excel workbook unless explicitly asked.

## Validation

For lightweight checks:

```bash
conda run -n auto_anno python -m unittest discover -s auto_annotate/tests -p "test_*.py"
```

For the data-backed feedback loop:

```bash
conda run -n auto_anno python -m unittest discover -s tests -p "test_*.py"
conda run -n auto_anno python -m auto_annotate.cli evaluate-ground-truth ground_truth -i input
```

The ground-truth tests currently assert approximate minimum scores:

```text
phase overlap > 0.87
occlusion overlap > 0.52
left_foot overlap > 0.70
right_foot overlap > 0.70
test overlap > 0.93
```

Annotation runs can be slower when pose or mask generation is enabled. If a change affects only documentation, do not force the full video-backed evaluation unless useful.

## Current Workspace State

The git repo has many uncommitted changes and untracked data/artifacts. Do not revert them casually. Notable current state:

- `README.md` is deleted in the worktree, though the previous README is still available with `git show HEAD:README.md`.
- Core Python files have large edits relative to `HEAD`, including `cli.py`, `pipeline_adapter.py`, `phase_rules.py`, `elan_export.py`, `subject_selection.py`, `smoothing.py`, `types.py`, and `config.py`.
- New/untracked areas include `evaluation.py`, `person_masks.py`, `pose3d.py`, `workbook_tracker.py`, `tests/`, `auto_annotate/tests/`, `ground_truth/`, `batch_annotations/`, `merge/`, `models/`, and `visfrailty_screening_v7.xlsx`.
- As of the last run, `conda run -n auto_anno python -m auto_annotate.cli` completed successfully and generated 14 `.eaf` files in `output/`.

When adding code, inspect nearby tests and the dirty file before editing. Preserve user-generated annotations and workbook state.
