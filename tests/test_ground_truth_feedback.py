from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from auto_annotate.evaluation import evaluate_ground_truth, find_input_for_ground_truth, track_occlusion_score
from auto_annotate.pipeline_adapter import extract_annotations


ROOT = Path(__file__).resolve().parents[1]


class GroundTruthFeedbackTests(unittest.TestCase):
    def test_ground_truth_files_match_input_folders(self) -> None:
        for gt_path in (ROOT / "ground_truth").glob("*_ground_truth.eaf"):
            with self.subTest(gt_path=gt_path.name):
                match = find_input_for_ground_truth(gt_path, ROOT / "input")
                self.assertIsNotNone(match)
                self.assertTrue(any(match.glob("rgb_video*.mp4")))

    def test_full_tandem_setup_is_not_marked_in_position(self) -> None:
        annotations = extract_annotations(str(ROOT / "input" / "multiple ra FT"))
        self.assertEqual(annotations["test_type"], "full_tandem")
        self.assertEqual(annotations["left_foot"], [])
        self.assertEqual(annotations["right_foot"], [])
        self.assertEqual(annotations["occlusion"], [])

        phases = annotations["phase"]
        self.assertGreaterEqual(len(phases), 2)
        self.assertEqual(phases[0]["label"], "out_of_pos")
        self.assertEqual(phases[1]["label"], "full-tandem")
        self.assertGreaterEqual(phases[1]["start_ms"], 1200)
        self.assertLessEqual(phases[1]["start_ms"], 2300)

        test = annotations["test"][0]
        self.assertEqual(test["start_ms"], phases[1]["start_ms"])
        self.assertLessEqual(test["end_ms"] - test["start_ms"], 10_000)

    def test_feedback_loop_evaluates_current_ground_truth(self) -> None:
        report = evaluate_ground_truth(ROOT / "ground_truth", ROOT / "input")
        expected_count = len(list((ROOT / "ground_truth").glob("*_ground_truth.eaf")))
        self.assertEqual(report["file_count"], expected_count)
        self.assertFalse(report["skipped"])
        self.assertGreater(report["tiers"]["phase"]["overlap_score"], 0.87)
        self.assertGreater(report["tiers"]["occlusion"]["overlap_score"], 0.52)
        self.assertGreater(report["tiers"]["left_foot"]["overlap_score"], 0.70)
        self.assertGreater(report["tiers"]["right_foot"]["overlap_score"], 0.70)
        self.assertGreater(report["tiers"]["test"]["overlap_score"], 0.93)

        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "occlusion_score_history.jsonl"
            entry = track_occlusion_score(report, history_path)
            self.assertEqual(entry["score"], report["tiers"]["occlusion"]["overlap_score"])
            self.assertEqual(entry["previous_score"], None)
            lines = history_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["score"], entry["score"])


if __name__ == "__main__":
    unittest.main()
