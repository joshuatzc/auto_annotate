from __future__ import annotations

import unittest

from auto_annotate.phase_rules import build_tug_annotation_bundle
from auto_annotate.types import GaitEvent, TUGInterval


class PhaseRulesSpecTests(unittest.TestCase):
    def test_tug_state_machine_covers_tug_interval_without_phase_gaps(self) -> None:
        tug = TUGInterval(1000, 9000, 8000)
        left = [
            GaitEvent(2500, "left", "stance_start"),
            GaitEvent(2700, "left", "swing_start"),
            GaitEvent(3500, "left", "stance_start"),
            GaitEvent(3700, "left", "swing_start"),
            GaitEvent(6000, "left", "stance_start"),
            GaitEvent(6200, "left", "swing_start"),
            GaitEvent(7000, "left", "stance_start"),
            GaitEvent(7200, "left", "swing_start"),
        ]
        right = [
            GaitEvent(3000, "right", "stance_start"),
            GaitEvent(3200, "right", "swing_start"),
            GaitEvent(4000, "right", "stance_start"),
            GaitEvent(4200, "right", "swing_start"),
            GaitEvent(6500, "right", "stance_start"),
            GaitEvent(6700, "right", "swing_start"),
            GaitEvent(7500, "right", "stance_start"),
            GaitEvent(7700, "right", "swing_start"),
        ]

        bundle = build_tug_annotation_bundle(tug, left, right)
        labels = [phase.label for phase in bundle.phases]

        self.assertEqual(bundle.phases[0].start_ms, tug.start_ms)
        self.assertEqual(bundle.phases[-1].end_ms, tug.end_ms)
        for previous, current in zip(bundle.phases, bundle.phases[1:]):
            self.assertEqual(previous.end_ms, current.start_ms)
        self.assertIn("sit-to-stand", labels)
        self.assertGreaterEqual(labels.count("walk"), 2)
        self.assertIn("turn", labels)
        self.assertIn("stand-to-sit", labels)
        self.assertNotIn("unknown", labels)

    def test_missing_foot_side_is_unknown_only_during_walk(self) -> None:
        tug = TUGInterval(1000, 6000, 5000)
        left = [
            GaitEvent(1800, "left", "stance_start"),
            GaitEvent(2100, "left", "swing_start"),
            GaitEvent(3000, "left", "stance_start"),
        ]
        right = [
            GaitEvent(2300, "right", "stance_start"),
            GaitEvent(2600, "right", "swing_start"),
            GaitEvent(4500, "right", "stance_start"),
        ]

        bundle = build_tug_annotation_bundle(tug, left, right)
        self.assertTrue(bundle.left)
        self.assertTrue(bundle.right)
        self.assertTrue(all(annotation.start_ms >= 2300 for annotation in bundle.left))
        self.assertTrue(all(annotation.label.startswith("left_") or annotation.label == "unknown" for annotation in bundle.left))


if __name__ == "__main__":
    unittest.main()
