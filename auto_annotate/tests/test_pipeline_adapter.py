from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from auto_annotate.pipeline_adapter import load_gait_events, load_tug_interval, select_primary_subject


class PipelineAdapterSpecTests(unittest.TestCase):
    def test_load_tug_interval_requires_valid_biomarker_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "biomarker.json"
            path.write_text(
                json.dumps({"tug": {"start_ms": 1000, "end_ms": 9200, "duration_ms": 8200}}),
                encoding="utf-8",
            )

            tug = load_tug_interval(path)

        self.assertEqual(tug.start_ms, 1000)
        self.assertEqual(tug.end_ms, 9200)
        self.assertEqual(tug.duration_ms, 8200)

    def test_load_tug_interval_rejects_missing_tug_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "biomarker.json"
            path.write_text(json.dumps({"tug": {"start_ms": 2000}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tug_interval(path)

    def test_load_gait_events_accepts_event_list_and_discards_negative_times(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "left_boundaries.json"
            path.write_text(
                json.dumps(
                    {
                        "events": [
                            {"time_ms": -5, "event_type": "stance_start"},
                            {"time_ms": 1000, "event_type": "stance_start"},
                            {"time_ms": 1300, "event_type": "toe_off"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            events = load_gait_events(path, "left")

        self.assertEqual([event.time_ms for event in events], [1000, 1300])
        self.assertEqual([event.event_type for event in events], ["stance_start", "swing_start"])

    def test_select_primary_subject_prefers_stable_centre_track(self) -> None:
        bboxes = [
            {"bbox": [0, 10, 100, 250], "track_id": 10},
            {"bbox": [430, 10, 530, 250], "track_id": 2},
            {"bbox": [435, 10, 535, 250], "track_id": 2},
        ]

        self.assertEqual(select_primary_subject(bboxes, frame_width=1000), 2)


if __name__ == "__main__":
    unittest.main()
