from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from auto_annotate.cli import main


class CliSpecTests(unittest.TestCase):
    def test_cli_writes_eaf_debug_and_run_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "sample.mp4"
            video.write_bytes(b"placeholder")
            pipeline_out = root / "pipeline"
            pipeline_out.mkdir()
            output = root / "out"
            (pipeline_out / "biomarker.json").write_text(
                json.dumps({"tug": {"start_ms": 1000, "end_ms": 9000, "duration_ms": 8000}}),
                encoding="utf-8",
            )
            (pipeline_out / "left_boundaries.json").write_text(
                json.dumps(
                    {
                        "events": [
                            {"time_ms": 2500, "event_type": "stance_start"},
                            {"time_ms": 2700, "event_type": "swing_start"},
                            {"time_ms": 3000, "event_type": "stance_start"},
                            {"time_ms": 3500, "event_type": "stance_start"},
                            {"time_ms": 3700, "event_type": "swing_start"},
                            {"time_ms": 6000, "event_type": "stance_start"},
                            {"time_ms": 6200, "event_type": "swing_start"},
                            {"time_ms": 7000, "event_type": "stance_start"},
                            {"time_ms": 7200, "event_type": "swing_start"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (pipeline_out / "right_boundaries.json").write_text(
                json.dumps(
                    {
                        "events": [
                            {"time_ms": 3000, "event_type": "stance_start"},
                            {"time_ms": 3200, "event_type": "swing_start"},
                            {"time_ms": 4000, "event_type": "stance_start"},
                            {"time_ms": 4200, "event_type": "swing_start"},
                            {"time_ms": 6000, "event_type": "stance_start"},
                            {"time_ms": 6500, "event_type": "stance_start"},
                            {"time_ms": 6700, "event_type": "swing_start"},
                            {"time_ms": 7500, "event_type": "stance_start"},
                            {"time_ms": 7700, "event_type": "swing_start"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            exit_code = main([
                "--video",
                str(video),
                "--pipeline-out",
                str(pipeline_out),
                "--out",
                str(output),
                "--debug",
            ])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "sample_auto.eaf").exists())
            self.assertTrue((output / "sample_debug.json").exists())
            with (output / "run_log.csv").open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["video_id"], "sample")
            self.assertEqual(rows[0]["tug_duration_ms"], "8000")
            self.assertEqual(rows[0]["eaf_written"], "True")


if __name__ == "__main__":
    unittest.main()
