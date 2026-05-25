from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from auto_annotate.elan_export import validate_eaf, write_eaf
from auto_annotate.types import AnnotationBundle, FootAnnotation, PhaseAnnotation, TUGInterval


class ElanExportSpecTests(unittest.TestCase):
    def test_write_and_validate_strict_bundle_eaf(self) -> None:
        bundle = AnnotationBundle(
            tug=TUGInterval(1000, 4000, 3000),
            phases=[
                PhaseAnnotation(1000, 1500, "sit"),
                PhaseAnnotation(1500, 2100, "sit-to-stand"),
                PhaseAnnotation(2100, 3000, "walk"),
                PhaseAnnotation(3000, 3400, "stand-to-sit"),
                PhaseAnnotation(3400, 4000, "sit"),
            ],
            left=[
                FootAnnotation(2100, 2500, "left_stance", "left"),
                FootAnnotation(2500, 3000, "left_swing", "left"),
            ],
            right=[
                FootAnnotation(2100, 2500, "right_swing", "right"),
                FootAnnotation(2500, 3000, "right_stance", "right"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "video.mp4"
            video.write_bytes(b"placeholder")
            eaf = Path(tmp) / "video_auto.eaf"
            write_eaf(bundle, video, eaf)
            errors = validate_eaf(eaf)

        self.assertEqual(errors, [])

    def test_writer_rejects_annotations_outside_tug(self) -> None:
        bundle = AnnotationBundle(
            tug=TUGInterval(1000, 4000, 3000),
            phases=[PhaseAnnotation(900, 1200, "sit")],
            left=[],
            right=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_eaf(bundle, Path(tmp) / "video.mp4", Path(tmp) / "bad.eaf")


if __name__ == "__main__":
    unittest.main()
