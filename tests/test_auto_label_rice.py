"""Verify label/export invariants without requiring GPU inference."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("auto_label_rice", Path(__file__).parents[1] / "scripts" / "auto_label_rice.py")
labeler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labeler)


class LabelExportTests(unittest.TestCase):
    def test_semantic_export_is_polygon_without_confidence_column(self):
        obj = {"class_id": 3, "score": 0.91, "polygon": [[0, 0], [1, 0], [0.5, 1]]}
        tokens = labeler.yolo_text([obj]).split()
        self.assertEqual(len(tokens), 7)
        self.assertEqual(tokens[0], "3")
        self.assertEqual(labeler.yolo_text([]), "")

    def test_invalid_polygons_cannot_be_exported_as_valid_labels(self):
        for polygon in [[[0, 0], [1, 1]], [[0, 0], [0, 0], [0, 0]],
                        [[0, 0], [float("nan"), 0], [0, 1]], [[0, 0], [2, 0], [0, 1]]]:
            self.assertIsNone(labeler.valid_polygon(polygon))
            with self.assertRaises(ValueError):
                labeler.yolo_text([{"class_id": 0, "polygon": polygon}])

    def test_generic_object_masks_are_not_semantic_labels(self):
        proposal = {"score": 0.9, "polygon": [[0, 0], [1, 0], [0, 1]]}
        with self.assertRaises(ValueError):
            labeler.yolo_text([proposal])
        status, reasons = labeler.classify_record(labeler.NAMES.index("Tungro"), [], [proposal])
        self.assertEqual(status, "unclassified_proposals")
        self.assertIn("class_not_supported_by_old_model", reasons)

    def test_no_detection_is_queued_not_treated_as_healthy(self):
        status, reasons = labeler.classify_record(0, [], [])
        self.assertEqual(status, "needs_annotation")
        self.assertIn("no_semantic_label", reasons)

    def test_original_class_ids_and_folder_disagreement_are_preserved(self):
        self.assertEqual(labeler.NAMES[:9], labeler.ORIGINAL_NAMES)
        self.assertEqual(labeler.NAMES.index("Red_Stripe"), 8)
        status, reasons = labeler.classify_record(0, [{"class_id": 1, "score": 0.95}], [], True)
        self.assertEqual(status, "semantic_draft")
        self.assertIn("model_class_differs_from_folder", reasons)
        self.assertIn("identical_content_in_different_classes", reasons)

    def test_duplicate_files_are_kept_in_inventory_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for folder in ["Brown Spot", "Narrow Brown Spot"]:
                p = root / "source" / folder / "same.jpg"
                p.parent.mkdir(parents=True)
                p.write_bytes(b"identical source content")
            out = root / "output"
            with labeler.open_db(out) as db:
                self.assertEqual(labeler.discover([root / "source"], db), 2)
                self.assertEqual(labeler.discover([root / "source"], db), 0)
                summary = labeler.report(out, db)
            self.assertEqual(summary["total_images"], 2)
            self.assertEqual(summary["unique_file_contents"], 1)
            self.assertEqual(summary["class_conflict_groups"], 1)
            records = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["reviewed"] is False for r in records))
            self.assertTrue(all(r["status"] == "pending" for r in records))


if __name__ == "__main__":
    unittest.main()
