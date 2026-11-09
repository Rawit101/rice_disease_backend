from pathlib import Path
import sys, tempfile, unittest
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_rice_10class as audit


class TenClassAuditTests(unittest.TestCase):
    def test_scope_uses_category_name_not_conflicting_catalog_number(self):
        mapping = {'Bakanae': 'Bakanae', 'Tungro': 'Tungro', 'Healthy': 'Healthy',
                   'Leaf Smut': 'Leaf_Smut', 'False Smut': 'False_Smut',
                   'Sheath Blight': 'Sheath_Blight', 'Sheath Rot': 'Sheath_Rot'}
        self.assertIsNone(audit.allowed_class('Bakanae', mapping))
        self.assertEqual(audit.allowed_class('Tungro', mapping), 9)
        for name in ['Healthy', 'Leaf Smut', 'Sheath Blight']:
            self.assertIsNone(audit.allowed_class(name, mapping))
        self.assertEqual(audit.allowed_class('False Smut', mapping), 4)
        self.assertEqual(audit.allowed_class('Sheath Rot', mapping), 6)

    def sample(self, base, label):
        root = base / 'dataset'; root.mkdir()
        Image.new('RGB', (224,224), 'white').save(root/'image.jpg')
        if label is not None: (root/'label.txt').write_text(label)
        return dict(id='example', dataset_root='dataset', image='image.jpg',
                    label='label.txt' if label is not None else None,
                    expected_class_id=6, quality='automatic_pseudo_label_unreviewed',
                    exact_content_source_categories=['Sheath_Rot'], reasons=[])

    def test_disagreement_is_held_without_renaming_to_folder(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); row=self.sample(base,'1 0.1 0.1 0.7 0.1 0.7 0.7 0.1 0.7\n')
            with patch.object(audit,'BASE',base): result=audit.one(row)
            self.assertEqual(result['class_ids'],[1])
            self.assertIn('annotation_class_disagrees_with_source_hint',result['audit_flags'])
            self.assertIn('mask_overlaps_near_uniform_background',result['audit_flags'])
            self.assertFalse(result['training_ready_after_automatic_audit'])
            self.assertTrue((base/'dataset/label.txt').read_text().startswith('1 '))

    def test_missing_detection_never_becomes_healthy_or_negative(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); row=self.sample(base,None)
            with patch.object(audit,'BASE',base): result=audit.one(row)
            self.assertIn('no_usable_mask_not_a_negative',result['audit_flags'])
            self.assertIsNone(result['label_sha256'])
            self.assertFalse((base/'dataset/label.txt').exists())
            self.assertFalse(result['training_ready_after_automatic_audit'])


if __name__ == '__main__': unittest.main()
