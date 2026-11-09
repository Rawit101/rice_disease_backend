import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
import json

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import prepare_rice_training as p


def obj(cls=1, score=.9, x=.1):
    return {'class_id':cls, 'score':score,
            'polygon':[[x,.1],[x+.25,.1],[x+.25,.4],[x,.4]]}


class SelectionTests(unittest.TestCase):
    def test_consensus_requires_semantics_and_mask_agreement(self):
        self.assertTrue(p.agreement([obj()],[obj()],1)[0])
        for other in [[obj(cls=2)],[],[obj(score=.3)],[obj(x=.6)], [obj(),obj(x=.5)]]:
            self.assertFalse(p.agreement([obj()],other,1)[0])

    def test_one_object_cannot_match_two_instances(self):
        self.assertFalse(p.agreement([obj(),obj(x=.6)], [obj(),obj()],1)[0])

    def test_groups_are_transitive_and_deterministic(self):
        groups = p.Groups(['b','a','c'])
        groups.join('c','b')
        groups.join('b','a')
        self.assertEqual([groups.find(x) for x in ['a','b','c']], ['a']*3)

    def test_empty_unknown_or_invalid_labels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'a.txt'
            for text in ['', '9 .1 .1 .4 .1 .4 .4', '1 .1 .1 .1 .1 .1 .1', '1 nan .1 .4 .1 .4 .4']:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    p.read_polygons(path)

    def test_merge_keeps_original_validation_and_blocks_eval_duplicate(self):
        import numpy as np
        from PIL import Image
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, extra = root/'old', root/'extra'
            for folder in [old/'images/train',old/'images/val',extra/'images/pool']:
                folder.mkdir(parents=True)
            rng = np.random.default_rng(230)
            for path in [old/'images/train/a.jpg',old/'images/val/b.jpg',extra/'images/pool/c.jpg']:
                Image.fromarray(rng.integers(0,256,(100,100,3),dtype=np.uint8)).save(path)
            import shutil
            shutil.copyfile(old/'images/val/b.jpg',extra/'images/pool/evaluation_copy.jpg')
            (extra/'train.txt').write_text('./images/pool/c.jpg\n./images/pool/evaluation_copy.jpg\n')
            original = {'train':'images/train','val':'images/val','names':p.ORIGINAL_NAMES}
            (old/'data.yaml').write_text(yaml.safe_dump(original))
            subprocess.run([sys.executable,str(SCRIPTS/'merge_rice_training.py'),
                            '--original-yaml',str(old/'data.yaml'),'--root',str(extra)],
                           check=True,capture_output=True,text=True)
            report = json.loads((extra/'merged_with_original/merge_report.json').read_text())
            self.assertEqual(report['added_train'],1)
            self.assertEqual(report['merged_train'],2)
            self.assertEqual(len(report['additions_held_as_evaluation_near_duplicates']),1)
            val = (extra/'merged_with_original/val.txt').read_text().splitlines()
            self.assertEqual(val,[str((old/'images/val/b.jpg').resolve())])


if __name__=='__main__':
    unittest.main()
