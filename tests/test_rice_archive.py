"""New source folders must never acquire a guessed disease or negative label."""
import sys
from pathlib import Path
import unittest
import argparse
import contextlib
import io
import json
import tempfile
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from auto_label_rice import NAMES, FOLDERS, ORIGINAL_NAMES
from label_rice_archive import choose_labels
import label_rice_archive as archive


def obj(cls, score=.9):
    return {'class_id': cls, 'score': score, 'polygon': [[.1,.1],[.5,.1],[.5,.5],[.1,.5]]}


def raw(a=None,b=None,recovery=None):
    return {'v8_800':a or [], 'v6_800':b or [], 'v8_1280':recovery or [],'v6_1280':[]}


class RiceArchiveTests(unittest.TestCase):
    def test_folder_mapping_preserves_head_and_distinguishes_blast_sites(self):
        self.assertEqual(NAMES[:9], ORIGINAL_NAMES)
        self.assertEqual(FOLDERS['Leaf Blast'], 'Rice_Blast')
        self.assertEqual(FOLDERS['Neck Blast'], 'Neck_Blast')
        self.assertGreaterEqual(NAMES.index(FOLDERS['Neck Blast']),9)

    def test_no_detection_is_not_negative(self):
        objects,quality,reasons,eligible,strict=choose_labels(1,raw())
        self.assertEqual(objects,[])
        self.assertEqual(quality,'no_detection_not_a_negative_label')
        self.assertFalse(eligible or strict)

    def test_new_disease_and_healthy_ignore_old_class_predictions(self):
        for name in ['Healthy','Hispa','Sheath_Blight','Bakanae','Neck_Blast']:
            objects,quality,reasons,eligible,strict=choose_labels(NAMES.index(name),raw([obj(2)],[obj(2)]))
            self.assertFalse(objects or eligible or strict)
            self.assertTrue(reasons)

    def test_conflicting_teacher_class_is_preserved_but_not_default_train(self):
        objects,quality,reasons,eligible,strict=choose_labels(1,raw([obj(2)],[obj(2)]))
        self.assertEqual(objects[0]['class_id'],2)
        self.assertIn('predicted_class_differs_from_source_folder',reasons)
        self.assertFalse(eligible or strict)

    def test_consensus_is_candidate_not_reviewed_truth(self):
        objects,quality,reasons,eligible,strict=choose_labels(1,raw([obj(1)],[obj(1)]))
        self.assertTrue(eligible and strict)
        self.assertEqual(quality,'automatic_pseudo_label_unreviewed')
        self.assertFalse(reasons)

    def test_recovery_and_source_conflicts_are_review_only(self):
        for data,conflict in [(raw(recovery=[obj(1)]),False),(raw([obj(1)],[obj(1)]),True)]:
            objects,quality,reasons,eligible,strict=choose_labels(1,data,conflict)
            self.assertTrue(objects)
            self.assertFalse(eligible or strict)

    def test_export_accounts_for_sources_and_excludes_unlabeled_from_yolo_lists(self):
        from PIL import Image
        from validate_rice_archive import validate
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'out';old=root/'previous'
            for d in [out/'images/pool',out/'labels/pool',out/'thumbnails',out/'cache',
                      old/'images/pool',old/'labels/pool']:
                d.mkdir(parents=True)
            records=[]
            cases=[('positive',1,1),('disagreement',0,1),('missing',1,None),
                   ('novel',13,2),('healthy',12,1),('reviewed_healthy',12,1),
                   ('shared_new',13,1),('shared_known',1,1)]
            for image_id,expected,predicted in cases:
                sha='shared' if image_id.startswith('shared_') else image_id
                image='images/pool/'+image_id+'.jpg'
                Image.new('RGB',(80,60),'green').save(out/image)
                thumb='thumbnails/'+image_id+'.jpg'
                Image.new('RGB',(40,30),'green').save(out/thumb)
                records.append(dict(id=image_id,source='/input/'+image_id+'.jpg',
                    source_folder=image_id,source_class_id=expected,sha256=sha,
                    image=image,thumbnail=thumb,error=None,canonical_id=image_id))
                for model in ['v8','v6']:
                    for recovery in [False,True]:
                        archive.write_json(archive.cache_path(out,sha,model,recovery),
                            [obj(predicted)] if predicted is not None else [])
            archive.write_json(out/'prepared.json',records)
            archive.write_json(out/'phash_cache.json',{})
            archive.write_json(old.parent/'phash_cache.json',{})
            old_record=dict(id='val',sha256='val_sha',image='images/pool/val.jpg',
                label='labels/pool/val.txt',default_split='val',reasons=[],
                quality='ai_visually_screened_mask',source_class_id=4)
            Image.new('RGB',(80,60),'green').save(old/old_record['image'])
            (old/old_record['label']).write_text(archive.yolo_text([obj(4)]))
            old_background=dict(id='old_background',sha256='reviewed_healthy',
                image='images/pool/old_background.jpg',label='labels/pool/old_background.txt',
                default_split='train',reasons=[],quality='ai_visually_screened_background',source_class_id=12)
            (old/old_background['label']).write_text('')
            archive.write_json(old/'manifest.json',[old_record,old_background])
            (root/'model.pt').write_bytes(b'checkpoint fixture')
            args=argparse.Namespace(output=out,previous=old,source=Path('/input'),model=root/'model.pt',secondary=root/'model.pt')
            with patch.object(archive,'leakage_groups',return_value={x:x for x in
                    [r['sha256'] for r in records]+['val_sha']}):
                with contextlib.redirect_stdout(io.StringIO()):
                    archive.export(args)
                    result=validate(out)
            self.assertEqual(result['source_images'],8)
            self.assertEqual(result['annotated_files'],4)
            self.assertEqual(result['train_images'],2)
            self.assertEqual(result['strict_train_images'],1)
            manifest={r['id']:r for r in archive.read_json(out/'manifest.json')}
            for name in ['novel','healthy','missing','shared_new']:
                self.assertIsNone(manifest[name]['label'])
                self.assertIsNone(manifest[name]['default_split'])
            self.assertEqual(manifest['reviewed_healthy']['quality'],'ai_visually_screened_background')
            self.assertEqual((out/manifest['reviewed_healthy']['label']).read_text(),'')
            self.assertIsNotNone(manifest['shared_known']['label'])
            self.assertIsNone(manifest['shared_known']['default_split'])
            self.assertEqual(manifest['shared_new']['canonical_id'],'shared_known')


if __name__=='__main__':unittest.main()
