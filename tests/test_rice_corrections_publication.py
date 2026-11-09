import json
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from publish_rice_corrections import build,verify_repairs
from auto_label_rice import yolo_text


class CorrectionPublicationTests(unittest.TestCase):
    def fixture(self,root,visual=False):
        base=root/'base';cor=root/'cor';base.mkdir();cor.mkdir()
        obj=dict(class_id=7 if visual else 4,polygon=[[.1,.1],[.8,.1],[.8,.8]])
        row=dict(id='one',sha256='abc',source_class_id=4,source_folder='False Smut',
            source_relative_path='False Smut/one.jpg',source='/source/one.jpg',
            image='images/pool/one.jpg',thumbnail='thumbnails/one.jpg',
            prediction_file='predictions/one.json',label='labels/pool/one.txt',
            quality='automatic_pseudo_label_unreviewed',exact_duplicate=False,
            reasons=['predicted_class_differs_from_source_folder','related_image_in_checked_validation'] if visual else [],
            default_split=None if visual else 'train',two_model_agreement_candidate=not visual,
            object_count=1,class_ids=[obj['class_id']])
        def put(folder,name,value):
            p=folder/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(value) if not isinstance(value,str) else value)
        put(base,'manifest.json',[row]);put(base,'validation_manifest.json',[])
        put(base,'report.json',{});put(base,row['prediction_file'],dict(objects=[obj]))
        put(base,row['label'],yolo_text([obj]))
        for name in [row['image'],row['thumbnail'],'val.txt','data.yaml','data_strict.yaml',
            'data_all_predictions.yaml','class_mapping.json','inference_config.json','requirements.txt',
            'restore_rice_images.py','merge_rice_training.py','validate_rice_archive.py',
            'README_LABELS_ONLY.md','weights/best.pt']:put(base,name,'fixture')
        # Deliberately harmful automatic proposals must never reach active labels.
        bad=dict(class_id=1,polygon=[[0,0],[1,0],[1,1]])
        put(cor,'boundary/abc.json',dict(schema=3,objects=[bad],geometry_changes=1,
            boundary_changes=1,object_audit=[dict(needs_visual_boundary_review=True)]))
        decision=dict(sha256='abc',annotation_source='accepted.json' if visual else None,
                      audit_index=1,status='manual_trace_accepted' if visual else 'not_accepted')
        put(cor,'visual_decisions.json',[decision])
        if visual:put(cor,'accepted.json',dict(objects=[dict(obj,class_id=4)],visually_accepted=True))
        return base,cor,root/'out'

    def test_unreviewed_color_and_instance_split_proposals_do_not_change_label(self):
        with tempfile.TemporaryDirectory() as d:
            base,cor,out=self.fixture(Path(d))
            build(base,cor,out);verify_repairs(base,out)
            self.assertEqual((base/'labels/pool/one.txt').read_bytes(),(out/'labels/pool/one.txt').read_bytes())
            r=json.loads((out/'manifest.json').read_text())[0]
            self.assertFalse(r['repair_changed'])
            self.assertEqual(r['object_count'],1)
            self.assertTrue(r['two_model_agreement_candidate'])

    def test_individually_accepted_annotation_keeps_evaluation_hold(self):
        with tempfile.TemporaryDirectory() as d:
            base,cor,out=self.fixture(Path(d),visual=True)
            build(base,cor,out);verify_repairs(base,out)
            r=json.loads((out/'manifest.json').read_text())[0]
            self.assertEqual(r['class_ids'],[4])
            self.assertTrue(r['repair_changed'])
            self.assertIsNone(r['default_split'])
            self.assertFalse(r['two_model_agreement_candidate'])
            self.assertFalse(r['expert_diagnosis_confirmed'])
            self.assertIn('related_image_in_checked_validation',r['reasons'])


if __name__=='__main__':unittest.main()
