from pathlib import Path
import sys,unittest
import argparse,contextlib,io,json,sqlite3,tempfile
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_full_rice_comparison import merge_predictions
import build_full_rice_comparison as builder


def obj(cls,score,x=.1):
    return {'class_id':cls,'score':score,'polygon':[[x,.1],[x+.2,.1],[x+.2,.4],[x,.4]]}


class FullComparisonTests(unittest.TestCase):
    def test_overlap_conflict_keeps_actual_predicted_class_and_provenance(self):
        result,conflicts=merge_predictions([('old',[obj(2,.7)]),('other',[obj(1,.8)])])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['class_id'],1)
        self.assertEqual(conflicts,1)
        self.assertEqual({x['model'] for x in result[0]['support']},{'old','other'})

    def test_separate_instances_are_retained_and_same_mask_is_not_duplicated(self):
        result,conflicts=merge_predictions([('old',[obj(2,.8),obj(2,.7,x=.6)]),
                                           ('other',[obj(2,.9)])])
        self.assertEqual(len(result),2)
        self.assertEqual(conflicts,0)

    def test_no_predictions_does_not_manufacture_a_mask(self):
        self.assertEqual(merge_predictions([('a',[]),('b',[])]),([],0))

    def test_export_preserves_teacher_ids_and_never_turns_missing_or_new_disease_into_negative(self):
        from PIL import Image
        from validate_full_rice import validate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for folder in ['source/images','ensemble','recovery','additional','checked/labels','curated']:
                (root/folder).mkdir(parents=True)
            records=[]
            for name,cls,objects in [('prediction',1,[obj(2,.9)]),('missing',0,[]),
                                      ('new_disease',20,[]),('healthy',12,[]),('val_image',4,[])]:
                r={'id':name,'source':'/input/'+name+'.jpg','folder':name,
                   'folder_class_id':cls,'sha256':name,'image':'images/'+name+'.jpg','objects':objects}
                records.append(r)
                Image.new('RGB',(80,60),'green').save(root/'source'/r['image'])
                (root/'ensemble'/(name+'.json')).write_text(json.dumps(objects))
                if cls>=9:
                    for model in ['v8','v6']:
                        (root/'additional'/(name+'_'+model+'.json')).write_text(json.dumps([obj(2,.8)]))
            db=sqlite3.connect(root/'source/progress.sqlite')
            db.execute('create table images (folder,source,record)')
            for r in records:db.execute('insert into images values (?,?,?)',(r['folder'],r['source'],json.dumps(r)))
            db.commit();db.close()
            (root/'ensemble/config.json').write_text(json.dumps({'model_sha256':'secondary'}))
            (root/'recovery/selection.json').write_text('{}')
            (root/'backgrounds.json').write_text(json.dumps([{'source_sha256':'healthy','accepted_background':False}]))
            (root/'decisions.json').write_text('[]')
            (root/'checked/manifest.json').write_text(json.dumps([{'id':'val_image','source_sha256':'val_image'}]))
            (root/'checked/labels/val_image.txt').write_text('4 .1 .1 .3 .1 .3 .4 .1 .4\n')
            (root/'curated/manifest.json').write_text(json.dumps([{'id':'val_image','disposition':'val','split_group':'val_image'}]))
            (root/'model.pt').write_bytes(b'fixture for checkpoint hash verification')
            args=argparse.Namespace(source=root/'source',ensemble=root/'ensemble',recovery=root/'recovery',
                additional=root/'additional',checked=root/'checked',curated=root/'curated',
                backgrounds=root/'backgrounds.json',decisions=root/'decisions.json',manual=root/'manual.json',model=root/'model.pt',output=root/'out')
            with patch.object(builder,'leakage_groups',return_value={r['sha256']:r['sha256'] for r in records}):
                with contextlib.redirect_stdout(io.StringIO()):builder.build(args)
            manifest={r['id']:r for r in json.loads((root/'out/manifest.json').read_text())}
            self.assertIsNone(manifest['missing']['label'])
            self.assertIsNone(manifest['new_disease']['label'])
            self.assertIsNone(manifest['healthy']['default_split'])
            self.assertEqual(manifest['prediction']['class_ids'],[2])
            self.assertIn('predicted_class_differs_from_source_folder',manifest['prediction']['reasons'])
            with contextlib.redirect_stdout(io.StringIO()):result=validate(root/'out')
            self.assertEqual(result['structural_validation'],'passed')


if __name__=='__main__':unittest.main()
