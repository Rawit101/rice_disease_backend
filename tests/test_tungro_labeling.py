from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from label_tungro import proposals
from publish_tungro import choose,draft_masks,NAMES
from auto_label_rice import ORIGINAL_NAMES


class TungroLabelingTests(unittest.TestCase):
    def test_new_class_appends_tungro_without_shifting_old_ids(self):
        self.assertEqual(NAMES[:9],ORIGINAL_NAMES)
        self.assertEqual(NAMES[9],'Tungro')
        self.assertEqual(len(NAMES),10)

    def test_visible_leaf_region_not_whole_frame_is_proposed(self):
        rgb=np.full((160,160,3),255,np.uint8)
        rgb[20:145,60:70]=[240,170,30]
        objects,_=proposals(rgb)
        self.assertEqual(len(objects),1)
        self.assertEqual(objects[0]['class_id'],9)
        self.assertAlmostEqual(min(p[0] for p in objects[0]['polygon']),60/160)
        self.assertLess(max(p[0] for p in objects[0]['polygon']),.5)
        self.assertFalse(objects[0]['diagnosis_confirmed'])

    def test_no_dummy_full_frame_labels_for_uniform_colors(self):
        for color in [[35,150,30],[240,170,30],[255,255,255]]:
            image=np.empty((128,128,3),np.uint8);image[:]=color
            objects,_=proposals(image)
            self.assertEqual(objects,[])

    def test_object_support_replaces_a_large_unsupported_color_region(self):
        large=dict(class_id=9,polygon=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]])
        leaf=dict(class_id=9,polygon=[[.45,.2],[.5,.2],[.5,.8],[.45,.8]])
        self.assertEqual(draft_masks([large],[leaf]),[leaf])

    def test_known_bad_mask_is_held_instead_of_exported_as_negative(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'proposals').mkdir();(root/'sam_t').mkdir()
            obj=dict(class_id=9,polygon=[[.1,.1],[.3,.1],[.3,.8]])
            (root/'proposals/x.json').write_text(json.dumps(dict(objects=[obj])))
            (root/'sam_t/x.json').write_text(json.dumps(dict(objects=[obj])))
            with patch('publish_tungro.WORK',root):
                objects,status,prediction=choose(dict(id='x'),{'x':dict(accepted=False)})
            self.assertEqual(objects,[])
            self.assertEqual(status,'visual_review_requires_reannotation')
            self.assertEqual(len(prediction['color_proposal']),1)
            self.assertFalse(prediction['expert_diagnosis_confirmed'])


if __name__=='__main__':unittest.main()
