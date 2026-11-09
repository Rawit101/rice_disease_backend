import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from repair_rice_annotations import cycles,geometry


class AnnotationRepairTests(unittest.TestCase):
    def test_valid_contour_and_class_are_unchanged(self):
        p=[[.1,.1],[.4,.1],[.4,.4],[.1,.4]]
        fixed,n=geometry([dict(class_id=8,polygon=p,score=.8)])
        self.assertEqual(n,0)
        self.assertEqual(fixed[0]['polygon'],p)
        self.assertEqual(fixed[0]['class_id'],8)

    def test_disconnected_loops_drop_zero_area_connecting_bridge(self):
        a=[[.1,.1],[.3,.1],[.3,.3],[.1,.3]]
        b=[[.6,.6],[.8,.6],[.8,.8],[.6,.8]]
        walk=a+[a[0]]+b+[b[0],a[0]]
        fixed,n=geometry([dict(class_id=1,polygon=walk,score=.7)])
        self.assertEqual(n,1)
        self.assertEqual(len(fixed),2)
        self.assertEqual({o['class_id'] for o in fixed},{1})
        for o in fixed:
            xs=[p[0] for p in o['polygon']]
            self.assertLess(max(xs)-min(xs),.25)

    def test_opposite_winding_hole_is_held_instead_of_silently_filled(self):
        a=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
        b=[[.3,.3],[.3,.7],[.7,.7],[.7,.3]]
        walk=a+[a[0]]+b+[b[0],a[0]]
        fixed,n=geometry([dict(class_id=2,polygon=walk)])
        self.assertEqual(n,0)
        self.assertEqual(len(fixed),1)
        self.assertTrue(fixed[0]['topology_requires_review'])
        self.assertEqual(fixed[0]['polygon'],walk)

    def test_pixel_boundary_trim_uses_native_image_coordinate_frame(self):
        import cv2,numpy as np
        from audit_rice_boundaries import audit_object
        rgb=np.full((20,20,3),255,np.uint8)
        rgb[3:16,3:19]=[140,70,25]
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        obj={'class_id':1,'score':.8,'polygon':[[3/20,3/20],[19/20,3/20],[19/20,15/20],[3/20,15/20]]}
        result=audit_object(obj,hsv,20,20)
        self.assertTrue(result['repair_applied'])
        self.assertAlmostEqual(max(x for p in result['objects'] for x,y in p['polygon']),18/20)
        self.assertEqual({p['class_id'] for p in result['objects']},{1})


if __name__=='__main__':unittest.main()
