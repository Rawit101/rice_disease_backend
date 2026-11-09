from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from recheck_rice_remaining import crop_boxes,map_crop_object


class CropCoordinateTests(unittest.TestCase):
    def test_crop_polygon_is_mapped_to_original_image_coordinates(self):
        obj={'class_id':1,'score':.8,'polygon':[[.2,.2],[.8,.2],[.8,.8],[.2,.8]]}
        mapped=map_crop_object(obj,(100,200,300,600),1000,1000)
        self.assertAlmostEqual(mapped['polygon'][0][0],.14)
        self.assertAlmostEqual(mapped['polygon'][0][1],.28)
        self.assertFalse(mapped['touches_internal_crop_edge'])

    def test_internal_crop_cut_is_flagged_but_original_image_edge_is_allowed(self):
        obj={'class_id':1,'score':.8,'polygon':[[0,.2],[.5,.2],[.5,.8],[0,.8]]}
        self.assertTrue(map_crop_object(obj,(100,100,500,500),1000,1000)['touches_internal_crop_edge'])
        self.assertFalse(map_crop_object(obj,(0,0,400,400),1000,1000)['touches_internal_crop_edge'])

    def test_all_zoom_windows_stay_inside_the_image(self):
        for width,height in [(1,1),(224,224),(1200,1600)]:
            boxes=crop_boxes(width,height)
            self.assertTrue(boxes)
            for x0,y0,x1,y1 in boxes:
                self.assertTrue(0<=x0<x1<=width)
                self.assertTrue(0<=y0<y1<=height)


if __name__=='__main__':unittest.main()
