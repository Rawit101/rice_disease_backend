"""Verify portable label handoff preserves the normalized coordinate frame."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from restore_rice_images import restore,within


class RestoreImagesTests(unittest.TestCase):
    def test_sha_verification_exif_rotation_and_duplicate_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); source=p/'original';out=p/'bundle'
            (source/'Brown Spot').mkdir(parents=True);out.mkdir()
            original=source/'Brown Spot/photo.jpg'
            exif=Image.Exif();exif[274]=6
            Image.new('RGB',(40,80),'green').save(original,exif=exif)
            sha=hashlib.sha256(original.read_bytes()).hexdigest()
            rows=[]
            for name in ['first','second']:
                rows.append(dict(sha256=sha,source_relative_path='Brown Spot/photo.jpg',
                    image='images/pool/'+name+'.jpg',thumbnail='thumbnails/'+name+'.jpg',
                    width=80,height=40,quality='automatic_pseudo_label_unreviewed'))
            (out/'manifest.json').write_text(json.dumps(rows))
            with contextlib.redirect_stdout(io.StringIO()):result=restore(source,out,workers=1)
            self.assertEqual(result['restored_source_files'],2)
            self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),sha)
            self.assertEqual((out/rows[0]['image']).read_bytes(),(out/rows[1]['image']).read_bytes())
            with Image.open(out/rows[0]['image']) as im:self.assertEqual(im.size,(80,40))
            original.write_bytes(b'wrong archive contents')
            with self.assertRaisesRegex(ValueError,'checksum'):
                with contextlib.redirect_stdout(io.StringIO()):restore(source,out,workers=1)

    def test_manifest_paths_cannot_escape_selected_directories(self):
        with self.assertRaises(ValueError):within(Path('/tmp/chosen'),'../other/image.jpg')
        with self.assertRaises(ValueError):within(Path('/tmp/chosen'),'/etc/passwd')


if __name__=='__main__':unittest.main()
