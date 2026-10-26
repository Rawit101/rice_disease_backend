"""Full application initialization with mocked YOLO; no AI modules or credentials."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP = textwrap.dedent('''
    import importlib.abc
    import io
    import runpy
    import sys
    import types
    from unittest.mock import Mock
    from PIL import Image
    import dotenv

    dotenv.load_dotenv = lambda *args, **kwargs: None
    ai_imports = []
    class BlockAI(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith(('chromadb', 'google.genai', 'google.generativeai', 'openai')):
                ai_imports.append(fullname)
                raise AssertionError('AI import attempted: ' + fullname)
    sys.meta_path.insert(0, BlockAI())
    class Vector(list):
        def tolist(self): return list(self)
    model = Mock()
    model.names = {0: 'Rice_Blast'}
    result = Mock(boxes=[Mock(cls=[0], conf=[0.9], xyxy=[Vector([1, 2, 3, 4])])])
    model.predict.return_value = [result]
    ultralytics = types.ModuleType('ultralytics')
    ultralytics.YOLO = lambda path: model
    sys.modules['ultralytics'] = ultralytics
    namespace = runpy.run_path(sys.argv[1])
    client = namespace['app'].test_client()
    data = io.BytesIO()
    Image.new('RGB', (8, 8)).save(data, 'JPEG')
    data.seek(0)
    response = client.post('/predict', data={
        'image': (data, 'rice.jpg'), 'include_annotated_image': 'false'
    })
    assert response.status_code == 200, response.data
    assert response.json['predictions'][0]['class'] == 'Rice_Blast'
    assert response.json['annotated_image'] is None
    result.plot.assert_not_called()
    assert not namespace.get('ENABLE_RAG', False)
    assert namespace.get('collection') is None
    assert ai_imports == [], ai_imports
    if '/search' in [rule.rule for rule in namespace['app'].url_map.iter_rules()]:
        assert client.post('/search', json={'question': 'test'}).status_code == 503
        assert client.post('/query', json={'question': 'test'}).status_code == 503
    print('STARTUP_AND_DETECTION_OK')
''')


class DetectorStartupTests(unittest.TestCase):
    def test_initialization_and_detection_never_import_ai_even_with_stale_keys(self):
        for filename in ['yolo_server.py', 'render-api/app.py', 'hf-space/app.py']:
            for stale_keys in [False, True]:
                with self.subTest(service=filename, stale_keys=stale_keys):
                    environment = {k: v for k, v in os.environ.items() if k not in [
                        'ENABLE_RAG', 'GEMINI_API_KEY', 'MAXPLUS_API_KEY', 'MAXPLUS_BASE_URL',
                        'RAG_API_URL', 'YOLO_CONFIDENCE_THRESHOLD']}
                    if stale_keys:
                        environment.update(GEMINI_API_KEY='unused', MAXPLUS_API_KEY='unused',
                                           RAG_API_URL='http://unavailable.invalid')
                    result = subprocess.run([sys.executable, '-c', BOOTSTRAP, filename],
                                            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('STARTUP_AND_DETECTION_OK', result.stdout)

    def test_menu_image_matches_published_action_dimensions(self):
        from PIL import Image
        menu = ROOT / 'image-only-richmenu.png'
        self.assertLess(menu.stat().st_size, 1024 * 1024)
        with Image.open(menu) as image:
            self.assertEqual(image.size, (2500, 843))


if __name__ == '__main__':
    unittest.main()
