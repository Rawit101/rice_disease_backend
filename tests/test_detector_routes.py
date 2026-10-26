"""Exercise the exact production /predict handlers with mocked model/annotation services.

Extract handlers with AST to avoid loading best.pt, ChromaDB, API clients or their
startup side effects. Flask request parsing, image decoding and response JSON are real.
"""
import ast
import base64
import io
import math
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from flask import Flask, jsonify, request
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ['yolo_server.py', 'render-api/app.py', 'hf-space/app.py']


class Vector(list):
    def tolist(self):
        return list(self)


def load_route(filename):
    tree = ast.parse((ROOT / filename).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'predict']
    model = Mock()
    model.names = {0: 'Rice_Blast', 1: 'Brown_Spot'}
    boxes = [Mock(cls=[0], conf=[0.8], xyxy=[Vector([1, 2, 3, 4])]),
             Mock(cls=[1], conf=[0.9], xyxy=[Vector([2, 3, 4, 5])])]
    result = Mock(boxes=boxes)
    model.predict.return_value = [result]
    cv2 = Mock(IMWRITE_JPEG_QUALITY=1)
    cv2.imencode.return_value = (True, b'annotation')
    app = Flask(__name__)
    namespace = dict(app=app, model=model, MODEL_PATH='best.pt',
                     request=request, jsonify=jsonify, io=io, Image=Image,
                     cv2=cv2, base64=base64, math=math, YOLO_CONFIDENCE_THRESHOLD=0.25)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), namespace)
    return app.test_client(), model, result, cv2


def image_request(client, **fields):
    data = io.BytesIO()
    Image.new('RGB', (8, 8)).save(data, 'JPEG')
    data.seek(0)
    return client.post('/predict', data={'image': (data, 'rice.jpg'), **fields})


class DetectorRouteTests(unittest.TestCase):
    def test_line_skips_all_annotation_and_uses_same_threshold(self):
        for filename in SERVICES:
            with self.subTest(service=filename):
                client, model, result, cv2 = load_route(filename)
                response = image_request(client, include_annotated_image='false', confidence_threshold='0.4')
                self.assertEqual(response.status_code, 200)
                self.assertIsNone(response.json['annotated_image'])
                self.assertEqual(response.json['predictions'][0]['class'], 'Brown_Spot')
                self.assertEqual(model.predict.call_args.kwargs, dict(conf=0.4, verbose=False))
                result.plot.assert_not_called()
                cv2.imencode.assert_not_called()

    def test_other_consumers_still_get_annotation_by_default(self):
        for filename in SERVICES:
            with self.subTest(service=filename):
                client, model, result, cv2 = load_route(filename)
                response = image_request(client)
                self.assertEqual(response.status_code, 200)
                self.assertIsNotNone(response.json['annotated_image'])
                self.assertEqual(model.predict.call_args.kwargs['conf'], 0.25)
                result.plot.assert_called_once()
                cv2.imencode.assert_called_once()

    def test_empty_results_and_detector_failures(self):
        for filename in SERVICES:
            with self.subTest(service=filename):
                client, model, result, _ = load_route(filename)
                result.boxes = []
                self.assertEqual(image_request(client).json['predictions'], [])
                result.plot.assert_not_called()
                model.predict.side_effect = RuntimeError('mock failure')
                self.assertEqual(image_request(client).status_code, 500)

    def test_invalid_thresholds_do_not_run_inference(self):
        for filename in SERVICES:
            for value in ['0', '-1', '1.1', 'nan', 'inf', 'invalid']:
                with self.subTest(service=filename, value=value):
                    client, model, _, _ = load_route(filename)
                    self.assertEqual(image_request(client, confidence_threshold=value).status_code, 400)
                    model.predict.assert_not_called()

    def test_model_unavailable_and_missing_image(self):
        for filename in SERVICES:
            with self.subTest(service=filename):
                client, model, _, _ = load_route(filename)
                self.assertEqual(client.post('/predict', data={}).status_code, 400)
                client.application.view_functions['predict'].__globals__['model'] = None
                self.assertEqual(image_request(client).status_code, 500)
                model.predict.assert_not_called()

    def test_production_threshold_configuration(self):
        for filename in SERVICES:
            tree = ast.parse((ROOT / filename).read_text())
            nodes = [n for n in tree.body if
                     isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'YOLO_CONFIDENCE_THRESHOLD' for t in n.targets)
                     or isinstance(n, ast.If) and 'YOLO_CONFIDENCE_THRESHOLD' in ast.unparse(n.test)]
            code = compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec')
            for value in ['0.25', '0.4', 'nan', '0']:
                with self.subTest(service=filename, value=value), patch.dict(os.environ, {'YOLO_CONFIDENCE_THRESHOLD': value}):
                    namespace = dict(os=os, math=math)
                    if value in ['nan', '0']:
                        with self.assertRaises(ValueError):
                            exec(code, namespace)
                    else:
                        exec(code, namespace)
                        self.assertEqual(namespace['YOLO_CONFIDENCE_THRESHOLD'], float(value))

    def test_render_startup_uses_pt_model(self):
        tree = ast.parse((ROOT / 'render-api/app.py').read_text())
        main = [n for n in tree.body if isinstance(n, ast.If) and '__name__' in ast.unparse(n.test)]
        app = Mock()
        namespace = dict(__name__='__main__', model=Mock(names={0: 'Rice_Blast'}), collection=None,
                         app=app, os=os, ENABLE_RAG=False)
        with patch.dict(os.environ, {'PORT': '5000'}):
            exec(compile(ast.Module(body=main, type_ignores=[]), 'render-api/app.py', 'exec'), namespace)
        app.run.assert_called_once_with(host='0.0.0.0', port=5000, debug=False)

    def test_render_docker_root_context_paths(self):
        text = (ROOT / 'render-api/Dockerfile').read_text()
        for line in text.splitlines():
            if line.startswith('COPY '):
                self.assertTrue((ROOT / line.split()[1]).exists(), line)
        self.assertIn('CMD ["python", "render-api/app.py"]', text)
        self.assertIn('COPY index.js database.js image-processing.js line-interface.js ./', (ROOT / 'Dockerfile.render').read_text())


if __name__ == '__main__':
    unittest.main()
