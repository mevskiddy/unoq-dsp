import inspect
from pathlib import Path
import unittest
from unoq_dsp import ImageClassifier, ObjectDetector, Session, load

class ReleaseTests(unittest.TestCase):
    def test_blocking_defaults(self):
        for cls in (ImageClassifier, ObjectDetector, Session):
            self.assertEqual(inspect.signature(cls).parameters['pipeline'].default, 'rpc')

    def test_bundled_models(self):
        root = Path(__file__).resolve().parents[1]
        for filename, kind in [('mobilenetv4-small.uqmodel','image_classification'), ('yolo26n-320.uqmodel','yolo_one_to_one')]:
            model = load(root/'models'/filename)
            self.assertEqual(model.metadata['task']['kind'], kind)
            self.assertGreater(len(model.plan), 1000000)
