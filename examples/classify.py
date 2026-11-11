"""Usage: ./run python examples/classify.py photo.jpg"""
from pathlib import Path
import sys
from PIL import Image, ImageOps
from unoq_dsp import ImageClassifier
root = Path(__file__).resolve().parents[1]
with Image.open(sys.argv[1]) as source:
    image = ImageOps.exif_transpose(source).convert("RGB")
with ImageClassifier(root / "models/mobilenetv4-small.uqmodel") as model:
    for result in model.classify(image, top_k=5):
        print(result)
