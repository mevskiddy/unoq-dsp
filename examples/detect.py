"""Usage: ./run python examples/detect.py photo.jpg [annotated.jpg]"""
from pathlib import Path
import sys
from PIL import Image, ImageOps, ImageDraw
from unoq_dsp import ObjectDetector
root = Path(__file__).resolve().parents[1]
with Image.open(sys.argv[1]) as source:
    image = ImageOps.exif_transpose(source).convert("RGB")
with ObjectDetector(root / "models/yolo26n-320.uqmodel") as model:
    results = model.detect(image, score_threshold=0.5)
    for result in results:
        print(result)
    print(model.last_timings)
if len(sys.argv) > 2:
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for result in results:
        x1, y1, x2, y2 = result.box
        box = (x1*w, y1*h, x2*w, y2*h)
        draw.rectangle(box, outline="#39c38b", width=3)
        draw.text((box[0]+4, box[1]+4), f"{result.label} {result.score:.0%}", fill="#39c38b")
    image.save(sys.argv[2])
