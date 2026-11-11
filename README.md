# UNO Q DSP

INT8 image classification and object detection on the Arduino UNO Q’s Hexagon DSP.
Includes ready-to-run models, a Python API, compiled runtime, and source for extending kernels and model support.

Inference uses **blocking FastRPC**: the calling CPU thread sleeps while the DSP executes the neural network. Image preparation and result decoding run on Linux.

| Model | Input | Task | Warm DSP inference per image |
|---|---|---|---:|
| MobileNetV4-Conv-Small | 224 × 224 | 1,000 ImageNet classes | ~50 ms |
| YOLO26n | 320 × 320 | 80 COCO object classes | ~192 ms |

Each timing is for **one image (batch size 1), with the model already loaded**, measured on UNO Q. It covers the complete neural-network invocation and input/output transfer, not a single layer or detected object. Image loading, preprocessing, result decoding and model initialization are excluded; these are not end-to-end application timings.

Calling-process CPU time during that invocation is about 1.1 ms per image for classification and 3.2 ms per image for detection. Keep a model open to process successive images.

Reproduce the measurements with `./run python examples/benchmark.py photo.jpg`.

## Start

Copy this folder to your UNO Q. Follow [DSP setup](docs/SETUP.md) once to enable FastRPC and authorize the board, then install:

```sh
./scripts/install.sh
```

Classify a photo or detect objects:

```sh
./run python examples/classify.py photo.jpg
./run python examples/detect.py photo.jpg annotated.jpg
```

The detector prints labels, confidence scores and boxes, and saves an annotated image when an output path is supplied. Both bundles include their preprocessing settings and labels.

## Python

Open an interpreter with `./run python`, or launch your script with `./run python app.py`:

```python
from PIL import Image
from unoq_dsp import ImageClassifier, ObjectDetector

image = Image.open("photo.jpg").convert("RGB")

with ImageClassifier("models/mobilenetv4-small.uqmodel") as model:
    print(model.classify(image, top_k=5))

with ObjectDetector("models/yolo26n-320.uqmodel") as model:
    for item in model.detect(image, score_threshold=0.5):
        print(item.label, item.score, item.box)
```

Boxes are normalized `(left, top, right, bottom)` coordinates relative to the original image. The API also accepts RGB `uint8` NumPy arrays. Use one active model per process; closing the context releases the model for the next session.

`./run` configures the native libraries, manages the FastRPC listener and serializes access to the DSP. No inference service starts at boot. The model and DSP executable are loaded automatically when a session opens.

## Extend

The [extension guide](docs/EXTENDING.md) covers TFLite compilation, operators, memory planning and rebuilding the native runtime. Kernel and compiler source are included; end users use the supplied binaries.

Model sources and re-export steps are in [the model guide](docs/MODELS.md).

Project code is Apache-2.0. MobileNetV4 weights are Apache-2.0; YOLO26 assets and export adaptations retain AGPL-3.0 terms. See [third-party notices](THIRD_PARTY_NOTICES.md).
