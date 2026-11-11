# Model sources

The two `.uqmodel` files are self-contained compiled INT8 graphs with weights,
quantization parameters, labels and task metadata. Their hashes and upstream
sources are recorded in `models/provenance.json`.

- **MobileNetV4-Conv-Small:** timm's `mobilenetv4_conv_small.e2400_r224_in1k`,
  224-pixel bicubic center crop, ImageNet normalization, 1,000 output labels.
- **YOLO26n:** Ultralytics 8.4.210, 320-pixel RGB letterbox, one-to-one prediction
  head, 80 COCO labels. The DSP runs the learned network; Python decodes boxes
  and selects detections.

## Re-export

The conversion scripts live in `tools/export/`. Use Python 3.12 on a workstation
and the versions recorded in the corresponding `models/*-export.json` file.
The detector export is an AGPL-3.0 adaptation of Ultralytics' model; its source
architecture is available at
https://github.com/ultralytics/ultralytics/tree/v8.4.210/ultralytics.

The scripts expect Imagenette under `local-artifacts/calibration/imagenette2-160`
and COCO128 under `local-artifacts/calibration/coco128`. These datasets are used
for calibration and are not bundled. The original YOLO26n checkpoint is included as `models/yolo26n.pt` for
re-export and modification. The classifier script downloads its checkpoint
through timm. Recorded hashes identify the inputs used for these bundles.

```sh
python tools/export/export_classifier.py
python tools/export/export_detector.py
python tools/export/package_models.py
```

The classifier uses the export's default moving-average calibration; the detector
uses observed min/max calibration. Export metadata records preprocessing,
calibration inputs, package versions and source hashes.

The [compiler and kernel guide](EXTENDING.md) covers converting additional TFLite
models and implementing operators.
