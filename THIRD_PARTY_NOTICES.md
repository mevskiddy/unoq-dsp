# Fixed-point arithmetic attribution

The exponent polynomial in `src/unoq_dsp/softmax.py` and fixed-point reciprocal
in `runtime/qnn.c` are adaptations of gemmlowp and TensorFlow Lite reference
algorithms, specialized to signed INT8 inputs and a compiled exponential table.

- Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
- Copyright 2017 The TensorFlow Authors. All Rights Reserved.
- Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
- Sources: https://github.com/google/gemmlowp/blob/master/fixedpoint/fixedpoint.h
  and https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/softmax.h

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an AS IS BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

# Model assets

Model checkpoints and calibration datasets retain their original licenses.
MobileNetV4 timm weights are identified by their Hugging Face model card.
Ultralytics YOLO26 code/weights are subject to Ultralytics' AGPL-3.0 or enterprise
licensing terms. The model export metadata records checkpoint hashes and sources.

# Release components

Project-authored runtime, compiler, API and examples are provided under the
Apache License 2.0 in `LICENSE`. The binary runtime corresponds to `runtime/`
and the flags in `runtime/vision_flags.sh`.

FastRPC binaries are built from Qualcomm FastRPC commit
`8f5927397d1b7c87e6724cb1ddbde956a7f34cab`, without source changes. They use
`/opt/unoq-dsp/config` as their configuration base. Its BSD-3-Clause notice is
in `licenses/FastRPC-BSD-3-Clause.txt`; source and rebuild instructions are in
`docs/EXTENDING.md`.

MobileNetV4-Conv-Small weights are from Ross Wightman's timm model
`mobilenetv4_conv_small.e2400_r224_in1k`, Apache-2.0:
https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k

YOLO26n weights and model architecture come from Ultralytics 8.4.210:
https://github.com/ultralytics/ultralytics/tree/v8.4.210
The detector bundle and detector export adaptation are provided under
AGPL-3.0, reproduced in `licenses/AGPL-3.0.txt`. The export adaptation exposes
the original one-to-one head and quantizes it to INT8 (November 2026).
Corresponding export/calibration code is in `tools/export/`; the upstream
architecture source is at the versioned link above. Model checkpoint and
export checksums are recorded in `models/yolo26n-320-minmax-export.json`.

The device-tree FastRPC node follows Arduino's Linux device-tree definition;
the overlay is provided under GPL-2.0-only (`licenses/GPL-2.0.txt`).

Qualcomm SDK signing tools and the per-device TestSig are not distributed in
this repository. The setup guide obtains the tools from Qualcomm and generates
the authorization on the user's own machine. The matching firmware shell is
fetched from its upstream repository during setup and retains its upstream terms.
