# Extend the runtime

## Compile a TFLite model

Install compiler dependencies in the environment used for conversion:

```sh
.venv/bin/pip install '.[compiler]'
.venv/bin/unoq-dsp inspect model.tflite
.venv/bin/unoq-dsp compile model.tflite -o model.uqmodel
```

Or use Python:

```python
from unoq_dsp import compile_model
compile_model("model.tflite").save("model.uqmodel")
```

Compilation reads the TFLite graph, lowers operators, packs weights and quantization parameters, plans activation memory and writes a checksummed `.uqmodel` bundle. It runs offline; inference does not need LiteRT. Task metadata supplies preprocessing, labels and output decoding for the image APIs.

For tensor-level inference, launch with `./run python`:

```python
import numpy as np
from unoq_dsp import Session

with Session("model.uqmodel") as model:
    result = model.run(np.load("input.npy"))
```

## Source map

| Location | Purpose |
|---|---|
| `runtime/qnn.c`, `depthwise*.h` | INT8 spatial operators and requantization |
| `runtime/gemm.c`, `gemm*.h`, `gemm2x4.S`, `requant2.S` | Packed dot products, matrix multiplication and assembly leaves |
| `runtime/graph.c`, `graph.h` | Plan validation, allocation and execution dispatch |
| `runtime/graph_dsp.c`, `bridge.c` | DSP entry points, blocking RPC, shared buffers and cache synchronization |
| `src/unoq_dsp/tflite.py` | Graph lowering and operator selection |
| `src/unoq_dsp/memory.py` | Activation lifetime planning |
| `src/unoq_dsp/bundle.py` | Versioned model container and integrity checks |
| `src/unoq_dsp/modern.py`, `vision.py` | Image preprocessing and prediction decoding |
| `tools/export/` | Model export, calibration and task packaging |

To add an operator, implement its kernel and graph descriptor/validation, add TFLite lowering, then compare its integer outputs with the TFLite reference kernel. Preserve quantization rounding, saturation and fused activation semantics. `tests/tflite_fixtures.py` builds small graphs for focused operator tests.

## Rebuild

On the UNO Q:

```sh
sudo apt-get install -y build-essential clang-19 lld-19 git \
  autoconf automake libtool pkg-config libyaml-dev libbsd-dev patchelf
./scripts/build-runtime.sh
./scripts/install.sh
```

The script fetches the pinned FastRPC source and builds the ARM64 bridge and Hexagon v66 module with the selected optimization flags. Binaries go into `prebuilt/`; checksums are updated for the new build. The kernel build uses LLVM and does not require Qualcomm’s proprietary compiler.
For a kernel-only change, run `./scripts/build-kernels.sh` and reinstall.

## Test

```sh
.venv/bin/pip install '.[compiler]'
.venv/bin/python -m unittest discover -s tests -v
```

To run execution tests on the DSP, install your rebuilt runtime and use:

```sh
./run env UNOQ_TEST_DSP=1 python -m unittest discover -s tests -v
```

Keep sessions open across images to reuse model state and transport allocations. Blocking RPC is the default for the public API and command-line tools.
