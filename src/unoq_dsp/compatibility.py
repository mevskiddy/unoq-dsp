"""Read-only model inventory and honest compatibility checks, without hardware.

An operator name alone never establishes support: the actual lowerer must also
accept shapes, options, quantization, accumulation bounds and graph limits.
"""
from collections import Counter
from pathlib import Path
import hashlib
import math

import numpy as np
import tflite

from .errors import UnsupportedModel
from .flatbuffer import buffer_data, builtin_code

SUPPORTED = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "RESHAPE",
             "AVERAGE_POOL_2D", "MAX_POOL_2D", "SOFTMAX", "ADD",
             "CONCATENATION", "RESIZE_NEAREST_NEIGHBOR", "QUANTIZE", "LOGISTIC",
             "TANH", "HARD_SWISH", "RELU", "RELU6", "DEQUANTIZE", "PAD", "MEAN",
             "MUL", "SLICE", "TRANSPOSE", "BATCH_MATMUL"}
OP_NAMES = {v: k for k, v in vars(tflite.BuiltinOperator).items() if isinstance(v, int)}
TYPE_NAMES = {v: k for k, v in vars(tflite.TensorType).items() if isinstance(v, int)}
WIDTHS = {0: 4, 1: 2, 2: 4, 3: 1, 4: 8, 6: 1, 7: 2, 9: 1, 10: 8, 15: 4, 16: 2}


def read_model(path):
    with Path(path).open("rb") as f:
        data = f.read(16*1024*1024+1)
    if len(data) > 16*1024*1024:
        raise UnsupportedModel("TFLite input exceeds 16 MiB parser limit")
    if len(data) < 8 or data[4:8] != b"TFL3":
        raise UnsupportedModel("Expected a TFLite flatbuffer")
    model = tflite.Model.GetRootAsModel(data, 0)
    if model.Version() != 3 or model.SubgraphsLength() != 1:
        raise UnsupportedModel("Schema version 3 and exactly one subgraph required")
    graph = model.Subgraphs(0)
    if not 0 < graph.TensorsLength() <= 10000 or not 0 < graph.OperatorsLength() <= 10000:
        raise UnsupportedModel("Invalid or excessive graph size")
    return data, model, graph


def inspect_model(path):
    """Return all missing operator families plus current compilation outcome.

    Memory figures are logical TFLite tensor sizes, not a promised DSP allocation.
    Packed weights, kernel scratch, alignment and transport buffers add overhead.
    """
    report = {"schema": 1, "source": str(path), "compilable": False,
              "hardware_validated": False, "cpu_nn_fallback": False,
              "issues": [], "notes": [], "operators": [], "inputs": [], "outputs": []}
    try:
        data, model, graph = read_model(path)
        report["sha256"] = hashlib.sha256(data).hexdigest()
        report["file_bytes"] = len(data)
        report["subgraph_tensors"] = graph.TensorsLength()
        specs = []
        constants = set()
        buffers = {}
        for i in range(graph.TensorsLength()):
            tensor = graph.Tensors(i)
            shape = [int(d) for d in tensor.ShapeAsNumpy()]
            if len(shape) > 8 or any(d <= 0 for d in shape):
                report["issues"].append(f"Tensor {i}: non-static or invalid shape {shape}")
            signature = tensor.ShapeSignatureAsNumpy()
            if not isinstance(signature, int) and np.any(signature < 0):
                report["notes"].append(f"Tensor {i}: dynamic signature frozen to allocated shape {shape}")
            if not 0 <= tensor.Buffer() < model.BuffersLength():
                raise UnsupportedModel(f"Tensor {i}: invalid buffer index")
            raw = buffer_data(data,model,tensor.Buffer())
            if len(raw):
                constants.add(i)
                buffers[tensor.Buffer()] = len(raw)
            q = tensor.Quantization()
            spec = {"index": i, "name": (tensor.Name() or b"").decode("utf8", "replace"),
                    "shape": shape, "dtype": TYPE_NAMES.get(tensor.Type(), str(tensor.Type())),
                    "logical_bytes": math.prod(shape)*WIDTHS.get(tensor.Type(), 0)}
            if q and q.ScaleLength():
                spec["quantization"] = {"scales": q.ScaleAsNumpy().tolist(),
                                        "zero_points": q.ZeroPointAsNumpy().tolist(),
                                        "axis": q.QuantizedDimension()}
            specs.append(spec)
        def indices(vector, optional=False):
            values = [int(i) for i in vector]
            if any(not (optional and i == -1) and not 0 <= i < len(specs) for i in values):
                raise UnsupportedModel("Invalid tensor index")
            return values
        inputs = indices(graph.InputsAsNumpy())
        outputs = indices(graph.OutputsAsNumpy())
        report["inputs"] = [specs[i] for i in inputs]
        report["outputs"] = [specs[i] for i in outputs]
        if len(inputs) != 1 or not 1 <= len(outputs) <= 32:
            report["issues"].append(f"Current compiler requires one input and 1..32 outputs; model has {len(inputs)}/{len(outputs)}")
        last_use = {i: graph.OperatorsLength() for i in outputs}
        operations = []
        for i in range(graph.OperatorsLength()):
            op = graph.Operators(i)
            if op.OpcodeIndex() >= model.OperatorCodesLength():
                raise UnsupportedModel("Invalid opcode index")
            code = model.OperatorCodes(op.OpcodeIndex())
            number=builtin_code(code)
            name = OP_NAMES.get(number, f"BUILTIN_{number}")
            if code.CustomCode():
                name = code.CustomCode().decode("utf8", "replace")
            ins = indices(op.InputsAsNumpy(), optional=True)
            outs = indices(op.OutputsAsNumpy())
            for j in ins:
                if j >= 0:
                    last_use[j] = max(i, last_use.get(j, -1))
            operations.append((ins, outs))
            report["operators"].append({"index": i, "name": name, "version": code.Version(),
                                        "kernel_available": name in SUPPORTED})
        counts = Counter(o["name"] for o in report["operators"])
        report["operator_counts"] = dict(sorted(counts.items()))
        for name in sorted(counts.keys()-SUPPORTED):
            nodes = [o["index"] for o in report["operators"] if o["name"] == name]
            report["issues"].append(f"Missing operator {name}: nodes {nodes}")
        # Lower bound for an out-of-place schedule, before allocator alignment.
        live = set(inputs)-constants
        peak = sum(specs[i]["logical_bytes"] for i in live)
        for index, (_, outs) in enumerate(operations):
            live.update(set(outs)-constants)
            peak = max(peak, sum(specs[i]["logical_bytes"] for i in live))
            live = {i for i in live if last_use.get(i, -1) > index}
        report["memory"] = {"constant_buffer_bytes": sum(buffers.values()),
                            "logical_peak_activation_bytes": peak,
                            "includes_dsp_packing_scratch_alignment": False}
        from .tflite import lower
        try:
            plan, meta = lower(path, reuse_memory=True)
        except UnsupportedModel as exc:
            report["issues"].append(str(exc))
        else:
            report["compilable"] = not report["issues"]
            report["compiled"] = {"plan_bytes": len(plan), "arena_bytes": meta["arena_bytes"],
                                  "macs_and_pool_work": meta["macs"], "dsp_threads": meta["dsp_threads"]}
    except (ValueError, TypeError, IndexError, OverflowError, AttributeError, RuntimeError) as exc:
        report["issues"].append(f"Model inspection failed: {exc}")
    # FlatBuffers reports many malformed lengths through struct.error.
    except __import__("struct").error as exc:
        report["issues"].append(f"Malformed flatbuffer: {exc}")
    return report
