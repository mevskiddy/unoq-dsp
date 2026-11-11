"""Exact finite-domain compilation for byte-wise TFLite operations.

The pinned reference runtime is used ONLY while compiling 256-entry tables.
Compiled inference executes the tables on DSP and does not import LiteRT.
This preserves the reference's integer approximations for sigmoid/tanh/h-swish.
"""
from functools import lru_cache
from importlib.metadata import version as package_version
import flatbuffers
import numpy as np
import tflite

from .errors import UnsupportedModel

UNARY_NAMES = ("QUANTIZE", "LOGISTIC", "TANH", "HARD_SWISH", "RELU", "RELU6")
UNARY_CODES = {getattr(tflite.BuiltinOperator, name) for name in UNARY_NAMES}


@lru_cache(maxsize=256)
def lookup_table(code, version, input_type, xs, xz, output_type, ys, yz):
    if code not in UNARY_CODES or input_type not in (3, 9) or output_type not in (3, 9):
        raise UnsupportedModel("Unsupported unary table contract")
    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
    except ImportError as exc:
        raise UnsupportedModel("This model requires the compiler extra: pip install 'unoq-dsp[compiler]'; compiled inference does not require LiteRT") from exc
    if package_version("ai-edge-litert") != "2.3.0":
        raise UnsupportedModel("Unary compilation requires the validated ai-edge-litert==2.3.0")
    b = flatbuffers.Builder(1024)
    def table(name, fields):
        getattr(tflite, name+"Start")(b)
        for key, value in fields.items():
            getattr(tflite, name+"Add"+key)(b, value)
        return getattr(tflite, name+"End")(b)
    def offsets(name, values):
        getattr(tflite, name)(b, len(values))
        for value in reversed(values):
            b.PrependUOffsetTRelative(value)
        return b.EndVector()
    def vec(values, dtype):
        return b.CreateNumpyVector(np.asarray(values, dtype=dtype))
    tensors = []
    for typ, scale, zero in ((input_type, xs, xz), (output_type, ys, yz)):
        shape = vec([256], "i4")
        sv, zv = vec([scale], "f4"), vec([zero], "i8")
        q = table("QuantizationParameters", {"Scale": sv, "ZeroPoint": zv})
        tensors.append(table("Tensor", {"Shape": shape, "Type": typ, "Quantization": q}))
    iv, ov = vec([0], "i4"), vec([1], "i4")
    op = table("Operator", {"Inputs": iv, "Outputs": ov})
    tv = offsets("SubGraphStartTensorsVector", tensors)
    ops = offsets("SubGraphStartOperatorsVector", [op])
    iv, ov = vec([0], "i4"), vec([1], "i4")
    graph = table("SubGraph", {"Tensors": tv, "Operators": ops, "Inputs": iv, "Outputs": ov})
    opcode = table("OperatorCode", {"BuiltinCode": code, "DeprecatedBuiltinCode": min(code, 127), "Version": version})
    buffer = table("Buffer", {})
    gv = offsets("ModelStartSubgraphsVector", [graph])
    cv = offsets("ModelStartOperatorCodesVector", [opcode])
    bv = offsets("ModelStartBuffersVector", [buffer])
    model = table("Model", {"Version": 3, "Subgraphs": gv, "OperatorCodes": cv, "Buffers": bv})
    b.Finish(model, file_identifier=b"TFL3")
    try:
        interpreter = Interpreter(model_content=bytes(b.Output()), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        interpreter.allocate_tensors()
        domain = np.arange(-128, 128, dtype="i1") if input_type == 9 else np.arange(256, dtype="u1")
        interpreter.set_tensor(0, domain)
        interpreter.invoke()
        return interpreter.get_tensor(1).view("u1").tobytes()
    except (ValueError, RuntimeError) as exc:
        raise UnsupportedModel(f"Reference could not prepare unary lookup: {exc}") from exc
