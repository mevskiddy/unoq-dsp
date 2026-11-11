"""Exhaust all 65,536 signed-byte pairs against independent LiteRT ADD."""
import ctypes as C
import os
from pathlib import Path
import tempfile
import unittest

import flatbuffers
import numpy as np
import tflite
from ai_edge_litert.interpreter import Interpreter, OpResolverType

from unoq_dsp import CompiledModel, Session
from unoq_dsp.tflite import lower
from unoq_dsp.memory import compact_plan


def add_model(xs, xz, bs, bz, ys, yz, act, multiple_outputs=False):
    b = flatbuffers.Builder(100000)
    def table(name, fields):
        getattr(tflite, name+"Start")(b)
        for key, value in fields.items():
            getattr(tflite, name+"Add"+key)(b, value)
        return getattr(tflite, name+"End")(b)
    def vector(name, values):
        getattr(tflite, name)(b, len(values))
        for value in reversed(values):
            b.PrependUOffsetTRelative(value)
        return b.EndVector()
    def vec(values, dtype):
        return b.CreateNumpyVector(np.asarray(values, dtype=dtype))
    def tensor(name, buffer, scale, zero):
        n = b.CreateString(name)
        sh = vec([1, 256, 256, 1], "i4")
        scales = vec([scale], "f4")
        zeros = vec([zero], "i8")
        q = table("QuantizationParameters", {"Scale": scales, "ZeroPoint": zeros})
        return table("Tensor", {"Name": n, "Shape": sh, "Type": 9, "Buffer": buffer, "Quantization": q})
    constants = np.tile(np.arange(-128, 128, dtype="i1"), 256)
    raw = b.CreateByteVector(constants.tobytes())
    buffers = [table("Buffer", {}), table("Buffer", {"Data": raw})]
    tensors = [tensor("x", 0, xs, xz), tensor("b", 1, bs, bz), tensor("y", 0, ys, yz)]
    options = table("AddOptions", {"FusedActivationFunction": act})
    inputs, outputs = vec([0, 1], "i4"), vec([2], "i4")
    op = table("Operator", {"Inputs": inputs, "Outputs": outputs,
                            "BuiltinOptionsType": tflite.BuiltinOptions.AddOptions,
                            "BuiltinOptions": options})
    tv = vector("SubGraphStartTensorsVector", tensors)
    ov = vector("SubGraphStartOperatorsVector", [op])
    iv, yv = vec([0], "i4"), vec([0, 2] if multiple_outputs else [2], "i4")
    graph = table("SubGraph", {"Tensors": tv, "Operators": ov, "Inputs": iv, "Outputs": yv})
    opcode = table("OperatorCode", {"BuiltinCode": tflite.BuiltinOperator.ADD,
                                    "DeprecatedBuiltinCode": tflite.BuiltinOperator.ADD, "Version": 2})
    gv = vector("ModelStartSubgraphsVector", [graph])
    cv = vector("ModelStartOperatorCodesVector", [opcode])
    bv = vector("ModelStartBuffersVector", buffers)
    model = table("Model", {"Version": 3, "Subgraphs": gv, "OperatorCodes": cv, "Buffers": bv})
    b.Finish(model, file_identifier=b"TFL3")
    return bytes(b.Output())


@unittest.skipUnless(os.environ.get("UNOQ_NATIVE_LIBRARY") or os.environ.get("UNOQ_TEST_DSP"), "Requires a native graph library or board")
class ResidualAddTests(unittest.TestCase):
    def test_all_byte_pairs_across_scales_zero_points_and_clamps(self):
        dsp = bool(os.environ.get("UNOQ_TEST_DSP"))
        if not dsp:
            lib = C.CDLL(os.environ["UNOQ_NATIVE_LIBRARY"])
            lib.ug_load.argtypes = [C.c_void_p, C.c_uint32]
            lib.ug_run.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint32]
        x = np.repeat(np.arange(-128, 128, dtype="i1"), 256).reshape(1, 256, 256, 1)
        cases = [(.0625, -27, .03125, 13, .125, -5),
                 (.0039215689, -128, .031, 127, .0078, 0),
                 (.017, 0, .019, -1, .011, 25),
                 (.5, 127, .0001, -128, .00001, -128)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"add.tflite"
            for spec in cases:
                for act in (0, 1, 3):
                    data = add_model(*spec, act)
                    path.write_bytes(data)
                    ref = Interpreter(model_content=data, experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
                    ref.allocate_tensors()
                    ref.set_tensor(0, x)
                    ref.invoke()
                    expected = ref.get_tensor(2)
                    plan, meta = lower(path)
                    for p, m in ((plan, meta), compact_plan(plan, meta)):
                        if dsp:
                            with Session(CompiledModel(p, m)) as session:
                                actual = session.run(x)
                        else:
                            buf = C.create_string_buffer(p)
                            self.assertEqual(lib.ug_load(buf, len(p)), 0)
                            actual = np.empty(x.shape, "i1")
                            self.assertEqual(lib.ug_run(x.ctypes.data, x.nbytes, actual.ctypes.data, actual.nbytes), 0)
                        np.testing.assert_array_equal(actual, expected, err_msg=str((spec, act)))


if __name__ == "__main__":
    unittest.main()
