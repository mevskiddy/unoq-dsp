import ctypes as C
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tflite
from ai_edge_litert.interpreter import Interpreter, OpResolverType

from unoq_dsp import compile_model, Session, load
from tflite_fixtures import single_op


@unittest.skipUnless(os.environ.get("UNOQ_NATIVE_LIBRARY") or os.environ.get("UNOQ_TEST_DSP"), "Requires native graph library or board")
class UnaryTests(unittest.TestCase):
    def test_all_byte_values_for_quantize_and_nonlinear_ops(self):
        cases = [("QUANTIZE", 2, 9, .031, -27, 9, .017, 0),
                 ("QUANTIZE", 2, 3, .0078125, 128, 9, .0078125, 0),
                 ("QUANTIZE", 2, 9, .031, -27, 3, .017, 128),
                 ("LOGISTIC", 2, 9, .061, 13, 9, 1/256, -128),
                 ("TANH", 2, 9, .019, -27, 9, 1/128, 0),
                 ("HARD_SWISH", 1, 9, .037, 31, 9, .041, -31),
                 ("RELU", 2, 9, .037, 31, 9, .041, -31),
                 ("RELU6", 2, 9, .037, 31, 9, .041, -31)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"model.tflite"
            bundle = Path(temp)/"model.uqmodel"
            for name, version, it, xs, xz, ot, ys, yz in cases:
                data = single_op(getattr(tflite.BuiltinOperator, name), None, {},
                                 [((2, 256), it, None, xs, xz), ((2, 256), ot, None, ys, yz)], [0], 1, version=version)
                path.write_bytes(data)
                domain = np.arange(-128, 128, dtype="i1") if it == 9 else np.arange(256, dtype="u1")
                x = np.stack((domain, domain[::-1])).copy()
                ref = Interpreter(model_content=data, experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
                ref.allocate_tensors()
                ref.set_tensor(0, x)
                ref.invoke()
                expected = ref.get_tensor(1)
                model = compile_model(path)
                model.save(bundle)
                model = load(bundle)
                if os.environ.get("UNOQ_TEST_DSP"):
                    with Session(model) as session:
                        actual = session.run(x)
                else:
                    lib = C.CDLL(os.environ["UNOQ_NATIVE_LIBRARY"])
                    lib.ug_load.argtypes = [C.c_void_p, C.c_uint32]
                    lib.ug_run.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint32]
                    buf = C.create_string_buffer(model.plan)
                    self.assertEqual(lib.ug_load(buf, len(model.plan)), 0)
                    actual = np.empty(expected.shape, expected.dtype)
                    self.assertEqual(lib.ug_run(x.ctypes.data, x.nbytes, actual.ctypes.data, actual.nbytes), 0)
                np.testing.assert_array_equal(actual, expected, err_msg=name)


if __name__ == "__main__":
    unittest.main()
