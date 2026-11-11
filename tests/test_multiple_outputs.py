import ctypes as C
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

from unoq_dsp import compile_model, Session, load, InvalidBundle
from test_residual_add import add_model


@unittest.skipUnless(os.environ.get("UNOQ_NATIVE_LIBRARY") or os.environ.get("UNOQ_TEST_DSP"), "Requires native graph library or board")
class MultipleOutputsTests(unittest.TestCase):
    def test_two_outputs_preserve_values_shapes_order_and_lifetimes(self):
        data = add_model(.017, 0, .019, -1, .011, 25, 0, multiple_outputs=True)
        ref = Interpreter(model_content=data, experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        ref.allocate_tensors()
        x = np.repeat(np.arange(-128, 128, dtype="i1"), 256).reshape(1, 256, 256, 1)
        ref.set_tensor(0, x)
        ref.invoke()
        expected = {"x": ref.get_tensor(0), "y": ref.get_tensor(2)}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"test.tflite"
            path.write_bytes(data)
            bundle = Path(temp)/"test.uqmodel"
            for compact in (False, True):
                model = compile_model(path, reuse_memory=compact)
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
                    y = np.empty(model.metadata["output"]["shape"], "i1")
                    self.assertEqual(lib.ug_run(x.ctypes.data, x.nbytes, y.ctypes.data, y.nbytes), 0)
                    actual = {s["name"]: y[s["packed_offset"]:s["packed_offset"]+s["bytes"]].reshape(s["shape"])
                              for s in model.metadata["outputs"]}
                self.assertEqual(list(actual), ["x", "y"])
                for name in expected:
                    np.testing.assert_array_equal(actual[name], expected[name])


if __name__ == "__main__":
    unittest.main()
