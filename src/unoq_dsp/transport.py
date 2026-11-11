"""Small hardware-only session; importing it does not load ONNX or LiteRT."""
from pathlib import Path
import ctypes as C
import os
import threading

import numpy as np

from .errors import DSPError


class HardwareSession:
    """One active model per process; calls are serialized, close is idempotent.

    The board must already have the DSP module, FastRPC listener and permissions
    configured. Launch applications with the repository's ./run wrapper.
    """

    def _open_plan(self, plan, metadata, library, pipeline):
        if pipeline not in ("mailbox", "rpc"):
            raise ValueError("pipeline must be mailbox or rpc")
        self.metadata = metadata
        self.lock = threading.Lock()
        self.handle = None
        configured = library or os.environ.get("UNOQ_DSP_LIBRARY")
        if configured is None:
            candidate = Path("/opt/unoq-dsp/lib/libuno_backend.so")
            if not candidate.is_file():
                raise DSPError("Set UNOQ_DSP_LIBRARY to the board's libuno_backend.so")
            configured = candidate
        try:
            self.lib = C.CDLL(str(configured))
        except OSError as exc:
            raise DSPError(f"Cannot load DSP host library: {exc}") from exc
        self.lib.ug_open_mode.argtypes = [C.c_void_p, C.c_uint32, C.c_int, C.POINTER(C.c_int)]
        self.lib.ug_open_mode.restype = C.c_void_p
        self.lib.ug_infer.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint32]
        self.lib.ug_infer.restype = C.c_int
        self.lib.ug_close.argtypes = [C.c_void_p]
        self.lib.ug_close.restype = None
        error = C.c_int()
        buf = C.create_string_buffer(plan)
        self.handle = self.lib.ug_open_mode(buf, len(plan), int(pipeline == "mailbox"), C.byref(error))
        if not self.handle:
            code = error.value & 0xffffffff
            detail = " (DSP module unavailable)" if code == 0x80000406 else ""
            raise DSPError(f"DSP model load failed: {code:#010x}{detail}; no CPU fallback")

    def run(self, x):
        with self.lock:
            if not self.handle:
                raise DSPError("Session closed")
            spec = self.metadata["input"]
            x = np.asarray(x)
            if x.shape != tuple(spec["shape"]) or x.dtype != np.dtype(spec["dtype"]):
                raise ValueError(f"Expected {tuple(spec['shape'])} {spec['dtype']}")
            x = np.ascontiguousarray(x)
            spec = self.metadata["output"]
            y = np.empty(spec["shape"], dtype=spec["dtype"])
            rc = self.lib.ug_infer(self.handle, x.ctypes.data, x.nbytes, y.ctypes.data, y.nbytes)
            if rc:
                raise DSPError(f"DSP inference failed: {rc & 0xffffffff:#010x}; no CPU fallback")
            if self.metadata.get("outputs"):
                outputs = {}
                wire = y.reshape(-1).view("u1")
                for spec in self.metadata["outputs"]:
                    value = wire[spec["packed_offset"]:spec["packed_offset"]+spec["bytes"]].view(spec["dtype"]).reshape(spec["shape"])
                    if spec.get("dequantize"):
                        q = spec["quantization"]
                        value = (value.astype("f4")-np.float32(q["zero_point"]))*np.float32(q["scale"])
                    outputs[spec["name"]] = value
                return next(iter(outputs.values())) if len(outputs)==1 else outputs
            return y

    def close(self):
        with self.lock:
            if self.handle:
                self.lib.ug_close(self.handle)
                self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
