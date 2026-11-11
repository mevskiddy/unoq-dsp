"""Bounds-checked access to inline and appended TFLite constant buffers."""
import numpy as np
import flatbuffers.number_types as number_types
from .errors import UnsupportedModel


def builtin_code(opcode):
    # The standalone tflite package's accessor ignores the actual int32 field
    # below 127. Modern exporters can omit the deprecated field entirely.
    offset=opcode._tab.Offset(10)
    current=opcode._tab.Get(number_types.Int32Flags,opcode._tab.Pos+offset) if offset else 0
    return max(current,opcode.DeprecatedBuiltinCode())


def buffer_data(data, model, index):
    if not 0 <= index < model.BuffersLength():
        raise UnsupportedModel('Invalid TFLite buffer index')
    buffer=model.Buffers(index)
    offset,size=int(buffer.Offset()),int(buffer.Size())
    if offset or size:
        if buffer.DataLength() or offset<8 or size<=0 or offset>len(data) or size>len(data)-offset:
            raise UnsupportedModel('Invalid appended TFLite buffer bounds')
        return np.frombuffer(data,dtype='u1',count=size,offset=offset)
    raw=buffer.DataAsNumpy()
    return np.empty(0,dtype='u1') if isinstance(raw,int) else raw
