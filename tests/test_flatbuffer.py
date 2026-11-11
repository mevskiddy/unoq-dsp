import unittest
import flatbuffers
import tflite
import numpy as np
from unoq_dsp.flatbuffer import builtin_code,buffer_data
from unoq_dsp.errors import UnsupportedModel

class FlatbufferTests(unittest.TestCase):
 def opcode(self,current,deprecated):
  b=flatbuffers.Builder(64);tflite.OperatorCodeStart(b)
  if current is not None:tflite.OperatorCodeAddBuiltinCode(b,current)
  if deprecated is not None:tflite.OperatorCodeAddDeprecatedBuiltinCode(b,deprecated)
  op=tflite.OperatorCodeEnd(b);b.Finish(op);return tflite.OperatorCode.GetRootAsOperatorCode(bytes(b.Output()),0)
 def test_modern_and_deprecated_opcode_fields(self):
  for current,deprecated,expected in ((114,None,114),(None,114,114),(3,3,3),(0,0,0),(150,127,150)):
   self.assertEqual(builtin_code(self.opcode(current,deprecated)),expected)
 def model(self,offset,size,inline=None):
  b=flatbuffers.Builder(128);data=b.CreateByteVector(inline) if inline is not None else None
  tflite.BufferStart(b);tflite.BufferAddOffset(b,offset);tflite.BufferAddSize(b,size)
  if data is not None:tflite.BufferAddData(b,data)
  buf=tflite.BufferEnd(b);tflite.ModelStartBuffersVector(b,1);b.PrependUOffsetTRelative(buf);v=b.EndVector()
  tflite.ModelStart(b);tflite.ModelAddBuffers(b,v);root=tflite.ModelEnd(b);b.Finish(root)
  data=bytes(b.Output());data+=b'\0'*(256-len(data))+b'abcd'
  return data,tflite.Model.GetRootAsModel(data,0)
 def test_appended_constant_and_bounds(self):
  data,m=self.model(256,4);self.assertEqual(buffer_data(data,m,0).tobytes(),b'abcd')
  for offset,size,inline in ((256,5,None),(1,4,None),(256,0,None),(0,4,None),(256,4,b'z')):
   data,m=self.model(offset,size,inline)
   with self.assertRaises(UnsupportedModel):buffer_data(data,m,0)
  with self.assertRaises(UnsupportedModel):buffer_data(data,m,1)

if __name__=='__main__':unittest.main()
