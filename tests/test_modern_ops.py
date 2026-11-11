import ctypes as C
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tflite
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from unoq_dsp import compile_model, Session
from tflite_fixtures import single_op

@unittest.skipUnless(os.environ.get('UNOQ_NATIVE_LIBRARY') or os.environ.get('UNOQ_TEST_DSP'),'Requires execution backend')
class ModernOpsTests(unittest.TestCase):
    def compare(self,model,shape):
        ref=Interpreter(model_content=model,experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        ref.allocate_tensors()
        rng=np.random.default_rng(291)
        fixtures=[np.full(shape,v,'i1') for v in (-128,0,127)]+[rng.integers(-128,128,shape,dtype='i1')]
        if np.prod(shape)>=256:fixtures.append(np.resize(np.arange(-128,128,dtype='i1'),shape))
        if shape==(256,):fixtures.append(np.arange(-128,128,dtype='i1'))
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'m.tflite';p.write_bytes(model)
            for compact in (False,True):
                compiled=compile_model(p,reuse_memory=compact)
                if os.environ.get('UNOQ_TEST_DSP'):
                    session=Session(compiled);run=session.run
                else:
                    lib=C.CDLL(os.environ['UNOQ_NATIVE_LIBRARY']);lib.ug_load.argtypes=[C.c_void_p,C.c_uint32];lib.ug_run.argtypes=[C.c_void_p,C.c_uint32,C.c_void_p,C.c_uint32]
                    buf=C.create_string_buffer(compiled.plan);self.assertEqual(lib.ug_load(buf,len(compiled.plan)),0)
                    def run(x):
                        y=np.empty(compiled.metadata['output']['shape'],'i1')
                        self.assertEqual(lib.ug_run(x.ctypes.data,x.nbytes,y.ctypes.data,y.nbytes),0);return y
                    session=None
                try:
                    for x in fixtures:
                        ref.set_tensor(0,x);ref.invoke();np.testing.assert_array_equal(run(x),ref.get_tensor(ref.get_output_details()[0]['index']))
                finally:
                    if session:session.close()

    def test_multiply_scalar_equal_shape_and_saturation(self):
        shape=(2,3,5,7);rng=np.random.default_rng(36)
        for bsh in (shape,(1,)):
            b=rng.integers(-128,128,bsh,dtype='i1')
            for xs,bs,ys,xz,bz,yz in ((.01,.02,.03,-128,127,-7),(.5,.25,.031,31,-98,41),(.003,.009,.5,0,0,0)):
                for act in (0,1,3):
                    model=single_op(tflite.BuiltinOperator.MUL,'MulOptions',{'FusedActivationFunction':act},[(shape,9,None,xs,xz),(bsh,9,b,bs,bz),(shape,9,None,ys,yz)],[0,1],2)
                    self.compare(model,shape)

    def test_exact_fused_silu_all_256_codes(self):
        import flatbuffers
        from ai_edge_litert import schema_py_generated as schema
        for xs,xz,ys,yz in ((.03,-71,.07,-128),(.8,-127,.6,-125),(.001,17,.0009,21)):
            raw=single_op(tflite.BuiltinOperator.MUL,'MulOptions',{},[((256,),9,None,xs,xz),((256,),9,None,1/256,-128),((256,),9,None,ys,yz)],[0,1],2)
            model=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(raw,0));graph=model.subgraphs[0]
            code=schema.OperatorCodeT();code.builtinCode=code.deprecatedBuiltinCode=tflite.BuiltinOperator.LOGISTIC;code.version=2;model.operatorCodes.append(code)
            op=schema.OperatorT();op.opcodeIndex=1;op.inputs=np.array([0],'i4');op.outputs=np.array([1],'i4');graph.operators.insert(0,op)
            b=flatbuffers.Builder(2048);root=model.Pack(b);b.Finish(root,file_identifier=b'TFL3');self.compare(bytes(b.Output()),(256,))

    def test_exact_softmax_scales_depths_and_ties(self):
        for width in (1,2,3,17,100,256):
            for scale in (.001,.03,.17,1.,16.):
                shape=(3,width)
                model=single_op(tflite.BuiltinOperator.SOFTMAX,'SoftmaxOptions',{'Beta':1.},[(shape,9,None,scale,-17),(shape,9,None,1/256,-128)],[0],1)
                self.compare(model,shape)

    def test_signed_requantization_negative_ties_and_biases(self):
        shape=(1,1,256,8);weights=np.eye(8,dtype='i1').reshape(8,1,1,8);bias=np.arange(-4,4,dtype='i4')
        for exponent in (1,2,3,7,15):
            for fraction in (.5,.75,.999):
                scale=fraction/(2**exponent)
                for zero in (-17,0,77):
                    model=single_op(tflite.BuiltinOperator.CONV_2D,'Conv2DOptions',{'Padding':1,'StrideH':1,'StrideW':1,'DilationHFactor':1,'DilationWFactor':1},[(shape,9,None,1.,0),((8,1,1,8),9,weights,1.,0),((8,),2,bias,1.,0),(shape,9,None,1/scale,zero)],[0,1,2],3)
                    self.compare(model,shape)

    def test_transpose_ranks(self):
        for shape,perm in (((2,3),(1,0)),((2,3,5),(0,2,1)),((2,3,5,7),(0,3,1,2)),((2,3,5,7),(0,2,3,1))):
            out=tuple(shape[i] for i in perm)
            model=single_op(tflite.BuiltinOperator.TRANSPOSE,'TransposeOptions',{},[(shape,9,None,.03,-19),((len(perm),),2,np.array(perm,'i4'),None,0),(out,9,None,.03,-19)],[0,1],2)
            self.compare(model,shape)

    def test_slice_bounds_and_contiguous_spans(self):
        for shape,begin,size in (((11,),(3,),(-1,)),((2,3,5,7),(0,1,2,1),(2,2,3,5)),((1,2,128,100),(0,0,32,0),(1,2,32,100)),((2,3,5),(1,0,1),(1,-1,3))):
            out=tuple(s-b if n==-1 else n for s,b,n in zip(shape,begin,size));r=len(shape)
            model=single_op(tflite.BuiltinOperator.SLICE,'SliceOptions',{},[(shape,9,None,.02,15),((r,),2,np.array(begin,'i4'),None,0),((r,),2,np.array(size,'i4'),None,0),(out,9,None,.02,15)],[0,1,2],3)
            self.compare(model,shape)

    def test_batched_matmul_transposes_padding_and_tiles(self):
        rng=np.random.default_rng(98)
        for batches,m,k,n in ((2,7,11,9),(2,33,32,35),(2,64,100,100)):
            for ax,ay in ((False,False),(True,False),(False,True),(True,True)):
                ash=(batches,k,m) if ax else (batches,m,k);bsh=(batches,n,k) if ay else (batches,k,n)
                b=rng.integers(-128,128,bsh,dtype='i1');out=(batches,m,n)
                model=single_op(tflite.BuiltinOperator.BATCH_MATMUL,'BatchMatMulOptions',{'AdjX':ax,'AdjY':ay},[(ash,9,None,.006,2),(bsh,9,b,.05,-17),(out,9,None,.08,-19)],[0,1],2,version=2)
                self.compare(model,ash)

if __name__=='__main__':unittest.main()
