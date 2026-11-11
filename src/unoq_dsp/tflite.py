"""Bounded signed-INT8 TFLite vision importer; no CPU inference fallback.
Uses the existing resident DSP transport. Quantized semantics are distinct from
ONNX QLinearConv (which has a different rounding contract).
"""
from pathlib import Path
import math,struct
import numpy as np
import tflite
from .errors import UnsupportedModel
from .flatbuffer import buffer_data, builtin_code
from .transport import HardwareSession as BaseSession
from .quantization import add_parameters, multiplier, multiply
from .unary import UNARY_CODES, lookup_table
from .limits import MAX_TENSORS, MAX_OPERATORS, ARENA_BYTES, MAX_WORK, MAX_CHANNELS


def lower(path, *, scalar=False, dsp_threads=2, reuse_memory=False):
    if dsp_threads not in (1,2):raise ValueError('Expected one or two DSP threads')
    with Path(path).open('rb') as f:data=f.read(16*1024*1024+1)
    if len(data)>16*1024*1024:raise UnsupportedModel('TFLite input exceeds 16 MiB parser limit')
    if data[4:8]!=b'TFL3':raise UnsupportedModel('Expected a TFLite flatbuffer')
    m=tflite.Model.GetRootAsModel(data,0)
    if m.Version()!=3:raise UnsupportedModel('TFLite schema version 3 required')
    if m.SubgraphsLength()!=1:raise UnsupportedModel('Exactly one subgraph required')
    g=m.Subgraphs(0)
    for i in range(g.TensorsLength()):
        t=g.Tensors(i)
        if t.IsVariable() or t.Sparsity() is not None:raise UnsupportedModel('Variable and sparse tensors are unsupported')
    if g.InputsLength()!=1 or not 1<=g.OutputsLength()<=32:raise UnsupportedModel('One input and 1..32 outputs required')
    model_outputs=[int(i) for i in g.OutputsAsNumpy()]
    arena=bytearray();ts=[];ops=[];ids={};snapshots={};work=0;dequantized={}
    def tensor(i):return g.Tensors(int(i))
    def shape(i):return tuple(int(x) for x in tensor(i).ShapeAsNumpy())
    def quant(i):
        q=tensor(i).Quantization()
        if q is None:raise UnsupportedModel('Quantization parameters required')
        scales=np.asarray(q.ScaleAsNumpy(),dtype=np.float64);zeros=np.asarray(q.ZeroPointAsNumpy(),dtype=np.int64)
        if scales.size==0 or not np.all(np.isfinite(scales)) or np.any(scales<=0) or scales.size!=zeros.size:raise UnsupportedModel('Invalid quantization')
        return scales,zeros
    def array(i,dtype):
        raw=buffer_data(data,m,tensor(i).Buffer())
        if not raw.size:raise UnsupportedModel('Constant tensor required')
        return np.frombuffer(raw.tobytes(),dtype=dtype).reshape(shape(i)).copy()
    def add(name,sh,dtype,values=None):
        dt=np.dtype(dtype);size=math.prod(sh)*dt.itemsize
        if not 0<size<=4*1024*1024 or len(sh)>4 or any(d<=0 for d in sh):raise UnsupportedModel('Invalid tensor size')
        off=(len(arena)+127)&~127
        if len(ts)>=MAX_TENSORS or off+size>(64*1024*1024 if reuse_memory else ARENA_BYTES):raise UnsupportedModel('Bounded tensor/arena limit exceeded')
        arena.extend(bytes(off-len(arena)));arena.extend(bytes(size) if values is None else np.asarray(values,dtype=dt).tobytes())
        typ={np.dtype('i1'):3,np.dtype('u1'):2,np.dtype('<i4'):4}[dt];i=len(ts);ts.append(dict(name=name,shape=sh,dtype=dt,type=typ,offset=off,bytes=size,constant=values is not None));return i
    def activation(i):
        if i not in ids:
            if tensor(i).Type() not in (tflite.TensorType.INT8,tflite.TensorType.UINT8):raise UnsupportedModel('Only INT8/UINT8 activations supported')
            dtype='i1' if tensor(i).Type()==tflite.TensorType.INT8 else 'u1'
            raw=buffer_data(data,m,tensor(i).Buffer())
            values=None if len(raw)==0 else array(i,dtype)
            ids[i]=add(tensor(i).Name().decode(),shape(i),dtype,values)
        return ids[i]
    def opts(o,cls):
        obj=cls();b=o.BuiltinOptions();obj.Init(b.Bytes,b.Pos);return obj
    def limits(act,scale,zero):
        def q(v):return int(math.floor(v/scale+0.5))+zero
        if act==0:return -128,127
        if act==1:return max(-128,zero),127
        if act==3:return max(-128,zero),min(127,q(6))
        raise UnsupportedModel(f'Unsupported fused activation {act}')
    inp=int(g.Inputs(0));out=int(g.Outputs(0));input_id=activation(inp)
    consumers={}; producers={}; fused_pad={}; fused_sigmoid={}; skip=set()
    for i in range(g.OperatorsLength()):
        op=g.Operators(i)
        for src in op.InputsAsNumpy():consumers.setdefault(int(src),[]).append(i)
        for dst in op.OutputsAsNumpy():producers[int(dst)]=i
    for i in range(g.OperatorsLength()):
        op=g.Operators(i);code=builtin_code(m.OperatorCodes(op.OpcodeIndex()))
        src=int(op.Inputs(0));dst=int(op.Outputs(0));users=consumers.get(dst,[])
        if dst in model_outputs or len(users)!=1:continue
        user=g.Operators(users[0]);uc=builtin_code(m.OperatorCodes(user.OpcodeIndex()))
        if code==tflite.BuiltinOperator.PAD and uc in (tflite.BuiltinOperator.CONV_2D,tflite.BuiltinOperator.DEPTHWISE_CONV_2D) and int(user.Inputs(0))==dst:
            pads=array(int(op.Inputs(1)),'<i4')
            if pads.shape==(4,2) and np.all(pads>=0) and np.all(pads<=4096) and not np.any(pads[[0,3]]) and tuple(np.array(shape(src))+pads.sum(axis=1))==shape(dst) and all(np.array_equal(a,b) for a,b in zip(quant(src),quant(dst))):
                fused_pad[users[0]]=(src,pads);skip.add(i)
        if code==tflite.BuiltinOperator.LOGISTIC and uc==tflite.BuiltinOperator.MUL and sorted(int(x) for x in user.InputsAsNumpy())==sorted([src,dst]) and shape(src)==shape(dst):
            fused_sigmoid[users[0]]=(src,dst,op);skip.add(i)
    def emit_lookup(ai,a,zi,b,index):
        # Only fuse when the pre-activation value has no other observable user.
        expected_users=2 if index in fused_sigmoid else 1
        if ops and ops[-1][0] in (14,15) and ops[-1][3]==a and ops[-1][5]==0 and ai not in model_outputs and len(consumers.get(ai,[]))==expected_users:
            old=ops[-1];ops[-1]=(*old[:5],b+1,0,0);ids[zi]=a
            ts[a]['name']=tensor(zi).Name().decode();snapshots.pop(ai,None)
            return a
        z=activation(zi);ops.append((22,a,b,z,0,0,0,0));return z
    for index in range(g.OperatorsLength()):
        if index in skip:continue
        o=g.Operators(index);code=builtin_code(m.OperatorCodes(o.OpcodeIndex()));ins=[int(i) for i in o.InputsAsNumpy()];outs=[int(i) for i in o.OutputsAsNumpy()]
        if len(outs)!=1:raise UnsupportedModel('One output per op required')
        ai,zi=ins[0],outs[0]
        if index in fused_pad:ai=fused_pad[index][0]
        if index in fused_sigmoid:ai=fused_sigmoid[index][0]
        a=activation(ai);xs,xz=quant(ai)
        if code==tflite.BuiltinOperator.DEQUANTIZE:
            if len(ins)!=1 or zi not in model_outputs or shape(ai)!=shape(zi) or tensor(zi).Type()!=tflite.TensorType.FLOAT32:raise UnsupportedModel('Only terminal FLOAT32 dequantization is supported')
            if xs.size!=1:raise UnsupportedModel('Output dequantization requires per-tensor scale')
            ids[zi]=a;dequantized[zi]=ai;snapshots[zi]=a;continue
        ys,yz=quant(zi)
        if code in UNARY_CODES:
            if len(ins)!=1 or shape(ai)!=shape(zi) or xs.size!=1 or ys.size!=1:raise UnsupportedModel('Unary lookup requires equal shapes and per-tensor quantization')
            for source,zero in ((ai,xz[0]),(zi,yz[0])):
                low,high=(-128,127) if tensor(source).Type()==tflite.TensorType.INT8 else (0,255)
                if not low<=zero<=high:raise UnsupportedModel('Invalid unary zero point')
            table=lookup_table(code,m.OperatorCodes(o.OpcodeIndex()).Version(),tensor(ai).Type(),float(xs[0]),int(xz[0]),tensor(zi).Type(),float(ys[0]),int(yz[0]))
            b=add(f'lookup-{index}',(256,),'u1',np.frombuffer(table,dtype='u1'));z=emit_lookup(ai,a,zi,b,index)
            work+=math.prod(shape(zi));snapshots[zi]=z;continue
        if tensor(ai).Type()!=tflite.TensorType.INT8 or tensor(zi).Type()!=tflite.TensorType.INT8:raise UnsupportedModel('This operation requires signed INT8 activations; UINT8 boundaries require QUANTIZE')
        if xs.size!=1 or ys.size!=1 or not (-128<=xz[0]<=127 and -128<=yz[0]<=127):raise UnsupportedModel('Per-tensor INT8 activations required')
        if code==tflite.BuiltinOperator.RESHAPE:
            if math.prod(shape(ai))!=math.prod(shape(zi)) or not np.array_equal(xs,ys) or not np.array_equal(xz,yz):raise UnsupportedModel('Invalid reshape')
            z=activation(zi);ops.append((5,a,a,z,0,0,0,0));snapshots[zi]=z;continue
        q=[1,1,1,1,1,1,0,0,0,0,1,1,int(xz[0]),int(yz[0]),-128,127,1,1,8,0]
        if code in (tflite.BuiltinOperator.CONV_2D,tflite.BuiltinOperator.DEPTHWISE_CONV_2D,tflite.BuiltinOperator.FULLY_CONNECTED):
            depth=code==tflite.BuiltinOperator.DEPTHWISE_CONV_2D;fc=code==tflite.BuiltinOperator.FULLY_CONNECTED
            weights=array(ins[1],'i1');ws,wz=quant(ins[1])
            if tensor(ins[1]).Type()!=tflite.TensorType.INT8 or np.any(wz):raise UnsupportedModel('Symmetric INT8 weights required')
            if fc:
                p=opts(o,tflite.FullyConnectedOptions)
                if p.WeightsFormat()!=0 or p.KeepNumDims() or len(shape(zi))!=2 or len(weights.shape)!=2:raise UnsupportedModel('Unsupported FC layout')
                batch,oc=shape(zi);ic=weights.shape[1]
                if math.prod(shape(ai))!=batch*ic:raise UnsupportedModel('Invalid flattened FC input')
                kh=kw=1;ash=(batch,1,1,ic);zsh=(batch,1,1,oc)
                aa=add(f'fc-input-{index}',ash,'i1');ops.append((5,a,a,aa,0,0,0,0));a=aa;weights=weights.reshape(oc,1,1,ic)
                q[10:12]=[1,1]
            else:
                p=opts(o,tflite.DepthwiseConv2DOptions if depth else tflite.Conv2DOptions)
                if depth and p.DepthMultiplier()!=1:raise UnsupportedModel('Depth multiplier must be one')
                ash=shape(ai);zsh=shape(zi)
                if len(ash)!=4 or len(zsh)!=4:raise UnsupportedModel('NHWC rank four required')
                batch,ih,iw,ic=ash;_,oh,ow,oc=zsh;kh,kw=weights.shape[1:3]
                q[2:6]=[p.StrideH(),p.StrideW(),p.DilationHFactor(),p.DilationWFactor()];q[10:12]=[oh,ow]
                if p.Padding()==tflite.Padding.SAME:
                    original=shape(ins[0]);ph=max(0,(oh-1)*q[2]+(kh-1)*q[4]+1-original[1]);pw=max(0,(ow-1)*q[3]+(kw-1)*q[5]+1-original[2]);q[6:10]=[ph//2,pw//2,ph-ph//2,pw-pw//2]
                elif p.Padding()!=tflite.Padding.VALID:raise UnsupportedModel('Unsupported padding')
                if index in fused_pad:
                    pads=fused_pad[index][1];q[6:10]=[q[6]+int(pads[1,0]),q[7]+int(pads[2,0]),q[8]+int(pads[1,1]),q[9]+int(pads[2,1])]
            if weights.shape!=((1,kh,kw,oc) if depth else (oc,kh,kw,ic)) or ws.size not in (1,oc):raise UnsupportedModel('Invalid weight shape/scales')
            if ws.size>1 and tensor(ins[1]).Quantization().QuantizedDimension()!=(3 if depth else 0):raise UnsupportedModel('Invalid weight quantization axis')
            q[:2]=[kh,kw];q[14:16]=limits(p.FusedActivationFunction(),float(ys[0]),int(yz[0]));q[16:18]=[ic,oc]
            k=kh*kw*(1 if depth else ic);kp=(k+7)&~7;q[18]=kp;q[19]=int(scalar)
            if kp>8192 or max(ic,oc)>MAX_CHANNELS:raise UnsupportedModel('Kernel exceeds bounded dimensions')
            bias=np.zeros(oc,dtype='i4') if len(ins)<3 or ins[2]<0 else array(ins[2],'<i4')
            if len(ins)>=3 and ins[2]>=0:
                bs,bz=quant(ins[2])
                if tensor(ins[2]).Type()!=tflite.TensorType.INT32 or bias.shape!=(oc,) or np.any(bz) or not np.allclose(bs,xs[0]*ws,rtol=1e-5,atol=0):raise UnsupportedModel('Invalid bias quantization')
            if depth:
                sums=weights.reshape(k,oc).astype('i8').sum(axis=0);packed=weights.reshape(-1)
            else:
                w=weights.reshape(oc,k).T;pw=np.pad(w,((0,kp-k),(0,(-oc)%4)));packed=pw.reshape(kp//8,8,-1,4).transpose(2,0,3,1).copy().reshape(-1);sums=w.astype('i8').sum(axis=0)
            corrected=bias.astype('i8')-(int(xz[0])+128)*sums;mult=[];shift=[]
            for value in np.broadcast_to(xs[0]*ws/ys[0],(oc,)):
                significand,exponent=math.frexp(float(value));integer=int(math.floor(significand*(1<<31)+0.5))
                if integer==1<<31:integer//=2;exponent+=1
                if exponent < -31:integer=0;exponent=0
                if exponent>15:raise UnsupportedModel('Quantization shift too large')
                mult.append(integer);shift.append(exponent)
            # Prove the actual immutable weights, including zero-weight channels
            # whose quantized bias may legitimately approach INT32 limits.
            rows=weights.reshape(k,oc).T if depth else weights.reshape(oc,k)
            low=corrected+255*np.minimum(rows.astype('i8'),0).sum(axis=1)
            high=corrected+255*np.maximum(rows.astype('i8'),0).sum(axis=1)
            factor=np.left_shift(np.int64(1),np.maximum(shift,0))
            if np.any(corrected<-2147483648) or np.any(corrected>2147483647) or np.any(low*factor<-2147483648) or np.any(high*factor>2147483647):raise UnsupportedModel('INT32 accumulation/requantization bound exceeded')
            config=np.array(q+corrected.tolist()+mult+shift,dtype='<i4');bi=add(f'packed-{index}',(len(packed),),'i1',packed);qi=add(f'quant-{index}',(len(config),),'<i4',config)
            z=add(f'fc-output-{index}',zsh,'i1') if fc else activation(zi);ops.append((15 if depth else 14,a,bi,z,qi,0,0,0));work+=math.prod(zsh[:3])*oc*k
            if fc:
                final=activation(zi);ops.append((5,z,z,final,0,0,0,0));z=final
        elif code==tflite.BuiltinOperator.CONCATENATION:
            p=opts(o,tflite.ConcatenationOptions);rank=len(shape(ai));axis=p.Axis()
            if axis<0:axis+=rank
            if not 0<=axis<rank or not 2<=len(ins)<=32 or p.FusedActivationFunction()!=0:raise UnsupportedModel('Unsupported concatenation options')
            joined=list(shape(ai))
            for j,source in enumerate(ins):
                ss,sz=quant(source);sh=shape(source)
                if not np.array_equal(ss,ys) or not np.array_equal(sz,yz) or len(sh)!=rank or any(sh[d]!=shape(ai)[d] for d in range(rank) if d!=axis):raise UnsupportedModel('Concatenation requires compatible shapes and identical quantization')
                if j==0:continue
                joined[axis]+=sh[axis];b=activation(source)
                if j==len(ins)-1:
                    if tuple(joined)!=shape(zi):raise UnsupportedModel('Invalid concatenation output shape')
                    z=activation(zi)
                else:z=add(f'concat-{index}-{j}',tuple(joined),'i1')
                ops.append((20,a,b,z,axis,0,0,0));work+=math.prod(joined);a=z
        elif code==tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
            if len(ins)!=2 or len(shape(ai))!=4 or len(shape(zi))!=4 or not np.array_equal(xs,ys) or not np.array_equal(xz,yz):raise UnsupportedModel('Resize requires NHWC and unchanged quantization')
            p=opts(o,tflite.ResizeNearestNeighborOptions)
            if p.AlignCorners() and p.HalfPixelCenters():raise UnsupportedModel('Resize cannot combine align_corners and half_pixel_centers')
            if tensor(ins[1]).Type()!=tflite.TensorType.INT32 or not np.array_equal(array(ins[1],'<i4'),shape(zi)[1:3]):raise UnsupportedModel('Resize size must be a matching constant INT32 pair')
            n,ih,iw,c=shape(ai);on,oh,ow,oc=shape(zi)
            if (n,c)!=(on,oc):raise UnsupportedModel('Resize cannot change batch or channels')
            def positions(size,output):
                factor=np.float32((size-1)/(output-1) if p.AlignCorners() and output>1 else size/output)
                values=(np.arange(output,dtype='f4')+np.float32(.5 if p.HalfPixelCenters() else 0))*factor
                values=np.floor(values+.5) if p.AlignCorners() else np.floor(values)
                return np.clip(values,0,size-1).astype('<i4')
            positions_=np.concatenate((positions(ih,oh)*iw*c,positions(iw,ow)*c))
            b=add(f'resize-{index}',(oh+ow,),'<i4',positions_);z=activation(zi)
            ops.append((21,a,b,z,0,0,0,0));work+=math.prod(shape(zi))
        elif code==tflite.BuiltinOperator.ADD:
            if len(ins)!=2 or shape(ins[1])!=shape(ai) or shape(ai)!=shape(zi):raise UnsupportedModel('ADD currently requires equal input/output shapes')
            b=activation(ins[1]);bs,bz=quant(ins[1])
            if bs.size!=1 or not -128<=bz[0]<=127:raise UnsupportedModel('ADD requires per-tensor INT8 quantization')
            p=opts(o,tflite.AddOptions);lo,hi=limits(p.FusedActivationFunction(),float(ys[0]),int(yz[0]))
            parameters=add_parameters(float(xs[0]),int(xz[0]),float(bs[0]),int(bz[0]),float(ys[0]),int(yz[0]),lo,hi)
            qi=add(f'add-{index}',(len(parameters),),'<i4',parameters);z=activation(zi)
            ops.append((18,a,b,z,qi,0,0,0));work+=math.prod(shape(zi))
        elif code==tflite.BuiltinOperator.MUL:
            if len(ins)!=2 or shape(ai)!=shape(zi):raise UnsupportedModel('MUL requires equal shape or scalar RHS')
            p=opts(o,tflite.MulOptions);lo,hi=limits(p.FusedActivationFunction(),float(ys[0]),int(yz[0]))
            source=ins[1] if ins[0]==ai else ins[0]
            if index in fused_sigmoid:source=fused_sigmoid[index][1]
            bs,bz=quant(source)
            if bs.size!=1 or not -128<=bz[0]<=127 or tensor(source).Type()!=tflite.TensorType.INT8:raise UnsupportedModel('MUL requires per-tensor INT8')
            qm,qs=multiplier(float(xs[0]*bs[0]/ys[0]))
            if not -31<=qs<=15 or 65025*(1<<max(qs,0))>2147483520:raise UnsupportedModel('MUL requantization exceeds safe bounds')
            raw=buffer_data(data,m,tensor(source).Buffer())
            if index in fused_sigmoid or raw.size==1:
                domain=np.arange(-128,128,dtype='i8')
                if index in fused_sigmoid:
                    sig=fused_sigmoid[index][2]
                    values=np.frombuffer(lookup_table(tflite.BuiltinOperator.LOGISTIC,m.OperatorCodes(sig.OpcodeIndex()).Version(),9,float(xs[0]),int(xz[0]),9,float(bs[0]),int(bz[0])),dtype='i1').astype('i8')
                else:values=int(array(source,'i1').reshape(-1)[0])
                products=(domain-int(xz[0]))*(values-int(bz[0]))
                table=np.clip(multiply(products*(1<<max(qs,0)),qm,min(qs,0))+int(yz[0]),lo,hi).astype('i1').view('u1')
                b=add(f'mul-table-{index}',(256,),'u1',table);z=emit_lookup(ai,a,zi,b,index)
            else:
                if shape(source)!=shape(ai):raise UnsupportedModel('MUL broadcast shape unsupported')
                b=activation(source);qi=add(f'mul-{index}',(7,),'<i4',[int(xz[0]),int(bz[0]),int(yz[0]),qm,qs,lo,hi]);z=activation(zi);ops.append((25,a,b,z,qi,0,0,0))
            work+=math.prod(shape(zi))
        elif code==tflite.BuiltinOperator.TRANSPOSE:
            if len(ins)!=2 or tensor(ins[1]).Type()!=tflite.TensorType.INT32 or not np.array_equal(xs,ys) or not np.array_equal(xz,yz):raise UnsupportedModel('TRANSPOSE requires constant INT32 permutation and unchanged quantization')
            perm=array(ins[1],'<i4').reshape(-1).tolist();rank=len(shape(ai))
            if sorted(perm)!=list(range(rank)) or tuple(shape(ai)[d] for d in perm)!=shape(zi):raise UnsupportedModel('Invalid transpose permutation')
            z=activation(zi);ops.append((10,a,a,z,*(perm+[0]*(4-rank))));work+=math.prod(shape(zi))
        elif code==tflite.BuiltinOperator.SLICE:
            if len(ins)!=3 or any(tensor(j).Type()!=tflite.TensorType.INT32 for j in ins[1:]) or not np.array_equal(xs,ys) or not np.array_equal(xz,yz):raise UnsupportedModel('SLICE requires constant INT32 begin/size and unchanged quantization')
            begin=array(ins[1],'<i4').reshape(-1);size=array(ins[2],'<i4').reshape(-1);ash=np.array(shape(ai));rank=len(ash)
            if len(begin)!=rank or len(size)!=rank:raise UnsupportedModel('Invalid slice rank')
            size=np.where(size==-1,ash-begin,size)
            if np.any(begin<0) or np.any(size<=0) or np.any(begin+size>ash) or tuple(size)!=shape(zi):raise UnsupportedModel('Invalid slice bounds')
            z=activation(zi);ops.append((26,a,a,z,*(begin.tolist()+[0]*(4-rank))));work+=math.prod(shape(zi))
        elif code==tflite.BuiltinOperator.BATCH_MATMUL:
            if len(ins)!=2 or len(shape(ai))!=3 or len(shape(ins[1]))!=3 or len(shape(zi))!=3:raise UnsupportedModel('BATCH_MATMUL requires equal-batch rank-three inputs')
            p=opts(o,tflite.BatchMatMulOptions);bs,bz=quant(ins[1]);ash=shape(ai);bsh=shape(ins[1]);zsh=shape(zi)
            rows,k=(ash[2],ash[1]) if p.AdjX() else ash[1:];kb,cols=(bsh[2],bsh[1]) if p.AdjY() else bsh[1:]
            if bs.size!=1 or tensor(ins[1]).Type()!=tflite.TensorType.INT8 or not -128<=bz[0]<=127 or ash[0]!=bsh[0] or zsh!=(ash[0],rows,cols) or k!=kb or k>8192 or cols>2048:raise UnsupportedModel('Unsupported BATCH_MATMUL dimensions or quantization')
            qm,qs=multiplier(float(xs[0]*bs[0]/ys[0]))
            if not -31<=qs<=15 or k*65025*(1<<max(qs,0))>2147483520:raise UnsupportedModel('BATCH_MATMUL accumulation exceeds safe bounds')
            b=activation(ins[1]);qi=add(f'bmm-{index}',(7,),'<i4',[int(xz[0]),int(bz[0]),int(yz[0]),qm,qs,int(p.AdjX()),int(p.AdjY())]);z=activation(zi);ops.append((27,a,b,z,qi,0,0,0));work+=ash[0]*rows*cols*k
        elif code==tflite.BuiltinOperator.PAD:
            if len(ins)!=2 or len(shape(ai))!=4 or len(shape(zi))!=4 or not np.array_equal(xs,ys) or not np.array_equal(xz,yz) or tensor(ins[1]).Type()!=tflite.TensorType.INT32:raise UnsupportedModel('PAD requires NHWC INT8 with unchanged quantization')
            pads=array(ins[1],'<i4')
            if pads.shape!=(4,2) or np.any(pads<0) or np.any(pads>4096) or tuple(np.array(shape(ai))+pads.sum(axis=1))!=shape(zi):raise UnsupportedModel('Invalid constant padding')
            b=add(f'pad-{index}',(8,),'<i4',pads.reshape(-1));z=activation(zi)
            ops.append((23,a,b,z,int(xz[0])&0xffffffff,0,0,0));work+=math.prod(shape(zi))
        elif code==tflite.BuiltinOperator.MEAN:
            if len(ins)!=2 or len(shape(ai))!=4 or tensor(ins[1]).Type()!=tflite.TensorType.INT32:raise UnsupportedModel('MEAN requires constant spatial axes on NHWC INT8')
            axes=array(ins[1],'<i4').reshape(-1);axes=np.where(axes<0,axes+4,axes)
            if set(axes.tolist())!={1,2}:raise UnsupportedModel('Only spatial global MEAN is supported')
            p=opts(o,tflite.ReducerOptions);n,h,w,c=shape(ai)
            if shape(zi)!=((n,1,1,c) if p.KeepDims() else (n,c)):raise UnsupportedModel('Invalid global MEAN shape')
            count=h*w;qm,qs=multiplier(float(xs[0])/float(ys[0]))
            adjust=min(count.bit_length()-1,32,31+qs)
            if adjust<0:raise UnsupportedModel('MEAN scale underflow')
            qm=(qm<<adjust)//count;qs-=adjust
            if not -31<=qs<=30 or count*max(128+int(xz[0]),127-int(xz[0]))*(1<<max(qs,0))>2147483520:raise UnsupportedModel('MEAN accumulation exceeds safe bounds')
            b=add(f'mean-{index}',(5,),'<i4',[count,int(xz[0]),int(yz[0]),qm,qs]);z=activation(zi)
            ops.append((24,a,b,z,0,0,0,0));work+=math.prod(shape(ai))
        elif code in (tflite.BuiltinOperator.AVERAGE_POOL_2D,tflite.BuiltinOperator.MAX_POOL_2D):
            if not np.array_equal(xs,ys) or not np.array_equal(xz,yz):raise UnsupportedModel('Pooling quantization must be unchanged')
            p=opts(o,tflite.Pool2DOptions);batch,ih,iw,ic=shape(ai);_,oh,ow,oc=shape(zi)
            q[:4]=[p.FilterHeight(),p.FilterWidth(),p.StrideH(),p.StrideW()];q[10:12]=[oh,ow];q[16:18]=[ic,oc];q[19]=int(code==tflite.BuiltinOperator.MAX_POOL_2D);q[14:16]=limits(p.FusedActivationFunction(),float(ys[0]),int(yz[0]))
            if p.Padding()==tflite.Padding.SAME:
                ph=max(0,(oh-1)*q[2]+q[0]-ih);pw=max(0,(ow-1)*q[3]+q[1]-iw);q[6:10]=[ph//2,pw//2,ph-ph//2,pw-pw//2]
            elif p.Padding()!=tflite.Padding.VALID:raise UnsupportedModel('Unsupported pooling padding')
            qi=add(f'pool-{index}',(20,),'<i4',q);z=activation(zi);ops.append((16,a,a,z,qi,0,0,0));work+=math.prod(shape(zi))*q[0]*q[1]
        elif code==tflite.BuiltinOperator.SOFTMAX:
            p=opts(o,tflite.SoftmaxOptions)
            if shape(ai)!=shape(zi) or ys[0]!=1/256 or yz[0]!=-128 or p.Beta()<=0:raise UnsupportedModel('Unsupported softmax quantization')
            from .softmax import exp_table
            params=[shape(ai)[-1]]+exp_table(float(xs[0]),p.Beta());qi=add(f'softmax-{index}',(len(params),),'<i4',params);z=activation(zi);ops.append((28,a,qi,z,0,0,0,0));work+=math.prod(shape(ai))
        else:raise UnsupportedModel(f'Unsupported TFLite builtin {code}')
        snapshots[zi]=z
    for source, target in [(inp,input_id)]+[(dequantized.get(i,i),ids[i]) for i in model_outputs]:
        scales, zeros = quant(source)
        if scales.size!=1:raise UnsupportedModel('Per-tensor output quantization required')
        ts[target]['quantization'] = {'scale': float(scales[0]), 'zero_point': int(zeros[0])}
    output_id=ids[out];outputs=[]
    if len(model_outputs)>1 or dequantized:
        packed_offset=0
        for source in model_outputs:
            spec=dict(ts[ids[source]]);spec['name']=tensor(source).Name().decode();spec['packed_offset']=packed_offset
            if source in dequantized:spec['dequantize']=True
            outputs.append(spec);packed_offset+=spec['bytes']
        if len({t['name'] for t in outputs})!=len(outputs):raise UnsupportedModel('Output tensor names must be unique')
        if len(model_outputs)>1:
            descriptor=add('output-list',(len(model_outputs),),'<i4',[ids[i] for i in model_outputs])
            output_id=add('packed-outputs',(packed_offset,),'i1');ops.append((19,ids[out],descriptor,output_id,0,0,0,0));work+=packed_offset
    if not ops or len(ts)>MAX_TENSORS or len(ops)>MAX_OPERATORS or work>MAX_WORK:raise UnsupportedModel('Model exceeds bounded graph limits or has no executable operators')
    h=struct.pack('<8I',0x55474e31,2,len(ts),len(ops),len(arena),input_id,output_id,int(dsp_threads==2))
    desc=b''.join(struct.pack('<12I',t['offset'],t['bytes'],t['type'],len(t['shape']),*(list(t['shape'])+[1]*(4-len(t['shape']))),int(t['constant']),0,0,0) for t in ts)
    plan=h+desc+b''.join(struct.pack('<8I',*op) for op in ops)+arena
    metadata={'input':ts[input_id],'output':ts[output_id],'arena_bytes':len(arena),'plan_bytes':len(plan),'tensors':ts,'snapshots':snapshots,'ops':ops,'macs':work,'scalar':scalar,'dsp_threads':dsp_threads}
    if outputs:metadata['outputs']=outputs
    if reuse_memory:
        from .memory import compact_plan
        plan,metadata=compact_plan(plan,metadata)
    if metadata['arena_bytes']>ARENA_BYTES:raise UnsupportedModel('Packed graph exceeds 16 MiB DSP arena')
    return bytes(plan),metadata

class Session(BaseSession):
    def __init__(self,model_path,library=None,pipeline='rpc',*,scalar=False,dsp_threads=2):
        plan,metadata=lower(model_path,scalar=scalar,dsp_threads=dsp_threads)
        self._open_plan(plan,metadata,library,pipeline)
