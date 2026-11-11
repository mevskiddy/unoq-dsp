#ifndef UNO_GRAPH_H
#define UNO_GRAPH_H
#include <stdint.h>
#define UG_MAGIC 0x55474e31u
#define UG_MAX_T 1024
#define UG_MAX_OP 512
#define UG_ARENA (16*1024*1024)
#define UG_MAX_PLAN (17*1024*1024)
#define UG_MAX_WORK UINT64_C(2000000000)
/* Version 2 header reserved: 0=one thread, 1=allow a second for large matrices.
   Version 3 additionally permits tensor buffers to overlap only when graph
   lifetimes do not intersect. Constants remain immutable. The loader checks
   this independently; same-operation input/output aliasing is forbidden.
   All fields little endian. Arena offsets and byte lengths; dimensions <=4. */
typedef struct {uint32_t magic,version,nt,no,arena_bytes,input,output,reserved;} UGHeader;
typedef struct {uint32_t offset,bytes,type,rank,d[4],flags,reserved[3];} UGTensor;
typedef struct {uint32_t code,a,b,out,p[4];} UGOp;
/* types: 1=float32, 2=uint8, 3=int8, 4=int32.
   ops: 1=FP matmul packed B, 2=Add, 3=Mul, 4=Relu, 5=Copy,
        6=MatMulInteger u8/i8 packed B, 7=Cast i32->f32. Version 2 also adds:
        8=FP four-channel packed matmul, p[0]=use FMA (0/1);
        9=INT8 four-channel/eight-reduction packed matmul;
        10=Transpose, p contains the axis permutation;
        11=NHWC im2col, 12=NHWC max pool, 13=last-axis float softmax;
        14..17=INT8 NHWC vision ops, see qnn.h;
        18=INT8 equal-shape residual ADD with prepared input tables;
        19=pack byte tensor outputs using a constant tensor-id list;
        20=INT8 same-quantization concatenation;
        21=nearest NHWC resize using prepared row/column offsets;
        22=byte-wise unary operation using a prepared 256-entry table;
        23=NHWC INT8 zero padding; 24=spatial global INT8 mean;
        25=equal-shape INT8 multiply; 26=INT8 slice, p=begin indices;
        27=rank-three INT8 batched matmul, p[0]=quantization/options;
        28=reference-exact INT8 softmax with Q31 exponent table.
   Conv/depthwise p[1] optionally names a fused post-LUT tensor id plus one.
   Spatial ops use a constant int32 b[12]: kh,kw,sh,sw,dh,dw,pt,pl,pb,pr,oh,ow.
   Packed tensor d[0] is output channels rounded up to four; z.d[1] is logical N.
   Integer params remain az, bz, correction tensor, 1. See gemm.h for layouts. */
int ug_load(const void*,uint32_t);
int ug_run(const void*,uint32_t,void*,uint32_t);
uint32_t ug_input_bytes(void);
uint32_t ug_output_bytes(void);
/* Diagnostic snapshot, used only after a synchronous inference has completed. */
int ug_snapshot(uint32_t,void*,uint32_t);
int ug_profile(uint32_t,void*,uint32_t);
#endif
