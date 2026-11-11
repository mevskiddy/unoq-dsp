#ifndef UNO_QNN_H
#define UNO_QNN_H
#include "graph.h"
/* NHWC signed INT8 vision ops, TFLite double-rounding quantization semantics.
   14=conv/pointwise/FC, 15=depthwise (multiplier 1), 16=average/max pool,
   17=softmax with precomputed fixed-point exponential table.
   p[0] names immutable INT32 parameters. No dynamic allocation during inference. */
#define UQ_HEAD 20
#define UQ_TILE 16
#define UQ_MAX_K 8192
#define UQ_MAX_C 2048
int uq_validate(const UGOp*,const UGTensor*,const UGTensor*,const UGTensor*,const UGTensor*,const void*,const void*,uint64_t*);
int uq_run(unsigned,const UGTensor*,const UGTensor*,const int8_t*,const int8_t*,int8_t*,const int32_t*,int);
/* Equal-shape residual ADD. q[0:8]=output multiplier, shift, zero, min, max,
   three zero reserved words; q[8:520]=two pre-scaled signed-byte tables. */
void uq_add(const int8_t*,const int8_t*,int8_t*,uint32_t,const int32_t*);
/* Spatial global mean. q=count, input/output zero points, Q31 multiplier, shift. */
void uq_mean(const int8_t*,int8_t*,unsigned,unsigned,const int32_t*);
void uq_mul(const int8_t*,const int8_t*,int8_t*,unsigned,const int32_t*);
int uq_bmm(const UGTensor*,const UGTensor*,const UGTensor*,const int8_t*,const int8_t*,int8_t*,const int32_t*,int);
void uq_post_reset(void);
void uq_post_register(const int32_t*,const uint8_t*);
void uq_softmax_exact(const int8_t*,int8_t*,unsigned,const int32_t*);
#ifdef UQ_FUSE_DW_PW
int uq_run_fused(const UGTensor*,const UGTensor*,const UGTensor*,const int8_t*,const int8_t*,const int8_t*,int8_t*,int8_t*,const int32_t*,const int32_t*,int);
#endif
#ifdef UQ_CONSTANT_CHANNELS
void uq_prepare_reset(void);
int uq_prepare(unsigned,const int8_t*,const int32_t*,const int16_t*,const int16_t*,int16_t*,int16_t*);
#endif
#endif
