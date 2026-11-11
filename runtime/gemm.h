#ifndef UNO_GEMM_H
#define UNO_GEMM_H
#include <stdint.h>
#ifdef UQ_CONSTANT_CHANNELS
int ug_matmul_u8i8_selected(const uint8_t*,const int8_t*,int32_t*,unsigned,unsigned,unsigned,unsigned,const uint16_t*,unsigned);
#endif
/* Dense inputs M*K, outputs M*N. Packed weights reserve ceil(N/4)*4*K values.
   FP32 offset: (j/4)*K*4 + c*4 + j%4.
   INT8 offset: (j/4)*K*4 + (c/8)*32 + (j%4)*8 + c%8; K must be a multiple of 8.
   Unused output channels contain zero. Integer pointers are 8-byte aligned.
   Legacy weights use N*K row-major (the transpose of ONNX's right matrix).
   No input/output/weight aliasing. Sizes and INT8 K<=8192 validated by graph.c.
 */
int ug_matmul_f32(const float*,const float*,float*,unsigned,unsigned,unsigned,int,int,int);
int ug_matmul_u8i8(const uint8_t*,const int8_t*,int32_t*,unsigned,unsigned,unsigned,int,int);
/* Serial packed vision kernel. Explicit activation row stride, in bytes;
   xs>=K and xs%8==0; N%4==0. Caller owns at least M*xs input bytes.
   Intended for already split worker jobs; never creates nested workers. */
int ug_matmul_u8i8_strided(const uint8_t*,const int8_t*,int32_t*,unsigned,unsigned,unsigned,unsigned);
#ifdef UQ_DIRECT_SMALL
int ug_matmul_s8i8_small(const int8_t*,const int8_t*,int32_t*,unsigned,unsigned,unsigned);
#endif
#ifdef UQ_GEMM_DOT4
int ug_matmul_u8i8_dot4(const uint8_t*,const int8_t*,int32_t*,unsigned,unsigned,unsigned,unsigned);
#endif
int ug_gemm_available(void);
/* Joined two-way DSP execution. Callbacks must not recursively create workers.
   -1 means no worker was created, allowing a serial retry. */
int ug_run_pair(int (*)(void*),void*,void*);
int ug_begin_frame(void);
int ug_end_frame(void);
#endif
