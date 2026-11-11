/* Generated register tiles selected by measured on-device experiments.
 * Weight layout is documented in gemm.h; caller validates all dimensions.
 * Compile with -ffp-contract=off. FMA is explicit and opt-in in the plan.
 */
#include "gemm.h"
#include "kernels.h"
static inline void prefetch_weight_group(const int8_t*w,unsigned k){
#if defined(UQ_WEIGHT_PREFETCH) && defined(__hexagon__)
 /* V66 L2FETCH register pair: stride[47:32], width[31:16], height[15:0].
  * A validated 4*K block, K divisible by 32, has no speculative overread. */
 if(k%32==0){uint64_t control=(UINT64_C(128)<<32)|(UINT64_C(128)<<16)|(k/32);__asm__ volatile("l2fetch(%0,%1)"::"r"(w),"r"(control));}
#else
 (void)w;(void)k;
#endif
}
static inline float fma_dsp(float s,float x,float w){
#ifdef __hexagon__
 return Q6_R_sfmpyacc_RR(s,x,w);
#else
 return __builtin_fmaf(x,w,s);
#endif
}
#ifndef __hexagon__
static inline long long packed_dot(long long sum,long long sw,long long ux){
 int8_t w[8];uint8_t x[8];int32_t lo=(int32_t)sum,hi=(int32_t)((uint64_t)sum>>32);
 __builtin_memcpy(w,&sw,8);__builtin_memcpy(x,&ux,8);
 for(unsigned i=0;i<4;i++){lo+=(int32_t)w[i]*x[i];hi+=(int32_t)w[i+4]*x[i+4];}
 return (long long)((uint64_t)(uint32_t)lo|((uint64_t)(uint32_t)hi<<32));
}
#define Q6_P_vrmpybsuacc_PP packed_dot
#endif

static __attribute__((noinline)) void base_f(const float*restrict x,const float*restrict w,float*restrict y,unsigned m,unsigned n,unsigned k){
 for(unsigned r=0;r<m;r++)for(unsigned j=0;j<n;j++){float s=0;for(unsigned c=0;c<k;c++)s+=x[r*k+c]*w[j*k+c];y[r*n+j]=s;}
}
static __attribute__((noinline)) void base_i(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n,unsigned k){
 for(unsigned r=0;r<m;r++)for(unsigned j=0;j<n;j++)y[r*n+j]=ug_dot_u8i8(x+r*k,w+j*k,k);
}
static __attribute__((noinline)) void pf1x4(const float*restrict x,const float*restrict w,float*restrict y,unsigned m,unsigned n,unsigned k){
for(unsigned r=0;r<m;r+=1){for(unsigned j=0;j<n;j+=4){
float s00=0;
float s01=0;
float s02=0;
float s03=0;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
for(unsigned c=0;c<k;c+=1){
float x0=r+0<m?x[(r+0)*k+c]:0;
float w0=w[(j/4)*k*4+c*4+0];
float w1=w[(j/4)*k*4+c*4+1];
float w2=w[(j/4)*k*4+c*4+2];
float w3=w[(j/4)*k*4+c*4+3];
s00+=x0*w0;
s01+=x0*w1;
s02+=x0*w2;
s03+=x0*w3;
}
if(r+0<m&&j+0<n)y[(r+0)*n+j+0]=s00;
if(r+0<m&&j+1<n)y[(r+0)*n+j+1]=s01;
if(r+0<m&&j+2<n)y[(r+0)*n+j+2]=s02;
if(r+0<m&&j+3<n)y[(r+0)*n+j+3]=s03;
}}}
static __attribute__((noinline)) void pf1x4_fma(const float*restrict x,const float*restrict w,float*restrict y,unsigned m,unsigned n,unsigned k){
for(unsigned r=0;r<m;r+=1){for(unsigned j=0;j<n;j+=4){
float s00=0;
float s01=0;
float s02=0;
float s03=0;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
for(unsigned c=0;c<k;c+=1){
float x0=r+0<m?x[(r+0)*k+c]:0;
float w0=w[(j/4)*k*4+c*4+0];
float w1=w[(j/4)*k*4+c*4+1];
float w2=w[(j/4)*k*4+c*4+2];
float w3=w[(j/4)*k*4+c*4+3];
s00=fma_dsp(s00,x0,w0);
s01=fma_dsp(s01,x0,w1);
s02=fma_dsp(s02,x0,w2);
s03=fma_dsp(s03,x0,w3);
}
if(r+0<m&&j+0<n)y[(r+0)*n+j+0]=s00;
if(r+0<m&&j+1<n)y[(r+0)*n+j+1]=s01;
if(r+0<m&&j+2<n)y[(r+0)*n+j+2]=s02;
if(r+0<m&&j+3<n)y[(r+0)*n+j+3]=s03;
}}}
static __attribute__((noinline)) void pf2x4(const float*restrict x,const float*restrict w,float*restrict y,unsigned m,unsigned n,unsigned k){
for(unsigned r=0;r<m;r+=2){for(unsigned j=0;j<n;j+=4){
float s00=0;
float s01=0;
float s02=0;
float s03=0;
float s10=0;
float s11=0;
float s12=0;
float s13=0;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
for(unsigned c=0;c<k;c+=1){
float x0=r+0<m?x[(r+0)*k+c]:0;
float x1=r+1<m?x[(r+1)*k+c]:0;
float w0=w[(j/4)*k*4+c*4+0];
float w1=w[(j/4)*k*4+c*4+1];
float w2=w[(j/4)*k*4+c*4+2];
float w3=w[(j/4)*k*4+c*4+3];
s00+=x0*w0;
s01+=x0*w1;
s02+=x0*w2;
s03+=x0*w3;
s10+=x1*w0;
s11+=x1*w1;
s12+=x1*w2;
s13+=x1*w3;
}
if(r+0<m&&j+0<n)y[(r+0)*n+j+0]=s00;
if(r+0<m&&j+1<n)y[(r+0)*n+j+1]=s01;
if(r+0<m&&j+2<n)y[(r+0)*n+j+2]=s02;
if(r+0<m&&j+3<n)y[(r+0)*n+j+3]=s03;
if(r+1<m&&j+0<n)y[(r+1)*n+j+0]=s10;
if(r+1<m&&j+1<n)y[(r+1)*n+j+1]=s11;
if(r+1<m&&j+2<n)y[(r+1)*n+j+2]=s12;
if(r+1<m&&j+3<n)y[(r+1)*n+j+3]=s13;
}}}
static __attribute__((noinline)) void pf2x4_fma(const float*restrict x,const float*restrict w,float*restrict y,unsigned m,unsigned n,unsigned k){
for(unsigned r=0;r<m;r+=2){for(unsigned j=0;j<n;j+=4){
float s00=0;
float s01=0;
float s02=0;
float s03=0;
float s10=0;
float s11=0;
float s12=0;
float s13=0;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
for(unsigned c=0;c<k;c+=1){
float x0=r+0<m?x[(r+0)*k+c]:0;
float x1=r+1<m?x[(r+1)*k+c]:0;
float w0=w[(j/4)*k*4+c*4+0];
float w1=w[(j/4)*k*4+c*4+1];
float w2=w[(j/4)*k*4+c*4+2];
float w3=w[(j/4)*k*4+c*4+3];
s00=fma_dsp(s00,x0,w0);
s01=fma_dsp(s01,x0,w1);
s02=fma_dsp(s02,x0,w2);
s03=fma_dsp(s03,x0,w3);
s10=fma_dsp(s10,x1,w0);
s11=fma_dsp(s11,x1,w1);
s12=fma_dsp(s12,x1,w2);
s13=fma_dsp(s13,x1,w3);
}
if(r+0<m&&j+0<n)y[(r+0)*n+j+0]=s00;
if(r+0<m&&j+1<n)y[(r+0)*n+j+1]=s01;
if(r+0<m&&j+2<n)y[(r+0)*n+j+2]=s02;
if(r+0<m&&j+3<n)y[(r+0)*n+j+3]=s03;
if(r+1<m&&j+0<n)y[(r+1)*n+j+0]=s10;
if(r+1<m&&j+1<n)y[(r+1)*n+j+1]=s11;
if(r+1<m&&j+2<n)y[(r+1)*n+j+2]=s12;
if(r+1<m&&j+3<n)y[(r+1)*n+j+3]=s13;
}}}
static __attribute__((noinline)) void pf2x4_jo(const float*restrict x,const float*restrict w,float*restrict y,unsigned m,unsigned n,unsigned k){
for(unsigned j=0;j<n;j+=4){for(unsigned r=0;r<m;r+=2){
float s00=0;
float s01=0;
float s02=0;
float s03=0;
float s10=0;
float s11=0;
float s12=0;
float s13=0;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
for(unsigned c=0;c<k;c+=1){
float x0=r+0<m?x[(r+0)*k+c]:0;
float x1=r+1<m?x[(r+1)*k+c]:0;
float w0=w[(j/4)*k*4+c*4+0];
float w1=w[(j/4)*k*4+c*4+1];
float w2=w[(j/4)*k*4+c*4+2];
float w3=w[(j/4)*k*4+c*4+3];
s00+=x0*w0;
s01+=x0*w1;
s02+=x0*w2;
s03+=x0*w3;
s10+=x1*w0;
s11+=x1*w1;
s12+=x1*w2;
s13+=x1*w3;
}
if(r+0<m&&j+0<n)y[(r+0)*n+j+0]=s00;
if(r+0<m&&j+1<n)y[(r+0)*n+j+1]=s01;
if(r+0<m&&j+2<n)y[(r+0)*n+j+2]=s02;
if(r+0<m&&j+3<n)y[(r+0)*n+j+3]=s03;
if(r+1<m&&j+0<n)y[(r+1)*n+j+0]=s10;
if(r+1<m&&j+1<n)y[(r+1)*n+j+1]=s11;
if(r+1<m&&j+2<n)y[(r+1)*n+j+2]=s12;
if(r+1<m&&j+3<n)y[(r+1)*n+j+3]=s13;
}}}
static __attribute__((noinline)) void pf2x4_fma_jo(const float*restrict x,const float*restrict w,float*restrict y,unsigned m,unsigned n,unsigned k){
for(unsigned j=0;j<n;j+=4){for(unsigned r=0;r<m;r+=2){
float s00=0;
float s01=0;
float s02=0;
float s03=0;
float s10=0;
float s11=0;
float s12=0;
float s13=0;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
for(unsigned c=0;c<k;c+=1){
float x0=r+0<m?x[(r+0)*k+c]:0;
float x1=r+1<m?x[(r+1)*k+c]:0;
float w0=w[(j/4)*k*4+c*4+0];
float w1=w[(j/4)*k*4+c*4+1];
float w2=w[(j/4)*k*4+c*4+2];
float w3=w[(j/4)*k*4+c*4+3];
s00=fma_dsp(s00,x0,w0);
s01=fma_dsp(s01,x0,w1);
s02=fma_dsp(s02,x0,w2);
s03=fma_dsp(s03,x0,w3);
s10=fma_dsp(s10,x1,w0);
s11=fma_dsp(s11,x1,w1);
s12=fma_dsp(s12,x1,w2);
s13=fma_dsp(s13,x1,w3);
}
if(r+0<m&&j+0<n)y[(r+0)*n+j+0]=s00;
if(r+0<m&&j+1<n)y[(r+0)*n+j+1]=s01;
if(r+0<m&&j+2<n)y[(r+0)*n+j+2]=s02;
if(r+0<m&&j+3<n)y[(r+0)*n+j+3]=s03;
if(r+1<m&&j+0<n)y[(r+1)*n+j+0]=s10;
if(r+1<m&&j+1<n)y[(r+1)*n+j+1]=s11;
if(r+1<m&&j+2<n)y[(r+1)*n+j+2]=s12;
if(r+1<m&&j+3<n)y[(r+1)*n+j+3]=s13;
}}}
static __attribute__((noinline)) void pi2x2(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n,unsigned k){

for(unsigned r=0;r<m;r+=2){for(unsigned j=0;j<n;j+=2){
long long s00=0;
long long s01=0;
long long s10=0;
long long s11=0;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
for(unsigned c=0;c<k;c+=8){
long long x0=0;if(r+0<m)__builtin_memcpy(&x0,__builtin_assume_aligned(x+(r+0)*k+c,8),8);
long long x1=0;if(r+1<m)__builtin_memcpy(&x1,__builtin_assume_aligned(x+(r+1)*k+c,8),8);
long long w0;__builtin_memcpy(&w0,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+0)%4)*8,8),8);
long long w1;__builtin_memcpy(&w1,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+1)%4)*8,8),8);
s00=Q6_P_vrmpybsuacc_PP(s00,w0,x0);
s01=Q6_P_vrmpybsuacc_PP(s01,w1,x0);
s10=Q6_P_vrmpybsuacc_PP(s10,w0,x1);
s11=Q6_P_vrmpybsuacc_PP(s11,w1,x1);
}
if(r+0<m&&j+0<n)y[(r+0)*n+j+0]=(int32_t)s00+(int32_t)((uint64_t)s00>>32);
if(r+0<m&&j+1<n)y[(r+0)*n+j+1]=(int32_t)s01+(int32_t)((uint64_t)s01>>32);
if(r+1<m&&j+0<n)y[(r+1)*n+j+0]=(int32_t)s10+(int32_t)((uint64_t)s10>>32);
if(r+1<m&&j+1<n)y[(r+1)*n+j+1]=(int32_t)s11+(int32_t)((uint64_t)s11>>32);
}}}
static __attribute__((noinline)) void pi2x4(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n,unsigned k){

for(unsigned r=0;r<m;r+=2){for(unsigned j=0;j<n;j+=4){
long long s00=0;
long long s01=0;
long long s02=0;
long long s03=0;
long long s10=0;
long long s11=0;
long long s12=0;
long long s13=0;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
for(unsigned c=0;c<k;c+=8){
long long x0=0;if(r+0<m)__builtin_memcpy(&x0,__builtin_assume_aligned(x+(r+0)*k+c,8),8);
long long x1=0;if(r+1<m)__builtin_memcpy(&x1,__builtin_assume_aligned(x+(r+1)*k+c,8),8);
long long w0;__builtin_memcpy(&w0,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+0)%4)*8,8),8);
long long w1;__builtin_memcpy(&w1,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+1)%4)*8,8),8);
long long w2;__builtin_memcpy(&w2,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+2)%4)*8,8),8);
long long w3;__builtin_memcpy(&w3,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+3)%4)*8,8),8);
s00=Q6_P_vrmpybsuacc_PP(s00,w0,x0);
s01=Q6_P_vrmpybsuacc_PP(s01,w1,x0);
s02=Q6_P_vrmpybsuacc_PP(s02,w2,x0);
s03=Q6_P_vrmpybsuacc_PP(s03,w3,x0);
s10=Q6_P_vrmpybsuacc_PP(s10,w0,x1);
s11=Q6_P_vrmpybsuacc_PP(s11,w1,x1);
s12=Q6_P_vrmpybsuacc_PP(s12,w2,x1);
s13=Q6_P_vrmpybsuacc_PP(s13,w3,x1);
}
if(r+0<m&&j+0<n)y[(r+0)*n+j+0]=(int32_t)s00+(int32_t)((uint64_t)s00>>32);
if(r+0<m&&j+1<n)y[(r+0)*n+j+1]=(int32_t)s01+(int32_t)((uint64_t)s01>>32);
if(r+0<m&&j+2<n)y[(r+0)*n+j+2]=(int32_t)s02+(int32_t)((uint64_t)s02>>32);
if(r+0<m&&j+3<n)y[(r+0)*n+j+3]=(int32_t)s03+(int32_t)((uint64_t)s03>>32);
if(r+1<m&&j+0<n)y[(r+1)*n+j+0]=(int32_t)s10+(int32_t)((uint64_t)s10>>32);
if(r+1<m&&j+1<n)y[(r+1)*n+j+1]=(int32_t)s11+(int32_t)((uint64_t)s11>>32);
if(r+1<m&&j+2<n)y[(r+1)*n+j+2]=(int32_t)s12+(int32_t)((uint64_t)s12>>32);
if(r+1<m&&j+3<n)y[(r+1)*n+j+3]=(int32_t)s13+(int32_t)((uint64_t)s13>>32);
}}}
/* Complete tiles avoid edge-condition machinery in the hot kernel.
 * Dispatch below proves even M and N divisible by four; tails keep pi2x4_jo. */
#ifdef UQ_GEMM_LEAF
#if defined(__hexagon__) && defined(UQ_GEMM_ASM)
extern void ug_dsp_gemm2x4(const uint8_t*,const uint8_t*,const int8_t*,int32_t*,int32_t*,unsigned);
#define pi2x4_leaf ug_dsp_gemm2x4
#else
/* Limit live state to a single 2x4 tile. Outer matrix counters and dimensions
 * no longer compete with eight packed accumulators in the reduction loop. */
static __attribute__((noinline)) void pi2x4_leaf(const uint8_t*restrict x0ptr,const uint8_t*restrict x1ptr,const int8_t*restrict w,int32_t*restrict y0,int32_t*restrict y1,unsigned k){
long long s00=0;
long long s01=0;
long long s02=0;
long long s03=0;
long long s10=0;
long long s11=0;
long long s12=0;
long long s13=0;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
for(unsigned c=0;c<k;c+=8){
long long x0=0;__builtin_memcpy(&x0,__builtin_assume_aligned(x0ptr+c,8),8);
long long x1=0;__builtin_memcpy(&x1,__builtin_assume_aligned(x1ptr+c,8),8);
long long w0;__builtin_memcpy(&w0,__builtin_assume_aligned(w+(c/8)*32+0,8),8);
long long w1;__builtin_memcpy(&w1,__builtin_assume_aligned(w+(c/8)*32+8,8),8);
long long w2;__builtin_memcpy(&w2,__builtin_assume_aligned(w+(c/8)*32+16,8),8);
long long w3;__builtin_memcpy(&w3,__builtin_assume_aligned(w+(c/8)*32+24,8),8);
s00=Q6_P_vrmpybsuacc_PP(s00,w0,x0);
s01=Q6_P_vrmpybsuacc_PP(s01,w1,x0);
s02=Q6_P_vrmpybsuacc_PP(s02,w2,x0);
s03=Q6_P_vrmpybsuacc_PP(s03,w3,x0);
s10=Q6_P_vrmpybsuacc_PP(s10,w0,x1);
s11=Q6_P_vrmpybsuacc_PP(s11,w1,x1);
s12=Q6_P_vrmpybsuacc_PP(s12,w2,x1);
s13=Q6_P_vrmpybsuacc_PP(s13,w3,x1);
}
y0[0]=(int32_t)s00+(int32_t)((uint64_t)s00>>32);
y0[1]=(int32_t)s01+(int32_t)((uint64_t)s01>>32);
y0[2]=(int32_t)s02+(int32_t)((uint64_t)s02>>32);
y0[3]=(int32_t)s03+(int32_t)((uint64_t)s03>>32);
y1[0]=(int32_t)s10+(int32_t)((uint64_t)s10>>32);
y1[1]=(int32_t)s11+(int32_t)((uint64_t)s11>>32);
y1[2]=(int32_t)s12+(int32_t)((uint64_t)s12>>32);
y1[3]=(int32_t)s13+(int32_t)((uint64_t)s13>>32);
}
#endif
#endif /* portable leaf versus explicitly scheduled DSP leaf */

static __attribute__((noinline)) void pi2x4_full(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n,unsigned k){

for(unsigned j=0;j<n;j+=4){for(unsigned r=0;r<m;r+=2){
long long s00=0;
long long s01=0;
long long s02=0;
long long s03=0;
long long s10=0;
long long s11=0;
long long s12=0;
long long s13=0;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
for(unsigned c=0;c<k;c+=8){
long long x0=0;__builtin_memcpy(&x0,__builtin_assume_aligned(x+(r+0)*k+c,8),8);
long long x1=0;__builtin_memcpy(&x1,__builtin_assume_aligned(x+(r+1)*k+c,8),8);
long long w0;__builtin_memcpy(&w0,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+0)%4)*8,8),8);
long long w1;__builtin_memcpy(&w1,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+1)%4)*8,8),8);
long long w2;__builtin_memcpy(&w2,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+2)%4)*8,8),8);
long long w3;__builtin_memcpy(&w3,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+3)%4)*8,8),8);
s00=Q6_P_vrmpybsuacc_PP(s00,w0,x0);
s01=Q6_P_vrmpybsuacc_PP(s01,w1,x0);
s02=Q6_P_vrmpybsuacc_PP(s02,w2,x0);
s03=Q6_P_vrmpybsuacc_PP(s03,w3,x0);
s10=Q6_P_vrmpybsuacc_PP(s10,w0,x1);
s11=Q6_P_vrmpybsuacc_PP(s11,w1,x1);
s12=Q6_P_vrmpybsuacc_PP(s12,w2,x1);
s13=Q6_P_vrmpybsuacc_PP(s13,w3,x1);
}
y[(r+0)*n+j+0]=(int32_t)s00+(int32_t)((uint64_t)s00>>32);
y[(r+0)*n+j+1]=(int32_t)s01+(int32_t)((uint64_t)s01>>32);
y[(r+0)*n+j+2]=(int32_t)s02+(int32_t)((uint64_t)s02>>32);
y[(r+0)*n+j+3]=(int32_t)s03+(int32_t)((uint64_t)s03>>32);
y[(r+1)*n+j+0]=(int32_t)s10+(int32_t)((uint64_t)s10>>32);
y[(r+1)*n+j+1]=(int32_t)s11+(int32_t)((uint64_t)s11>>32);
y[(r+1)*n+j+2]=(int32_t)s12+(int32_t)((uint64_t)s12>>32);
y[(r+1)*n+j+3]=(int32_t)s13+(int32_t)((uint64_t)s13>>32);
}}}

/* Separate activation row stride avoids measured dual-load bank conflicts.
 * K and packed weights remain unchanged; caller bounds the padded workspace. */
static __attribute__((noinline)) void pi2x4_stride(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n,unsigned k,unsigned xs){

for(unsigned j=0;j<n;j+=4){for(unsigned r=0;r<m;r+=2){
long long s00=0;
long long s01=0;
long long s02=0;
long long s03=0;
long long s10=0;
long long s11=0;
long long s12=0;
long long s13=0;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
for(unsigned c=0;c<k;c+=8){
long long x0=0;__builtin_memcpy(&x0,__builtin_assume_aligned(x+(r+0)*xs+c,8),8);
long long x1=0;__builtin_memcpy(&x1,__builtin_assume_aligned(x+(r+1)*xs+c,8),8);
long long w0;__builtin_memcpy(&w0,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+0)%4)*8,8),8);
long long w1;__builtin_memcpy(&w1,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+1)%4)*8,8),8);
long long w2;__builtin_memcpy(&w2,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+2)%4)*8,8),8);
long long w3;__builtin_memcpy(&w3,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+3)%4)*8,8),8);
s00=Q6_P_vrmpybsuacc_PP(s00,w0,x0);
s01=Q6_P_vrmpybsuacc_PP(s01,w1,x0);
s02=Q6_P_vrmpybsuacc_PP(s02,w2,x0);
s03=Q6_P_vrmpybsuacc_PP(s03,w3,x0);
s10=Q6_P_vrmpybsuacc_PP(s10,w0,x1);
s11=Q6_P_vrmpybsuacc_PP(s11,w1,x1);
s12=Q6_P_vrmpybsuacc_PP(s12,w2,x1);
s13=Q6_P_vrmpybsuacc_PP(s13,w3,x1);
}
y[(r+0)*n+j+0]=(int32_t)s00+(int32_t)((uint64_t)s00>>32);
y[(r+0)*n+j+1]=(int32_t)s01+(int32_t)((uint64_t)s01>>32);
y[(r+0)*n+j+2]=(int32_t)s02+(int32_t)((uint64_t)s02>>32);
y[(r+0)*n+j+3]=(int32_t)s03+(int32_t)((uint64_t)s03>>32);
y[(r+1)*n+j+0]=(int32_t)s10+(int32_t)((uint64_t)s10>>32);
y[(r+1)*n+j+1]=(int32_t)s11+(int32_t)((uint64_t)s11>>32);
y[(r+1)*n+j+2]=(int32_t)s12+(int32_t)((uint64_t)s12>>32);
y[(r+1)*n+j+3]=(int32_t)s13+(int32_t)((uint64_t)s13>>32);
}}}

static __attribute__((noinline)) void pi2x4_jo(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n,unsigned k){

for(unsigned j=0;j<n;j+=4){for(unsigned r=0;r<m;r+=2){
long long s00=0;
long long s01=0;
long long s02=0;
long long s03=0;
long long s10=0;
long long s11=0;
long long s12=0;
long long s13=0;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
for(unsigned c=0;c<k;c+=8){
long long x0=0;if(r+0<m)__builtin_memcpy(&x0,__builtin_assume_aligned(x+(r+0)*k+c,8),8);
long long x1=0;if(r+1<m)__builtin_memcpy(&x1,__builtin_assume_aligned(x+(r+1)*k+c,8),8);
long long w0;__builtin_memcpy(&w0,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+0)%4)*8,8),8);
long long w1;__builtin_memcpy(&w1,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+1)%4)*8,8),8);
long long w2;__builtin_memcpy(&w2,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+2)%4)*8,8),8);
long long w3;__builtin_memcpy(&w3,__builtin_assume_aligned(w+(j/4)*k*4+(c/8)*32+((j+3)%4)*8,8),8);
s00=Q6_P_vrmpybsuacc_PP(s00,w0,x0);
s01=Q6_P_vrmpybsuacc_PP(s01,w1,x0);
s02=Q6_P_vrmpybsuacc_PP(s02,w2,x0);
s03=Q6_P_vrmpybsuacc_PP(s03,w3,x0);
s10=Q6_P_vrmpybsuacc_PP(s10,w0,x1);
s11=Q6_P_vrmpybsuacc_PP(s11,w1,x1);
s12=Q6_P_vrmpybsuacc_PP(s12,w2,x1);
s13=Q6_P_vrmpybsuacc_PP(s13,w3,x1);
}
if(r+0<m&&j+0<n)y[(r+0)*n+j+0]=(int32_t)s00+(int32_t)((uint64_t)s00>>32);
if(r+0<m&&j+1<n)y[(r+0)*n+j+1]=(int32_t)s01+(int32_t)((uint64_t)s01>>32);
if(r+0<m&&j+2<n)y[(r+0)*n+j+2]=(int32_t)s02+(int32_t)((uint64_t)s02>>32);
if(r+0<m&&j+3<n)y[(r+0)*n+j+3]=(int32_t)s03+(int32_t)((uint64_t)s03>>32);
if(r+1<m&&j+0<n)y[(r+1)*n+j+0]=(int32_t)s10+(int32_t)((uint64_t)s10>>32);
if(r+1<m&&j+1<n)y[(r+1)*n+j+1]=(int32_t)s11+(int32_t)((uint64_t)s11>>32);
if(r+1<m&&j+2<n)y[(r+1)*n+j+2]=(int32_t)s12+(int32_t)((uint64_t)s12>>32);
if(r+1<m&&j+3<n)y[(r+1)*n+j+3]=(int32_t)s13+(int32_t)((uint64_t)s13>>32);
}}}

/* FP32 packed kernels selected by measured v66 convolution workloads.
 * Preserve reduction order; partial rows reuse a valid input and suppress stores.
 * K=128 panels bound input/weight working sets for long reductions. */
static __attribute__((noinline)) void conv_0(const float*restrict x,const float*restrict w,float*restrict y,unsigned m,unsigned n,unsigned k){
for(unsigned rb=0;rb<m;rb+=m){unsigned re=rb+m;if(re>m)re=m;
for(unsigned jb=0;jb<n;jb+=n){unsigned je=jb+n;if(je>n)je=n;
for(unsigned cb=0;cb<k;cb+=k){unsigned ce=cb+k;if(ce>k)ce=k;
for(unsigned j=jb;j<je;j+=4)for(unsigned r=rb;r<re;r+=2){
float s00=0;
float s01=0;
float s02=0;
float s03=0;
float s10=0;
float s11=0;
float s12=0;
float s13=0;
const float *x0p=x+(r+0<re?r+0:r)*k;
const float *x1p=x+(r+1<re?r+1:r)*k;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
for(unsigned c=cb;c<ce;c++){
float x0=x0p[c];
float x1=x1p[c];
float w0=w[((j+0)/4)*k*4+c*4+0];
float w1=w[((j+1)/4)*k*4+c*4+1];
float w2=w[((j+2)/4)*k*4+c*4+2];
float w3=w[((j+3)/4)*k*4+c*4+3];
s00=fma_dsp(s00,x0,w0);
s01=fma_dsp(s01,x0,w1);
s02=fma_dsp(s02,x0,w2);
s03=fma_dsp(s03,x0,w3);
s10=fma_dsp(s10,x1,w0);
s11=fma_dsp(s11,x1,w1);
s12=fma_dsp(s12,x1,w2);
s13=fma_dsp(s13,x1,w3);
}
if(r+0<re&&j+0<je)y[(r+0)*n+j+0]=s00;
if(r+0<re&&j+1<je)y[(r+0)*n+j+1]=s01;
if(r+0<re&&j+2<je)y[(r+0)*n+j+2]=s02;
if(r+0<re&&j+3<je)y[(r+0)*n+j+3]=s03;
if(r+1<re&&j+0<je)y[(r+1)*n+j+0]=s10;
if(r+1<re&&j+1<je)y[(r+1)*n+j+1]=s11;
if(r+1<re&&j+2<je)y[(r+1)*n+j+2]=s12;
if(r+1<re&&j+3<je)y[(r+1)*n+j+3]=s13;
}}}}}
static __attribute__((noinline)) void conv_10(const float*restrict x,const float*restrict w,float*restrict y,unsigned m,unsigned n,unsigned k){
for(unsigned i=0;i<m*n;i++)y[i]=0;
for(unsigned rb=0;rb<m;rb+=8){unsigned re=rb+8;if(re>m)re=m;
for(unsigned jb=0;jb<n;jb+=16){unsigned je=jb+16;if(je>n)je=n;
for(unsigned cb=0;cb<k;cb+=128){unsigned ce=cb+128;if(ce>k)ce=k;
for(unsigned j=jb;j<je;j+=4)for(unsigned r=rb;r<re;r+=2){
float s00=(r+0<re&&j+0<je?y[(r+0)*n+j+0]:0);
float s01=(r+0<re&&j+1<je?y[(r+0)*n+j+1]:0);
float s02=(r+0<re&&j+2<je?y[(r+0)*n+j+2]:0);
float s03=(r+0<re&&j+3<je?y[(r+0)*n+j+3]:0);
float s10=(r+1<re&&j+0<je?y[(r+1)*n+j+0]:0);
float s11=(r+1<re&&j+1<je?y[(r+1)*n+j+1]:0);
float s12=(r+1<re&&j+2<je?y[(r+1)*n+j+2]:0);
float s13=(r+1<re&&j+3<je?y[(r+1)*n+j+3]:0);
const float *x0p=x+(r+0<re?r+0:r)*k;
const float *x1p=x+(r+1<re?r+1:r)*k;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
for(unsigned c=cb;c<ce;c++){
float x0=x0p[c];
float x1=x1p[c];
float w0=w[((j+0)/4)*k*4+c*4+0];
float w1=w[((j+1)/4)*k*4+c*4+1];
float w2=w[((j+2)/4)*k*4+c*4+2];
float w3=w[((j+3)/4)*k*4+c*4+3];
s00=fma_dsp(s00,x0,w0);
s01=fma_dsp(s01,x0,w1);
s02=fma_dsp(s02,x0,w2);
s03=fma_dsp(s03,x0,w3);
s10=fma_dsp(s10,x1,w0);
s11=fma_dsp(s11,x1,w1);
s12=fma_dsp(s12,x1,w2);
s13=fma_dsp(s13,x1,w3);
}
if(r+0<re&&j+0<je)y[(r+0)*n+j+0]=s00;
if(r+0<re&&j+1<je)y[(r+0)*n+j+1]=s01;
if(r+0<re&&j+2<je)y[(r+0)*n+j+2]=s02;
if(r+0<re&&j+3<je)y[(r+0)*n+j+3]=s03;
if(r+1<re&&j+0<je)y[(r+1)*n+j+0]=s10;
if(r+1<re&&j+1<je)y[(r+1)*n+j+1]=s11;
if(r+1<re&&j+2<je)y[(r+1)*n+j+2]=s12;
if(r+1<re&&j+3<je)y[(r+1)*n+j+3]=s13;
}}}}}

static void serial_f32(const float*x,const float*w,float*y,unsigned m,unsigned n,unsigned k,int packed,int fused){
 if(!packed){base_f(x,w,y,m,n,k);return;}
 if(fused && m>=4 && n%4==0 && k>=256){
  if(k>=512)conv_10(x,w,y,m,n,k);else conv_0(x,w,y,m,n,k);
  return;
 }
 if(m==1){if(fused)pf1x4_fma(x,w,y,m,n,k);else pf1x4(x,w,y,m,n,k);}
 else if(m<8){if(fused)pf2x4_fma(x,w,y,m,n,k);else pf2x4(x,w,y,m,n,k);}
 else{if(fused)pf2x4_fma_jo(x,w,y,m,n,k);else pf2x4_jo(x,w,y,m,n,k);}
}
#ifdef UQ_GEMM_TINY
#include "gemm_tiny.h"
#ifdef UQ_GEMM_MEDIUM
#include "gemm_medium.h"
#endif
#endif
#ifdef UQ_DIRECT_SMALL
#include "gemm_direct.h"
int ug_matmul_s8i8_small(const int8_t*x,const int8_t*w,int32_t*y,unsigned m,unsigned n,unsigned k){
 if(!m||!n||m%2||n%4||(k!=8&&k!=16))return 21;
 if(k==8)pi2x4_full_k8_signed((const uint8_t*)x,w,y,m,n);
 else pi2x4_full_k16_signed((const uint8_t*)x,w,y,m,n);
 return 0;
}
#endif
static void serial_u8i8(const uint8_t*x,const int8_t*w,int32_t*y,unsigned m,unsigned n,unsigned k,int packed){
 if(!packed){base_i(x,w,y,m,n,k);return;}
#if defined(UQ_GEMV_LEAF) && defined(UQ_GEMM_LEAF)
 if(m==1&&n%4==0&&k>=64){
  int32_t unused[4] __attribute__((aligned(8)));
  prefetch_weight_group(w,k);
  for(unsigned j=0;j<n;j+=4){if(j+4<n)prefetch_weight_group(w+(j/4+1)*k*4,k);pi2x4_leaf(x,x,w+(j/4)*k*4,y+j,unused,k);}
  return;
 }
#endif
#ifdef UQ_GEMM_TINY
 if(m%2==0&&n%4==0){
  if(k==8){pi2x4_full_k8(x,w,y,m,n);return;}
  if(k==16){pi2x4_full_k16(x,w,y,m,n);return;}
 }
#endif
 if(m%2==0 && n%4==0)pi2x4_full(x,w,y,m,n,k);
 else if(m==1 && n*k<=4096)pi2x2(x,w,y,m,n,k);
 else if(m<8)pi2x4(x,w,y,m,n,k);
 else pi2x4_jo(x,w,y,m,n,k);
}

/* Every worker is joinable and joined before its stack, job or outputs are reused.
   No persistent DSP threads and no priority/frequency boosts. */
static int poisoned;
int ug_gemm_available(void){return !poisoned;}
#ifdef __hexagon__
typedef struct {
 char name[16]; uint8_t partition,stid; uint16_t priority;
 uint8_t flags,bus_priority; uint16_t timetest;
 uint32_t stack_size; void*stack; uint16_t detach;
} UGThreadAttr;
extern int qurt_thread_create(unsigned*,UGThreadAttr*,void(*)(void*),void*);
extern int qurt_thread_join(unsigned,int*);
extern unsigned qurt_thread_get_id(void);
extern int qurt_thread_get_priority(unsigned);
extern void qurt_thread_exit(int);
extern int qurt_sysenv_get_max_hw_threads(void*);
static unsigned char worker_stack[16384] __attribute__((aligned(128)));
typedef struct {const void*x,*w;void*y;unsigned m,n,k;int integer,packed,fused;} UGJob;
static void run_job(UGJob*j){
 if(j->integer)serial_u8i8(j->x,j->w,j->y,j->m,j->n,j->k,j->packed);
 else serial_f32(j->x,j->w,j->y,j->m,j->n,j->k,j->packed,j->fused);
}
static void worker_entry(void*arg){run_job(arg);qurt_thread_exit(0);}
/* Return -1 only if no worker was created, allowing a serial DSP retry. */
static int parallel(UGJob job){
 unsigned hardware_threads=0;
 if(qurt_sysenv_get_max_hw_threads(&hardware_threads)||hardware_threads<2)return -1;
 unsigned first=(job.m/2)&~1u;UGJob second=job;second.m-=first;
 second.x=(const unsigned char*)job.x+first*job.k*(job.integer?1:4);
 second.y=(unsigned char*)job.y+first*job.n*4;job.m=first;
 UGThreadAttr attr={.name="uno-gemm",.priority=254,.bus_priority=255,
  .timetest=65534,.stack_size=sizeof(worker_stack),.stack=worker_stack,.detach=1};
 int priority=qurt_thread_get_priority(qurt_thread_get_id());
 if(priority>=128&&priority<=254)attr.priority=(uint16_t)priority;
 unsigned tid;int rc=qurt_thread_create(&tid,&attr,worker_entry,&second);
 if(rc)return -1;
 run_job(&job);
 int status=0;rc=qurt_thread_join(tid,&status);
 if(rc||status){poisoned=1;return 24;}
 return 0;
}
#endif
int ug_matmul_f32(const float*x,const float*w,float*y,unsigned m,unsigned n,unsigned k,int packed,int fused,int threads){
 if(poisoned)return 24;
#ifdef __hexagon__
 if(threads==2&&packed&&m>=8&&(uint64_t)m*n*k>=131072){
  int rc=parallel((UGJob){x,w,y,m,n,k,0,packed,fused});if(rc!=-1)return rc;
 }
#else
 (void)threads;
#endif
 serial_f32(x,w,y,m,n,k,packed,fused);return 0;
}
int ug_matmul_u8i8(const uint8_t*x,const int8_t*w,int32_t*y,unsigned m,unsigned n,unsigned k,int packed,int threads){
 if(poisoned)return 24;
#ifdef __hexagon__
 if(threads==2&&packed&&m>=16&&(uint64_t)m*n*k>=524288){
  int rc=parallel((UGJob){x,w,y,m,n,k,1,packed,0});if(rc!=-1)return rc;
 }
#else
 (void)threads;
#endif
 serial_u8i8(x,w,y,m,n,k,packed);return 0;
}


#ifdef UQ_CONSTANT_CHANNELS
int ug_matmul_u8i8_selected(const uint8_t*x,const int8_t*w,int32_t*y,unsigned m,unsigned n,unsigned k,unsigned xs,const uint16_t*groups,unsigned count){
 if(poisoned)return 24;
 if(!m||!n||k<64||k>8192||n%4||k%8||xs<k||xs%8||count>n/4)return 18;
 unsigned even=m&~1u;
 for(unsigned g=0;g<count;g++){
  unsigned j=groups[g];if(j%4||j>=n)return 18;
  const int8_t*wg=w+(j/4)*k*4;
  if(g+1<count&&groups[g+1]<n)prefetch_weight_group(w+(groups[g+1]/4)*k*4,k);
  for(unsigned r=0;r<even;r+=2){
#ifdef UQ_GEMM_LEAF
   pi2x4_leaf(x+r*xs,x+(r+1)*xs,wg,y+r*n+j,y+(r+1)*n+j,k);
#else
   int32_t temp[8];pi2x4_stride(x+r*xs,wg,temp,2,4,k,xs);
   for(unsigned c=0;c<4;c++){y[r*n+j+c]=temp[c];y[(r+1)*n+j+c]=temp[c+4];}
#endif
  }
  if(m!=even){
#if defined(UQ_GEMV_LEAF) && defined(UQ_GEMM_LEAF)
   int32_t unused[4] __attribute__((aligned(8)));pi2x4_leaf(x+even*xs,x+even*xs,wg,y+even*n+j,unused,k);
#else
   pi2x2(x+even*xs,wg,y+even*n+j,1,4,k);
#endif
  }
 }
 return 0;
}
#endif
int ug_matmul_u8i8_strided(const uint8_t*x,const int8_t*w,int32_t*y,unsigned m,unsigned n,unsigned k,unsigned xs){
 if(poisoned)return 24;
 if(!m||!n||!k||k>8192||n%4||k%8||xs<k||xs%8)return 18;
 unsigned even=m&~1u;
#ifdef UQ_GEMM_MEDIUM
 if(even&&k==64)pi2x4_stride_k64(x,w,y,even,n,xs);
#if UQ_GEMM_MEDIUM >= 128
 else if(even&&k==128)pi2x4_stride_k128(x,w,y,even,n,xs);
#endif
 else
#endif
#ifdef UQ_GEMM_TINY
 if(even&&k==32)pi2x4_stride_k32(x,w,y,even,n,xs);
 else if(even&&k==8&&xs==k)pi2x4_full_k8(x,w,y,even,n);
 else if(even&&k==16&&xs==k)pi2x4_full_k16(x,w,y,even,n);
 else
#endif
#ifdef UQ_GEMM_LEAF
 if(k>=64){prefetch_weight_group(w,k);for(unsigned j=0;j<n;j+=4){
  if(j+4<n)prefetch_weight_group(w+(j/4+1)*k*4,k);
  for(unsigned r=0;r<even;r+=2)pi2x4_leaf(x+r*xs,x+(r+1)*xs,w+(j/4)*k*4,y+r*n+j,y+(r+1)*n+j,k);
 }
 }else
#endif
 if(even)pi2x4_stride(x,w,y,even,n,k,xs);
 if(m!=even){
#if defined(UQ_GEMV_LEAF) && defined(UQ_GEMM_LEAF)
  if(k>=64){
   int32_t unused[4] __attribute__((aligned(8)));
   prefetch_weight_group(w,k);
   for(unsigned j=0;j<n;j+=4){if(j+4<n)prefetch_weight_group(w+(j/4+1)*k*4,k);pi2x4_leaf(x+even*xs,x+even*xs,w+(j/4)*k*4,y+even*n+j,unused,k);}
  }else pi2x2(x+even*xs,w,y+even*n,1,n,k);
#else
  pi2x2(x+even*xs,w,y+even*n,1,n,k);
#endif
 }
 return 0;
}

#ifdef __hexagon__
typedef struct {int(*fn)(void*);void*arg;int result;} CallbackJob;
static unsigned frame_tid;
static int frame_active;
static int frame_command;
static CallbackJob frame_job;
#ifdef UQ_FRAME_FUTEX
/* Public QuRT futex ABI. The kernel compares the word before sleeping, so a
 * completion/submission between the acquire load and wait cannot be lost. */
extern int qurt_futex_wait(void*,int);
extern int qurt_futex_wake(void*,int);
#endif
#ifdef UQ_FRAME_HANDSHAKE
static void frame_entry(void*unused){
 (void)unused;unsigned spins=0;
 /* One atomic state word closes the sleep/notify race:
  * 0 idle, 1 work, 2 stop, 3 idle sleeper, 4 work with parent sleeping. */
 for(;;){
  int state=__atomic_load_n(&frame_command,__ATOMIC_ACQUIRE);
  if(state==2)break;
  if(state==1||state==4){
   frame_job.result=frame_job.fn(frame_job.arg);
   int previous=__atomic_exchange_n(&frame_command,0,__ATOMIC_ACQ_REL);
   if(previous==4)qurt_futex_wake(&frame_command,1);
   spins=0;
  }else{
   if(state==3){qurt_futex_wait(&frame_command,3);spins=0;continue;}
   if(spins++<512)continue;
   int expected=0;
   if(__atomic_compare_exchange_n(&frame_command,&expected,3,0,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE))
    qurt_futex_wait(&frame_command,3);
   spins=0;
  }
 }
 qurt_thread_exit(0);
}
#else
static void frame_entry(void*unused){
 (void)unused;
#ifdef UQ_FRAME_SPIN
 unsigned spins=0;
#endif
 for(;;){
  int command=__atomic_load_n(&frame_command,__ATOMIC_ACQUIRE);
  if(command==2)break;
  if(command==1){
#ifdef UQ_FRAME_SPIN
   spins=0;
#endif
   frame_job.result=frame_job.fn(frame_job.arg);__atomic_store_n(&frame_command,0,__ATOMIC_RELEASE);
#ifdef UQ_FRAME_FUTEX
   qurt_futex_wake(&frame_command,1);
#endif
  }
#ifdef UQ_FRAME_FUTEX
  else {
#ifdef UQ_FRAME_SPIN
   if(spins++<UQ_FRAME_SPIN)continue;
   spins=0;
#endif
   qurt_futex_wait(&frame_command,0);
  }
#endif
 }
 qurt_thread_exit(0);
}
#endif
static void callback_entry(void*arg){CallbackJob*j=arg;j->result=j->fn(j->arg);qurt_thread_exit(0);}
#endif
int ug_run_pair(int(*fn)(void*),void*first,void*second){
 if(poisoned)return 24;
#ifdef __hexagon__
 if(frame_active){
  frame_job.fn=fn;frame_job.arg=second;frame_job.result=0;
#ifdef UQ_FRAME_HANDSHAKE
  int previous=__atomic_exchange_n(&frame_command,1,__ATOMIC_ACQ_REL);
  if(previous==3)qurt_futex_wake(&frame_command,1);
  int rc=fn(first);unsigned spins=0;
  for(;;){
   int state=__atomic_load_n(&frame_command,__ATOMIC_ACQUIRE);
   if(state==0||state==3)break;
   if(spins++<512)continue;
   int expected=1;
   if(__atomic_compare_exchange_n(&frame_command,&expected,4,0,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE)||expected==4)
    qurt_futex_wait(&frame_command,4);
   spins=0;
  }
#else
  __atomic_store_n(&frame_command,1,__ATOMIC_RELEASE);
#ifdef UQ_FRAME_FUTEX
  qurt_futex_wake(&frame_command,1);
#endif
  int rc=fn(first);
#ifdef UQ_FRAME_SPIN
  unsigned spins=0;
#endif
  while(__atomic_load_n(&frame_command,__ATOMIC_ACQUIRE)!=0){
#ifdef UQ_FRAME_FUTEX
#ifdef UQ_FRAME_SPIN
   if(spins++<UQ_FRAME_SPIN)continue;
   spins=0;
#endif
   qurt_futex_wait(&frame_command,1);
#endif
  }
#endif
  return rc?rc:frame_job.result;
 }
 unsigned count=0;if(qurt_sysenv_get_max_hw_threads(&count)||count<2)return -1;
 CallbackJob job={fn,second,0};
 UGThreadAttr attr={.name="uno-vision",.priority=254,.bus_priority=255,.timetest=65534,.stack_size=sizeof(worker_stack),.stack=worker_stack,.detach=1};
 int priority=qurt_thread_get_priority(qurt_thread_get_id());if(priority>=128&&priority<=254)attr.priority=(uint16_t)priority;
 unsigned tid;if(qurt_thread_create(&tid,&attr,callback_entry,&job))return -1;
 int rc=fn(first),status=0;int joined=qurt_thread_join(tid,&status);
 if(joined||status){poisoned=1;return 24;}
 return rc?rc:job.result;
#else
 (void)fn;(void)first;(void)second;return -1;
#endif
}

/* A worker lives for one inference only. This amortizes layer dispatch while
 * retaining a join before model replacement or RPC completion. */
int ug_begin_frame(void){
 if(poisoned)return 24;
#ifdef __hexagon__
 if(frame_active)return 24;
 unsigned count=0;if(qurt_sysenv_get_max_hw_threads(&count)||count<2)return 0;
 frame_command=0;
 UGThreadAttr attr={.name="uno-frame",.priority=254,.bus_priority=255,.timetest=65534,.stack_size=sizeof(worker_stack),.stack=worker_stack,.detach=1};
 int priority=qurt_thread_get_priority(qurt_thread_get_id());if(priority>=128&&priority<=254)attr.priority=(uint16_t)priority;
 if(qurt_thread_create(&frame_tid,&attr,frame_entry,0))return 0;
 frame_active=1;
#endif
 return 0;
}
int ug_end_frame(void){
#ifdef __hexagon__
 if(frame_active){
#ifdef UQ_FRAME_HANDSHAKE
  int previous=__atomic_exchange_n(&frame_command,2,__ATOMIC_ACQ_REL);
  if(previous==3)qurt_futex_wake(&frame_command,1);
#else
  __atomic_store_n(&frame_command,2,__ATOMIC_RELEASE);
#ifdef UQ_FRAME_FUTEX
  qurt_futex_wake(&frame_command,1);
#endif
#endif
  int status=0;int rc=qurt_thread_join(frame_tid,&status);frame_active=0;
  if(rc||status){poisoned=1;return 24;}
 }
#endif
 return 0;
}

#ifdef UQ_GEMM_DOT4
/* Each vrmpy lane accumulates a complete output channel. Four-byte input
 * broadcasts avoid the final horizontal reduction and halve live accumulators. */
#ifdef UQ_DOT4_TINY
#include "gemm_dot4_tiny.h"
#endif
#ifdef UQ_DOT4_WIDE
#include "gemm_dot4_wide.h"
#endif
int ug_matmul_u8i8_dot4(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n,unsigned k,unsigned xs){
 if(!m||m%2||!n||n%4||!k||k%8||k>8192||xs<k||xs%8)return 18;
#ifdef UQ_DOT4_TINY
 if(k==8)return dot4_k8(x,w,y,m,n,xs);
 if(k==16)return dot4_k16(x,w,y,m,n,xs);
 if(k==32)return dot4_k32(x,w,y,m,n,xs);
#if UQ_DOT4_MEDIUM >= 64
 if(k==64)return dot4_k64(x,w,y,m,n,xs);
#endif
#if UQ_DOT4_MEDIUM >= 128
 if(k==128)return dot4_k128(x,w,y,m,n,xs);
#endif
#endif
#ifdef UQ_DOT4_WIDE
 if(n%8==0)return dot4_2x8(x,w,y,m,n,k,xs);
#endif
 for(unsigned j=0;j<n;j+=4)for(unsigned r=0;r<m;r+=2){
  long long a=0,b=0,c=0,d=0;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
  for(unsigned t=0;t<k;t+=4){
   uint32_t v0,v1;long long w01,w23;
   __builtin_memcpy(&v0,__builtin_assume_aligned(x+r*xs+t,4),4);
   __builtin_memcpy(&v1,__builtin_assume_aligned(x+(r+1)*xs+t,4),4);
   long long x0=(long long)((uint64_t)v0|((uint64_t)v0<<32)),x1=(long long)((uint64_t)v1|((uint64_t)v1<<32));
   __builtin_memcpy(&w01,__builtin_assume_aligned(w+j*k+t*4,8),8);
   __builtin_memcpy(&w23,__builtin_assume_aligned(w+j*k+t*4+8,8),8);
   a=Q6_P_vrmpybsuacc_PP(a,w01,x0);b=Q6_P_vrmpybsuacc_PP(b,w23,x0);
   c=Q6_P_vrmpybsuacc_PP(c,w01,x1);d=Q6_P_vrmpybsuacc_PP(d,w23,x1);
  }
  __builtin_memcpy(__builtin_assume_aligned(y+r*n+j,8),&a,8);
  __builtin_memcpy(__builtin_assume_aligned(y+r*n+j+2,8),&b,8);
  __builtin_memcpy(__builtin_assume_aligned(y+(r+1)*n+j,8),&c,8);
  __builtin_memcpy(__builtin_assume_aligned(y+(r+1)*n+j+2,8),&d,8);
 }
 return 0;
}
#endif
