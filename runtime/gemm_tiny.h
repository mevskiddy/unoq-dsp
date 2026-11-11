/* Fixed reduction lengths remove reduction-loop overhead; generated from the
 * checked full-tile kernels by the fixed-reduction generator. */

static __attribute__((noinline)) void pi2x4_full_k8(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n){
 const unsigned k=8;

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
#pragma clang loop unroll(full)
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

static __attribute__((noinline)) void pi2x4_full_k16(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n){
 const unsigned k=16;

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
#pragma clang loop unroll(full)
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

static __attribute__((noinline)) void pi2x4_stride_k32(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n,unsigned xs){
 const unsigned k=32;

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
#pragma clang loop unroll(full)
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
