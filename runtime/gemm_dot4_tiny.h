/* Generated exact fixed-reduction kernels from the dot4 implementation. */

static int dot4_k8(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n,unsigned xs){
 const unsigned k=8;
 if(!m||m%2||!n||n%4||!k||k%8||k>8192||xs<k||xs%8)return 18;
 for(unsigned j=0;j<n;j+=4)for(unsigned r=0;r<m;r+=2){
  long long a=0,b=0,c=0,d=0;
#ifdef __clang__
#pragma clang loop unroll(full)
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

static int dot4_k16(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n,unsigned xs){
 const unsigned k=16;
 if(!m||m%2||!n||n%4||!k||k%8||k>8192||xs<k||xs%8)return 18;
 for(unsigned j=0;j<n;j+=4)for(unsigned r=0;r<m;r+=2){
  long long a=0,b=0,c=0,d=0;
#ifdef __clang__
#pragma clang loop unroll(full)
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

static int dot4_k32(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n,unsigned xs){
 const unsigned k=32;
 if(!m||m%2||!n||n%4||!k||k%8||k>8192||xs<k||xs%8)return 18;
 for(unsigned j=0;j<n;j+=4)for(unsigned r=0;r<m;r+=2){
  long long a=0,b=0,c=0,d=0;
#ifdef __clang__
#pragma clang loop unroll(full)
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

#if UQ_DOT4_MEDIUM >= 64
static int dot4_k64(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n,unsigned xs){
 const unsigned k=64;
 if(!m||m%2||!n||n%4||!k||k%8||k>8192||xs<k||xs%8)return 18;
 for(unsigned j=0;j<n;j+=4)for(unsigned r=0;r<m;r+=2){
  long long a=0,b=0,c=0,d=0;
#ifdef __clang__
#pragma clang loop unroll(full)
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

#if UQ_DOT4_MEDIUM >= 128
static int dot4_k128(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n,unsigned xs){
 const unsigned k=128;
 if(!m||m%2||!n||n%4||!k||k%8||k>8192||xs<k||xs%8)return 18;
 for(unsigned j=0;j<n;j+=4)for(unsigned r=0;r<m;r+=2){
  long long a=0,b=0,c=0,d=0;
#ifdef __clang__
#pragma clang loop unroll(full)
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
