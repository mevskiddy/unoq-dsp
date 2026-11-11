static int dot4_2x8(const uint8_t*restrict x,const int8_t*restrict w,int32_t*restrict y,unsigned m,unsigned n,unsigned k,unsigned xs){
 for(unsigned j=0;j<n;j+=8)for(unsigned r=0;r<m;r+=2){
  long long s00=0,s01=0,s02=0,s03=0,s10=0,s11=0,s12=0,s13=0;
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
  for(unsigned t=0;t<k;t+=4){
   uint32_t v0,v1;
   __builtin_memcpy(&v0,__builtin_assume_aligned(x+r*xs+t,4),4);
   __builtin_memcpy(&v1,__builtin_assume_aligned(x+(r+1)*xs+t,4),4);
   long long x0=(long long)((uint64_t)v0|((uint64_t)v0<<32)),x1=(long long)((uint64_t)v1|((uint64_t)v1<<32));
   long long w0;__builtin_memcpy(&w0,__builtin_assume_aligned(w+(j+0)*k+t*4+0,8),8);
   long long w1;__builtin_memcpy(&w1,__builtin_assume_aligned(w+(j+0)*k+t*4+8,8),8);
   long long w2;__builtin_memcpy(&w2,__builtin_assume_aligned(w+(j+4)*k+t*4+0,8),8);
   long long w3;__builtin_memcpy(&w3,__builtin_assume_aligned(w+(j+4)*k+t*4+8,8),8);
   s00=Q6_P_vrmpybsuacc_PP(s00,w0,x0);
   s01=Q6_P_vrmpybsuacc_PP(s01,w1,x0);
   s02=Q6_P_vrmpybsuacc_PP(s02,w2,x0);
   s03=Q6_P_vrmpybsuacc_PP(s03,w3,x0);
   s10=Q6_P_vrmpybsuacc_PP(s10,w0,x1);
   s11=Q6_P_vrmpybsuacc_PP(s11,w1,x1);
   s12=Q6_P_vrmpybsuacc_PP(s12,w2,x1);
   s13=Q6_P_vrmpybsuacc_PP(s13,w3,x1);
  }
  __builtin_memcpy(__builtin_assume_aligned(y+(r+0)*n+j+0,8),&s00,8);
  __builtin_memcpy(__builtin_assume_aligned(y+(r+0)*n+j+2,8),&s01,8);
  __builtin_memcpy(__builtin_assume_aligned(y+(r+0)*n+j+4,8),&s02,8);
  __builtin_memcpy(__builtin_assume_aligned(y+(r+0)*n+j+6,8),&s03,8);
  __builtin_memcpy(__builtin_assume_aligned(y+(r+1)*n+j+0,8),&s10,8);
  __builtin_memcpy(__builtin_assume_aligned(y+(r+1)*n+j+2,8),&s11,8);
  __builtin_memcpy(__builtin_assume_aligned(y+(r+1)*n+j+4,8),&s12,8);
  __builtin_memcpy(__builtin_assume_aligned(y+(r+1)*n+j+6,8),&s13,8);
 }
 return 0;
}
