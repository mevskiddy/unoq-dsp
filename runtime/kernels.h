#ifndef UNO_KERNELS_H
#define UNO_KERNELS_H
#include <stdint.h>
#ifdef __hexagon__
#include <hexagon_protos.h>
#endif
static inline int32_t ug_dot_u8i8(const uint8_t*x,const int8_t*w,uint32_t n){
 uint32_t i=0;int32_t sum=0;
#ifdef __hexagon__
 long long acc=0;
 /* memcpy remains alias-safe; explicit alignment permits paired 64-bit loads. */
 if(!(((uintptr_t)x|(uintptr_t)w)&7)){
  const uint8_t*ax=__builtin_assume_aligned(x,8);
  const int8_t*aw=__builtin_assume_aligned(w,8);
  for(;i+8<=n;i+=8){long long a,b;__builtin_memcpy(&a,ax+i,8);__builtin_memcpy(&b,aw+i,8);acc=Q6_P_vrmpybsuacc_PP(acc,b,a);}
 }else
 for(;i+8<=n;i+=8){long long a,b;__builtin_memcpy(&a,x+i,8);__builtin_memcpy(&b,w+i,8);acc=Q6_P_vrmpybsuacc_PP(acc,b,a);}
 sum=(int32_t)acc+(int32_t)((uint64_t)acc>>32);
#endif
 for(;i<n;i++)sum+=(int32_t)x[i]*w[i];
 return sum;
}
#endif
