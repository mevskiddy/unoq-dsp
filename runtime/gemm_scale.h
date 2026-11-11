/* Exact fused dot products and prepared affine scaling; no INT32 tensor spill. */

static void fused_k8_r1(const uint8_t*restrict x,const int8_t*restrict w,int8_t*restrict y,unsigned m,unsigned n,unsigned xs,const int32_t*q,const UQAffine*restrict table){
 for(unsigned j=0;j<n;j+=4)for(unsigned r=0;r<m;r+=1){
  long long s00=0,s01=0;
#ifdef __clang__
#pragma clang loop unroll(full)
#endif
  for(unsigned t=0;t<8;t+=4){
   long long w0,w1;
   __builtin_memcpy(&w0,__builtin_assume_aligned(w+j*8+t*4,8),8);
   __builtin_memcpy(&w1,__builtin_assume_aligned(w+j*8+t*4+8,8),8);
   uint32_t a0;__builtin_memcpy(&a0,__builtin_assume_aligned(x+(r+0)*xs+t,4),4);
   long long x0=(long long)((uint64_t)a0|((uint64_t)a0<<32));
   s00=Q6_P_vrmpybsuacc_PP(s00,w0,x0);
   s01=Q6_P_vrmpybsuacc_PP(s01,w1,x0);
  }
  {
   const UQAffine*a=table+j+0;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s00,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s00>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+0)*n+j+0]=(int8_t)v;
   y[(r+0)*n+j+1]=(int8_t)((uint64_t)v>>32);
  }
  {
   const UQAffine*a=table+j+2;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s01,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s01>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+0)*n+j+2]=(int8_t)v;
   y[(r+0)*n+j+3]=(int8_t)((uint64_t)v>>32);
  }
 }
}

static void fused_k8_r2(const uint8_t*restrict x,const int8_t*restrict w,int8_t*restrict y,unsigned m,unsigned n,unsigned xs,const int32_t*q,const UQAffine*restrict table){
 for(unsigned j=0;j<n;j+=4)for(unsigned r=0;r<m;r+=2){
  long long s00=0,s01=0,s10=0,s11=0;
#ifdef __clang__
#pragma clang loop unroll(full)
#endif
  for(unsigned t=0;t<8;t+=4){
   long long w0,w1;
   __builtin_memcpy(&w0,__builtin_assume_aligned(w+j*8+t*4,8),8);
   __builtin_memcpy(&w1,__builtin_assume_aligned(w+j*8+t*4+8,8),8);
   uint32_t a0;__builtin_memcpy(&a0,__builtin_assume_aligned(x+(r+0)*xs+t,4),4);
   long long x0=(long long)((uint64_t)a0|((uint64_t)a0<<32));
   s00=Q6_P_vrmpybsuacc_PP(s00,w0,x0);
   s01=Q6_P_vrmpybsuacc_PP(s01,w1,x0);
   uint32_t a1;__builtin_memcpy(&a1,__builtin_assume_aligned(x+(r+1)*xs+t,4),4);
   long long x1=(long long)((uint64_t)a1|((uint64_t)a1<<32));
   s10=Q6_P_vrmpybsuacc_PP(s10,w0,x1);
   s11=Q6_P_vrmpybsuacc_PP(s11,w1,x1);
  }
  {
   const UQAffine*a=table+j+0;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s00,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s00>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+0)*n+j+0]=(int8_t)v;
   y[(r+0)*n+j+1]=(int8_t)((uint64_t)v>>32);
  }
  {
   const UQAffine*a=table+j+2;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s01,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s01>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+0)*n+j+2]=(int8_t)v;
   y[(r+0)*n+j+3]=(int8_t)((uint64_t)v>>32);
  }
  {
   const UQAffine*a=table+j+0;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s10,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s10>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+1)*n+j+0]=(int8_t)v;
   y[(r+1)*n+j+1]=(int8_t)((uint64_t)v>>32);
  }
  {
   const UQAffine*a=table+j+2;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s11,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s11>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+1)*n+j+2]=(int8_t)v;
   y[(r+1)*n+j+3]=(int8_t)((uint64_t)v>>32);
  }
 }
}

static void fused_k16_r1(const uint8_t*restrict x,const int8_t*restrict w,int8_t*restrict y,unsigned m,unsigned n,unsigned xs,const int32_t*q,const UQAffine*restrict table){
 for(unsigned j=0;j<n;j+=4)for(unsigned r=0;r<m;r+=1){
  long long s00=0,s01=0;
#ifdef __clang__
#pragma clang loop unroll(full)
#endif
  for(unsigned t=0;t<16;t+=4){
   long long w0,w1;
   __builtin_memcpy(&w0,__builtin_assume_aligned(w+j*16+t*4,8),8);
   __builtin_memcpy(&w1,__builtin_assume_aligned(w+j*16+t*4+8,8),8);
   uint32_t a0;__builtin_memcpy(&a0,__builtin_assume_aligned(x+(r+0)*xs+t,4),4);
   long long x0=(long long)((uint64_t)a0|((uint64_t)a0<<32));
   s00=Q6_P_vrmpybsuacc_PP(s00,w0,x0);
   s01=Q6_P_vrmpybsuacc_PP(s01,w1,x0);
  }
  {
   const UQAffine*a=table+j+0;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s00,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s00>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+0)*n+j+0]=(int8_t)v;
   y[(r+0)*n+j+1]=(int8_t)((uint64_t)v>>32);
  }
  {
   const UQAffine*a=table+j+2;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s01,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s01>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+0)*n+j+2]=(int8_t)v;
   y[(r+0)*n+j+3]=(int8_t)((uint64_t)v>>32);
  }
 }
}

static void fused_k16_r2(const uint8_t*restrict x,const int8_t*restrict w,int8_t*restrict y,unsigned m,unsigned n,unsigned xs,const int32_t*q,const UQAffine*restrict table){
 for(unsigned j=0;j<n;j+=4)for(unsigned r=0;r<m;r+=2){
  long long s00=0,s01=0,s10=0,s11=0;
#ifdef __clang__
#pragma clang loop unroll(full)
#endif
  for(unsigned t=0;t<16;t+=4){
   long long w0,w1;
   __builtin_memcpy(&w0,__builtin_assume_aligned(w+j*16+t*4,8),8);
   __builtin_memcpy(&w1,__builtin_assume_aligned(w+j*16+t*4+8,8),8);
   uint32_t a0;__builtin_memcpy(&a0,__builtin_assume_aligned(x+(r+0)*xs+t,4),4);
   long long x0=(long long)((uint64_t)a0|((uint64_t)a0<<32));
   s00=Q6_P_vrmpybsuacc_PP(s00,w0,x0);
   s01=Q6_P_vrmpybsuacc_PP(s01,w1,x0);
   uint32_t a1;__builtin_memcpy(&a1,__builtin_assume_aligned(x+(r+1)*xs+t,4),4);
   long long x1=(long long)((uint64_t)a1|((uint64_t)a1<<32));
   s10=Q6_P_vrmpybsuacc_PP(s10,w0,x1);
   s11=Q6_P_vrmpybsuacc_PP(s11,w1,x1);
  }
  {
   const UQAffine*a=table+j+0;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s00,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s00>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+0)*n+j+0]=(int8_t)v;
   y[(r+0)*n+j+1]=(int8_t)((uint64_t)v>>32);
  }
  {
   const UQAffine*a=table+j+2;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s01,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s01>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+0)*n+j+2]=(int8_t)v;
   y[(r+0)*n+j+3]=(int8_t)((uint64_t)v>>32);
  }
  {
   const UQAffine*a=table+j+0;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s10,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s10>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+1)*n+j+0]=(int8_t)v;
   y[(r+1)*n+j+1]=(int8_t)((uint64_t)v>>32);
  }
  {
   const UQAffine*a=table+j+2;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s11,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s11>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+1)*n+j+2]=(int8_t)v;
   y[(r+1)*n+j+3]=(int8_t)((uint64_t)v>>32);
  }
 }
}

static void fused_k32_r1(const uint8_t*restrict x,const int8_t*restrict w,int8_t*restrict y,unsigned m,unsigned n,unsigned xs,const int32_t*q,const UQAffine*restrict table){
 for(unsigned j=0;j<n;j+=4)for(unsigned r=0;r<m;r+=1){
  long long s00=0,s01=0;
#ifdef __clang__
#pragma clang loop unroll(full)
#endif
  for(unsigned t=0;t<32;t+=4){
   long long w0,w1;
   __builtin_memcpy(&w0,__builtin_assume_aligned(w+j*32+t*4,8),8);
   __builtin_memcpy(&w1,__builtin_assume_aligned(w+j*32+t*4+8,8),8);
   uint32_t a0;__builtin_memcpy(&a0,__builtin_assume_aligned(x+(r+0)*xs+t,4),4);
   long long x0=(long long)((uint64_t)a0|((uint64_t)a0<<32));
   s00=Q6_P_vrmpybsuacc_PP(s00,w0,x0);
   s01=Q6_P_vrmpybsuacc_PP(s01,w1,x0);
  }
  {
   const UQAffine*a=table+j+0;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s00,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s00>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+0)*n+j+0]=(int8_t)v;
   y[(r+0)*n+j+1]=(int8_t)((uint64_t)v>>32);
  }
  {
   const UQAffine*a=table+j+2;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s01,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s01>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+0)*n+j+2]=(int8_t)v;
   y[(r+0)*n+j+3]=(int8_t)((uint64_t)v>>32);
  }
 }
}

static void fused_k32_r2(const uint8_t*restrict x,const int8_t*restrict w,int8_t*restrict y,unsigned m,unsigned n,unsigned xs,const int32_t*q,const UQAffine*restrict table){
 for(unsigned j=0;j<n;j+=4)for(unsigned r=0;r<m;r+=2){
  long long s00=0,s01=0,s10=0,s11=0;
#ifdef __clang__
#pragma clang loop unroll(full)
#endif
  for(unsigned t=0;t<32;t+=4){
   long long w0,w1;
   __builtin_memcpy(&w0,__builtin_assume_aligned(w+j*32+t*4,8),8);
   __builtin_memcpy(&w1,__builtin_assume_aligned(w+j*32+t*4+8,8),8);
   uint32_t a0;__builtin_memcpy(&a0,__builtin_assume_aligned(x+(r+0)*xs+t,4),4);
   long long x0=(long long)((uint64_t)a0|((uint64_t)a0<<32));
   s00=Q6_P_vrmpybsuacc_PP(s00,w0,x0);
   s01=Q6_P_vrmpybsuacc_PP(s01,w1,x0);
   uint32_t a1;__builtin_memcpy(&a1,__builtin_assume_aligned(x+(r+1)*xs+t,4),4);
   long long x1=(long long)((uint64_t)a1|((uint64_t)a1<<32));
   s10=Q6_P_vrmpybsuacc_PP(s10,w0,x1);
   s11=Q6_P_vrmpybsuacc_PP(s11,w1,x1);
  }
  {
   const UQAffine*a=table+j+0;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s00,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s00>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+0)*n+j+0]=(int8_t)v;
   y[(r+0)*n+j+1]=(int8_t)((uint64_t)v>>32);
  }
  {
   const UQAffine*a=table+j+2;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s01,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s01>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+0)*n+j+2]=(int8_t)v;
   y[(r+0)*n+j+3]=(int8_t)((uint64_t)v>>32);
  }
  {
   const UQAffine*a=table+j+0;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s10,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s10>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+1)*n+j+0]=(int8_t)v;
   y[(r+1)*n+j+1]=(int8_t)((uint64_t)v>>32);
  }
  {
   const UQAffine*a=table+j+2;
   int64_t lo=Q6_P_mpyacc_RR(a[0].constant,(int32_t)s11,a[0].multiplier);
   int64_t hi=Q6_P_mpyacc_RR(a[1].constant,(int32_t)((uint64_t)s11>>32),a[1].multiplier);
   int32_t v0=Q6_R_asr_RR((int32_t)(lo>>32),a[0].shift-32),v1=Q6_R_asr_RR((int32_t)(hi>>32),a[1].shift-32);
   long long v=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long low=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
   long long high=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   v=Q6_P_vmaxw_PP(low,Q6_P_vminw_PP(high,v));
   y[(r+1)*n+j+2]=(int8_t)v;
   y[(r+1)*n+j+3]=(int8_t)((uint64_t)v>>32);
  }
 }
}
