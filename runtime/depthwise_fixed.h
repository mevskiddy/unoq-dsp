/* Fixed-channel specializations retain the generic border and arithmetic rules. */
#if UQ_DW_FIXED >= 8
static __attribute__((noinline)) void depthwise3_c8(const UGTensor*a,const UGTensor*z,const int8_t*UQ_NOALIAS x,const int8_t*UQ_NOALIAS w,int8_t*UQ_NOALIAS y,const int32_t*UQ_NOALIAS q,unsigned first,unsigned end,int32_t*UQ_NOALIAS acc,unsigned scratch){
 unsigned channels=8,ih=a->d[1],iw=a->d[2];
#ifdef UQ_DW_SCALE
 const UQAffine*direct_scale=lookup_affine(q);
#endif
#ifdef UQ_RANGE_DW_SKIP
 const UQPrepared*prep=lookup(q);
#endif
 int8_t pad[8] __attribute__((aligned(8)));
 for(unsigned c=0;c<channels;c++)pad[c]=(int8_t)q[12];
#if defined(UQ_DEPTHWISE_HALFWORD) && !defined(UQ_DW_DOT)
 /* Reuse this worker's otherwise unused gather buffer; no new allocation.
  * At most 9*1024*2 bytes. Expand weights once per layer invocation, rather
  * than widening four products and adding them separately at every tap. */
 uint8_t*wide_weights=columns[scratch];
 for(unsigned c=0;c<9*channels;c+=4){
#ifdef __hexagon__
  int packed;__builtin_memcpy(&packed,__builtin_assume_aligned(w+c,4),4);
  long long wide=Q6_P_vsxtbh_R(packed);
  __builtin_memcpy(wide_weights+2*c,&wide,8);
#else
  int16_t wide[4];for(unsigned lane=0;lane<4;lane++)wide[lane]=w[c+lane];
  __builtin_memcpy(wide_weights+2*c,wide,8);
#endif
 }
#endif
 unsigned oh=z->d[1],ow=z->d[2];
 unsigned input_channels=channels;
#ifdef UQ_DW_PAD
 if(x==dw_padded_input)input_channels+=8;
#endif
#ifdef UQ_DW_DOT
 /* Two four-tap dot groups for each channel, plus a wide ninth-tap product.
  * This changes data arrangement only; no partial sum is narrowed. */
 uint8_t*dot_weights=columns[scratch];
 for(unsigned c=0;c<channels;c+=4){
  for(unsigned block=0;block<2;block++)for(unsigned lane=0;lane<4;lane++)for(unsigned t=0;t<4;t++)
   dot_weights[(c/4)*40+block*16+lane*4+t]=(uint8_t)w[(block*4+t)*channels+c+lane];
  int16_t tail[4];for(unsigned lane=0;lane<4;lane++)tail[lane]=w[8*channels+c+lane];
  __builtin_memcpy(dot_weights+(c/4)*40+32,tail,8);
 }
#endif
 unsigned n=first/(oh*ow),h=(first/ow)%oh,v=first%ow;
 const int sh=q[2],sw=q[3],dh=q[4],dw=q[5],top=q[6],left=q[7];
 for(unsigned tile=first,tile_size=tile_rows(8,channels);tile<end;tile+=tile_size){
 unsigned stop=tile+tile_size;if(stop>end)stop=end;
 PHASE_START();
 for(unsigned row=tile;row<stop;row++){
  const int8_t*ptr[9];
  int yy=(int)h*sh-top,xx=(int)v*sw-left;
  if(yy>=0&&xx>=0&&(unsigned)(yy+2*dh)<ih&&(unsigned)(xx+2*dw)<iw){
   const int8_t*base=x+((n*ih+yy)*iw+xx)*input_channels;
   for(int kh=0;kh<3;kh++)for(int kw=0;kw<3;kw++)ptr[kh*3+kw]=base+(kh*dh*iw+kw*dw)*input_channels;
  }else{
   for(int kh=0;kh<3;kh++)for(int kw=0;kw<3;kw++){
    int sy=yy+kh*dh,sx=xx+kw*dw;
    ptr[kh*3+kw]=sy>=0&&sx>=0&&(unsigned)sy<ih&&(unsigned)sx<iw?x+((n*ih+sy)*iw+sx)*input_channels:pad;
   }
  }
  if(++v==ow){v=0;if(++h==oh){h=0;n++;}}
  int32_t*dst=acc+(row-tile)*channels;
#if defined(UQ_DW_LEAF) && defined(UQ_DW_DOT) && defined(__hexagon__)
  if(channels>=UQ_DW_LEAF)depthwise_dot_row(ptr,dot_weights,channels,dst);
  else
#endif
#ifdef __clang__
#pragma clang loop unroll(full)
#endif
  for(unsigned c=0;c<channels;c+=4){
#ifdef UQ_RANGE_DW_SKIP
   if(prep&&prep->value[c]!=-256&&prep->value[c+1]!=-256&&prep->value[c+2]!=-256&&prep->value[c+3]!=-256){dst[c]=dst[c+1]=dst[c+2]=dst[c+3]=0;continue;}
#endif
#ifdef __hexagon__
   long long lo=0,hi=0;
#ifdef UQ_DW_DOT
   #pragma clang loop unroll(full)
   for(unsigned block=0;block<2;block++){
    int v0,v1,v2,v3;unsigned t=block*4;
    __builtin_memcpy(&v0,__builtin_assume_aligned(ptr[t]+c,4),4);
    __builtin_memcpy(&v1,__builtin_assume_aligned(ptr[t+1]+c,4),4);
    __builtin_memcpy(&v2,__builtin_assume_aligned(ptr[t+2]+c,4),4);
    __builtin_memcpy(&v3,__builtin_assume_aligned(ptr[t+3]+c,4),4);
    long long a=(long long)(((uint64_t)(uint32_t)v2<<32)|(uint32_t)v0)^INT64_C(-9187201950435737472);
    long long b=(long long)(((uint64_t)(uint32_t)v3<<32)|(uint32_t)v1)^INT64_C(-9187201950435737472);
    long long even=Q6_P_shuffeb_PP(b,a),odd=Q6_P_shuffob_PP(b,a);
    long long x01=Q6_P_vtrunewh_PP(odd,even),x23=Q6_P_vtrunowh_PP(odd,even),w01,w23;
    __builtin_memcpy(&w01,__builtin_assume_aligned(dot_weights+(c/4)*40+block*16,8),8);
    __builtin_memcpy(&w23,__builtin_assume_aligned(dot_weights+(c/4)*40+block*16+8,8),8);
    lo=Q6_P_vrmpybsuacc_PP(lo,w01,x01);hi=Q6_P_vrmpybsuacc_PP(hi,w23,x23);
   }
   int xv;long long tail;
   __builtin_memcpy(&xv,__builtin_assume_aligned(ptr[8]+c,4),4);
   __builtin_memcpy(&tail,__builtin_assume_aligned(dot_weights+(c/4)*40+32,8),8);
   long long xx=Q6_P_vzxtbh_R(xv^(int)0x80808080u);
   lo=Q6_P_vmpyhacc_RR(lo,(int32_t)tail,(int32_t)xx);
   hi=Q6_P_vmpyhacc_RR(hi,(int32_t)((uint64_t)tail>>32),(int32_t)((uint64_t)xx>>32));
#else
#ifdef __clang__
#pragma clang loop unroll(full)
#endif
   for(unsigned t=0;t<9;t++){
#if defined(UQ_DEPTHWISE_HALFWORD) && !defined(UQ_DW_DOT)
    int xv;long long ww;
    __builtin_memcpy(&xv,__builtin_assume_aligned(ptr[t]+c,4),4);
    __builtin_memcpy(&ww,__builtin_assume_aligned(wide_weights+2*(t*channels+c),8),8);
    long long xx=Q6_P_vzxtbh_R(xv^(int)0x80808080u);
    /* Each partial sum is bounded by 9*255*128 = 293760; no saturation,
     * overflow, or precision loss. This still evaluates the INT8 model. */
    lo=Q6_P_vmpyhacc_RR(lo,(int32_t)ww,(int32_t)xx);
    hi=Q6_P_vmpyhacc_RR(hi,(int32_t)((uint64_t)ww>>32),(int32_t)((uint64_t)xx>>32));
#else
    int xv,wv;__builtin_memcpy(&xv,__builtin_assume_aligned(ptr[t]+c,4),4);__builtin_memcpy(&wv,__builtin_assume_aligned(w+t*channels+c,4),4);
    long long prod=Q6_P_vmpybsu_RR(wv,xv^(int)0x80808080u);
    lo=Q6_P_vaddw_PP(lo,Q6_P_vsxthw_R((int32_t)prod));
    hi=Q6_P_vaddw_PP(hi,Q6_P_vsxthw_R((int32_t)((uint64_t)prod>>32)));
#endif
   }
#endif
   int32_t v0=(int32_t)lo,v1=(int32_t)((uint64_t)lo>>32),v2=(int32_t)hi,v3=(int32_t)((uint64_t)hi>>32);
#else
   int32_t v0=0,v1=0,v2=0,v3=0;
   for(unsigned t=0;t<9;t++){
    const int8_t*xx=ptr[t]+c;
#if defined(UQ_DEPTHWISE_HALFWORD) && !defined(UQ_DW_DOT)
    int16_t ww[4];__builtin_memcpy(ww,wide_weights+2*(t*channels+c),8);
#else
    const int8_t*ww=w+t*channels+c;
#endif
    v0+=((int)xx[0]+128)*ww[0];v1+=((int)xx[1]+128)*ww[1];v2+=((int)xx[2]+128)*ww[2];v3+=((int)xx[3]+128)*ww[3];
   }
#endif
#ifdef UQ_DW_SCALE
   if(direct_scale){
    int32_t vv[4]={v0,v1,v2,v3};
#ifdef __clang__
#pragma clang loop unroll(full)
#endif
    for(unsigned j=0;j<4;j+=2){
     const UQAffine*t=direct_scale+c+j;
#ifdef __hexagon__
     int64_t p0=Q6_P_mpyacc_RR(t[0].constant,vv[j],t[0].multiplier),p1=Q6_P_mpyacc_RR(t[1].constant,vv[j+1],t[1].multiplier);
     int32_t a0=Q6_R_asr_RR((int32_t)(p0>>32),t[0].shift-32),a1=Q6_R_asr_RR((int32_t)(p1>>32),t[1].shift-32);
     long long both=(long long)((uint64_t)(uint32_t)a0|((uint64_t)(uint32_t)a1<<32));
     long long lo=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
     long long hi=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
     both=Q6_P_vmaxw_PP(lo,Q6_P_vminw_PP(hi,both));
     uint16_t bytes=(uint16_t)Q6_R_vsathb_R(Q6_R_vsatwh_P(both));
#else
     int32_t a0=(int32_t)((t[0].constant+(int64_t)vv[j]*t[0].multiplier)>>t[0].shift),a1=(int32_t)((t[1].constant+(int64_t)vv[j+1]*t[1].multiplier)>>t[1].shift);
     uint16_t bytes=(uint16_t)((uint8_t)clamp(a0,q)|((uint16_t)(uint8_t)clamp(a1,q)<<8));
#endif
     __builtin_memcpy(__builtin_assume_aligned(y+row*channels+c+j,2),&bytes,2);
    }
   }else
#endif
   {dst[c]=v0;dst[c+1]=v1;dst[c+2]=v2;dst[c+3]=v3;}
  }
 }
 PHASE_END(scratch,3);
#ifdef UQ_DW_SCALE
 if(!direct_scale)
#endif
 requantize(acc,y+tile*channels,stop-tile,channels,q);
 PHASE_END(scratch,2);
 }
}
#endif
#if UQ_DW_FIXED >= 16
static __attribute__((noinline)) void depthwise3_c16(const UGTensor*a,const UGTensor*z,const int8_t*UQ_NOALIAS x,const int8_t*UQ_NOALIAS w,int8_t*UQ_NOALIAS y,const int32_t*UQ_NOALIAS q,unsigned first,unsigned end,int32_t*UQ_NOALIAS acc,unsigned scratch){
 unsigned channels=16,ih=a->d[1],iw=a->d[2];
#ifdef UQ_DW_SCALE
 const UQAffine*direct_scale=lookup_affine(q);
#endif
#ifdef UQ_RANGE_DW_SKIP
 const UQPrepared*prep=lookup(q);
#endif
 int8_t pad[16] __attribute__((aligned(8)));
 for(unsigned c=0;c<channels;c++)pad[c]=(int8_t)q[12];
#if defined(UQ_DEPTHWISE_HALFWORD) && !defined(UQ_DW_DOT)
 /* Reuse this worker's otherwise unused gather buffer; no new allocation.
  * At most 9*1024*2 bytes. Expand weights once per layer invocation, rather
  * than widening four products and adding them separately at every tap. */
 uint8_t*wide_weights=columns[scratch];
 for(unsigned c=0;c<9*channels;c+=4){
#ifdef __hexagon__
  int packed;__builtin_memcpy(&packed,__builtin_assume_aligned(w+c,4),4);
  long long wide=Q6_P_vsxtbh_R(packed);
  __builtin_memcpy(wide_weights+2*c,&wide,8);
#else
  int16_t wide[4];for(unsigned lane=0;lane<4;lane++)wide[lane]=w[c+lane];
  __builtin_memcpy(wide_weights+2*c,wide,8);
#endif
 }
#endif
 unsigned oh=z->d[1],ow=z->d[2];
 unsigned input_channels=channels;
#ifdef UQ_DW_PAD
 if(x==dw_padded_input)input_channels+=8;
#endif
#ifdef UQ_DW_DOT
 /* Two four-tap dot groups for each channel, plus a wide ninth-tap product.
  * This changes data arrangement only; no partial sum is narrowed. */
 uint8_t*dot_weights=columns[scratch];
 for(unsigned c=0;c<channels;c+=4){
  for(unsigned block=0;block<2;block++)for(unsigned lane=0;lane<4;lane++)for(unsigned t=0;t<4;t++)
   dot_weights[(c/4)*40+block*16+lane*4+t]=(uint8_t)w[(block*4+t)*channels+c+lane];
  int16_t tail[4];for(unsigned lane=0;lane<4;lane++)tail[lane]=w[8*channels+c+lane];
  __builtin_memcpy(dot_weights+(c/4)*40+32,tail,8);
 }
#endif
 unsigned n=first/(oh*ow),h=(first/ow)%oh,v=first%ow;
 const int sh=q[2],sw=q[3],dh=q[4],dw=q[5],top=q[6],left=q[7];
 for(unsigned tile=first,tile_size=tile_rows(8,channels);tile<end;tile+=tile_size){
 unsigned stop=tile+tile_size;if(stop>end)stop=end;
 PHASE_START();
 for(unsigned row=tile;row<stop;row++){
  const int8_t*ptr[9];
  int yy=(int)h*sh-top,xx=(int)v*sw-left;
  if(yy>=0&&xx>=0&&(unsigned)(yy+2*dh)<ih&&(unsigned)(xx+2*dw)<iw){
   const int8_t*base=x+((n*ih+yy)*iw+xx)*input_channels;
   for(int kh=0;kh<3;kh++)for(int kw=0;kw<3;kw++)ptr[kh*3+kw]=base+(kh*dh*iw+kw*dw)*input_channels;
  }else{
   for(int kh=0;kh<3;kh++)for(int kw=0;kw<3;kw++){
    int sy=yy+kh*dh,sx=xx+kw*dw;
    ptr[kh*3+kw]=sy>=0&&sx>=0&&(unsigned)sy<ih&&(unsigned)sx<iw?x+((n*ih+sy)*iw+sx)*input_channels:pad;
   }
  }
  if(++v==ow){v=0;if(++h==oh){h=0;n++;}}
  int32_t*dst=acc+(row-tile)*channels;
#if defined(UQ_DW_LEAF) && defined(UQ_DW_DOT) && defined(__hexagon__)
  if(channels>=UQ_DW_LEAF)depthwise_dot_row(ptr,dot_weights,channels,dst);
  else
#endif
#ifdef __clang__
#pragma clang loop unroll(full)
#endif
  for(unsigned c=0;c<channels;c+=4){
#ifdef UQ_RANGE_DW_SKIP
   if(prep&&prep->value[c]!=-256&&prep->value[c+1]!=-256&&prep->value[c+2]!=-256&&prep->value[c+3]!=-256){dst[c]=dst[c+1]=dst[c+2]=dst[c+3]=0;continue;}
#endif
#ifdef __hexagon__
   long long lo=0,hi=0;
#ifdef UQ_DW_DOT
   #pragma clang loop unroll(full)
   for(unsigned block=0;block<2;block++){
    int v0,v1,v2,v3;unsigned t=block*4;
    __builtin_memcpy(&v0,__builtin_assume_aligned(ptr[t]+c,4),4);
    __builtin_memcpy(&v1,__builtin_assume_aligned(ptr[t+1]+c,4),4);
    __builtin_memcpy(&v2,__builtin_assume_aligned(ptr[t+2]+c,4),4);
    __builtin_memcpy(&v3,__builtin_assume_aligned(ptr[t+3]+c,4),4);
    long long a=(long long)(((uint64_t)(uint32_t)v2<<32)|(uint32_t)v0)^INT64_C(-9187201950435737472);
    long long b=(long long)(((uint64_t)(uint32_t)v3<<32)|(uint32_t)v1)^INT64_C(-9187201950435737472);
    long long even=Q6_P_shuffeb_PP(b,a),odd=Q6_P_shuffob_PP(b,a);
    long long x01=Q6_P_vtrunewh_PP(odd,even),x23=Q6_P_vtrunowh_PP(odd,even),w01,w23;
    __builtin_memcpy(&w01,__builtin_assume_aligned(dot_weights+(c/4)*40+block*16,8),8);
    __builtin_memcpy(&w23,__builtin_assume_aligned(dot_weights+(c/4)*40+block*16+8,8),8);
    lo=Q6_P_vrmpybsuacc_PP(lo,w01,x01);hi=Q6_P_vrmpybsuacc_PP(hi,w23,x23);
   }
   int xv;long long tail;
   __builtin_memcpy(&xv,__builtin_assume_aligned(ptr[8]+c,4),4);
   __builtin_memcpy(&tail,__builtin_assume_aligned(dot_weights+(c/4)*40+32,8),8);
   long long xx=Q6_P_vzxtbh_R(xv^(int)0x80808080u);
   lo=Q6_P_vmpyhacc_RR(lo,(int32_t)tail,(int32_t)xx);
   hi=Q6_P_vmpyhacc_RR(hi,(int32_t)((uint64_t)tail>>32),(int32_t)((uint64_t)xx>>32));
#else
#ifdef __clang__
#pragma clang loop unroll(full)
#endif
   for(unsigned t=0;t<9;t++){
#if defined(UQ_DEPTHWISE_HALFWORD) && !defined(UQ_DW_DOT)
    int xv;long long ww;
    __builtin_memcpy(&xv,__builtin_assume_aligned(ptr[t]+c,4),4);
    __builtin_memcpy(&ww,__builtin_assume_aligned(wide_weights+2*(t*channels+c),8),8);
    long long xx=Q6_P_vzxtbh_R(xv^(int)0x80808080u);
    /* Each partial sum is bounded by 9*255*128 = 293760; no saturation,
     * overflow, or precision loss. This still evaluates the INT8 model. */
    lo=Q6_P_vmpyhacc_RR(lo,(int32_t)ww,(int32_t)xx);
    hi=Q6_P_vmpyhacc_RR(hi,(int32_t)((uint64_t)ww>>32),(int32_t)((uint64_t)xx>>32));
#else
    int xv,wv;__builtin_memcpy(&xv,__builtin_assume_aligned(ptr[t]+c,4),4);__builtin_memcpy(&wv,__builtin_assume_aligned(w+t*channels+c,4),4);
    long long prod=Q6_P_vmpybsu_RR(wv,xv^(int)0x80808080u);
    lo=Q6_P_vaddw_PP(lo,Q6_P_vsxthw_R((int32_t)prod));
    hi=Q6_P_vaddw_PP(hi,Q6_P_vsxthw_R((int32_t)((uint64_t)prod>>32)));
#endif
   }
#endif
   int32_t v0=(int32_t)lo,v1=(int32_t)((uint64_t)lo>>32),v2=(int32_t)hi,v3=(int32_t)((uint64_t)hi>>32);
#else
   int32_t v0=0,v1=0,v2=0,v3=0;
   for(unsigned t=0;t<9;t++){
    const int8_t*xx=ptr[t]+c;
#if defined(UQ_DEPTHWISE_HALFWORD) && !defined(UQ_DW_DOT)
    int16_t ww[4];__builtin_memcpy(ww,wide_weights+2*(t*channels+c),8);
#else
    const int8_t*ww=w+t*channels+c;
#endif
    v0+=((int)xx[0]+128)*ww[0];v1+=((int)xx[1]+128)*ww[1];v2+=((int)xx[2]+128)*ww[2];v3+=((int)xx[3]+128)*ww[3];
   }
#endif
#ifdef UQ_DW_SCALE
   if(direct_scale){
    int32_t vv[4]={v0,v1,v2,v3};
#ifdef __clang__
#pragma clang loop unroll(full)
#endif
    for(unsigned j=0;j<4;j+=2){
     const UQAffine*t=direct_scale+c+j;
#ifdef __hexagon__
     int64_t p0=Q6_P_mpyacc_RR(t[0].constant,vv[j],t[0].multiplier),p1=Q6_P_mpyacc_RR(t[1].constant,vv[j+1],t[1].multiplier);
     int32_t a0=Q6_R_asr_RR((int32_t)(p0>>32),t[0].shift-32),a1=Q6_R_asr_RR((int32_t)(p1>>32),t[1].shift-32);
     long long both=(long long)((uint64_t)(uint32_t)a0|((uint64_t)(uint32_t)a1<<32));
     long long lo=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
     long long hi=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
     both=Q6_P_vmaxw_PP(lo,Q6_P_vminw_PP(hi,both));
     uint16_t bytes=(uint16_t)Q6_R_vsathb_R(Q6_R_vsatwh_P(both));
#else
     int32_t a0=(int32_t)((t[0].constant+(int64_t)vv[j]*t[0].multiplier)>>t[0].shift),a1=(int32_t)((t[1].constant+(int64_t)vv[j+1]*t[1].multiplier)>>t[1].shift);
     uint16_t bytes=(uint16_t)((uint8_t)clamp(a0,q)|((uint16_t)(uint8_t)clamp(a1,q)<<8));
#endif
     __builtin_memcpy(__builtin_assume_aligned(y+row*channels+c+j,2),&bytes,2);
    }
   }else
#endif
   {dst[c]=v0;dst[c+1]=v1;dst[c+2]=v2;dst[c+3]=v3;}
  }
 }
 PHASE_END(scratch,3);
#ifdef UQ_DW_SCALE
 if(!direct_scale)
#endif
 requantize(acc,y+tile*channels,stop-tile,channels,q);
 PHASE_END(scratch,2);
 }
}
#endif
#if UQ_DW_FIXED >= 32
static __attribute__((noinline)) void depthwise3_c32(const UGTensor*a,const UGTensor*z,const int8_t*UQ_NOALIAS x,const int8_t*UQ_NOALIAS w,int8_t*UQ_NOALIAS y,const int32_t*UQ_NOALIAS q,unsigned first,unsigned end,int32_t*UQ_NOALIAS acc,unsigned scratch){
 unsigned channels=32,ih=a->d[1],iw=a->d[2];
#ifdef UQ_DW_SCALE
 const UQAffine*direct_scale=lookup_affine(q);
#endif
#ifdef UQ_RANGE_DW_SKIP
 const UQPrepared*prep=lookup(q);
#endif
 int8_t pad[32] __attribute__((aligned(8)));
 for(unsigned c=0;c<channels;c++)pad[c]=(int8_t)q[12];
#if defined(UQ_DEPTHWISE_HALFWORD) && !defined(UQ_DW_DOT)
 /* Reuse this worker's otherwise unused gather buffer; no new allocation.
  * At most 9*1024*2 bytes. Expand weights once per layer invocation, rather
  * than widening four products and adding them separately at every tap. */
 uint8_t*wide_weights=columns[scratch];
 for(unsigned c=0;c<9*channels;c+=4){
#ifdef __hexagon__
  int packed;__builtin_memcpy(&packed,__builtin_assume_aligned(w+c,4),4);
  long long wide=Q6_P_vsxtbh_R(packed);
  __builtin_memcpy(wide_weights+2*c,&wide,8);
#else
  int16_t wide[4];for(unsigned lane=0;lane<4;lane++)wide[lane]=w[c+lane];
  __builtin_memcpy(wide_weights+2*c,wide,8);
#endif
 }
#endif
 unsigned oh=z->d[1],ow=z->d[2];
 unsigned input_channels=channels;
#ifdef UQ_DW_PAD
 if(x==dw_padded_input)input_channels+=8;
#endif
#ifdef UQ_DW_DOT
 /* Two four-tap dot groups for each channel, plus a wide ninth-tap product.
  * This changes data arrangement only; no partial sum is narrowed. */
 uint8_t*dot_weights=columns[scratch];
 for(unsigned c=0;c<channels;c+=4){
  for(unsigned block=0;block<2;block++)for(unsigned lane=0;lane<4;lane++)for(unsigned t=0;t<4;t++)
   dot_weights[(c/4)*40+block*16+lane*4+t]=(uint8_t)w[(block*4+t)*channels+c+lane];
  int16_t tail[4];for(unsigned lane=0;lane<4;lane++)tail[lane]=w[8*channels+c+lane];
  __builtin_memcpy(dot_weights+(c/4)*40+32,tail,8);
 }
#endif
 unsigned n=first/(oh*ow),h=(first/ow)%oh,v=first%ow;
 const int sh=q[2],sw=q[3],dh=q[4],dw=q[5],top=q[6],left=q[7];
 for(unsigned tile=first,tile_size=tile_rows(8,channels);tile<end;tile+=tile_size){
 unsigned stop=tile+tile_size;if(stop>end)stop=end;
 PHASE_START();
 for(unsigned row=tile;row<stop;row++){
  const int8_t*ptr[9];
  int yy=(int)h*sh-top,xx=(int)v*sw-left;
  if(yy>=0&&xx>=0&&(unsigned)(yy+2*dh)<ih&&(unsigned)(xx+2*dw)<iw){
   const int8_t*base=x+((n*ih+yy)*iw+xx)*input_channels;
   for(int kh=0;kh<3;kh++)for(int kw=0;kw<3;kw++)ptr[kh*3+kw]=base+(kh*dh*iw+kw*dw)*input_channels;
  }else{
   for(int kh=0;kh<3;kh++)for(int kw=0;kw<3;kw++){
    int sy=yy+kh*dh,sx=xx+kw*dw;
    ptr[kh*3+kw]=sy>=0&&sx>=0&&(unsigned)sy<ih&&(unsigned)sx<iw?x+((n*ih+sy)*iw+sx)*input_channels:pad;
   }
  }
  if(++v==ow){v=0;if(++h==oh){h=0;n++;}}
  int32_t*dst=acc+(row-tile)*channels;
#if defined(UQ_DW_LEAF) && defined(UQ_DW_DOT) && defined(__hexagon__)
  if(channels>=UQ_DW_LEAF)depthwise_dot_row(ptr,dot_weights,channels,dst);
  else
#endif
#ifdef __clang__
#pragma clang loop unroll(full)
#endif
  for(unsigned c=0;c<channels;c+=4){
#ifdef UQ_RANGE_DW_SKIP
   if(prep&&prep->value[c]!=-256&&prep->value[c+1]!=-256&&prep->value[c+2]!=-256&&prep->value[c+3]!=-256){dst[c]=dst[c+1]=dst[c+2]=dst[c+3]=0;continue;}
#endif
#ifdef __hexagon__
   long long lo=0,hi=0;
#ifdef UQ_DW_DOT
   #pragma clang loop unroll(full)
   for(unsigned block=0;block<2;block++){
    int v0,v1,v2,v3;unsigned t=block*4;
    __builtin_memcpy(&v0,__builtin_assume_aligned(ptr[t]+c,4),4);
    __builtin_memcpy(&v1,__builtin_assume_aligned(ptr[t+1]+c,4),4);
    __builtin_memcpy(&v2,__builtin_assume_aligned(ptr[t+2]+c,4),4);
    __builtin_memcpy(&v3,__builtin_assume_aligned(ptr[t+3]+c,4),4);
    long long a=(long long)(((uint64_t)(uint32_t)v2<<32)|(uint32_t)v0)^INT64_C(-9187201950435737472);
    long long b=(long long)(((uint64_t)(uint32_t)v3<<32)|(uint32_t)v1)^INT64_C(-9187201950435737472);
    long long even=Q6_P_shuffeb_PP(b,a),odd=Q6_P_shuffob_PP(b,a);
    long long x01=Q6_P_vtrunewh_PP(odd,even),x23=Q6_P_vtrunowh_PP(odd,even),w01,w23;
    __builtin_memcpy(&w01,__builtin_assume_aligned(dot_weights+(c/4)*40+block*16,8),8);
    __builtin_memcpy(&w23,__builtin_assume_aligned(dot_weights+(c/4)*40+block*16+8,8),8);
    lo=Q6_P_vrmpybsuacc_PP(lo,w01,x01);hi=Q6_P_vrmpybsuacc_PP(hi,w23,x23);
   }
   int xv;long long tail;
   __builtin_memcpy(&xv,__builtin_assume_aligned(ptr[8]+c,4),4);
   __builtin_memcpy(&tail,__builtin_assume_aligned(dot_weights+(c/4)*40+32,8),8);
   long long xx=Q6_P_vzxtbh_R(xv^(int)0x80808080u);
   lo=Q6_P_vmpyhacc_RR(lo,(int32_t)tail,(int32_t)xx);
   hi=Q6_P_vmpyhacc_RR(hi,(int32_t)((uint64_t)tail>>32),(int32_t)((uint64_t)xx>>32));
#else
#ifdef __clang__
#pragma clang loop unroll(full)
#endif
   for(unsigned t=0;t<9;t++){
#if defined(UQ_DEPTHWISE_HALFWORD) && !defined(UQ_DW_DOT)
    int xv;long long ww;
    __builtin_memcpy(&xv,__builtin_assume_aligned(ptr[t]+c,4),4);
    __builtin_memcpy(&ww,__builtin_assume_aligned(wide_weights+2*(t*channels+c),8),8);
    long long xx=Q6_P_vzxtbh_R(xv^(int)0x80808080u);
    /* Each partial sum is bounded by 9*255*128 = 293760; no saturation,
     * overflow, or precision loss. This still evaluates the INT8 model. */
    lo=Q6_P_vmpyhacc_RR(lo,(int32_t)ww,(int32_t)xx);
    hi=Q6_P_vmpyhacc_RR(hi,(int32_t)((uint64_t)ww>>32),(int32_t)((uint64_t)xx>>32));
#else
    int xv,wv;__builtin_memcpy(&xv,__builtin_assume_aligned(ptr[t]+c,4),4);__builtin_memcpy(&wv,__builtin_assume_aligned(w+t*channels+c,4),4);
    long long prod=Q6_P_vmpybsu_RR(wv,xv^(int)0x80808080u);
    lo=Q6_P_vaddw_PP(lo,Q6_P_vsxthw_R((int32_t)prod));
    hi=Q6_P_vaddw_PP(hi,Q6_P_vsxthw_R((int32_t)((uint64_t)prod>>32)));
#endif
   }
#endif
   int32_t v0=(int32_t)lo,v1=(int32_t)((uint64_t)lo>>32),v2=(int32_t)hi,v3=(int32_t)((uint64_t)hi>>32);
#else
   int32_t v0=0,v1=0,v2=0,v3=0;
   for(unsigned t=0;t<9;t++){
    const int8_t*xx=ptr[t]+c;
#if defined(UQ_DEPTHWISE_HALFWORD) && !defined(UQ_DW_DOT)
    int16_t ww[4];__builtin_memcpy(ww,wide_weights+2*(t*channels+c),8);
#else
    const int8_t*ww=w+t*channels+c;
#endif
    v0+=((int)xx[0]+128)*ww[0];v1+=((int)xx[1]+128)*ww[1];v2+=((int)xx[2]+128)*ww[2];v3+=((int)xx[3]+128)*ww[3];
   }
#endif
#ifdef UQ_DW_SCALE
   if(direct_scale){
    int32_t vv[4]={v0,v1,v2,v3};
#ifdef __clang__
#pragma clang loop unroll(full)
#endif
    for(unsigned j=0;j<4;j+=2){
     const UQAffine*t=direct_scale+c+j;
#ifdef __hexagon__
     int64_t p0=Q6_P_mpyacc_RR(t[0].constant,vv[j],t[0].multiplier),p1=Q6_P_mpyacc_RR(t[1].constant,vv[j+1],t[1].multiplier);
     int32_t a0=Q6_R_asr_RR((int32_t)(p0>>32),t[0].shift-32),a1=Q6_R_asr_RR((int32_t)(p1>>32),t[1].shift-32);
     long long both=(long long)((uint64_t)(uint32_t)a0|((uint64_t)(uint32_t)a1<<32));
     long long lo=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32));
     long long hi=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
     both=Q6_P_vmaxw_PP(lo,Q6_P_vminw_PP(hi,both));
     uint16_t bytes=(uint16_t)Q6_R_vsathb_R(Q6_R_vsatwh_P(both));
#else
     int32_t a0=(int32_t)((t[0].constant+(int64_t)vv[j]*t[0].multiplier)>>t[0].shift),a1=(int32_t)((t[1].constant+(int64_t)vv[j+1]*t[1].multiplier)>>t[1].shift);
     uint16_t bytes=(uint16_t)((uint8_t)clamp(a0,q)|((uint16_t)(uint8_t)clamp(a1,q)<<8));
#endif
     __builtin_memcpy(__builtin_assume_aligned(y+row*channels+c+j,2),&bytes,2);
    }
   }else
#endif
   {dst[c]=v0;dst[c+1]=v1;dst[c+2]=v2;dst[c+3]=v3;}
  }
 }
 PHASE_END(scratch,3);
#ifdef UQ_DW_SCALE
 if(!direct_scale)
#endif
 requantize(acc,y+tile*channels,stop-tile,channels,q);
 PHASE_END(scratch,2);
 }
}
#endif
