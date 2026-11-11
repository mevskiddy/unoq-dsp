/* Horizontal windows packed four channels at a time. Each lane holds three
 * unsigned pixels plus a zero tap. Workers prepare disjoint source-row ranges
 * except the bounded two-row halo. NHWC output and exact scaling are retained. */
static uint8_t dw_window4[2][98304] __attribute__((aligned(128)));
static uint64_t dw_window4_pad[2][UQ_MAX_C/2] __attribute__((aligned(128)));
static void depthwise_window4(const UGTensor*a,const UGTensor*z,const int8_t*x,const int8_t*w,int8_t*y,const int32_t*q,unsigned first,unsigned end,int32_t*acc,unsigned scratch){
 unsigned channels=q[17],ih=a->d[1],iw=a->d[2],ow=z->d[2];
 uint8_t*buf=dw_window4[scratch],*weights=columns[scratch];
 uint32_t zz=(uint32_t)(q[12]+128)*UINT32_C(0x00010101);
 uint64_t pad=(uint64_t)zz|((uint64_t)zz<<32);
 for(unsigned c=0;c<channels;c+=4)for(unsigned h=0;h<3;h++)for(unsigned lane=0;lane<4;lane++){
  for(unsigned t=0;t<3;t++)weights[c*12+h*16+lane*4+t]=(uint8_t)w[(h*3+t)*channels+c+lane];
  weights[c*12+h*16+lane*4+3]=0;
 }
 int sy0=(int)(first/ow)*q[2]-q[6],sy1=(int)((end-1)/ow)*q[2]-q[6]+3;
 if(sy0<0)sy0=0;if(sy1>(int)ih)sy1=(int)ih;
 int8_t pixel_pad[UQ_MAX_C] __attribute__((aligned(8)));
 for(unsigned c=0;c<channels;c++)pixel_pad[c]=(int8_t)q[12];
 for(unsigned c=0;c<channels/2;c++)dw_window4_pad[scratch][c]=pad;
 for(int sy=sy0;sy<sy1;sy++)for(unsigned v=0;v<ow;v++){
  int sx=(int)v*q[3]-q[7];const int8_t*src[3];
  for(unsigned t=0;t<3;t++){
   int xx=sx+(int)t;src[t]=xx>=0&&(unsigned)xx<iw?x+(sy*iw+xx)*channels:pixel_pad;
  }
  for(unsigned c=0;c<channels;c+=4){
   uint32_t v0,v1,v2;
   __builtin_memcpy(&v0,__builtin_assume_aligned(src[0]+c,4),4);
   __builtin_memcpy(&v1,__builtin_assume_aligned(src[1]+c,4),4);
   __builtin_memcpy(&v2,__builtin_assume_aligned(src[2]+c,4),4);
   v0^=UINT32_C(0x80808080);v1^=UINT32_C(0x80808080);v2^=UINT32_C(0x80808080);
   long long aa=(long long)((uint64_t)v0|((uint64_t)v2<<32)),bb=(long long)(uint64_t)v1;
   long long even=Q6_P_shuffeb_PP(bb,aa),odd=Q6_P_shuffob_PP(bb,aa);
   long long x01=Q6_P_vtrunewh_PP(odd,even),x23=Q6_P_vtrunowh_PP(odd,even);
   uint8_t*dst=buf+((unsigned)sy*ow+v)*channels*4+c*4;
   __builtin_memcpy(__builtin_assume_aligned(dst,8),&x01,8);
   __builtin_memcpy(__builtin_assume_aligned(dst+8,8),&x23,8);
  }
 }
 unsigned yy=first/ow,v=first%ow;
 for(unsigned tile=first,size=tile_rows(8,channels);tile<end;tile+=size){
  unsigned stop=tile+size;if(stop>end)stop=end;
  for(unsigned r=tile;r<stop;r++){
   int sy=(int)yy*q[2]-q[6];const uint8_t*src[3];
   for(unsigned h=0;h<3;h++)src[h]=sy+(int)h>=0&&(unsigned)(sy+(int)h)<ih?
    buf+((unsigned)(sy+(int)h)*ow+v)*channels*4:(const uint8_t*)dw_window4_pad[scratch];
   for(unsigned c=0;c<channels;c+=4){
    long long a0=0,a1=0;
#pragma clang loop unroll(full)
    for(unsigned h=0;h<3;h++){
     long long x0,x1,w0,w1;
     __builtin_memcpy(&x0,__builtin_assume_aligned(src[h]+c*4,8),8);
     __builtin_memcpy(&x1,__builtin_assume_aligned(src[h]+c*4+8,8),8);
     __builtin_memcpy(&w0,__builtin_assume_aligned(weights+c*12+h*16,8),8);
     __builtin_memcpy(&w1,__builtin_assume_aligned(weights+c*12+h*16+8,8),8);
     a0=Q6_P_vrmpybsuacc_PP(a0,w0,x0);a1=Q6_P_vrmpybsuacc_PP(a1,w1,x1);
    }
    __builtin_memcpy(__builtin_assume_aligned(acc+(r-tile)*channels+c,8),&a0,8);
    __builtin_memcpy(__builtin_assume_aligned(acc+(r-tile)*channels+c+2,8),&a1,8);
   }
   if(++v==ow){v=0;yy++;}
  }
  requantize(acc,y+tile*channels,stop-tile,channels,q);
 }
}
