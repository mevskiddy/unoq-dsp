#include "qnn.h"
#include "gemm.h"
#include <limits.h>
#ifdef UQ_RESTRICT
#define UQ_NOALIAS restrict
#else
#define UQ_NOALIAS
#endif
#ifdef __hexagon__
#include <hexagon_protos.h>
#endif
/* q[0:12]=kh,kw,sh,sw,dh,dw,pt,pl,pb,pr,oh,ow;
   q[12:20]=signed xzp,yzp,activation min,max,IC,OC,padded K,flags;
   q[20:]=corrected bias[OC], Q31 multipliers[OC], shifts[OC].
   flags: conv 0 packed / 1 scalar oracle; pool 0 average / 1 max.
   Softmax uses 256 positive Q24 LUT entries at q[20:]. */
static uint8_t columns[2][UQ_TILE*UQ_MAX_K] __attribute__((aligned(128)));
static int32_t accumulators[2][UQ_TILE*UQ_MAX_C] __attribute__((aligned(128)));
/* Reuse the bounded scratch more efficiently on small vision layers. No
 * allocation or larger maximum workspace: large K/C still choose small tiles. */
#ifndef UQ_TILE_TARGET
#define UQ_TILE_TARGET 32
#endif
static unsigned tile_rows(unsigned k,unsigned channels){
 unsigned n=UQ_TILE_TARGET;
#ifdef UQ_ADAPTIVE_TILE
 n=channels>=32?16:64;
#endif
#ifdef UQ_TILE_BUDGET
 unsigned resident=UQ_TILE_BUDGET/k;if(resident<2)resident=2;
 if(n>resident)n=resident&~1u;
#endif
 unsigned x=UQ_TILE*UQ_MAX_K/k,y=UQ_TILE*UQ_MAX_C/channels;
 if(n>x)n=x;if(n>y)n=y;return n;
}
#ifdef UQ_DIAGNOSTICS
/* Separate counters per worker; enabled only in diagnostic builds. */
static struct {uint64_t ticks[4];uint8_t pad[96];} phases[2] __attribute__((aligned(128)));
static int measuring;
static uint64_t phase_time(void){
#ifdef __hexagon__
 extern uint64_t qurt_timer_get_ticks(void);return measuring?qurt_timer_get_ticks():0;
#else
 return 0;
#endif
}
void uq_measurement(unsigned command,uint64_t*out){
 if(command==1){for(unsigned w=0;w<2;w++)for(unsigned k=0;k<4;k++)phases[w].ticks[k]=0;}
 if(command<2)measuring=command;
 for(unsigned w=0;w<2;w++)for(unsigned k=0;k<4;k++)out[w*4+k]=phases[w].ticks[k];
}
#define PHASE_START() uint64_t phase_start=phase_time()
#define PHASE_END(worker,kind) do{if(measuring){uint64_t end=phase_time();phases[worker].ticks[kind]+=end-phase_start;phase_start=end;}}while(0)
#else
#define PHASE_START() ((void)0)
#define PHASE_END(worker,kind) ((void)(worker),(void)(kind))
#endif
static int8_t clamp(int32_t x,const int32_t*q){return (int8_t)(x<q[14]?q[14]:x>q[15]?q[15]:x);}
/* Explicit integer definitions avoid signed left-shift and overflow UB.
   Loader proves the pre-shift accumulator bound fits INT32. */
static int32_t scale(int32_t v,int32_t m,int shift){
 if(shift>0)v=(int32_t)((uint32_t)v<<shift);
#ifdef __hexagon__
 /* Force a signed 32x32 -> 64 multiply-accumulate. The generic C expression
    can be widened/hoisted into three multiplies by LLVM in the pixel loop. */
 int64_t p=Q6_P_mpyacc_RR((int64_t)1<<30,v,m);
 int32_t h=(int32_t)(p>>31);
#else
 int64_t p=(int64_t)v*m;
 int32_t h=(int32_t)((p+((int64_t)1<<30))>>31);
#endif
 if(shift>=0)return h;
#ifdef __hexagon__
 /* round() uses a wide intermediate and ties toward +infinity. Subtract one
    from negative h to obtain TFLite ties away from zero. Since m>=0 and
    m<=INT32_MAX, h cannot be INT32_MIN, even for v==INT32_MIN. */
 return Q6_R_round_RR(h-(h<0),-shift);
#else
 unsigned s=(unsigned)-shift;uint32_t mask=((uint32_t)1<<s)-1;
 int32_t base=h>>s;uint32_t remainder=(uint32_t)h&mask;
 return base+(remainder>((mask>>1)+(h<0)));
#endif
}
/* Fixed-point reciprocal from gemmlowp (Apache-2.0), specialized to positive
 * Q12 sums and Q0 exponential terms. This removes per-element 64-bit division
 * and preserves reference rounding in attention rather than approximating it. */
void uq_softmax_exact(const int8_t*x,int8_t*y,unsigned bytes,const int32_t*q){
 unsigned width=q[0];const int32_t*lut=q+1;
 for(unsigned base=0;base<bytes;base+=width){
  int maximum=-128;for(unsigned j=0;j<width;j++)if(x[base+j]>maximum)maximum=x[base+j];
  uint32_t sum=0;for(unsigned j=0;j<width;j++)sum+=((uint32_t)lut[maximum-x[base+j]]+2048u)>>12;
  unsigned leading=(unsigned)__builtin_clz(sum),exponent=35-leading;
  uint32_t normalized=(sum<<leading)-0x80000000u;
  int32_t half=(int32_t)(((uint64_t)normalized+2147483648u)>>1);
  int32_t reciprocal=1515870810+scale(half,-1010580540,0);
  for(unsigned step=0;step<3;step++){
   int32_t error=536870912-scale(half,reciprocal,0);
   reciprocal+=scale(reciprocal,error,0)*4;
  }
  int64_t doubled=(int64_t)reciprocal*2;reciprocal=doubled>2147483647?2147483647:(int32_t)doubled;
  for(unsigned j=0;j<width;j++){
   uint32_t value=(uint32_t)scale(reciprocal,lut[maximum-x[base+j]],0);
   unsigned rounded=exponent>=32?0:(value+(1u<<(exponent-1)))>>exponent;
   y[base+j]=(int8_t)(rounded>255?127:(int)rounded-128);
  }
 }
}

/* Runtime RHS attention matrices use the same packed dot-product kernel as
 * convolution. Bounded column tiles avoid a full-image INT32 intermediate. */
static int8_t bmm_weights[2][32*UQ_MAX_K] __attribute__((aligned(128)));
void uq_mul(const int8_t*x,const int8_t*b,int8_t*y,unsigned count,const int32_t*q){
 for(unsigned i=0;i<count;i++){
  int32_t value=scale(((int)x[i]-q[0])*((int)b[i]-q[1]),q[3],q[4])+q[2];
  y[i]=(int8_t)(value<q[5]?q[5]:value>q[6]?q[6]:value);
 }
}
typedef struct {const UGTensor*a,*b,*z;const int8_t*x,*w;int8_t*y;const int32_t*q;unsigned first,end,scratch;} BMMJob;
static int bmm_job(void*opaque){
 BMMJob*j=opaque;const int32_t*q=j->q;unsigned m=j->z->d[1],n=j->z->d[2],k=j->a->d[q[5]?1:2],kp=(k+7)&~7u;
 unsigned xs=kp+(kp%32==0?8:0);uint8_t*col=columns[j->scratch];int32_t*acc=accumulators[j->scratch];int8_t*pack=bmm_weights[j->scratch];
 for(unsigned batch=j->first/m;batch<(j->end+m-1)/m;batch++){
  unsigned begin=j->first>batch*m?j->first-batch*m:0,end=j->end-batch*m;if(end>m)end=m;
  const int8_t*x=j->x+batch*m*k,*w=j->w+batch*k*n;int8_t*y=j->y+batch*m*n;
  for(unsigned base=0;base<n;base+=32){
   unsigned nc=n-base;if(nc>32)nc=32;unsigned np=(nc+3)&~3u;int32_t sums[32]={0};
   for(unsigned c=0;c<np;c++)for(unsigned t=0;t<kp;t++){
    int v=c<nc&&t<k?w[q[6]?(base+c)*k+t:t*n+base+c]:0;
    pack[(c/4)*kp*4+(t/8)*32+(c%4)*8+t%8]=(int8_t)v;sums[c]+=v;
   }
   unsigned tile=tile_rows(xs,np);
   for(unsigned row=begin;row<end;row+=tile){
    unsigned nr=end-row;if(nr>tile)nr=tile;int32_t sx[UQ_TILE_TARGET];
    for(unsigned r=0;r<nr;r++){
     int32_t sum=0;
     for(unsigned t=0;t<k;t++){unsigned v=(uint8_t)x[q[5]?t*m+row+r:(row+r)*k+t]^128u;col[r*xs+t]=(uint8_t)v;sum+=v;}
     for(unsigned t=k;t<kp;t++)col[r*xs+t]=0;sx[r]=sum;
    }
    int rc=ug_matmul_u8i8_strided(col,pack,acc,nr,np,kp,xs);if(rc)return rc;
    for(unsigned r=0;r<nr;r++)for(unsigned c=0;c<nc;c++){
     int32_t v=acc[r*np+c]-(q[0]+128)*sums[c]-q[1]*sx[r]+(int32_t)k*(q[0]+128)*q[1];
     v=scale(v,q[3],q[4])+q[2];y[(row+r)*n+base+c]=(int8_t)(v<-128?-128:v>127?127:v);
    }
   }
  }
 }
 return 0;
}
int uq_bmm(const UGTensor*a,const UGTensor*b,const UGTensor*z,const int8_t*x,const int8_t*w,int8_t*y,const int32_t*q,int threads){
 unsigned rows=z->d[0]*z->d[1];BMMJob jobs[2]={{a,b,z,x,w,y,q,0,rows,0},{a,b,z,x,w,y,q,0,rows,1}};
 if(threads==2&&rows>=8){jobs[0].end=rows/2;jobs[1].first=rows/2;int rc=ug_run_pair(bmm_job,jobs,jobs+1);if(rc!=-1)return rc;jobs[0].end=rows;}
 return bmm_job(jobs);
}

/* Fused post-activation tables are immutable per loaded graph. Worker threads
 * apply them while the just-produced quantized tile is still cache resident. */
static const int32_t*post_keys[UG_MAX_OP];
static const uint8_t*post_tables[UG_MAX_OP];
static unsigned post_count;
void uq_post_reset(void){post_count=0;}
void uq_post_register(const int32_t*q,const uint8_t*table){if(post_count<UG_MAX_OP){post_keys[post_count]=q;post_tables[post_count++]=table;}}
static void uq_post(const int32_t*q,int8_t*y,unsigned count){
 const uint8_t*table=0;
 for(unsigned i=0;i<post_count;i++)if(post_keys[i]==q){table=post_tables[i];break;}
 if(!table)return;
 unsigned j=0;
 for(;j+4<=count;j+=4){
  uint32_t word;__builtin_memcpy(&word,y+j,4);word^=0x80808080u;
  uint32_t result=table[word&255]|((uint32_t)table[(word>>8)&255]<<8)|((uint32_t)table[(word>>16)&255]<<16)|((uint32_t)table[word>>24]<<24);
  __builtin_memcpy(y+j,&result,4);
 }
 for(;j<count;j++)y[j]=(int8_t)table[(int)y[j]+128];
}

void uq_add(const int8_t*UQ_NOALIAS x,const int8_t*UQ_NOALIAS b,
            int8_t*UQ_NOALIAS y,uint32_t count,const int32_t*q){
 const int32_t*tx=q+8,*tb=q+264;
 for(uint32_t i=0;i<count;i++){
  int32_t sum=tx[(int)x[i]+128]+tb[(int)b[i]+128];
  int32_t value=scale(sum,q[0],q[1])+q[2];
  y[i]=(int8_t)(value<q[3]?q[3]:value>q[4]?q[4]:value);
 }
}
void uq_mean(const int8_t*UQ_NOALIAS x,int8_t*UQ_NOALIAS y,unsigned batches,unsigned channels,const int32_t*q){
 /* Contiguous channel tiles avoid a complete strided image scan per channel. */
 for(unsigned n=0;n<batches;n++)for(unsigned base=0;base<channels;base+=32){
  unsigned count=channels-base;if(count>32)count=32;
  int32_t sums[32]={0};
  for(unsigned p=0;p<(unsigned)q[0];p++){
   const int8_t*row=x+((n*q[0]+p)*channels)+base;
   for(unsigned c=0;c<count;c++)sums[c]+=row[c];
  }
  for(unsigned c=0;c<count;c++){
   int32_t value=scale(sums[c]-q[0]*q[1],q[3],q[4])+q[2];
   y[n*channels+base+c]=(int8_t)(value<-128?-128:value>127?127:value);
  }
 }
}
#ifdef __hexagon__
/* When doubling the pre-shifted accumulator is proven safe, the rounded high
 * product instruction computes exactly (v*m + 2^30)>>31 in one instruction.
 * The caller proves this bound once per channel, not for each output pixel. */
static int32_t scale_narrow(int32_t v,int32_t m,int shift){
 unsigned left=(shift>0?(unsigned)shift:0)+1;
 int32_t h=Q6_R_mpy_RR_rnd((int32_t)((uint32_t)v<<left),m);
 return shift<0?Q6_R_round_RR(h-(h<0),-shift):h;
}
#endif
int uq_validate(const UGOp*o,const UGTensor*a,const UGTensor*b,const UGTensor*z,const UGTensor*t,const void*data,const void*weights,uint64_t*work){
 const int32_t*q=data;
 if(a->type!=3||z->type!=3||!t->flags||t->type!=4||t->bytes<UQ_HEAD*4||((o->code!=14&&o->code!=15)&&o->p[1])||o->p[2]||o->p[3])return 18;
 if(q[12]<-128||q[12]>127||q[13]<-128||q[13]>127||q[14]<-128||q[15]>127||q[14]>q[15]||q[19]<0||q[19]>1)return 18;
 if(o->code==17){
  if(q[13]!=-128||q[14]!=-128||q[15]!=127||a->rank!=z->rank||a->bytes!=z->bytes||!a->rank||q[16]!=(int32_t)a->d[a->rank-1]||q[16]<1||q[16]>UQ_MAX_C||t->bytes!=(UQ_HEAD+256)*4)return 18;
  for(unsigned i=0;i<256;i++)if(q[UQ_HEAD+i]<0||q[UQ_HEAD+i]>(1<<24))return 18;
  for(unsigned d=0;d<a->rank;d++)if(a->d[d]!=z->d[d])return 18;
  if(q[UQ_HEAD]!=(1<<24))return 18;
  *work+=a->bytes;return 0;
 }
 if(a->rank!=4||z->rank!=4||a->d[0]!=z->d[0]||q[16]!=(int32_t)a->d[3]||q[17]!=(int32_t)z->d[3]||q[16]<1||q[16]>UQ_MAX_C||q[17]<1||q[17]>UQ_MAX_C)return 18;
 for(unsigned i=0;i<6;i++)if(q[i]<1||q[i]>4096)return 18;
 for(unsigned i=6;i<12;i++)if(q[i]<0||q[i]>4096)return 18;
 int64_t hh=(int64_t)a->d[1]+q[6]+q[8]-(int64_t)(q[0]-1)*q[4]-1;
 int64_t ww=(int64_t)a->d[2]+q[7]+q[9]-(int64_t)(q[1]-1)*q[5]-1;
 if(hh<0||ww<0||hh/q[2]+1!=q[10]||ww/q[3]+1!=q[11]||q[10]!=(int32_t)z->d[1]||q[11]!=(int32_t)z->d[2])return 18;
 uint64_t area=(uint64_t)q[0]*q[1],rows=(uint64_t)z->d[0]*z->d[1]*z->d[2];
 if(o->code==16){
  if(q[16]!=q[17]||q[12]!=q[13]||q[4]!=1||q[5]!=1||area>8192||t->bytes!=UQ_HEAD*4)return 18;
  /* Reject empty pool windows, including excessive explicit padding. */
  if(q[6]>=q[0]||q[7]>=q[1]||(q[10]-1)*(int64_t)q[2]-q[6]>=a->d[1]||(q[11]-1)*(int64_t)q[3]-q[7]>=a->d[2])return 18;
  *work+=rows*area*q[16];return 0;
 }
 uint64_t k=area*(o->code==15?1:q[16]);
 if(k>UQ_MAX_K||q[18]!=(int32_t)((k+7)&~7ULL)||!b->flags||b->type!=3||t->bytes!=(uint32_t)(UQ_HEAD+3*q[17])*4)return 18;
 if(o->code==15){if(q[16]!=q[17]||b->bytes!=area*q[17])return 18;}
 else if(b->bytes!=(uint32_t)((q[17]+3)&~3)*q[18])return 18;
 for(int c=0;c<q[17];c++){
  int shift=q[UQ_HEAD+2*q[17]+c];
  if(q[UQ_HEAD+q[17]+c]<0||shift < -31||shift>15)return 18;
  const int8_t*w=weights;int64_t lo=q[UQ_HEAD+c],hi=lo;
  unsigned terms=o->code==15?(unsigned)k:(unsigned)q[18];
  for(unsigned j=0;j<terms;j++){
   int v=o->code==15?w[j*q[17]+c]:w[(c/4)*terms*4+(j/8)*32+(c%4)*8+j%8];
   if(v<0)lo+=255*v;else hi+=255*v;
  }
  int64_t factor=1LL<<(shift>0?shift:0);
  if(lo*factor<INT32_MIN||hi*factor>INT32_MAX)return 18;
  int m=q[UQ_HEAD+q[17]+c];
  if((int64_t)scale((int32_t)lo,m,shift)+q[13]<INT32_MIN||(int64_t)scale((int32_t)hi,m,shift)+q[13]>INT32_MAX)return 18;
 }
 *work+=rows*k*q[17];return 0;
}
#ifdef UQ_CONSTANT_CHANNELS
/* A load-time interval proof, independent of any inference fixture. Every
 * unsigned input including padding lies in [0,255]. Monotone requantization
 * and clamping make equal endpoint outputs a proof of a constant channel. */
#ifdef UQ_PREPARED_SCALE
typedef struct {int64_t constant;int32_t multiplier;unsigned shift;} UQAffine;
static UQAffine affine_storage[32768] __attribute__((aligned(128)));
static unsigned affine_used;
static const int32_t*affine_keys[2*UG_MAX_OP];
static const UQAffine*affine_values[2*UG_MAX_OP];
static unsigned affine_count;
static void prepare_affine(const int32_t*q,const int16_t*values){
 unsigned oc=q[17];
 if(oc%2||q[14]<q[13]||affine_used+oc>32768||affine_count>=2*UG_MAX_OP)return;
 for(unsigned c=0;c<oc;c++){
  int m=q[UQ_HEAD+oc+c],s=q[UQ_HEAD+2*oc+c];
#ifdef UQ_SCALE_CONSTANTS
  if(values&&values[c]!=-256)continue;
#endif
  if((values&&values[c]!=-256)||!((s<0&&s>=-15)||m==0))return;
 }
 UQAffine*table=affine_storage+affine_used;
 for(unsigned c=0;c<oc;c++){
#ifdef UQ_SCALE_CONSTANTS
  if(values&&values[c]!=-256){
   table[c].constant=(int64_t)values[c]*INT64_C(4294967296);
   table[c].multiplier=0;table[c].shift=32;continue;
  }
#endif
  int m=q[UQ_HEAD+oc+c];unsigned n=m?(unsigned)-q[UQ_HEAD+2*oc+c]:1;
  table[c].constant=(int64_t)q[UQ_HEAD+c]*m+((int64_t)1<<30)+((int64_t)1<<(30+n))+(int64_t)q[13]*((int64_t)1<<(31+n));
  table[c].multiplier=m;table[c].shift=31+n;
 }
 affine_keys[affine_count]=q;affine_values[affine_count++]=table;affine_used+=oc;
}
static const UQAffine*lookup_affine(const int32_t*q){for(unsigned i=0;i<affine_count;i++)if(affine_keys[i]==q)return affine_values[i];return 0;}
#endif
#ifdef UQ_SIGNED_AFFINE
/* Collapse both quantization rounds even when activations may be negative.
 * H=floor(((acc+bias)*m+2^30)/2^31). For n=-shift>0, ties-away
 * is floor((H+2^(n-1)-(H<0))/2^n). The exact 64-bit affine form
 * therefore needs a 2^31 correction iff acc <= floor((-2^30-1)/m)-bias.
 * Raw dot accumulators are bounded strictly inside INT32 endpoints by K<=8192,
 * so clamping the cutoff to INT32 preserves always/never predicates. */
typedef struct {int64_t constant;int32_t multiplier;unsigned shift;int32_t cutoff;int32_t reserved[3];} UQSignedAffine;
static UQSignedAffine signed_storage[32768] __attribute__((aligned(128)));
static const int32_t*signed_keys[2*UG_MAX_OP];
static const UQSignedAffine*signed_values[2*UG_MAX_OP];
static unsigned signed_count,signed_used;
static void prepare_signed(const int32_t*q,const int16_t*values){
 unsigned oc=q[17];if(oc%2||q[14]>=q[13]||signed_used+oc>32768||signed_count>=2*UG_MAX_OP)return;
 for(unsigned c=0;c<oc;c++)if((!values||values[c]==-256)&&q[UQ_HEAD+oc+c]&&(q[UQ_HEAD+2*oc+c]>=0||q[UQ_HEAD+2*oc+c]<-15))return;
 UQSignedAffine*table=signed_storage+signed_used;
 for(unsigned c=0;c<oc;c++){
  if(values&&values[c]!=-256){table[c]=(UQSignedAffine){(int64_t)values[c]*INT64_C(4294967296),0,32,INT32_MIN,{0}};continue;}
  int m=q[UQ_HEAD+oc+c];unsigned n=m?(unsigned)-q[UQ_HEAD+2*oc+c]:1;
  int64_t constant=(int64_t)q[UQ_HEAD+c]*m+((int64_t)1<<30)+((int64_t)1<<(30+n))+(int64_t)q[13]*((int64_t)1<<(31+n));
  int64_t cutoff=m?-(((int64_t)1<<30)+m)/m-q[UQ_HEAD+c]:INT32_MIN;
  table[c]=(UQSignedAffine){constant,m,31+n,cutoff<INT32_MIN?INT32_MIN:cutoff>INT32_MAX?INT32_MAX:(int32_t)cutoff,{0}};
 }
 signed_keys[signed_count]=q;signed_values[signed_count++]=table;signed_used+=oc;
}
static const UQSignedAffine*lookup_signed(const int32_t*q){for(unsigned i=0;i<signed_count;i++)if(signed_keys[i]==q)return signed_values[i];return 0;}
static __attribute__((noinline)) void requantize_signed(const int32_t*restrict acc,int8_t*restrict y,unsigned rows,unsigned channels,const int32_t*q,const UQSignedAffine*restrict table){
 const int minimum=q[14],maximum=q[15];
 for(unsigned c=0;c<channels;c+=2){
  int64_t k0=table[c].constant,k1=table[c+1].constant;int m0=table[c].multiplier,m1=table[c+1].multiplier,t0=table[c].cutoff,t1=table[c+1].cutoff;
  unsigned s0=table[c].shift,s1=table[c+1].shift;
  for(unsigned r=0;r<rows;r++){
   int32_t a0=acc[r*channels+c],a1=acc[r*channels+c+1];
#ifdef __hexagon__
   int64_t p0=Q6_P_mpyacc_RR(k0,a0,m0),p1=Q6_P_mpyacc_RR(k1,a1,m1);
#else
   int64_t p0=k0+(int64_t)a0*m0,p1=k1+(int64_t)a1*m1;
#endif
#ifdef UQ_SIGNED_HIGH
   /* All prepared shifts are >=32. Subtracting 2^31 only changes the
    * surviving high word when the low word borrows. Keep that correction
    * in scalar predicates rather than two conditional 64-bit subtracts. */
   int32_t h0=(int32_t)(p0>>32),h1=(int32_t)(p1>>32);
   h0-=(a0<=t0)&((uint32_t)p0<UINT32_C(2147483648));
   h1-=(a1<=t1)&((uint32_t)p1<UINT32_C(2147483648));
#else
   if(a0<=t0)p0-=INT64_C(2147483648);if(a1<=t1)p1-=INT64_C(2147483648);
   int32_t h0=(int32_t)(p0>>32),h1=(int32_t)(p1>>32);
#endif
#ifdef __hexagon__
   int32_t v0=Q6_R_asr_RR(h0,s0-32),v1=Q6_R_asr_RR(h1,s1-32);
   long long both=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long lo=(long long)((uint64_t)(uint32_t)minimum|((uint64_t)(uint32_t)minimum<<32)),hi=(long long)((uint64_t)(uint32_t)maximum|((uint64_t)(uint32_t)maximum<<32));
   both=Q6_P_vmaxw_PP(lo,Q6_P_vminw_PP(hi,both));uint16_t result=(uint16_t)Q6_R_vsathb_R(Q6_R_vsatwh_P(both));
#else
   int32_t v0=h0>>(s0-32),v1=h1>>(s1-32);v0=v0<minimum?minimum:v0>maximum?maximum:v0;v1=v1<minimum?minimum:v1>maximum?maximum:v1;
   uint16_t result=(uint16_t)((uint8_t)v0|((uint16_t)(uint8_t)v1<<8));
#endif
   __builtin_memcpy(y+r*channels+c,&result,2);
  }
 }
}
#endif
typedef struct {
#ifdef UQ_GEMM_DOT4
 const int8_t*dot4_weights;
#endif
 const int32_t*q;int16_t value[UQ_MAX_C];uint16_t groups[UQ_MAX_C/4];unsigned active,pair_eligible;
#ifdef UQ_COMPACT_CHANNELS
 const int8_t*weights;const int32_t*quant;uint16_t map[UQ_MAX_C];unsigned live;
#ifdef UQ_RANGE_PROP
 uint16_t input_map[UQ_MAX_C];unsigned inputs;
#endif
#endif
} UQPrepared;
static UQPrepared prepared[UG_MAX_OP];
static const int32_t*prepared_keys[UG_MAX_OP];
static unsigned prepared_count;
#ifdef UQ_COMPACT_CHANNELS
static uint8_t prepared_storage[1024*1024] __attribute__((aligned(128)));
static unsigned prepared_used;
static int8_t compact_output[2][UQ_TILE*UQ_MAX_C] __attribute__((aligned(128)));
#ifdef UQ_COMPACT_DW
static int8_t compact_dw_input[16384] __attribute__((aligned(128)));
static int8_t compact_dw_output[16384] __attribute__((aligned(128)));
#endif
#endif
void uq_prepare_reset(void){prepared_count=0;
#ifdef UQ_SIGNED_AFFINE
 signed_count=signed_used=0;
#endif
#ifdef UQ_PREPARED_SCALE
 affine_used=affine_count=0;
#endif
#ifdef UQ_COMPACT_CHANNELS
 prepared_used=0;
#endif
}
int uq_prepare(unsigned code,const int8_t*w,const int32_t*q,const int16_t*inputlo,const int16_t*inputhi,int16_t*outlo,int16_t*outhi){
 if(prepared_count==UG_MAX_OP)return 18;
 prepared_keys[prepared_count]=q;
 UQPrepared*p=prepared+prepared_count++;p->q=q;p->active=0;p->pair_eligible=q[17]%2==0&&q[14]>=q[13];
 unsigned oc=q[17],k=code==15?(unsigned)(q[0]*q[1]):(unsigned)q[18];
 for(unsigned c=0;c<oc;c++){
  int64_t lo=q[UQ_HEAD+c],hi=lo;
  for(unsigned t=0;t<k;t++){
   int v=code==15?w[t*oc+c]:w[(c/4)*k*4+(t/8)*32+(c%4)*8+t%8];
   int il=0,ih=255;
   if(inputlo&&inputhi&&(code==15||t<(unsigned)(q[0]*q[1]*q[16]))){
    unsigned ic=code==15?c:t%(unsigned)q[16];il=inputlo[ic]+128;ih=inputhi[ic]+128;
    if(q[6]||q[7]||q[8]||q[9]){int pad=q[12]+128;if(il>pad)il=pad;if(ih<pad)ih=pad;}
   }
   if(v<0){lo+=ih*v;hi+=il*v;}else{lo+=il*v;hi+=ih*v;}
  }
  /* Loader bounds use logical K; padded dense weights may be nonzero in a
   * hostile plan. Disable the proof if its actual interval exceeds INT32. */
  p->value[c]=-256;
  if(outlo)outlo[c]=-128;if(outhi)outhi[c]=127;
  int s=q[UQ_HEAD+2*oc+c];int64_t factor=(int64_t)1<<(s>0?s:0);
  if(lo*factor<INT32_MIN||hi*factor>INT32_MAX)continue;
  int32_t m=q[UQ_HEAD+oc+c];
  int64_t av=(int64_t)scale((int32_t)lo,m,s)+q[13],bv=(int64_t)scale((int32_t)hi,m,s)+q[13];
  int8_t a=(int8_t)(av<q[14]?q[14]:av>q[15]?q[15]:av),b=(int8_t)(bv<q[14]?q[14]:bv>q[15]?q[15]:bv);
  if(outlo)outlo[c]=a;if(outhi)outhi[c]=b;
  if(a==b)p->value[c]=a;
 }
 for(unsigned c=0;c<oc;c++){int s=q[UQ_HEAD+2*oc+c];p->pair_eligible&=s<0&&s>=-15&&p->value[c]==-256;}
 for(unsigned c=0;c<oc;c+=4){
  int active=0;for(unsigned j=c;j<c+4&&j<oc;j++)active|=p->value[j]==-256;
  if(active)p->groups[p->active++]=(uint16_t)c;
 }
#ifdef UQ_COMPACT_CHANNELS
 p->weights=0;p->quant=0;p->live=0;
 for(unsigned c=0;c<oc;c++)if(p->value[c]==-256)p->map[p->live++]=(uint16_t)c;
 unsigned padded=(p->live+3)&~3u;
 unsigned compact_k=k;
#ifdef UQ_RANGE_PROP
 p->inputs=(unsigned)q[16];
 if(code==14&&inputlo&&inputhi&&q[0]==1&&q[1]==1&&q[2]==1&&q[3]==1&&!q[6]&&!q[7]&&!q[8]&&!q[9]){
  p->inputs=0;for(unsigned c=0;c<(unsigned)q[16];c++)if(inputlo[c]!=inputhi[c])p->input_map[p->inputs++]=(uint16_t)c;
  compact_k=(p->inputs+7)&~7u;if(!compact_k)compact_k=8;
 }
#endif
 int compactable=code==14;
#ifdef UQ_COMPACT_DW
 compactable|=code==15;
#endif
 if(compactable&&!q[19]&&padded&&padded*compact_k*4<=oc*k*3){
  unsigned weightbytes=compact_k*padded,parambytes=(UQ_HEAD+3*padded)*4;
  unsigned off=(prepared_used+127)&~127u;
  unsigned paramoff=(off+weightbytes+127)&~127u;
  if(paramoff+parambytes<=sizeof(prepared_storage)){
   int8_t*packed=(int8_t*)prepared_storage+off;int32_t*quant=(int32_t*)(prepared_storage+paramoff);
   for(unsigned j=0;j<padded;j++)for(unsigned t=0;t<compact_k;t++){
    unsigned c=j<p->live?p->map[j]:0;
    unsigned source=t;int valid=j<p->live;
    #ifdef UQ_RANGE_PROP
    if(p->inputs<(unsigned)q[16]){valid&=t<p->inputs;source=t<p->inputs?p->input_map[t]:0;}
    #endif
    if(code==15)packed[t*padded+j]=valid?w[t*oc+c]:0;
    else packed[(j/4)*compact_k*4+(t/8)*32+(j%4)*8+t%8]=valid?w[(c/4)*k*4+(source/8)*32+(c%4)*8+source%8]:0;
   }
   for(unsigned j=0;j<UQ_HEAD;j++)quant[j]=q[j];quant[17]=padded;quant[18]=compact_k;
   if(code==15){quant[16]=padded;quant[18]=(k+7)&~7u;}
   for(unsigned j=0;j<padded;j++)for(unsigned field=0;field<3;field++)
    quant[UQ_HEAD+field*padded+j]=j<p->live?q[UQ_HEAD+field*oc+p->map[j]]:0;
   #ifdef UQ_RANGE_PROP
   if(p->inputs<(unsigned)q[16])for(unsigned j=0;j<p->live;j++){
    unsigned c=p->map[j];int64_t bias=quant[UQ_HEAD+j];
    for(unsigned t=0;t<(unsigned)q[16];t++)if(inputlo[t]==inputhi[t])bias+=(inputlo[t]+128)*w[(c/4)*k*4+(t/8)*32+(c%4)*8+t%8];
    if(bias<INT32_MIN||bias>INT32_MAX)return 18;
    quant[UQ_HEAD+j]=(int32_t)bias;
   }
   #endif
   p->weights=packed;p->quant=quant;prepared_used=paramoff+parambytes;
  }
 }
#endif
#ifdef UQ_GEMM_DOT4
 p->dot4_weights=0;
 if(code==14&&!q[19]){
  unsigned nn=p->quant?(unsigned)p->quant[17]:oc,kk=p->quant?(unsigned)p->quant[18]:k;
  unsigned off=(prepared_used+127)&~127u;
#ifdef UQ_DOT4_SKEW
  off+=UQ_DOT4_SKEW;
#endif
  if(nn%4==0&&off+nn*kk<=sizeof(prepared_storage)
#ifdef UQ_DOT4_LIMIT
     &&kk<=UQ_DOT4_LIMIT
#endif
     ){
   const int8_t*src=p->quant?p->weights:w;int8_t*dst=(int8_t*)prepared_storage+off;
   for(unsigned j=0;j<nn;j+=4)for(unsigned t=0;t<kk;t+=4)for(unsigned c=0;c<4;c++)for(unsigned v=0;v<4;v++)
    dst[j*kk+t*4+c*4+v]=src[j*kk+(t/8)*32+c*8+t%8+v];
   p->dot4_weights=dst;prepared_used=off+nn*kk;
  }
 }
#endif
#ifdef UQ_PREPARED_SCALE
 prepare_affine(q,p->value);
#ifdef UQ_SIGNED_AFFINE
 prepare_signed(q,p->value);
#ifdef UQ_COMPACT_CHANNELS
 if(p->quant)prepare_signed(p->quant,0);
#endif
#endif
#ifdef UQ_COMPACT_CHANNELS
 if(p->quant)prepare_affine(p->quant,0);
#endif
#endif
 return 0;
}
static const UQPrepared*lookup(const int32_t*q){for(unsigned i=0;i<prepared_count;i++)if(prepared_keys[i]==q)return prepared+i;return 0;}
#endif
#ifdef UQ_REQUANT_BLOCK4
/* Experimental: keep four adjacent channels together, amortize metadata over
 * pixels, and write one complete word. Enabled only for on-device A/B tests.
 * The graph loader aligns bases; channels%4 proves every output word aligned.
 * General scaling is retained unless ALL four narrow-product bounds hold. */
static __attribute__((noinline)) void requantize4(const int32_t*restrict acc,int8_t*restrict y,unsigned rows,unsigned channels,const int32_t*restrict q){
 const int32_t*b=q+UQ_HEAD,*mult=b+channels,*shift=mult+channels;
 for(unsigned c=0;c<channels;c+=4){
#define META(i) const int32_t b##i=b[c+i],m##i=mult[c+i];const int s##i=shift[c+i]
  META(0);META(1);META(2);META(3);
#undef META
  const int offset=q[13],minimum=q[14],maximum=q[15];
#define LANE(i,fn) int32_t v##i=fn(acc[r*channels+c+i]+b##i,m##i,s##i)+offset; \
  bytes[i]=(uint8_t)(int8_t)(v##i<minimum?minimum:v##i>maximum?maximum:v##i)
#define PIXEL(fn) uint8_t bytes[4];LANE(0,fn);LANE(1,fn);LANE(2,fn);LANE(3,fn); \
  __builtin_memcpy(__builtin_assume_aligned(y+r*channels+c,4),bytes,4)
#ifdef __hexagon__
#define SAFE(i) ((b##i<0?-(int64_t)b##i:b##i)+(int64_t)q[18]*32640 <= (INT32_MAX >> ((s##i>0?s##i:0)+1)))
  if(SAFE(0)&&SAFE(1)&&SAFE(2)&&SAFE(3)){
#pragma clang loop unroll(disable)
   for(unsigned r=0;r<rows;r++){PIXEL(scale_narrow);}
   continue;
  }
#undef SAFE
#endif
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
  for(unsigned r=0;r<rows;r++){PIXEL(scale);}
#undef PIXEL
#undef LANE
 }
}
#endif
/* Quantization metadata is invariant across the pixel tile. Channel-major
 * traversal loads it once and keeps the compact accumulators in local memory. */
#ifdef UQ_AFFINE_PAIR
static __attribute__((noinline)) void requantize_affine_pair(const int32_t*restrict acc,int8_t*restrict y,unsigned rows,unsigned channels,const int32_t*restrict q){
 const int32_t*b=q+UQ_HEAD,*mult=b+channels,*shift=mult+channels;
 const int offset=q[13],minimum=q[14],maximum=q[15];
 for(unsigned c=0;c<channels;c+=2){
  int m0=mult[c],m1=mult[c+1];unsigned n0=(unsigned)-shift[c],n1=(unsigned)-shift[c+1];
  int64_t c0=(int64_t)b[c]*m0+((int64_t)1<<30)+((int64_t)1<<(30+n0))+(int64_t)offset*((int64_t)1<<(31+n0));
  int64_t c1=(int64_t)b[c+1]*m1+((int64_t)1<<30)+((int64_t)1<<(30+n1))+(int64_t)offset*((int64_t)1<<(31+n1));
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
  for(unsigned r=0;r<rows;r++){
#ifdef __hexagon__
   int64_t p0=Q6_P_mpyacc_RR(c0,acc[r*channels+c],m0),p1=Q6_P_mpyacc_RR(c1,acc[r*channels+c+1],m1);
   int32_t v0=(int32_t)(p0>>(31+n0)),v1=(int32_t)(p1>>(31+n1));
   long long both=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long lo=(long long)((uint64_t)(uint32_t)minimum|((uint64_t)(uint32_t)minimum<<32));
   long long hi=(long long)((uint64_t)(uint32_t)maximum|((uint64_t)(uint32_t)maximum<<32));
   both=Q6_P_vmaxw_PP(lo,Q6_P_vminw_PP(hi,both));
   uint16_t bytes=(uint16_t)Q6_R_vsathb_R(Q6_R_vsatwh_P(both));
#else
   int32_t v0=(int32_t)((c0+(int64_t)acc[r*channels+c]*m0)>>(31+n0));
   int32_t v1=(int32_t)((c1+(int64_t)acc[r*channels+c+1]*m1)>>(31+n1));
   v0=v0<minimum?minimum:v0>maximum?maximum:v0;v1=v1<minimum?minimum:v1>maximum?maximum:v1;
   uint16_t bytes=(uint16_t)((uint8_t)v0|((uint16_t)(uint8_t)v1<<8));
#endif
   __builtin_memcpy(__builtin_assume_aligned(y+r*channels+c,2),&bytes,2);
  }
 }
}
#endif
#ifdef UQ_PREPARED_SCALE
static __attribute__((noinline)) void requantize_prepared(const int32_t*restrict acc,int8_t*restrict y,unsigned rows,unsigned channels,const int32_t*q,const UQAffine*restrict table){
 const int minimum=q[14],maximum=q[15];
#ifdef UQ_SCALE_ROW
 for(unsigned r=0;r<rows;r++)for(unsigned c=0;c<channels;c+=2){
#else
 for(unsigned c=0;c<channels;c+=2){
#endif
  int64_t c0=table[c].constant,c1=table[c+1].constant;
  int m0=table[c].multiplier,m1=table[c+1].multiplier;
  unsigned n0=table[c].shift,n1=table[c+1].shift;
#ifdef UQ_SCALE_EQUAL
#ifdef __hexagon__
  if(n0==n1){
#pragma clang loop unroll(disable)
   for(unsigned r=0;r<rows;r++){
 #ifdef UQ_SCALE_DOUBLE_LOAD
   /* Even channel count and paired c prove eight-byte alignment. Force one
    * load of the pair instead of two scalar loads in the generated loop. */
   long long av;__asm__("%0 = memd(%1)" : "=r"(av) : "r"(acc+r*channels+c));
   int64_t p0=Q6_P_mpyacc_RR(c0,(int32_t)av,m0),p1=Q6_P_mpyacc_RR(c1,(int32_t)((uint64_t)av>>32),m1);
#else
   int64_t p0=Q6_P_mpyacc_RR(c0,acc[r*channels+c],m0),p1=Q6_P_mpyacc_RR(c1,acc[r*channels+c+1],m1);
#endif
    long long both=(long long)((uint64_t)(uint32_t)(p0>>32)|((uint64_t)(uint32_t)(p1>>32)<<32));
    both=Q6_P_vasrw_PR(both,n0-32);
    long long lo=(long long)((uint64_t)(uint32_t)minimum|((uint64_t)(uint32_t)minimum<<32));
    long long hi=(long long)((uint64_t)(uint32_t)maximum|((uint64_t)(uint32_t)maximum<<32));
    both=Q6_P_vmaxw_PP(lo,Q6_P_vminw_PP(hi,both));
    uint16_t bytes=(uint16_t)Q6_R_vsathb_R(Q6_R_vsatwh_P(both));
    __builtin_memcpy(__builtin_assume_aligned(y+r*channels+c,2),&bytes,2);
   }
   continue;
  }
#endif
#endif
#ifndef UQ_SCALE_ROW
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
  for(unsigned r=0;r<rows;r++){
#endif
#ifdef __hexagon__
#ifdef UQ_SCALE_DOUBLE_LOAD
   /* Even channel count and paired c prove eight-byte alignment. Force one
    * load of the pair instead of two scalar loads in the generated loop. */
   long long av;__asm__("%0 = memd(%1)" : "=r"(av) : "r"(acc+r*channels+c));
   int64_t p0=Q6_P_mpyacc_RR(c0,(int32_t)av,m0),p1=Q6_P_mpyacc_RR(c1,(int32_t)((uint64_t)av>>32),m1);
#else
   int64_t p0=Q6_P_mpyacc_RR(c0,acc[r*channels+c],m0),p1=Q6_P_mpyacc_RR(c1,acc[r*channels+c+1],m1);
#endif
#ifdef UQ_SCALE_WORD
   int32_t v0=Q6_R_asr_RR((int32_t)(p0>>32),n0-32),v1=Q6_R_asr_RR((int32_t)(p1>>32),n1-32);
#else
   int32_t v0=(int32_t)(p0>>n0),v1=(int32_t)(p1>>n1);
#endif
   long long both=(long long)((uint64_t)(uint32_t)v0|((uint64_t)(uint32_t)v1<<32));
   long long lo=(long long)((uint64_t)(uint32_t)minimum|((uint64_t)(uint32_t)minimum<<32));
   long long hi=(long long)((uint64_t)(uint32_t)maximum|((uint64_t)(uint32_t)maximum<<32));
   both=Q6_P_vmaxw_PP(lo,Q6_P_vminw_PP(hi,both));
#ifdef UQ_SCALE_BYTE_STORE
   y[r*channels+c]=(int8_t)both;
   y[r*channels+c+1]=(int8_t)((uint64_t)both>>32);
   continue;
#endif
   uint16_t bytes=(uint16_t)Q6_R_vsathb_R(Q6_R_vsatwh_P(both));
#else
   int32_t v0=(int32_t)((c0+(int64_t)acc[r*channels+c]*m0)>>n0),v1=(int32_t)((c1+(int64_t)acc[r*channels+c+1]*m1)>>n1);
   v0=v0<minimum?minimum:v0>maximum?maximum:v0;v1=v1<minimum?minimum:v1>maximum?maximum:v1;
   uint16_t bytes=(uint16_t)((uint8_t)v0|((uint16_t)(uint8_t)v1<<8));
#endif
   __builtin_memcpy(__builtin_assume_aligned(y+r*channels+c,2),&bytes,2);
#ifndef UQ_SCALE_ROW
  }
#endif
 }
}
#endif
static __attribute__((noinline)) void requantize(const int32_t*restrict acc,int8_t*restrict y,unsigned rows,unsigned channels,const int32_t*restrict q){
#ifdef UQ_REQUANT_CACHE
 /* Channel-wise scaling reuses parameters, but a large row stride can evict
  * accumulator lines before the neighboring channel consumes them. Restrict
  * each scaling pass to a bounded byte window, independently of GEMM tiling. */
 unsigned block=UQ_REQUANT_CACHE/(channels*4);if(!block)block=1;
 if(rows>block&&channels>=64){
  for(unsigned first=0;first<rows;first+=block){unsigned count=rows-first;if(count>block)count=block;requantize(acc+first*channels,y+first*channels,count,channels,q);}
  return;
 }
#endif
#ifdef UQ_SIGNED_AFFINE
 const UQSignedAffine*signed_table=lookup_signed(q);if(signed_table){requantize_signed(acc,y,rows,channels,q,signed_table);return;}
#endif
#ifdef UQ_PREPARED_SCALE
 const UQAffine*table=lookup_affine(q);
 if(table){
#if defined(UQ_SCALE_ASM) && defined(__hexagon__)
  extern void uq_requant2_asm(const int32_t*,int8_t*,unsigned,unsigned,const int32_t*,const UQAffine*);
  if(rows>=3){uq_requant2_asm(acc,y,rows,channels,q,table);return;}
#endif
  requantize_prepared(acc,y,rows,channels,q,table);return;
 }
#endif
#ifdef UQ_CONSTANT_CHANNELS
 const UQPrepared*prep=lookup(q);
#endif
#ifdef UQ_AFFINE_PAIR
 if(channels%2==0&&q[14]>=q[13]){
  int eligible=1;
#ifdef UQ_CONSTANT_CHANNELS
  if(prep)eligible=prep->pair_eligible;
  else
#endif
  for(unsigned c=0;c<channels;c++){int s=q[UQ_HEAD+2*channels+c];eligible&=s<0&&s>=-15;}
  if(eligible){requantize_affine_pair(acc,y,rows,channels,q);return;}
 }
#endif
#ifdef UQ_REQUANT_BLOCK4
 if(channels%4==0){requantize4(acc,y,rows,channels,q);return;}
#endif
 const int32_t*b=q+UQ_HEAD,*mult=b+channels,*shift=mult+channels;
 const int offset=q[13],minimum=q[14],maximum=q[15];
 for(unsigned c=0;c<channels;c++){
#ifdef UQ_CONSTANT_CHANNELS
  if(prep&&prep->value[c]!=-256){for(unsigned r=0;r<rows;r++)y[r*channels+c]=(int8_t)prep->value[c];continue;}
#endif
  int32_t bias=b[c],m=mult[c];int s=shift[c];
#ifdef UQ_REQUANT_AFFINE
  /* Exact double-rounding collapse for nonnegative fused activations.
   * Negative rounded values are discarded by the activation clamp, so their
   * ties-away correction cannot affect the result. Fold bias and zero point
   * into the wide product constant. Restrict the shift so every term and the
   * final product remain safely inside int64 under the loader's INT32 bound. */
  if(s<0&&s>=-15&&minimum>=offset){
   unsigned n=(unsigned)-s;
   int64_t constant=(int64_t)bias*m+((int64_t)1<<30)+((int64_t)1<<(30+n))+(int64_t)offset*((int64_t)1<<(31+n));
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
   for(unsigned r=0;r<rows;r++){
#ifdef __hexagon__
    int64_t product=Q6_P_mpyacc_RR(constant,acc[r*channels+c],m);
#else
    int64_t product=constant+(int64_t)acc[r*channels+c]*m;
#endif
    int32_t v=(int32_t)(product>>(31+n));
    y[r*channels+c]=(int8_t)(v<minimum?minimum:v>maximum?maximum:v);
   }
   continue;
  }
#endif
#ifdef __hexagon__
  int64_t bound=(bias<0?-(int64_t)bias:bias)+(int64_t)q[18]*32640;
  if(bound <= (INT32_MAX >> ((s>0?s:0)+1))){
#pragma clang loop unroll(disable)
   for(unsigned r=0;r<rows;r++){
    int32_t v=scale_narrow(acc[r*channels+c]+bias,m,s)+offset;
    y[r*channels+c]=(int8_t)(v<minimum?minimum:v>maximum?maximum:v);
   }
   continue;
  }
#endif
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
  for(unsigned r=0;r<rows;r++){
   int32_t v=scale(acc[r*channels+c]+bias,m,s)+offset;
   y[r*channels+c]=(int8_t)(v<minimum?minimum:v>maximum?maximum:v);
  }
 }
}
/* All graph tensors start on 128-byte boundaries; channels and c are multiples
 * of four on this path, including row/tap offsets. pad is aligned to eight.
 * Therefore these word loads are aligned for every admitted plan.
 * Four independent channels per packed multiply; widen each signed 16-bit
 * product before adding, so the nine taps cannot overflow a halfword lane. */
#if defined(UQ_DW_LEAF) && defined(UQ_DW_DOT) && defined(__hexagon__)
/* Keep spatial bookkeeping out of the channel accumulation loop so the
 * compiler can retain more pointers and operands in registers. */
static __attribute__((noinline)) void depthwise_dot_row(const int8_t*const*restrict ptr,const uint8_t*restrict dot_weights,unsigned channels,int32_t*restrict dst){
 for(unsigned c=0;c<channels;c+=4){
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
#pragma clang loop unroll(disable)
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
   dst[c]=v0;dst[c+1]=v1;dst[c+2]=v2;dst[c+3]=v3;
 }
}
#endif
#ifdef UQ_DW_PAD
static int8_t dw_padded_input[65536] __attribute__((aligned(128)));
#endif
#ifdef UQ_DW_FIXED
#include "depthwise_fixed.h"
#endif
static __attribute__((noinline)) void depthwise3(const UGTensor*a,const UGTensor*z,const int8_t*UQ_NOALIAS x,const int8_t*UQ_NOALIAS w,int8_t*UQ_NOALIAS y,const int32_t*UQ_NOALIAS q,unsigned first,unsigned end,int32_t*UQ_NOALIAS acc,unsigned scratch){
 unsigned channels=q[17],ih=a->d[1],iw=a->d[2];
#if UQ_DW_FIXED >= 8
 if(channels==8){depthwise3_c8(a,z,x,w,y,q,first,end,acc,scratch);uq_post(q,y+first*channels,(end-first)*channels);return;}
#endif
#if UQ_DW_FIXED >= 16
 if(channels==16){depthwise3_c16(a,z,x,w,y,q,first,end,acc,scratch);uq_post(q,y+first*channels,(end-first)*channels);return;}
#endif
#if UQ_DW_FIXED >= 32
 if(channels==32){depthwise3_c32(a,z,x,w,y,q,first,end,acc,scratch);uq_post(q,y+first*channels,(end-first)*channels);return;}
#endif

#ifdef UQ_DW_SCALE
 const UQAffine*direct_scale=lookup_affine(q);
#endif
#ifdef UQ_RANGE_DW_SKIP
 const UQPrepared*prep=lookup(q);
#endif
 int8_t pad[UQ_MAX_C] __attribute__((aligned(8)));
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
#pragma clang loop unroll(disable)
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
#pragma clang loop unroll(disable)
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
 uq_post(q,y+tile*channels,(stop-tile)*channels);
 PHASE_END(scratch,2);
 }
}
#ifdef UQ_DW5_DOT
#include "depthwise5.h"
#endif
/* RGB stem gather: fixed 3x3x3 payload, arbitrary validated stride/dilation.
 * Interior pixels avoid nine independent bounds checks; borders still use the
 * exact input zero point. No four-byte read crosses a three-byte pixel. */
static __attribute__((noinline)) void gather_rgb3(uint8_t*UQ_NOALIAS dst,const int8_t*UQ_NOALIAS x,const UGTensor*a,const int32_t*UQ_NOALIAS q,unsigned first,unsigned count,unsigned dst_stride){
 unsigned ih=a->d[1],iw=a->d[2],oh=q[10],ow=q[11];
 unsigned n=first/(oh*ow),h=(first/ow)%oh,v=first%ow;
 const int sh=q[2],sw=q[3],dh=q[4],dw=q[5],top=q[6],left=q[7];
 const uint8_t zero=q[12]+128;
 for(unsigned r=0;r<count;r++,dst+=dst_stride){
  int yy=(int)h*sh-top,xx=(int)v*sw-left;
  if(yy>=0&&xx>=0&&(unsigned)(yy+2*dh)<ih&&(unsigned)(xx+2*dw)<iw){
   const int8_t*base=x+((n*ih+yy)*iw+xx)*3;
#ifdef UQ_RGB_WIDE
   /* Aligned loads with an explicit allocation-end check. Each admitted row
    * supplies nine contiguous bytes; the aligned pair never escapes a->bytes.
    * Last-row edge cases and dilated columns retain the byte-safe fallback. */
   unsigned last=(unsigned)(base-x)+2*(unsigned)dh*iw*3;
   if(dw==1&&(last&~7u)+16<=a->bytes){
    uint64_t row[3];uint8_t tail[3];
    for(unsigned ky=0;ky<3;ky++){
     unsigned off=(unsigned)(base-x)+ky*(unsigned)dh*iw*3,aligned=off&~7u;
     uint64_t lo,hi;
     __builtin_memcpy(&lo,__builtin_assume_aligned(x+aligned,8),8);
     __builtin_memcpy(&hi,__builtin_assume_aligned(x+aligned+8,8),8);
#ifdef __hexagon__
     row[ky]=(uint64_t)Q6_P_valignb_PPp((long long)hi,(long long)lo,off&7u);
#else
     unsigned shift=(off&7u)*8;row[ky]=shift?(lo>>shift)|(hi<<(64-shift)):lo;
#endif
     tail[ky]=(uint8_t)x[off+8];
    }
    uint64_t packed[4]={row[0],(uint64_t)tail[0]|(row[1]<<8),
     (row[1]>>56)|((uint64_t)tail[1]<<8)|(row[2]<<16),
     (row[2]>>48)|((uint64_t)tail[2]<<16)|(((uint64_t)(uint8_t)q[12]*UINT64_C(0x0101010101))<<24)};
    for(unsigned i=0;i<4;i++){packed[i]^=UINT64_C(0x8080808080808080);__builtin_memcpy(__builtin_assume_aligned(dst+i*8,8),packed+i,8);}
   }else
#endif
   for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++){
    const int8_t*src=base+(ky*dh*iw+kx*dw)*3;unsigned c=(ky*3+kx)*3;
    dst[c]=(uint8_t)src[0]^128u;dst[c+1]=(uint8_t)src[1]^128u;dst[c+2]=(uint8_t)src[2]^128u;
   }
  }else{
   for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++){
    int sy=yy+ky*dh,sx=xx+kx*dw;unsigned c=(ky*3+kx)*3;
    if(sy>=0&&sx>=0&&(unsigned)sy<ih&&(unsigned)sx<iw){const int8_t*src=x+((n*ih+sy)*iw+sx)*3;
     dst[c]=(uint8_t)src[0]^128u;dst[c+1]=(uint8_t)src[1]^128u;dst[c+2]=(uint8_t)src[2]^128u;
    }else{dst[c]=zero;dst[c+1]=zero;dst[c+2]=zero;}
   }
  }
#ifdef UQ_RGB_PAD_STORE
  dst[27]=zero;
  uint32_t fill=(uint32_t)zero*UINT32_C(0x01010101);
  __builtin_memcpy(__builtin_assume_aligned(dst+28,4),&fill,4);
#else
  for(unsigned c=27;c<32;c++)dst[c]=zero;
#endif
  if(++v==ow){v=0;if(++h==oh){h=0;n++;}}
 }
}
#if defined(UQ_GEMM_SCALE) && defined(__hexagon__)
#include "gemm_scale.h"
#endif
#if defined(UQ_DW_WINDOW4) && defined(__hexagon__)
#include "depthwise_windows4.h"
#endif
#ifdef UQ_GLOBAL_POOL
static int global_pool9(const UGTensor*a,const int8_t*x,int8_t*y,const int32_t*q,unsigned first,unsigned end){
 unsigned channels=q[16];
 for(unsigned n=first;n<end;n++)for(unsigned c=0;c<channels;c+=4){
#ifdef __hexagon__
  long long sum=0;
#pragma clang loop unroll(full)
  for(unsigned r=0;r<9;r++){
   int v;__builtin_memcpy(&v,__builtin_assume_aligned(x+(n*9+r)*channels+c,4),4);
   sum=Q6_P_vaddh_PP(sum,Q6_P_vsxtbh_R(v));
  }
  int32_t sums[4];
  for(unsigned j=0;j<4;j++)sums[j]=(int16_t)((uint64_t)sum>>(j*16));
#else
  int32_t sums[4]={0};
  for(unsigned r=0;r<9;r++)for(unsigned j=0;j<4;j++)sums[j]+=x[(n*9+r)*channels+c+j];
#endif
  for(unsigned j=0;j<4;j++)y[n*channels+c+j]=clamp((sums[j]+(sums[j]>=0?4:-4))/9,q);
 }
 (void)a;return 0;
}
#endif
static int run_range(unsigned code,const UGTensor*a,const UGTensor*z,const int8_t*UQ_NOALIAS x,const int8_t*UQ_NOALIAS w,int8_t*UQ_NOALIAS y,const int32_t*UQ_NOALIAS q,unsigned first_row,unsigned end_row,unsigned scratch){
 uint8_t*column=columns[scratch];int32_t*accum=accumulators[scratch];
 unsigned ic=q[16],oc=q[17],ih=a->rank==4?a->d[1]:1,iw=a->rank==4?a->d[2]:1;
#ifdef UQ_GLOBAL_POOL
 if(code==16&&!q[19]&&ic%4==0&&ih==3&&iw==3&&q[0]==3&&q[1]==3&&!q[6]&&!q[7]&&!q[8]&&!q[9]&&q[10]==1&&q[11]==1)
  return global_pool9(a,x,y,q,first_row,end_row);
#endif

 if(code==17){
  for(unsigned base=0;base<a->bytes;base+=ic){
   int max=-128;for(unsigned c=0;c<ic;c++)if(x[base+c]>max)max=x[base+c];
   uint64_t sum=0;for(unsigned c=0;c<ic;c++)sum+=(uint32_t)q[UQ_HEAD+max-x[base+c]];
   for(unsigned c=0;c<ic;c++){uint64_t v=(uint64_t)(uint32_t)q[UQ_HEAD+max-x[base+c]]*256;int32_t r=(int32_t)((v+sum/2)/sum)-128;y[base+c]=clamp(r,q);}
  }return 0;
 }
#if defined(UQ_DW_WINDOW4) && defined(__hexagon__)
 if(code==15&&!q[19]&&q[0]==3&&q[1]==3&&q[4]==1&&q[5]==1&&oc%4==0&&a->d[0]==1&&first_row<end_row&&
    (uint64_t)a->d[1]*z->d[2]*oc*4<=sizeof(dw_window4[0])){
  depthwise_window4(a,z,x,w,y,q,first_row,end_row,accum,scratch);return 0;
 }
#endif
#ifdef UQ_DW5_DOT
 if(code==15&&!q[19]&&q[0]==5&&q[1]==5&&oc%4==0){depthwise5(a,z,x,w,y,q,first_row,end_row,accum,scratch);return 0;}
#endif
 if(code==15&&!q[19]&&q[0]==3&&q[1]==3&&oc%4==0){depthwise3(a,z,x,w,y,q,first_row,end_row,accum,scratch);return 0;}
 unsigned rows=end_row,oh=q[10],ow=q[11],kp=q[18];
 const int32_t*bias=q+UQ_HEAD,*mult=bias+oc,*shift=mult+oc;
 if(code==14){
#ifdef UQ_CONSTANT_CHANNELS
  const UQPrepared*prep=lookup(q);
#endif
#ifdef UQ_COMPACT_CHANNELS
  if(prep&&!prep->live){for(unsigned r=first_row;r<end_row;r++)for(unsigned c=0;c<oc;c++)y[r*oc+c]=(int8_t)prep->value[c];uq_post(q,y+first_row*oc,(end_row-first_row)*oc);return 0;}
  unsigned compute_oc=prep&&prep->quant?(unsigned)prep->quant[17]:oc;
#else
  unsigned compute_oc=oc;
#endif
  unsigned compute_k=kp;
#ifdef UQ_COMPACT_CHANNELS
  if(prep&&prep->quant)compute_k=(unsigned)prep->quant[18];
#endif
  unsigned input_stride=compute_k;
#ifdef UQ_BANK_PADDED
  /* Fixed-address probes measured a fivefold dual-load penalty at nonzero
   * multiples of 32 bytes; an eight-byte skew removes it. Pad scratch rows,
   * not K/weights. Never read padding or change the numerical operation. */
  if(!q[19]&&compute_oc%4==0&&compute_k%32==0)input_stride+=8;
#endif
  /* Bounded tile expansion; pointwise path is a contiguous signed-to-unsigned
     conversion, without a full-image im2col or INT32 intermediate tensor. */
  int pointwise=q[0]==1&&q[1]==1&&q[2]==1&&q[3]==1&&!q[6]&&!q[7]&&!q[8]&&!q[9]&&ic==kp;
  for(unsigned first=first_row,tile_size=tile_rows(input_stride,oc);first<rows;first+=tile_size){unsigned nr=rows-first;if(nr>tile_size)nr=tile_size;
   PHASE_START();
#ifdef UQ_DIRECT_SMALL
   int direct=pointwise&&!q[19]&&nr%2==0&&oc%4==0&&(ic==8||ic==16)&&!(prep&&prep->quant);
   if(!direct){
#endif
#ifdef UQ_RANGE_PROP
   if(prep&&prep->quant&&prep->inputs<ic){
    for(unsigned r=0;r<nr;r++){
     for(unsigned t=0;t<prep->inputs;t++)column[r*input_stride+t]=(uint8_t)x[(first+r)*ic+prep->input_map[t]]^128u;
     for(unsigned t=prep->inputs;t<compute_k;t++)column[r*input_stride+t]=0;
    }
   }else
#endif
   if(pointwise){
    /* ic==kp is divisible by eight; both bases and every pixel are aligned. */
    const int8_t*src=x+first*ic;
    if(input_stride==kp){
     for(unsigned t=0;t<nr*ic;t+=8){uint64_t v;
      __builtin_memcpy(&v,__builtin_assume_aligned(src+t,8),8);
      v^=UINT64_C(0x8080808080808080);
      __builtin_memcpy(__builtin_assume_aligned(column+t,8),&v,8);
     }
    }else for(unsigned r=0;r<nr;r++)for(unsigned t=0;t<ic;t+=8){uint64_t v;
      __builtin_memcpy(&v,__builtin_assume_aligned(src+r*ic+t,8),8);
      v^=UINT64_C(0x8080808080808080);
      __builtin_memcpy(__builtin_assume_aligned(column+r*input_stride+t,8),&v,8);
    }
   }
   else if(ic==3&&q[0]==3&&q[1]==3)gather_rgb3(column,x,a,q,first,nr,input_stride);
   else for(unsigned r=0;r<nr;r++){
    unsigned row=first+r,batch=row/(oh*ow),yy=(row/ow)%oh,xx=row%ow;uint8_t*dst=column+r*input_stride;unsigned c=0;
    for(int ky=0;ky<q[0];ky++)for(int kx=0;kx<q[1];kx++){
     int sy=(int)yy*q[2]-q[6]+ky*q[4],sx=(int)xx*q[3]-q[7]+kx*q[5];
     if(sy>=0&&sx>=0&&(unsigned)sy<ih&&(unsigned)sx<iw){const int8_t*src=x+((batch*ih+sy)*iw+sx)*ic;
#ifdef UQ_GATHER_WIDE
      if(ic%8==0){for(unsigned ch=0;ch<ic;ch+=8){uint64_t v;__builtin_memcpy(&v,__builtin_assume_aligned(src+ch,8),8);v^=UINT64_C(0x8080808080808080);__builtin_memcpy(__builtin_assume_aligned(dst+c,8),&v,8);c+=8;}}
      else
#endif
      for(unsigned ch=0;ch<ic;ch++)dst[c++]=(uint8_t)src[ch]^128u;
     }
     else for(unsigned ch=0;ch<ic;ch++)dst[c++]=(uint8_t)(q[12]+128);
    }
    for(;c<kp;c++)dst[c]=(uint8_t)(q[12]+128);
   }
#ifdef UQ_DIRECT_SMALL
   }
#endif
   PHASE_END(scratch,0);
#if defined(UQ_GEMM_SCALE) && defined(__hexagon__)
   if(!q[19]&&prep&&prep->dot4_weights&&!prep->quant&&oc%4==0&&nr%UQ_GEMM_SCALE==0){
    const UQAffine*table=lookup_affine(q);
    if(table&&(compute_k==8||compute_k==16||compute_k==32)){
#if UQ_GEMM_SCALE == 1
     if(compute_k==8)fused_k8_r1(column,prep->dot4_weights,y+first*oc,nr,oc,input_stride,q,table);
     if(compute_k==16)fused_k16_r1(column,prep->dot4_weights,y+first*oc,nr,oc,input_stride,q,table);
     if(compute_k==32)fused_k32_r1(column,prep->dot4_weights,y+first*oc,nr,oc,input_stride,q,table);
#endif
#if UQ_GEMM_SCALE == 2
     if(compute_k==8)fused_k8_r2(column,prep->dot4_weights,y+first*oc,nr,oc,input_stride,q,table);
     if(compute_k==16)fused_k16_r2(column,prep->dot4_weights,y+first*oc,nr,oc,input_stride,q,table);
     if(compute_k==32)fused_k32_r2(column,prep->dot4_weights,y+first*oc,nr,oc,input_stride,q,table);
#endif
     PHASE_END(scratch,1);continue;
    }
   }
#endif

#ifdef UQ_DIRECT_SMALL
   if(direct){int rc=ug_matmul_s8i8_small(x+first*ic,w,accum,nr,oc,ic);if(rc)return rc;}
   else
#endif
   if(q[19]){for(unsigned r=0;r<nr;r++)for(unsigned j=0;j<oc;j++){int32_t sum=0;for(unsigned c=0;c<kp;c++)sum+=(int32_t)column[r*kp+c]*w[(j/4)*kp*4+(c/8)*32+(j%4)*8+c%8];accum[r*oc+j]=sum;}}
   else{int rc;
#ifdef UQ_GEMM_DOT4
    if(prep&&prep->dot4_weights&&nr%2==0)rc=ug_matmul_u8i8_dot4(column,prep->dot4_weights,accum,nr,compute_oc,compute_k,input_stride);
    else
#endif
#ifdef UQ_COMPACT_CHANNELS
    if(prep&&prep->quant)rc=ug_matmul_u8i8_strided(column,prep->weights,accum,nr,compute_oc,compute_k,input_stride);
    else
#endif
#ifdef UQ_CONSTANT_CHANNELS
    if(prep&&oc%4==0&&kp>=64&&prep->active<oc/4)
     rc=ug_matmul_u8i8_selected(column,w,accum,nr,oc,kp,input_stride,prep->groups,prep->active);
    else
#endif
    rc=input_stride!=kp?ug_matmul_u8i8_strided(column,w,accum,nr,oc,kp,input_stride):ug_matmul_u8i8(column,w,accum,nr,oc,kp,1,1);if(rc)return rc;}
   PHASE_END(scratch,1);
#ifdef UQ_COMPACT_CHANNELS
   if(prep&&prep->quant){
    requantize(accum,compact_output[scratch],nr,compute_oc,prep->quant);
    for(unsigned r=0;r<nr;r++){
     for(unsigned c=0;c<oc;c++)y[(first+r)*oc+c]=(int8_t)prep->value[c];
     for(unsigned c=0;c<prep->live;c++)y[(first+r)*oc+prep->map[c]]=compact_output[scratch][r*compute_oc+c];
    }
   }else
#endif
   requantize(accum,y+first*oc,nr,oc,q);
   uq_post(q,y+first*oc,nr*oc);
   PHASE_END(scratch,2);
  }return 0;
 }
 for(unsigned row=first_row;row<rows;row++){
  unsigned batch=row/(oh*ow),yy=(row/ow)%oh,xx=row%ow;
  for(unsigned ch=0;ch<oc;ch++)accum[ch]=code==15?bias[ch]:q[19]?-128:0;
  unsigned valid_count=0;
  for(int ky=0;ky<q[0];ky++)for(int kx=0;kx<q[1];kx++){
   int sy=(int)yy*q[2]-q[6]+ky*q[4],sx=(int)xx*q[3]-q[7]+kx*q[5];int valid=sy>=0&&sx>=0&&(unsigned)sy<ih&&(unsigned)sx<iw;
   const int8_t*src=valid?x+((batch*ih+sy)*iw+sx)*ic:0;valid_count+=valid;
   if(code==15){const int8_t*wt=w+(ky*q[1]+kx)*oc;if(valid){for(unsigned ch=0;ch<oc;ch++)accum[ch]+=((int)src[ch]+128)*wt[ch];}else for(unsigned ch=0;ch<oc;ch++)accum[ch]+=(q[12]+128)*wt[ch];}
   else if(valid){if(q[19]){for(unsigned ch=0;ch<oc;ch++)if(src[ch]>accum[ch])accum[ch]=src[ch];}else for(unsigned ch=0;ch<oc;ch++)accum[ch]+=src[ch];}
  }
  for(unsigned ch=0;ch<oc;ch++){
   int32_t value=accum[ch];if(code==15)value=scale((int32_t)value,mult[ch],shift[ch])+q[13];
   else if(!q[19])value=(value+(value>0?(int)valid_count/2:-(int)valid_count/2))/(int)valid_count;
   y[row*oc+ch]=clamp(value,q);
  }
 }
 if(code==15)uq_post(q,y+first_row*oc,(end_row-first_row)*oc);
 return 0;
}

typedef struct {unsigned code;const UGTensor*a,*z;const int8_t*x,*w;int8_t*y;const int32_t*q;unsigned first,end,scratch;} QJob;
static int run_job(void*ptr){QJob*j=ptr;return run_range(j->code,j->a,j->z,j->x,j->w,j->y,j->q,j->first,j->end,j->scratch);}
#ifdef UQ_FUSE_DW_PW
typedef struct {QJob dw,pw;} FusedJob;
static int run_fused_job(void*ptr){
 FusedJob*j=ptr;
#ifndef UQ_FUSE_STRIP
#define UQ_FUSE_STRIP 128
#endif
 unsigned strip=UQ_FUSE_STRIP?UQ_FUSE_STRIP:j->dw.end-j->dw.first;
 for(unsigned first=j->dw.first;first<j->dw.end;first+=strip){
  unsigned end=first+strip;if(end>j->dw.end)end=j->dw.end;
  int rc=run_range(15,j->dw.a,j->dw.z,j->dw.x,j->dw.w,j->dw.y,j->dw.q,first,end,j->dw.scratch);
  if(rc)return rc;
  rc=run_range(14,j->pw.a,j->pw.z,j->pw.x,j->pw.w,j->pw.y,j->pw.q,first,end,j->pw.scratch);
  if(rc)return rc;
 }
 return 0;
}
int uq_run_fused(const UGTensor*a,const UGTensor*mid,const UGTensor*z,const int8_t*x,const int8_t*wd,const int8_t*wp,int8_t*yd,int8_t*yp,const int32_t*qd,const int32_t*qp,int threads){
 unsigned rows=mid->d[0]*mid->d[1]*mid->d[2];
#ifdef UQ_COMPACT_DW
 const UQPrepared*prepared_dw=lookup(qd);
 if(prepared_dw&&prepared_dw->quant)return -1;
#endif
 if(rows<128||qp[0]!=1||qp[1]!=1||qp[2]!=1||qp[3]!=1||qp[6]||qp[7]||qp[8]||qp[9]||qd[19]||qp[19])return -1;
 if(mid->d[0]!=z->d[0]||mid->d[1]!=z->d[1]||mid->d[2]!=z->d[2])return -1;
 FusedJob one={{15,a,mid,x,wd,yd,qd,0,rows,0},{14,mid,z,yd,wp,yp,qp,0,rows,0}};
 if(threads==2){
  FusedJob two=one;one.dw.end=one.pw.end=(rows/2)&~1u;
  two.dw.first=two.pw.first=one.dw.end;two.dw.scratch=two.pw.scratch=1;
  int rc=ug_run_pair(run_fused_job,&one,&two);if(rc!=-1)return rc;
  one.dw.end=one.pw.end=rows;
 }
 return run_fused_job(&one);
}
#endif
#ifdef UQ_DW_WINDOWS
/* Each horizontal 3-tap window is packed once and reused by three output
 * rows. Four-byte zero-padded windows make two adjacent pixels one dot pair.
 * Channel partitioning keeps the scratch private; outputs retain NHWC. */
static uint32_t dw_windows[2][16384] __attribute__((aligned(128)));
typedef struct {const UGTensor*a,*z;const int8_t*x,*w;int8_t*y;const int32_t*q;const UQAffine*scale;unsigned first,end,scratch;} DWWindowsJob;
static int run_dw_windows(void*arg){
 DWWindowsJob*j=arg;const int32_t*q=j->q;
 unsigned ic=q[16],ih=j->a->d[1],iw=j->a->d[2],oh=q[10],ow=q[11],ph=ih+q[6]+q[8];
 uint32_t*buf=dw_windows[j->scratch];uint32_t zero=(uint32_t)(q[12]+128)*0x010101u;
 for(unsigned c=j->first;c<j->end;c++){
  for(unsigned h=0;h<ph;h++)for(unsigned v=0;v<ow;v++){
   int sy=(int)h-q[6],sx=(int)v*q[3]-q[7];uint32_t packed=zero;
   if(sy>=0&&(unsigned)sy<ih){
    packed=0;
    for(unsigned t=0;t<3;t++){int xx=sx+(int)t;unsigned val=xx>=0&&(unsigned)xx<iw?(uint8_t)j->x[((unsigned)sy*iw+(unsigned)xx)*ic+c]^128u:(unsigned)(q[12]+128);packed|=val<<(8*t);}
   }
   buf[h*ow+v]=packed;
  }
  uint64_t wr[3];
  for(unsigned t=0;t<3;t++){uint32_t v=(uint8_t)j->w[(t*3)*ic+c]|((uint32_t)(uint8_t)j->w[(t*3+1)*ic+c]<<8)|((uint32_t)(uint8_t)j->w[(t*3+2)*ic+c]<<16);wr[t]=v|((uint64_t)v<<32);}
  const UQAffine*sc=j->scale+c;int64_t constant=sc->constant;int mult=sc->multiplier;unsigned shift=sc->shift;
  for(unsigned h=0;h<oh;h++)for(unsigned v=0;v<ow;v+=2){
#ifdef __hexagon__
   long long sum=0;
#pragma clang loop unroll(full)
   for(unsigned t=0;t<3;t++){long long input;__builtin_memcpy(&input,__builtin_assume_aligned(buf+(h*q[2]+t)*ow+v,8),8);sum=Q6_P_vrmpybsuacc_PP(sum,(long long)wr[t],input);}
   int64_t p0=Q6_P_mpyacc_RR(constant,(int32_t)sum,mult),p1=Q6_P_mpyacc_RR(constant,(int32_t)((uint64_t)sum>>32),mult);
   long long both=(long long)((uint64_t)(uint32_t)(p0>>32)|((uint64_t)(uint32_t)(p1>>32)<<32));
   both=Q6_P_vasrw_PR(both,shift-32);
   long long lo=(long long)((uint64_t)(uint32_t)q[14]|((uint64_t)(uint32_t)q[14]<<32)),hi=(long long)((uint64_t)(uint32_t)q[15]|((uint64_t)(uint32_t)q[15]<<32));
   both=Q6_P_vmaxw_PP(lo,Q6_P_vminw_PP(hi,both));
   uint16_t result=(uint16_t)Q6_R_vsathb_R(Q6_R_vsatwh_P(both));
   j->y[(h*ow+v)*ic+c]=(int8_t)result;j->y[(h*ow+v+1)*ic+c]=(int8_t)(result>>8);
#else
   for(unsigned lane=0;lane<2;lane++){int sum=0;for(unsigned kh=0;kh<3;kh++)for(unsigned kw=0;kw<3;kw++)sum+=(int8_t)(wr[kh]>>(8*kw))*((buf[(h*q[2]+kh)*ow+v+lane]>>(8*kw))&255);
    int32_t val=(int32_t)((constant+(int64_t)sum*mult)>>shift);j->y[(h*ow+v+lane)*ic+c]=clamp(val,q);}
#endif
  }
 }
 return 0;
}
#endif
int uq_run(unsigned code,const UGTensor*a,const UGTensor*z,const int8_t*x,const int8_t*w,int8_t*y,const int32_t*q,int threads){
 unsigned rows=code==17?1:z->d[0]*z->d[1]*z->d[2];
#ifdef UQ_COMPACT_DW
 if(code==15){
  const UQPrepared*p=lookup(q);unsigned oc=q[17];
  if(p&&!p->live){for(unsigned r=0;r<rows;r++)for(unsigned c=0;c<oc;c++)y[r*oc+c]=(int8_t)p->value[c];uq_post(q,y,rows*oc);return 0;}
  if(p&&p->quant){
   unsigned pc=p->quant[17],inputrows=a->d[0]*a->d[1]*a->d[2];
   if((uint64_t)inputrows*pc<=sizeof(compact_dw_input)&&(uint64_t)rows*pc<=sizeof(compact_dw_output)){
    for(unsigned r=0;r<inputrows;r++){
     for(unsigned c=0;c<p->live;c++)compact_dw_input[r*pc+c]=x[r*oc+p->map[c]];
     for(unsigned c=p->live;c<pc;c++)compact_dw_input[r*pc+c]=(int8_t)q[12];
    }
    UGTensor ca=*a,cz=*z;ca.d[3]=cz.d[3]=pc;ca.bytes=inputrows*pc;cz.bytes=rows*pc;
    int rc=run_range(code,&ca,&cz,compact_dw_input,p->weights,compact_dw_output,p->quant,0,rows,0);if(rc)return rc;
    for(unsigned r=0;r<rows;r++){
     for(unsigned c=0;c<oc;c++)y[r*oc+c]=(int8_t)p->value[c];
     for(unsigned c=0;c<p->live;c++)y[r*oc+p->map[c]]=compact_dw_output[r*pc+c];
    }
    uq_post(q,y,rows*oc);return 0;
   }
  }
 }
#endif
#ifdef UQ_DW_PAD
 if(code==15&&!q[19]&&q[0]==3&&q[1]==3&&q[17]%UQ_DW_PAD==0){
  unsigned c=q[17],pixels=a->d[0]*a->d[1]*a->d[2];
  if((uint64_t)pixels*(c+8)<=sizeof(dw_padded_input)){
   for(unsigned r=0;r<pixels;r++)for(unsigned t=0;t<c;t+=8){
    uint64_t v;__builtin_memcpy(&v,__builtin_assume_aligned(x+r*c+t,8),8);
    __builtin_memcpy(__builtin_assume_aligned(dw_padded_input+r*(c+8)+t,8),&v,8);
   }
   x=dw_padded_input;
  }
 }
#endif
#ifdef UQ_DW_WINDOWS
 if(code==15&&!q[19]&&q[0]==3&&q[1]==3&&q[4]==1&&q[5]==1&&a->d[0]==1&&q[11]%2==0&&q[6]<=2&&q[8]<=2&&q[7]<=2&&q[9]<=2&&
    (uint64_t)(a->d[1]+q[6]+q[8])*(unsigned)q[11]<=16384){
  const UQAffine*sc=lookup_affine(q);
  if(sc){DWWindowsJob one={a,z,x,w,y,q,sc,0,(unsigned)q[17],0};
   if(threads==2&&q[17]>=2){DWWindowsJob two=one;one.end=(unsigned)q[17]/2;two.first=one.end;two.scratch=1;
    int rc=ug_run_pair(run_dw_windows,&one,&two);if(rc!=-1)return rc;one.end=q[17];}
   return run_dw_windows(&one);
  }
 }
#endif
 QJob one={code,a,z,x,w,y,q,0,rows,0};
 if(threads==2&&(code==14||code==15)&&rows>=8&&(uint64_t)rows*q[17]>=2048){
  QJob two=one;one.end=(rows/2)&~1u;two.first=one.end;two.scratch=1;
  int rc=ug_run_pair(run_job,&one,&two);if(rc!=-1)return rc;
  one.end=rows;
 }
 return run_job(&one);
}
