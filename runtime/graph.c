#include "graph.h"
#include "gemm.h"
#include "qnn.h"
static UGHeader hdr;
static UGTensor ts[UG_MAX_T];
static UGOp ops[UG_MAX_OP];
static unsigned char arena[UG_ARENA] __attribute__((aligned(128)));
static int ready,profile_enabled,vision_only;
static uint64_t profile_ticks[UG_MAX_OP+2];
#ifdef UQ_RANGE_PROP
static int16_t range_lo[UG_MAX_T][UQ_MAX_C],range_hi[UG_MAX_T][UQ_MAX_C];
#endif
static uint64_t tick(void){
#ifdef __hexagon__
 extern uint64_t qurt_timer_get_ticks(void);return qurt_timer_get_ticks();
#else
 return 0;
#endif
}
#define PROFILE_OP_END() do{if(profile_enabled)profile_ticks[i+2]+=tick()-op_start;}while(0)
static void copy(void*d,const void*s,uint32_t n){
 unsigned char*y=d;const unsigned char*x=s;uint32_t i=0;
#ifdef UQ_WIDE_COPY
 if(!(((uintptr_t)x|(uintptr_t)y)&7))for(;i+8<=n;i+=8){
  uint64_t v;__builtin_memcpy(&v,__builtin_assume_aligned(x+i,8),8);
  __builtin_memcpy(__builtin_assume_aligned(y+i,8),&v,8);
 }
#endif
 for(;i<n;i++)y[i]=x[i];
}
static uint32_t width(uint32_t t){return t==1||t==4?4:t==2||t==3?1:0;}
static int same(const UGTensor*a,const UGTensor*b){if(a->type!=b->type||a->rank!=b->rank)return 0;for(uint32_t i=0;i<a->rank;i++)if(a->d[i]!=b->d[i])return 0;return 1;}
static int broadcast(const UGTensor*a,const UGTensor*b){
 if(a->type!=b->type||b->rank>a->rank)return 0;
 for(uint32_t i=0;i<b->rank;i++)if(b->d[i]!=1 && b->d[i]!=a->d[a->rank-b->rank+i])return 0;
 /* Restrict broadcast to scalar, last-axis vector, or equal shape. */
 if(b->bytes==width(b->type))return 1;
 if(b->bytes==a->d[a->rank-1]*width(b->type)){for(uint32_t i=0;i+1<b->rank;i++)if(b->d[i]!=1)return 0;return 1;}
 return same(a,b);
}
/* Stable softmax needs exp only on nonpositive arguments. Range reduction and
   a degree-7 Taylor polynomial on [-ln(2)/2,ln(2)/2]; tiny tails flush to zero. */
static float negexp(float x){
 if(x < -87.0f)return 0.0f;
 int n=(int)(x*1.4426950408889634f-0.5f);
 float r=(x-n*0.693145751953125f)-n*1.428606765330187e-6f;
 float v=1.0f+r*(1.0f+r*(0.5f+r*(0.1666666666666667f+r*(0.0416666666666667f+r*(0.0083333333333333f+r*(0.0013888888888889f+r*0.0001984126984127f))))));
 union{uint32_t u;float f;}scale;scale.u=(uint32_t)(n+127)<<23;return v*scale.f;
}
int ug_load(const void*blob,uint32_t len){
 ready=0;profile_enabled=0;if(!ug_gemm_available())return 24;if(!blob||len<sizeof(hdr))return 10;copy(&hdr,blob,sizeof(hdr));
 if(hdr.magic!=UG_MAGIC||(hdr.version<1||hdr.version>3)||!hdr.nt||hdr.nt>UG_MAX_T||!hdr.no||hdr.no>UG_MAX_OP||!hdr.arena_bytes||hdr.arena_bytes>UG_ARENA||hdr.input>=hdr.nt||hdr.output>=hdr.nt||(hdr.version==1?hdr.reserved!=0:hdr.reserved>1))return 11;
 uint32_t meta=sizeof(hdr)+hdr.nt*sizeof(UGTensor)+hdr.no*sizeof(UGOp);
 if(len!=meta+hdr.arena_bytes)return 12;
 const unsigned char*p=blob;copy(ts,p+sizeof(hdr),hdr.nt*sizeof(UGTensor));copy(ops,p+sizeof(hdr)+hdr.nt*sizeof(UGTensor),hdr.no*sizeof(UGOp));
 unsigned char defined[UG_MAX_T]={0};uint32_t end=0;
 int birth[UG_MAX_T],death[UG_MAX_T];
 for(uint32_t i=0;i<hdr.nt;i++){
  UGTensor*t=ts+i;uint32_t w=width(t->type);uint64_t count=1;
  if(!w||t->rank>4||t->offset%128||(hdr.version<3&&t->offset<end)||t->offset>hdr.arena_bytes||t->flags>1||t->reserved[0]||t->reserved[1]||t->reserved[2])return 13;
  for(uint32_t d=0;d<t->rank;d++){if(!t->d[d]||t->d[d]>UG_ARENA)return 13;count*=t->d[d];if(count>UG_ARENA)return 13;}
  if(count*w!=t->bytes||t->bytes>hdr.arena_bytes-t->offset)return 13;
  end=t->offset+t->bytes;defined[i]=t->flags;
  birth[i]=-1;death[i]=t->flags?(int)hdr.no:-1;
 }
 if(ts[hdr.input].flags)return 14;
 defined[hdr.input]=1;
 uint64_t work=0;vision_only=1;int parallel_vision=0;
 for(uint32_t i=0;i<hdr.no;i++){
  UGOp*o=ops+i;
  if(o->code!=5&&o->code!=10&&(o->code<14||o->code>28))vision_only=0;
  if(o->a>=hdr.nt||o->b>=hdr.nt||o->out>=hdr.nt||!defined[o->a]||defined[o->out])return 15;
  UGTensor*a=ts+o->a,*b=ts+o->b,*z=ts+o->out;
  if(o->code==1||o->code==6||o->code==8||o->code==9){
   int integer=o->code==6||o->code==9,packed=o->code==8||o->code==9;
   if(packed&&hdr.version<2)return 16;
   if(!defined[o->b]||!b->flags||a->rank!=2||b->rank!=2||z->rank!=2||a->d[1]!=b->d[1]||z->d[0]!=a->d[0]||(packed?((z->d[1]+3)/4*4!=b->d[0]):z->d[1]!=b->d[0]))return 16;
   if(!integer && (a->type!=1||b->type!=1||z->type!=1))return 16;
   if(integer && (a->type!=2||b->type!=3||z->type!=4||a->d[1]>8192||o->p[0]>255||(int32_t)o->p[1]<-128||(int32_t)o->p[1]>127))return 16;
   if(integer && (o->p[3]!=1||o->p[2]>=hdr.nt||!defined[o->p[2]]||!ts[o->p[2]].flags||ts[o->p[2]].type!=4||ts[o->p[2]].bytes!=z->d[1]*4))return 16;
   if(o->code==8&&(o->p[0]>1||o->p[1]||o->p[2]||o->p[3]))return 16;
   if(o->code==9&&a->d[1]%8)return 16;
   work+=(uint64_t)a->d[0]*a->d[1]*z->d[1];
  }else if(o->code==2||o->code==3){
   if(!defined[o->b]||!same(a,z)||!broadcast(a,b)||(a->type!=1&&a->type!=4))return 17;
   work+=a->bytes/width(a->type);
  }else if(o->code==4){if(!same(a,z)||(a->type!=1&&a->type!=4))return 18;work+=a->bytes/4;}
  else if(o->code==5){if(a->type!=z->type||a->bytes!=z->bytes)return 18;work+=a->bytes;}
  else if(o->code==7){if(a->type!=4||z->type!=1||a->bytes!=z->bytes)return 18;work+=a->bytes/4;}
  else if(o->code==10){
   if(a->rank!=z->rank||a->type!=z->type||a->bytes!=z->bytes)return 18;
   unsigned seen=0;
   for(unsigned d=0;d<a->rank;d++){unsigned q=o->p[d];if(q>=a->rank||(seen&(1u<<q))||z->d[d]!=a->d[q])return 18;seen|=1u<<q;}
   work+=z->bytes;
  }else if(o->code==11||o->code==12){
   if(a->rank!=4||a->type!=1||z->type!=1||!defined[o->b]||!b->flags||b->type!=4||b->bytes!=48)return 18;
   int32_t q[12];copy(q,p+meta+b->offset,48);
   for(unsigned d=0;d<6;d++)if(q[d]<1||q[d]>4096)return 18;
   for(unsigned d=6;d<12;d++)if(q[d]<0||q[d]>4096)return 18;
   int64_t hh=(int64_t)a->d[1]+q[6]+q[8]-(q[0]-1)*(int64_t)q[4]-1;
   int64_t ww=(int64_t)a->d[2]+q[7]+q[9]-(q[1]-1)*(int64_t)q[5]-1;
   if(hh<0||ww<0||hh/q[2]+1!=q[10]||ww/q[3]+1!=q[11])return 18;
   uint64_t rows=(uint64_t)a->d[0]*q[10]*q[11],k=(uint64_t)q[0]*q[1]*a->d[3];
   if(o->code==11){if(z->rank!=2||z->d[0]!=rows||z->d[1]!=k)return 18;}
   else if(z->rank!=4||z->d[0]!=a->d[0]||z->d[1]!=(uint32_t)q[10]||z->d[2]!=(uint32_t)q[11]||z->d[3]!=a->d[3])return 18;
   work+=rows*k;
  }else if(o->code==13){if(!same(a,z)||a->type!=1||!a->rank)return 18;work+=a->bytes/4;}
  else if(o->code>=14&&o->code<=17){
   if(hdr.version<2||o->p[0]>=hdr.nt||!defined[o->p[0]]||!defined[o->b])return 18;
   const UGTensor*t=ts+o->p[0];
   int rc=uq_validate(o,a,b,z,t,p+meta+t->offset,p+meta+b->offset,&work);if(rc)return rc;
   if(o->p[1]){
    unsigned id=o->p[1]-1;if(id>=hdr.nt||!ts[id].flags||ts[id].type!=2||ts[id].bytes!=256)return 18;
    death[id]=(int)i;work+=z->bytes;
   }
   if((o->code==14||o->code==15)&&z->bytes>=2048&&(uint64_t)z->d[0]*z->d[1]*z->d[2]>=8)parallel_vision=1;
  }else if(o->code==18){
   if(hdr.version<2||!defined[o->b]||a->type!=3||!same(a,b)||!same(a,z)||o->p[0]>=hdr.nt||o->p[1]||o->p[2]||o->p[3])return 18;
   UGTensor*t=ts+o->p[0];
   if(!t->flags||t->type!=4||t->bytes!=520*4)return 18;
   const int32_t*q=(const int32_t*)(p+meta+t->offset);
   if(q[0]<0||q[1]<-31||q[1]>0||q[2]<-128||q[2]>127||q[3]<-128||q[4]>127||q[3]>q[4]||q[5]||q[6]||q[7])return 18;
   for(unsigned j=8;j<520;j++)if(q[j]<-267386880||q[j]>267386880)return 18;
   work+=z->bytes;
  }else if(o->code==19){
   if(hdr.version<2||!b->flags||b->type!=4||!b->bytes||b->bytes>32*4||b->bytes%4||z->type!=3||z->rank!=1||o->p[0]||o->p[1]||o->p[2]||o->p[3])return 18;
   const uint32_t*list=(const uint32_t*)(p+meta+b->offset);uint64_t total=0;
   if(list[0]!=o->a)return 18;
   for(unsigned j=0;j<b->bytes/4;j++){
    unsigned id=list[j];if(id>=hdr.nt||!defined[id]||(ts[id].type!=2&&ts[id].type!=3))return 18;
    total+=ts[id].bytes;death[id]=(int)i;
   }
   if(total!=z->bytes)return 18;
   work+=total;
  }else if(o->code==20){
   unsigned axis=o->p[0];
   if(hdr.version<2||!defined[o->b]||a->type!=3||b->type!=3||z->type!=3||a->rank!=b->rank||a->rank!=z->rank||axis>=a->rank||o->p[1]||o->p[2]||o->p[3])return 18;
   for(unsigned d=0;d<a->rank;d++){
    if(d==axis){if((uint64_t)a->d[d]+b->d[d]!=z->d[d])return 18;}
    else if(a->d[d]!=b->d[d]||a->d[d]!=z->d[d])return 18;
   }
   work+=z->bytes;
  }else if(o->code==21){
   if(hdr.version<2||a->type!=3||z->type!=3||a->rank!=4||z->rank!=4||a->d[0]!=z->d[0]||a->d[3]!=z->d[3]||!b->flags||b->type!=4||b->bytes!=(z->d[1]+z->d[2])*4||o->p[0]||o->p[1]||o->p[2]||o->p[3])return 18;
   const uint32_t*q=(const uint32_t*)(p+meta+b->offset);uint32_t row=a->d[2]*a->d[3],batch=a->d[1]*row;
   for(unsigned d=0;d<z->d[1];d++)if(q[d]>=batch||q[d]%row)return 18;
   for(unsigned d=0;d<z->d[2];d++)if(q[z->d[1]+d]>=row||q[z->d[1]+d]%a->d[3])return 18;
   work+=z->bytes;
  }else if(o->code==22){
   if(hdr.version<2||(a->type!=2&&a->type!=3)||(z->type!=2&&z->type!=3)||a->rank!=z->rank||a->bytes!=z->bytes||!b->flags||b->type!=2||b->bytes!=256||o->p[0]||o->p[1]||o->p[2]||o->p[3])return 18;
   for(unsigned d=0;d<a->rank;d++)if(a->d[d]!=z->d[d])return 18;
   work+=z->bytes;
  }else if(o->code==23){
   if(hdr.version<2||a->type!=3||z->type!=3||a->rank!=4||z->rank!=4||!b->flags||b->type!=4||b->bytes!=32||(int32_t)o->p[0]<-128||(int32_t)o->p[0]>127||o->p[1]||o->p[2]||o->p[3])return 18;
   const int32_t*q=(const int32_t*)(p+meta+b->offset);
   for(unsigned d=0;d<4;d++)if(q[d*2]<0||q[d*2]>4096||q[d*2+1]<0||q[d*2+1]>4096||(uint64_t)a->d[d]+q[d*2]+q[d*2+1]!=z->d[d])return 18;
   work+=z->bytes;
  }else if(o->code==24){
   if(hdr.version<2||a->type!=3||z->type!=3||a->rank!=4||(z->rank!=2&&z->rank!=4)||!b->flags||b->type!=4||b->bytes!=20||o->p[0]||o->p[1]||o->p[2]||o->p[3])return 18;
   if(z->d[0]!=a->d[0]||z->d[z->rank-1]!=a->d[3]||(z->rank==4&&(z->d[1]!=1||z->d[2]!=1)))return 18;
   const int32_t*q=(const int32_t*)(p+meta+b->offset);
   if(q[0]<1||(uint64_t)a->d[1]*a->d[2]!=(uint32_t)q[0]||q[1]<-128||q[1]>127||q[2]<-128||q[2]>127||q[3]<0||q[4]<-31||q[4]>30)return 18;
   uint64_t bound=(uint64_t)q[0]*(q[1]>=0?128+q[1]:127-q[1]);
   if((bound<<(q[4]>0?q[4]:0))>2147483520u)return 18;
   work+=a->bytes;
  }else if(o->code==25||o->code==27){
   if(hdr.version<2||a->type!=3||b->type!=3||z->type!=3||!defined[o->b]||o->p[0]>=hdr.nt||o->p[1]||o->p[2]||o->p[3])return 18;
   const UGTensor*t=ts+o->p[0];if(!t->flags||t->type!=4||t->bytes!=28)return 18;
   const int32_t*q=(const int32_t*)(p+meta+t->offset);
   for(unsigned j=0;j<3;j++)if(q[j]<-128||q[j]>127)return 18;
   if(q[3]<0||q[4]<-31||q[4]>15)return 18;
   uint64_t k=1;
   if(o->code==25){if(!same(a,b)||!same(a,z)||q[5]<-128||q[6]>127||q[5]>q[6])return 18;work+=z->bytes;}
   else{
    if(a->rank!=3||b->rank!=3||z->rank!=3||q[5]<0||q[5]>1||q[6]<0||q[6]>1||a->d[0]!=b->d[0]||a->d[0]!=z->d[0])return 18;
    unsigned m=a->d[q[5]?2:1],n=b->d[q[6]?1:2];k=a->d[q[5]?1:2];
    if(k!=b->d[q[6]?2:1]||k>8192||n>2048||z->d[1]!=m||z->d[2]!=n)return 18;
    work+=(uint64_t)z->bytes*k;
   }
   if(k*65025*(1ULL<<(q[4]>0?q[4]:0))>2147483520u)return 18;
  }else if(o->code==26){
   if(hdr.version<2||a->type!=3||z->type!=3||a->rank!=z->rank||!a->rank)return 18;
   for(unsigned d=0;d<4;d++){
    if(d>=a->rank){if(o->p[d])return 18;}
    else if((uint64_t)o->p[d]+z->d[d]>a->d[d])return 18;
   }
   work+=z->bytes;
  }else if(o->code==28){
   if(hdr.version<2||a->type!=3||!same(a,z)||!a->rank||a->d[a->rank-1]>2048||!b->flags||b->type!=4||b->bytes!=257*4||o->p[0]||o->p[1]||o->p[2]||o->p[3])return 18;
   const int32_t*q=(const int32_t*)(p+meta+b->offset);
   if(q[0]!=(int32_t)a->d[a->rank-1]||q[1]!=2147483647)return 18;
   for(unsigned d=2;d<=256;d++)if(q[d]<0||q[d]>q[d-1])return 18;
   work+=z->bytes;
  }else return 19;
  if(work>UG_MAX_WORK)return 20;
  death[o->a]=(int)i;death[o->b]=(int)i;birth[o->out]=(int)i;death[o->out]=(int)i;
  if(o->code>=14&&o->code<=18)death[o->p[0]]=(int)i;
  if(o->code==25||o->code==27)death[o->p[0]]=(int)i;
  if(o->code==6||o->code==9)death[o->p[2]]=(int)i;
  defined[o->out]=1;
 }
 vision_only=vision_only&&parallel_vision;
 if(!defined[hdr.output])return 21;
 if(hdr.version==3){
  death[hdr.output]=(int)hdr.no;
  for(unsigned a=0;a<hdr.nt;a++){
   /* Constants are always immutable, even after their last graph consumer. */
   if(ts[a].flags){birth[a]=-1;death[a]=(int)hdr.no;}
   if(!defined[a])return 15;
  }
  for(unsigned a=0;a<hdr.nt;a++)for(unsigned b=a+1;b<hdr.nt;b++){
   int memory_overlap=ts[a].offset<ts[b].offset+ts[b].bytes&&ts[b].offset<ts[a].offset+ts[a].bytes;
   int live_overlap=birth[a]<=death[b]&&birth[b]<=death[a];
   if(memory_overlap&&live_overlap)return 25;
  }
 }
 copy(arena,p+meta,hdr.arena_bytes);
 uq_post_reset();
 for(unsigned i=0;i<hdr.no;i++)if((ops[i].code==14||ops[i].code==15)&&ops[i].p[1])
  uq_post_register((const int32_t*)(arena+ts[ops[i].p[0]].offset),arena+ts[ops[i].p[1]-1].offset);
#ifdef UQ_CONSTANT_CHANNELS
 uq_prepare_reset();
 #ifdef UQ_RANGE_PROP
 for(unsigned t=0;t<hdr.nt;t++)for(unsigned c=0;c<UQ_MAX_C;c++){range_lo[t][c]=-128;range_hi[t][c]=127;}
 #endif
 for(unsigned i=0;i<hdr.no;i++){
  const UGOp*o=ops+i;
  if(o->code==14||o->code==15){
   const int16_t*lo=0,*hi=0;int16_t*ol=0,*oh=0;
   #ifdef UQ_RANGE_PROP
   lo=range_lo[o->a];hi=range_hi[o->a];ol=range_lo[o->out];oh=range_hi[o->out];
   #endif
   int rc=uq_prepare(o->code,(const int8_t*)arena+ts[o->b].offset,(const int32_t*)(arena+ts[o->p[0]].offset),lo,hi,ol,oh);
   if(rc)return rc;
   #ifdef UQ_RANGE_PROP
   if(o->p[1]){
    const int8_t*lut=(const int8_t*)arena+ts[o->p[1]-1].offset;
    for(unsigned c=0;c<ts[o->out].d[3];c++){
     int lower=127,upper=-128;
     for(int v=ol[c];v<=oh[c];v++){int value=lut[v+128];if(value<lower)lower=value;if(value>upper)upper=value;}
     ol[c]=(int16_t)lower;oh[c]=(int16_t)upper;
    }
   }
   #endif
  }
  #ifdef UQ_RANGE_PROP
  else if(o->code==5&&ts[o->a].rank&&ts[o->out].rank){
   unsigned ac=ts[o->a].d[ts[o->a].rank-1],oc=ts[o->out].d[ts[o->out].rank-1];
   if(ac==oc&&ac<=UQ_MAX_C)for(unsigned c=0;c<oc;c++){range_lo[o->out][c]=range_lo[o->a][c];range_hi[o->out][c]=range_hi[o->a][c];}
  }else if(o->code==16){
   const int32_t*q=(const int32_t*)(arena+ts[o->p[0]].offset);
   for(unsigned c=0;c<(unsigned)q[17];c++){
    int lo=range_lo[o->a][c],hi=range_hi[o->a][c];
    range_lo[o->out][c]=(int16_t)(lo<q[14]?q[14]:lo>q[15]?q[15]:lo);
    range_hi[o->out][c]=(int16_t)(hi<q[14]?q[14]:hi>q[15]?q[15]:hi);
   }
  }
  #endif
 }
#endif
 ready=1;return 0;
}
uint32_t ug_input_bytes(void){return ready?ts[hdr.input].bytes:0;}
uint32_t ug_output_bytes(void){return ready?ts[hdr.output].bytes:0;}
int ug_snapshot(uint32_t index,void*out,uint32_t bytes){
 if(!ready||index>=hdr.nt||!out||bytes!=ts[index].bytes)return 22;
 copy(out,arena+ts[index].offset,bytes);return 0;
}
int ug_profile(uint32_t command,void*out,uint32_t bytes){
 if(!ready||command>2||!out||(bytes!=sizeof(profile_ticks)&&!(hdr.no<=128&&bytes==130*8)))return 22;
 if(command==1)for(unsigned i=0;i<UG_MAX_OP+2;i++)profile_ticks[i]=0;
 if(command<2)profile_enabled=command;
 copy(out,profile_ticks,bytes);
#ifdef UQ_DIAGNOSTICS
 if(hdr.no<=UG_MAX_OP-8){
  extern void uq_measurement(unsigned,uint64_t*);uint64_t phases[8];uq_measurement(command,phases);
  if(bytes==sizeof(profile_ticks))copy((uint64_t*)out+UG_MAX_OP+2-8,phases,sizeof(phases));
  /* Preserve the original small-graph diagnostic layout without overrunning
   * callers that still supply the supported 130-counter buffer. */
  if(hdr.no<=120)copy((uint64_t*)out+122,phases,sizeof(phases));
 }
#endif
 return 0;
}
static int run_inner(const void*input,uint32_t ni,void*output,uint32_t no){
 if(!ready||!input||!output||ni!=ug_input_bytes()||no!=ug_output_bytes())return 22;
 uint64_t run_start=profile_enabled?tick():0;
 copy(arena+ts[hdr.input].offset,input,ni);
 for(uint32_t i=0;i<hdr.no;i++){
  uint64_t op_start=profile_enabled?tick():0;
  const UGOp*o=ops+i;const UGTensor*a=ts+o->a,*b=ts+o->b,*z=ts+o->out;
  void*ap=arena+a->offset,*bp=arena+b->offset,*zp=arena+z->offset;uint32_t count=z->bytes/width(z->type);
  if(o->code>=14&&o->code<=17){
#ifdef UQ_FUSE_DW_PW
   if(o->code==15&&i+1<hdr.no&&ops[i+1].code==14&&ops[i+1].a==o->out){
    const UGOp*next=ops+i+1;
    int rc=uq_run_fused(a,z,ts+next->out,ap,bp,(const int8_t*)arena+ts[next->b].offset,zp,(int8_t*)arena+ts[next->out].offset,(const int32_t*)(arena+ts[o->p[0]].offset),(const int32_t*)(arena+ts[next->p[0]].offset),hdr.reserved?2:1);
    if(rc!=-1){if(rc){ready=0;return rc;}PROFILE_OP_END();i++;continue;}
   }
#endif
   int rc=uq_run(o->code,a,z,ap,bp,zp,(const int32_t*)(arena+ts[o->p[0]].offset),hdr.reserved?2:1);if(rc){ready=0;return rc;}
  }else if(o->code==28){
   uq_softmax_exact(ap,zp,a->bytes,bp);
  }else if(o->code==27){
   int rc=uq_bmm(a,b,z,ap,bp,zp,(const int32_t*)(arena+ts[o->p[0]].offset),hdr.reserved?2:1);if(rc){ready=0;return rc;}
  }else if(o->code==26){
   unsigned stride[4],step=1,offset=0,index[4]={0};
   for(int d=(int)a->rank-1;d>=0;d--){stride[d]=step;offset+=o->p[d]*step;step*=a->d[d];}
   unsigned last=a->rank-1,span=z->d[last];
   for(unsigned j=0;j<z->bytes;j+=span){
    copy((uint8_t*)zp+j,(uint8_t*)ap+offset,span);
    for(int d=(int)last-1;d>=0;d--){offset+=stride[d];if(++index[d]<z->d[d])break;offset-=index[d]*stride[d];index[d]=0;}
   }
  }else if(o->code==25){
   uq_mul(ap,bp,zp,z->bytes,(const int32_t*)(arena+ts[o->p[0]].offset));
  }else if(o->code==24){
   uq_mean(ap,zp,a->d[0],a->d[3],bp);
  }else if(o->code==23){
   const int32_t*q=bp;int8_t*dest=zp;const int8_t*src=ap;
   for(unsigned j=0;j<z->bytes;j++)dest[j]=(int8_t)(int32_t)o->p[0];
   if(!q[6]&&!q[7]){
    unsigned row=a->d[2]*a->d[3];
    for(unsigned n=0;n<a->d[0];n++)for(unsigned h=0;h<a->d[1];h++){
     unsigned offset=(((n+q[0])*z->d[1]+h+q[2])*z->d[2]+q[4])*z->d[3];
     copy(dest+offset,src+(n*a->d[1]+h)*row,row);
    }
   }else for(unsigned n=0;n<a->d[0];n++)for(unsigned h=0;h<a->d[1];h++)for(unsigned w=0;w<a->d[2];w++){
    unsigned offset=(((n+q[0])*z->d[1]+h+q[2])*z->d[2]+w+q[4])*z->d[3]+q[6];
    copy(dest+offset,src+((n*a->d[1]+h)*a->d[2]+w)*a->d[3],a->d[3]);
   }
  }else if(o->code==22){
   const uint8_t*table=bp;uint8_t*dest=zp;
   unsigned j=0;const uint8_t*src=ap;unsigned flip=a->type==3?128:0;
#ifdef UQ_LOOKUP_WORD
   for(;j+4<=a->bytes;j+=4){
    uint32_t word;__builtin_memcpy(&word,__builtin_assume_aligned(src+j,4),4);word^=flip*0x01010101u;
    uint32_t result=table[word&255]|((uint32_t)table[(word>>8)&255]<<8)|((uint32_t)table[(word>>16)&255]<<16)|((uint32_t)table[word>>24]<<24);
    __builtin_memcpy(__builtin_assume_aligned(dest+j,4),&result,4);
   }
#endif
   for(;j<a->bytes;j++)dest[j]=table[src[j]^flip];
  }else if(o->code==21){
   const uint32_t*q=bp;unsigned c=a->d[3],stride=a->d[1]*a->d[2]*c,offset=0;
   for(unsigned n=0;n<a->d[0];n++)for(unsigned h=0;h<z->d[1];h++)for(unsigned w=0;w<z->d[2];w++){
    copy((unsigned char*)zp+offset,(unsigned char*)ap+n*stride+q[h]+q[z->d[1]+w],c);offset+=c;
   }
  }else if(o->code==20){
   unsigned outer=1,inner=1,axis=o->p[0];
   for(unsigned d=0;d<axis;d++)outer*=a->d[d];
   for(unsigned d=axis+1;d<a->rank;d++)inner*=a->d[d];
   unsigned ac=a->d[axis]*inner,bc=b->d[axis]*inner;
   for(unsigned r=0;r<outer;r++){
    copy((unsigned char*)zp+r*(ac+bc),(unsigned char*)ap+r*ac,ac);
    copy((unsigned char*)zp+r*(ac+bc)+ac,(unsigned char*)bp+r*bc,bc);
   }
  }else if(o->code==19){
   const uint32_t*list=bp;unsigned offset=0;
   for(unsigned j=0;j<b->bytes/4;j++){
    const UGTensor*t=ts+list[j];copy((unsigned char*)zp+offset,arena+t->offset,t->bytes);offset+=t->bytes;
   }
  }else if(o->code==18){
   uq_add(ap,bp,zp,z->bytes,(const int32_t*)(arena+ts[o->p[0]].offset));
  }else if(o->code==1||o->code==8){
   const float*x=ap,*w=bp;float*y=zp;uint32_t m=a->d[0],k=a->d[1],n=z->d[1];
   int rc=ug_matmul_f32(x,w,y,m,n,k,o->code==8,o->code==8&&o->p[0],hdr.reserved?2:1);if(rc){ready=0;return rc;}
  }else if(o->code==6||o->code==9){
   const uint8_t*x=ap;const int8_t*w=bp;int32_t*y=zp;uint32_t m=a->d[0],k=a->d[1],n=z->d[1];int az=o->p[0],bz=(int32_t)o->p[1];
   (void)az;const int32_t*correction=(const int32_t*)(arena+ts[o->p[2]].offset);
   int rc=ug_matmul_u8i8(x,w,y,m,n,k,o->code==9,hdr.reserved?2:1);if(rc){ready=0;return rc;}
   for(uint32_t r=0;r<m;r++){
    int32_t sx=0;if(bz)for(uint32_t c=0;c<k;c++)sx+=x[r*k+c];
    for(uint32_t j=0;j<n;j++)y[r*n+j]=y[r*n+j]+correction[j]-bz*sx;
   }
  }else if(o->code==10){
   if(a->rank==4&&a->type==1&&o->p[0]==0&&o->p[1]==3&&o->p[2]==1&&o->p[3]==2){
    const float*x=ap;float*y=zp;uint32_t h=a->d[1],w=a->d[2],c=a->d[3];
    for(uint32_t n=0;n<a->d[0];n++)for(uint32_t ch=0;ch<c;ch++)for(uint32_t row=0;row<h;row++)for(uint32_t col=0;col<w;col++)y[((n*c+ch)*h+row)*w+col]=x[((n*h+row)*w+col)*c+ch];
    PROFILE_OP_END();continue;
   }
   if(a->rank==4&&a->type==1&&o->p[0]==0&&o->p[1]==2&&o->p[2]==3&&o->p[3]==1){
    const float*x=ap;float*y=zp;uint32_t c=a->d[1],h=a->d[2],w=a->d[3];
    for(uint32_t n=0;n<a->d[0];n++)for(uint32_t row=0;row<h;row++)for(uint32_t col=0;col<w;col++)for(uint32_t ch=0;ch<c;ch++)y[((n*h+row)*w+col)*c+ch]=x[((n*c+ch)*h+row)*w+col];
    PROFILE_OP_END();continue;
   }
   uint32_t stride[4],acc=1;for(int d=(int)a->rank-1;d>=0;d--){stride[d]=acc;acc*=a->d[d];}
   /* Odometer indexing avoids division/modulo helpers inside the DSP loop. */
   uint32_t w=width(a->type),index[4]={0},src=0;
   for(uint32_t j=0;j<count;j++){
    copy((unsigned char*)zp+j*w,(unsigned char*)ap+src*w,w);
    for(int d=(int)z->rank-1;d>=0;d--){uint32_t step=stride[o->p[d]];src+=step;if(++index[d]<z->d[d])break;src-=index[d]*step;index[d]=0;}
   }
  }else if(o->code==11||o->code==12){
   const int32_t*q=bp;const float*x=ap;float*y=zp;
   uint32_t channels=a->d[3],ih=a->d[1],iw=a->d[2];
   for(uint32_t batch=0;batch<a->d[0];batch++)for(int h=0;h<q[10];h++)for(int w=0;w<q[11];w++){
    uint32_t row=(batch*q[10]+h)*q[11]+w;
    if(o->code==12)for(uint32_t c=0;c<channels;c++)y[row*channels+c]=-3.402823466e38f;
    for(int kh=0;kh<q[0];kh++)for(int kw=0;kw<q[1];kw++){
     int sh=h*q[2]-q[6]+kh*q[4],sw=w*q[3]-q[7]+kw*q[5];int valid=sh>=0&&sw>=0&&(uint32_t)sh<ih&&(uint32_t)sw<iw;
     const float*source=valid?x+((batch*ih+sh)*iw+sw)*channels:0;
     if(o->code==11){float*target=y+(row*q[0]*q[1]+kh*q[1]+kw)*channels;for(uint32_t c=0;c<channels;c++)target[c]=valid?source[c]:0.0f;}
     else if(valid){float*target=y+row*channels;for(uint32_t c=0;c<channels;c++)if(source[c]>target[c])target[c]=source[c];}
    }
   }
  }else if(o->code==13){
   const float*x=ap;float*y=zp;uint32_t n=a->d[a->rank-1];
   for(uint32_t base=0;base<count;base+=n){float max=x[base],sum=0;for(uint32_t j=1;j<n;j++)if(x[base+j]>max)max=x[base+j];for(uint32_t j=0;j<n;j++){y[base+j]=negexp(x[base+j]-max);sum+=y[base+j];}for(uint32_t j=0;j<n;j++)y[base+j]/=sum;}
  }else if(o->code==5)copy(zp,ap,z->bytes);
  else if(o->code==7){for(uint32_t j=0;j<count;j++)((float*)zp)[j]=(float)((int32_t*)ap)[j];}
  else if(z->type==1){const float*x=ap,*v=bp;float*y=zp;uint32_t nb=b->bytes/4;
   if(o->code==4){for(uint32_t j=0;j<count;j++)y[j]=x[j]<0?0:x[j];}
   else if(nb==1){float scalar=v[0];if(o->code==2){for(uint32_t j=0;j<count;j++)y[j]=x[j]+scalar;}else{for(uint32_t j=0;j<count;j++)y[j]=x[j]*scalar;}}
   else if(o->code==2){for(uint32_t base=0;base<count;base+=nb)for(uint32_t j=0;j<nb;j++)y[base+j]=x[base+j]+v[j];}
   else{for(uint32_t base=0;base<count;base+=nb)for(uint32_t j=0;j<nb;j++)y[base+j]=x[base+j]*v[j];}
  }else{const uint32_t*x=ap,*v=bp;uint32_t*y=zp;uint32_t nb=b->bytes/4;
   for(uint32_t j=0;j<count;j++)y[j]=o->code==4?((int32_t)x[j]<0?0:x[j]):o->code==2?x[j]+v[j%nb]:x[j]*v[j%nb];
  }
  PROFILE_OP_END();
 }
 copy(output,arena+ts[hdr.output].offset,no);
 if(profile_enabled){profile_ticks[0]++;profile_ticks[1]+=tick()-run_start;}
 return 0;
}

#ifdef UQ_STREAM_WORKER
static int stream_active;
int ug_stream_begin(void){
 if(stream_active)return 24;
 if(!ready)return 22;
 if(vision_only&&hdr.reserved){int rc=ug_begin_frame();if(rc){ready=0;return rc;}stream_active=1;}
 return 0;
}
int ug_stream_end(void){
 if(!stream_active)return 0;
 int rc=ug_end_frame();stream_active=0;if(rc)ready=0;return rc;
}
#endif
int ug_run(const void*input,uint32_t ni,void*output,uint32_t no){
 if(!ready||!input||!output||ni!=ug_input_bytes()||no!=ug_output_bytes())return 22;
#ifdef UQ_STREAM_WORKER
 if(stream_active){int rc=run_inner(input,ni,output,no);if(rc)ready=0;return rc;}
#endif
 if(!vision_only||!hdr.reserved)return run_inner(input,ni,output,no);
 int rc=ug_begin_frame();if(rc){ready=0;return rc;}
 rc=run_inner(input,ni,output,no);
 int joined=ug_end_frame();if(rc||joined){ready=0;return rc?rc:joined;}
 return 0;
}
