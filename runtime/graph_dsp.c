#include "graph.h"
extern uint64_t qurt_timer_get_ticks(void);
extern int qurt_mem_cache_clean(uint32_t,uint32_t,int,int);
typedef union{struct{void*data;uint32_t length;}buf;uint64_t handle;} Arg;
#ifdef UQ_TRANSPORT_TRACE
static uint32_t trace[4];
static uint32_t trace_hash(const void*p,uint32_t n){const uint8_t*b=p;uint32_t h=2166136261u;for(uint32_t i=0;i<n;i++)h=(h^b[i])*16777619u;return h;}
#endif
#ifdef UQ_STREAM_WORKER
extern int ug_stream_begin(void);
extern int ug_stream_end(void);
#endif
#ifdef UQ_POWER_VOTE
/* Minimal public HAP_power.h DCVS-v2 ABI; boolean is one byte, union is
 * eight-byte aligned. A distinct client is destroyed on every mailbox exit.
 * This asks firmware for a supported performance corner, never a raw clock. */
typedef struct {uint8_t enable,pad0[3];int option;uint8_t set_latency,pad1[3];uint32_t latency;uint8_t set_params,pad2[3];int target,minimum,maximum;uint32_t reserved[3];} UQPowerV2;
typedef struct {int type;union {uint64_t alignment;UQPowerV2 v2;uint8_t storage[128];}u;} UQPowerRequest;
extern int HAP_power_set(void*,void*);
extern int HAP_power_destroy_client(void*) __attribute__((weak));
static uint8_t power_context;
static int power_begin(void){
 if(!HAP_power_destroy_client)return 25;
 UQPowerRequest r={0};r.type=7;r.u.v2.enable=1;r.u.v2.option=2;
 r.u.v2.set_latency=1;r.u.v2.latency=10;r.u.v2.set_params=1;
 r.u.v2.target=r.u.v2.minimum=r.u.v2.maximum=255;
 int rc=HAP_power_set(&power_context,&r);
 if(rc)HAP_power_destroy_client(&power_context);
 return rc;
}
#endif
/* The worker is retained only inside this bounded mailbox RPC and joined on
 * every exit, including timeout, stop, malformed buffers and inference error. */
static int mailbox_loop(Arg*a){
  uint32_t ni=ug_input_bytes(),no=ug_output_bytes();
  if(!ni||!no||a[0].buf.length<128+ni||a[1].buf.length<128+no||((uint32_t)a[0].buf.data&127)||((uint32_t)a[1].buf.data&127))return 2;
  volatile uint32_t*in=a[0].buf.data;volatile uint32_t*out=a[1].buf.data;
  uint32_t seen=0;uint64_t start=qurt_timer_get_ticks(),last=start;
  out[0]=0;out[1]=0;out[2]=1;
  int rc=qurt_mem_cache_clean((uint32_t)out,128,0,1);if(rc)return rc;
  /* Poll for at most 3 seconds, or 100 ms idle. Linux restarts on demand. */
  while(1){
   uint64_t now=qurt_timer_get_ticks();if(now-start>=57600000ULL||now-last>=1920000ULL)return 0;
   rc=qurt_mem_cache_clean((uint32_t)in,128,1,1);if(rc)return rc;
   if(in[1])return 0;uint32_t seq=in[0];
   if(seq && seq!=seen){
    rc=qurt_mem_cache_clean((uint32_t)in+128,ni,1,1);if(rc)return rc;
    rc=ug_run((const unsigned char*)in+128,ni,(unsigned char*)out+128,no);
    out[1]=rc;if(rc)return rc;
    rc=qurt_mem_cache_clean((uint32_t)out+128,no,0,1);if(rc)return rc;
    asm volatile("" ::: "memory");out[0]=seq;
    rc=qurt_mem_cache_clean((uint32_t)out,128,0,1);if(rc)return rc;
    seen=seq;last=qurt_timer_get_ticks();
   }
  }
}
__attribute__((visibility("default"))) int graph_skel_invoke(uint32_t sc,Arg*a){
 if(!a||!a[0].buf.data||!a[1].buf.data)return 2;
 if(sc==0x10100){if(a[1].buf.length!=12)return 2;int rc=ug_load(a[0].buf.data,a[0].buf.length);uint32_t*out=a[1].buf.data;out[0]=rc;out[1]=ug_input_bytes();out[2]=ug_output_bytes();return rc;}
 if(sc==0x2010100){
#ifdef UQ_STREAM_WORKER
#ifdef UQ_POWER_VOTE
  int vote=power_begin();if(vote)return vote;
#endif
  int rc=ug_stream_begin();
  if(!rc){rc=mailbox_loop(a);int joined=ug_stream_end();if(!rc)rc=joined;}
#ifdef UQ_POWER_VOTE
  int released=HAP_power_destroy_client(&power_context);if(!rc)rc=released;
#endif
  return rc;
#else
  return mailbox_loop(a);
#endif
 }
 if(sc==0x1010100){
  if(a[0].buf.length!=ug_input_bytes()||a[1].buf.length!=ug_output_bytes())return 22;
  /* The same physical pages can return under a new RPC mapping after model
   * close. Explicitly acquire input and publish output on the DSP side too,
   * as the mailbox path already does. Do not rely on cached mapping state. */
#ifdef UQ_TRANSPORT_TRACE
  trace[0]=trace_hash(a[0].buf.data,a[0].buf.length);
#endif
  int rc=qurt_mem_cache_clean((uint32_t)a[0].buf.data,a[0].buf.length,1,1);if(rc)return rc;
#ifdef UQ_TRANSPORT_TRACE
  trace[1]=trace_hash(a[0].buf.data,a[0].buf.length);
#endif
  rc=ug_run(a[0].buf.data,a[0].buf.length,a[1].buf.data,a[1].buf.length);if(rc)return rc;
#ifdef UQ_TRANSPORT_TRACE
  trace[2]=trace_hash(a[1].buf.data,a[1].buf.length);trace[3]++;
#endif
  return qurt_mem_cache_clean((uint32_t)a[1].buf.data,a[1].buf.length,0,1);
 }
 if(sc==0x3010100&&a[0].buf.length==4)return ug_snapshot(*(uint32_t*)a[0].buf.data,a[1].buf.data,a[1].buf.length);
 if(sc==0x4010100&&a[0].buf.length==4)return ug_profile(*(uint32_t*)a[0].buf.data,a[1].buf.data,a[1].buf.length);
#ifdef UQ_TRANSPORT_TRACE
 if(sc==0x5010100&&a[1].buf.length==16){uint32_t*out=a[1].buf.data;for(unsigned i=0;i<4;i++)out[i]=trace[i];return 0;}
#endif
 return 2;
}
