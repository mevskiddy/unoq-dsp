#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <sys/ioctl.h>
#include <linux/dma-buf.h>
#include "remote.h"
#include "rpcmem.h"
#include "graph.h"
typedef struct {remote_handle64 h;void*input,*output;uint32_t ni,no,ib,ob,seq;int fi,fo,pipeline,started,rc;pthread_t thread;_Atomic int done;} Session;
static pthread_mutex_t guard=PTHREAD_MUTEX_INITIALIZER;
static int active;
/* Keep one FastRPC module per host process. Reloading a model must not repeatedly
   tear down and recreate the entire DSP protection domain and listener threads. */
static remote_handle64 resident_handle;
static Session*live_session;
static int cleanup_registered;
/* Keep DMA mappings with the resident transport. Repeated model changes should
 * not recycle input/output pages and FastRPC registrations on every close. */
typedef struct {void*data;uint32_t capacity;} TransportBuffer;
static TransportBuffer transport_buffers[3]; /* plan, input, output */
static void*reserve_buffer(unsigned slot,uint32_t bytes){
 TransportBuffer*b=transport_buffers+slot;
 if(b->capacity>=bytes)return b->data;
 uint32_t capacity=(bytes+65535u)&~65535u;
 void*next=rpcmem_alloc(25,1,capacity);if(!next)return NULL;
 /* Allocate before releasing the old registration, avoiding immediate VA
  * reuse on a growth boundary. Capacity is bounded by validated plan/I/O. */
 if(b->data)rpcmem_free(b->data);
 b->data=next;b->capacity=capacity;return next;
}
void ug_close(void*);
static void release_transport(void){
 if(live_session)ug_close(live_session);
 if(resident_handle){remote_handle64_close(resident_handle);resident_handle=0;}
 for(unsigned i=0;i<3;i++){if(transport_buffers[i].data)rpcmem_free(transport_buffers[i].data);transport_buffers[i]=(TransportBuffer){0};}
}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int syncbuf(int fd,uint64_t flags){struct dma_buf_sync s={.flags=flags};return ioctl(fd,DMA_BUF_IOCTL_SYNC,&s);}
static int invoke(remote_handle64 h,unsigned method,void*in,unsigned ni,void*out,unsigned no){remote_arg a[2]={0};a[0].buf.pv=in;a[0].buf.nLen=ni;a[1].buf.pv=out;a[1].buf.nLen=no;return remote_handle64_invoke(h,REMOTE_SCALARS_MAKEX(0,method,1,1,0,0),a);}
static void*worker(void*ptr){Session*s=ptr;s->rc=invoke(s->h,2,s->input,s->ib,s->output,s->ob);atomic_store(&s->done,1);return NULL;}
static int start_worker(Session*s){
 if(s->started){pthread_join(s->thread,NULL);s->started=0;if(s->rc)return s->rc;}
 if(syncbuf(s->fi,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE))return 1004;
 memset(s->input,0,128);
 if(syncbuf(s->fi,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE))return 1004;
 if(syncbuf(s->fo,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE))return 1004;
 memset(s->output,0,128);
 if(syncbuf(s->fo,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE))return 1004;
 s->seq=0;s->rc=0;atomic_store(&s->done,0);
 if(pthread_create(&s->thread,NULL,worker,s))return 1005;
 s->started=1;
 double begin=now();
 while(now()-begin<1.0 && !atomic_load(&s->done)){
  if(syncbuf(s->fo,DMA_BUF_SYNC_START|DMA_BUF_SYNC_READ))return 1004;
  int ready=((volatile uint32_t*)s->output)[2]==1;
  if(syncbuf(s->fo,DMA_BUF_SYNC_END|DMA_BUF_SYNC_READ))return 1004;
  if(ready)return 0;
 }
 /* A clean idle expiry before readiness is safe to restart: no input sequence
    has been submitted. Keep this distinct from an ambiguous inference timeout. */
 if(atomic_load(&s->done))return s->rc?s->rc:1007;
 fprintf(stderr,"DSP mailbox readiness timed out before input submission\n");
 return 1006;
}
void ug_close(void*ptr){Session*s=ptr;if(!s)return;
 if(s->started){
  if(!atomic_load(&s->done)&&!syncbuf(s->fi,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE)){((volatile uint32_t*)s->input)[1]=1;syncbuf(s->fi,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}
  pthread_join(s->thread,NULL);
 }
 /* The transport and its DMA buffers survive model close; process exit
  * releases them after the final worker and remote module have stopped. */
 live_session=NULL;
 free(s);
 pthread_mutex_lock(&guard);active=0;pthread_mutex_unlock(&guard);
}
/* A loader EUNABLETOLOAD may be transient; its cause is not assumed here.
 * Retry module OPEN only;
 * never replay inference or hide permanent failures. A missing/broken module
 * still fails after two bounded delays, with the original error preserved. */
static int open_transport(remote_handle64*handle){
 int rc;
 for(unsigned attempt=0;attempt<3;attempt++){
  *handle=0;
  rc=remote_handle64_open("file:///libuno_graph.so?graph_skel_invoke&_dom=adsp",handle);
  if(!rc)return 0;
  *handle=0;
  if((uint32_t)rc!=UINT32_C(0x80000406)||attempt==2)return rc;
  fprintf(stderr,"DSP module loader returned 0x%08x; retry %u/2\n",(unsigned)rc,attempt+1);
  struct timespec delay={.tv_sec=0,.tv_nsec=50000000L*(attempt+1)};
  nanosleep(&delay,NULL);
 }
 return rc;
}
void*ug_open_mode(const void*plan,uint32_t bytes,int pipeline,int*error){
 *error=1;if(!plan||bytes>UG_MAX_PLAN||pipeline<0||pipeline>1)return NULL;
 pthread_mutex_lock(&guard);if(active){pthread_mutex_unlock(&guard);*error=1001;return NULL;}active=1;pthread_mutex_unlock(&guard);
 Session*s=calloc(1,sizeof(*s));if(!s){pthread_mutex_lock(&guard);active=0;pthread_mutex_unlock(&guard);return NULL;}s->pipeline=pipeline;live_session=s;
 int rc=0;
 if(!resident_handle){
  rc=open_transport(&resident_handle);
  if(rc){resident_handle=0;*error=rc;ug_close(s);return NULL;}
  if(!cleanup_registered){
   if(atexit(release_transport)){remote_handle64_close(resident_handle);resident_handle=0;ug_close(s);return NULL;}
   cleanup_registered=1;
  }
 }
 s->h=resident_handle;
 void*buf=reserve_buffer(0,bytes);if(!buf){*error=1002;ug_close(s);return NULL;}
 int planfd=rpcmem_to_fd(buf);uint32_t result[3]={0};
 if(syncbuf(planfd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE))rc=1004;
 else{
  memcpy(buf,plan,bytes);
  if(syncbuf(planfd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE))rc=1004;
  else rc=invoke(s->h,0,buf,bytes,result,12);
 }
 if(rc||result[0]||!result[1]||!result[2]||result[1]>4*1024*1024||result[2]>4*1024*1024){*error=rc?rc:(int)result[0];ug_close(s);return NULL;}
 s->ni=result[1];s->no=result[2];s->ib=(s->ni+128+4095)&~4095u;s->ob=(s->no+128+4095)&~4095u;
 s->input=reserve_buffer(1,s->ib);s->output=reserve_buffer(2,s->ob);
 if(!s->input||!s->output){*error=1002;ug_close(s);return NULL;}s->fi=rpcmem_to_fd(s->input);s->fo=rpcmem_to_fd(s->output);*error=0;return s;
}
void*ug_open(const void*plan,uint32_t bytes,int*error){return ug_open_mode(plan,bytes,0,error);}
int ug_snapshot_tensor(void*ptr,uint32_t index,void*out,uint32_t bytes){
 Session*s=ptr;if(!s||s->pipeline||!out||!bytes||bytes>4*1024*1024)return 1003;
 return invoke(s->h,3,&index,4,out,bytes);
}
#ifdef UQ_TRANSPORT_TRACE
int ug_transport_trace(void*ptr,uint32_t*out){Session*s=ptr;uint32_t dummy=0;return invoke(s->h,5,&dummy,4,out,16);}
#endif
int ug_profile_control(void*ptr,uint32_t command,void*out,uint32_t bytes){
 Session*s=ptr;if(!s||!out||(bytes!=130*8&&bytes!=(UG_MAX_OP+2)*8)||command>2)return 1003;
 /* Profiling is serialized with inference. Join the bounded mailbox RPC before
    touching graph counters; the next inference starts a fresh worker. */
 if(s->started){
  if(!atomic_load(&s->done)){
   if(syncbuf(s->fi,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE))return 1004;
   ((volatile uint32_t*)s->input)[1]=1;
   if(syncbuf(s->fi,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE))return 1004;
  }
  pthread_join(s->thread,NULL);s->started=0;if(s->rc)return s->rc;
 }
 return invoke(s->h,4,&command,4,out,bytes);
}
int ug_infer(void*ptr,const void*in,uint32_t ni,void*out,uint32_t no){
 Session*s=ptr;if(!s||!in||!out||ni!=s->ni||no!=s->no)return 1003;
 if(!s->pipeline){
  /* rpcmem is DMA-backed memory. Explicitly bracket CPU access even for
   * blocking RPC: recycled buffers may retain CPU cache lines across loads. */
  if(syncbuf(s->fi,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE))return 1004;
  memcpy((char*)s->input+128,in,ni);
  if(syncbuf(s->fi,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE))return 1004;
  int rc=invoke(s->h,1,(char*)s->input+128,ni,(char*)s->output+128,no);if(rc)return rc;
  if(syncbuf(s->fo,DMA_BUF_SYNC_START|DMA_BUF_SYNC_READ))return 1004;
  memcpy(out,(char*)s->output+128,no);
  return syncbuf(s->fo,DMA_BUF_SYNC_END|DMA_BUF_SYNC_READ)?1004:0;
 }
 /* A worker can expire at the boundary of a submission; retry once after joining. */
 for(int attempt=0;attempt<2;attempt++){
  if(!s->started||atomic_load(&s->done)){
   int rc=start_worker(s);if(rc==1007)continue;if(rc)return rc;
  }
  if(syncbuf(s->fi,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE))return 1004;
  memcpy((char*)s->input+128,in,ni);
  if(syncbuf(s->fi,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE))return 1004;
  if(syncbuf(s->fi,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE))return 1004;
  if(++s->seq==0)s->seq=1;
  atomic_thread_fence(memory_order_release);((volatile uint32_t*)s->input)[0]=s->seq;
  if(syncbuf(s->fi,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE))return 1004;
  double begin=now();
  while(now()-begin<4.0){
   if(syncbuf(s->fo,DMA_BUF_SYNC_START|DMA_BUF_SYNC_READ))return 1004;
   int found=((volatile uint32_t*)s->output)[0]==s->seq,rc=((volatile uint32_t*)s->output)[1];
   if(found && !rc)memcpy(out,(char*)s->output+128,no);
   if(syncbuf(s->fo,DMA_BUF_SYNC_END|DMA_BUF_SYNC_READ))return 1004;
   if(found)return rc;
   if(atomic_load(&s->done)){if(s->rc)return s->rc;break;}
  }
  if(!atomic_load(&s->done)){fprintf(stderr,"DSP mailbox response timed out after input submission; not replaying\n");return 1006;}
 }
 return 1006;
}
