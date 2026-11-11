/* Linux mock: independent CPU/device views expose missing DMA sync. Also check
 * that repeated model close/open retains bounded transport registrations. */
#define ioctl mock_ioctl
#define remote_handle64_invoke mock_invoke
#define remote_handle64_open mock_open
#define remote_handle64_close mock_close
#include "../runtime/bridge.c"
#include <assert.h>
#include <stdarg.h>

typedef struct {void*cpu;unsigned char*device;unsigned size;int fd;} Buffer;
static Buffer buffers[16];
static unsigned allocations,frees,invocations;
static int fail_sync;
void*rpcmem_alloc(int heapid,uint32_t flags,int size){
 (void)heapid;(void)flags;assert(allocations<16);
 Buffer*b=buffers+allocations;b->cpu=calloc(1,size);b->device=calloc(1,size);
 assert(b->cpu&&b->device);b->size=size;b->fd=100+(int)allocations++;
 return b->cpu;
}
void rpcmem_free(void*p){for(unsigned i=0;i<allocations;i++)if(buffers[i].cpu==p){free(p);free(buffers[i].device);buffers[i].cpu=NULL;buffers[i].device=NULL;frees++;return;}assert(0);}
int rpcmem_to_fd(void*p){for(unsigned i=0;i<allocations;i++)if(buffers[i].cpu==p)return buffers[i].fd;assert(0);return -1;}
static unsigned char*device_address(void*p){
 for(unsigned i=0;i<allocations;i++){Buffer*b=buffers+i;if(b->cpu&&(uintptr_t)p>=(uintptr_t)b->cpu&&(uintptr_t)p<(uintptr_t)b->cpu+b->size)return b->device+((uintptr_t)p-(uintptr_t)b->cpu);}
 assert(0);return NULL;
}
int mock_ioctl(int fd,unsigned long request,...){
 assert(request==DMA_BUF_IOCTL_SYNC);if(fail_sync){fail_sync=0;return -1;}
 va_list ap;va_start(ap,request);struct dma_buf_sync*s=va_arg(ap,struct dma_buf_sync*);va_end(ap);
 for(unsigned i=0;i<allocations;i++)if(buffers[i].cpu&&buffers[i].fd==fd){Buffer*b=buffers+i;
  if(s->flags==(DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE))memcpy(b->device,b->cpu,b->size);
  if(s->flags==(DMA_BUF_SYNC_START|DMA_BUF_SYNC_READ))memcpy(b->cpu,b->device,b->size);
  return 0;
 }
 assert(0);return -1;
}
int mock_open(const char*uri,remote_handle64*h){(void)uri;*h=123;return 0;}
int mock_close(remote_handle64 h){assert(h==123);return 0;}
int mock_invoke(remote_handle64 h,uint32_t sc,remote_arg*a){
 assert(h==123);unsigned method=sc>>24;invocations++;
 if(method==0){uint32_t*p=(uint32_t*)device_address(a[0].buf.pv);assert(p[0]==0x12345678);uint32_t*r=a[1].buf.pv;r[0]=0;r[1]=p[1];r[2]=p[2];return 0;}
 assert(method==1);assert(a[0].buf.nLen==a[1].buf.nLen);
 memcpy(device_address(a[1].buf.pv),device_address(a[0].buf.pv),a[0].buf.nLen);return 0;
}
int main(void){
 uint32_t plan[]={0x12345678,1024,1024};int error;uint8_t input[1024],output[1024];
 for(unsigned cycle=0;cycle<100;cycle++){
  Session*s=ug_open_mode(plan,sizeof(plan),0,&error);assert(s&&error==0);
  for(unsigned i=0;i<sizeof(input);i++)input[i]=(uint8_t)(cycle+i);
  memset(output,0,sizeof(output));assert(!ug_infer(s,input,sizeof(input),output,sizeof(output)));assert(!memcmp(input,output,sizeof(input)));
  unsigned before=invocations;fail_sync=1;assert(ug_infer(s,input,sizeof(input),output,sizeof(output))==1004);assert(invocations==before);
  ug_close(s);assert(allocations==3&&frees==0);
 }
 plan[1]=plan[2]=70000;
 uint8_t*large_in=malloc(70000),*large_out=malloc(70000);assert(large_in&&large_out);memset(large_in,0x9b,70000);
 for(unsigned cycle=0;cycle<2;cycle++){
  Session*s=ug_open_mode(plan,sizeof(plan),0,&error);assert(s&&error==0);
  assert(!ug_infer(s,large_in,70000,large_out,70000));assert(!memcmp(large_in,large_out,70000));ug_close(s);
  assert(allocations==5&&frees==2);
 }
 free(large_in);free(large_out);release_transport();assert(frees==5);
 puts("PASS blocking DMA synchronization, bounded buffer reuse, and pre-submit error propagation");return 0;
}
