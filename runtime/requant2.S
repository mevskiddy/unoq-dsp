/* Exact affine requantization, two adjacent channels and a software-pipelined
 * pixel loop. Inputs are validated by requantize(): rows >= 3, even channels,
 * aligned accumulators and 16-byte affine records. No speculative tail loads.
 * ABI: acc, output, rows, channels, quantization, affine table. */
.text
.globl uq_requant2_asm
.hidden uq_requant2_asm
.type uq_requant2_asm,@function
.p2align 6
uq_requant2_asm:
 { allocframe(#48) }
 { memd(r29+#0)=r17:16; memd(r29+#8)=r19:18 }
 { memd(r29+#16)=r21:20; memd(r29+#24)=r23:22 }
 { memd(r29+#32)=r25:24; memd(r29+#40)=r27:26 }
 { r8=memw(r4+#56); r10=memw(r4+#60); r28=asl(r3,#2); r6=lsr(r3,#1) }
 { m0=r28; r4=add(r2,#-2) }
 { loop1(.Lrq_channel,r6) }
 { r9=r8; r11=r10 }
.p2align 4
.Lrq_channel:
 { r20=r0; r21=r1; r13:12=memd(r5+#0); r15:14=memd(r5+#16) }
 { r17:16=memd(r5+#8); r19:18=memd(r5+#24) }
 { r27:26=memd(r20++m0) }
 { r23:22=combine(r13,r12); r25:24=combine(r15,r14) }
 { r17=add(r17,#-32); r19=add(r19,#-32) }
 { r23:22+=mpy(r26,r16); r25:24+=mpy(r27,r18); r27:26=memd(r20++m0) }
 { r6=asr(r23,r17); r7=asr(r25,r19) }
 { loop0(.Lrq_pixels,r4) }
.p2align 4
.Lrq_pixels:
 { r7:6=vminw(r7:6,r11:10); r23:22=combine(r13,r12); r25:24=combine(r15,r14) }
 { r7:6=vmaxw(r7:6,r9:8) }
 { r23:22+=mpy(r26,r16); r25:24+=mpy(r27,r18); memb(r21+#0)=r6; memb(r21+#1)=r7 }
 { r6=asr(r23,r17); r7=asr(r25,r19); r27:26=memd(r20++m0); r21=add(r21,r3) } :endloop0
 /* Compute the last row; the last legal input pair is already loaded. */
 { r7:6=vminw(r7:6,r11:10); r23:22=combine(r13,r12); r25:24=combine(r15,r14) }
 { r7:6=vmaxw(r7:6,r9:8) }
 { r23:22+=mpy(r26,r16); r25:24+=mpy(r27,r18); memb(r21+#0)=r6; memb(r21+#1)=r7 }
 { r6=asr(r23,r17); r7=asr(r25,r19); r21=add(r21,r3) }
 { r7:6=vminw(r7:6,r11:10) }
 { r7:6=vmaxw(r7:6,r9:8) }
 { memb(r21+#0)=r6; memb(r21+#1)=r7; r5=add(r5,#32); r0=add(r0,#8) }
 { r1=add(r1,#2) } :endloop1
 { r17:16=memd(r29+#0); r19:18=memd(r29+#8) }
 { r21:20=memd(r29+#16); r23:22=memd(r29+#24) }
 { r25:24=memd(r29+#32); r27:26=memd(r29+#40) }
 { dealloc_return }
.size uq_requant2_asm,.-uq_requant2_asm
