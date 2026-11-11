/* Hexagon v66, unsigned INT8 activations and signed INT8 packed weights.
 * Arguments: x0, x1, weights, y0, y1, K. K >= 16, divisible by 8.
 * Caller currently dispatches only K >= 64. No read beyond the K-byte rows
 * or 4*K weight block: n-1 pipelined iterations followed by a drain.
 * r0..r15: eight two-lane accumulators; r16..r19: inputs;
 * r20..r23: streamed weights; r24/r25: input pointers; r28: weight pointer;
 * r26..r27: next x0, then output pointers. All callee-saved registers restored.
 * Five-packet steady-state loop; dependent loads are separated from use.
 * Packet reads see old register values, allowing safe streamed replacements.
 */
.text
.globl ug_dsp_gemm2x4
.hidden ug_dsp_gemm2x4
.type ug_dsp_gemm2x4,@function
.p2align 6
ug_dsp_gemm2x4:
 { allocframe(#64) }
 { memd(r29+#16) = r17:16; memd(r29+#24) = r19:18 }
 { memd(r29+#32) = r21:20; memd(r29+#40) = r23:22 }
 { memd(r29+#48) = r25:24; memd(r29+#56) = r27:26 }
 { r24 = r0; r25 = r1; r28 = r2; r5 = lsr(r5,#3) }
 { memw(r29+#0) = r3; memw(r29+#4) = r4; r5 = add(r5,#-1) }
 { loop0(.Lgemm2x4,r5) }
 { r17:16 = memd(r24++#8); r19:18 = memd(r25++#8) }
 { r21:20 = memd(r28+#0); r23:22 = memd(r28+#8) }
 { r1:0 = combine(#0,#0); r3:2 = combine(#0,#0) }
 { r5:4 = combine(#0,#0); r7:6 = combine(#0,#0) }
 { r9:8 = combine(#0,#0); r11:10 = combine(#0,#0) }
 { r13:12 = combine(#0,#0); r15:14 = combine(#0,#0) }
.p2align 6
.Lgemm2x4:
 { r1:0 += vrmpybsu(r21:20,r17:16)
   r9:8 += vrmpybsu(r21:20,r19:18)
   r21:20 = memd(r28+#16)
   r27:26 = memd(r24++#8) }
 { r3:2 += vrmpybsu(r23:22,r17:16)
   r11:10 += vrmpybsu(r23:22,r19:18)
   r23:22 = memd(r28+#24) }
 { r5:4 += vrmpybsu(r21:20,r17:16)
   r13:12 += vrmpybsu(r21:20,r19:18)
   r21:20 = memd(r28+#32) }
 { r7:6 += vrmpybsu(r23:22,r17:16)
   r15:14 += vrmpybsu(r23:22,r19:18)
   r19:18 = memd(r25++#8) }
 { r17:16 = combine(r27,r26)
   r23:22 = memd(r28+#40)
   r28 = add(r28,#32) } :endloop0
 /* Last K block: do not prefetch another row/weight block. */
 { r1:0 += vrmpybsu(r21:20,r17:16)
   r9:8 += vrmpybsu(r21:20,r19:18)
   r21:20 = memd(r28+#16) }
 { r3:2 += vrmpybsu(r23:22,r17:16)
   r11:10 += vrmpybsu(r23:22,r19:18)
   r23:22 = memd(r28+#24) }
 { r27:26 = memd(r29+#0) }
 { r5:4 += vrmpybsu(r21:20,r17:16)
   r13:12 += vrmpybsu(r21:20,r19:18) }
 { r7:6 += vrmpybsu(r23:22,r17:16)
   r15:14 += vrmpybsu(r23:22,r19:18) }
 { r0 = add(r0,r1); r2 = add(r2,r3); r4 = add(r4,r5); r6 = add(r6,r7) }
 { r8 = add(r8,r9); r10 = add(r10,r11); r12 = add(r12,r13); r14 = add(r14,r15) }
 { memw(r26+#0) = r0; memw(r26+#4) = r2 }
 { memw(r26+#8) = r4; memw(r26+#12) = r6 }
 { memw(r27+#0) = r8; memw(r27+#4) = r10 }
 { memw(r27+#8) = r12; memw(r27+#12) = r14 }
 { r17:16 = memd(r29+#16); r19:18 = memd(r29+#24) }
 { r21:20 = memd(r29+#32); r23:22 = memd(r29+#40) }
 { r25:24 = memd(r29+#48); r27:26 = memd(r29+#56) }
 { dealloc_return }
.size ug_dsp_gemm2x4,.-ug_dsp_gemm2x4
