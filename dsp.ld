/* QuRT-compatible layout: two page-aligned LOAD segments and DYNAMIC.
 * The default LLD layout failed on this firmware even after signing.
 */
PHDRS { text PT_LOAD FILEHDR PHDRS FLAGS(5); data PT_LOAD FLAGS(6); dynamic PT_DYNAMIC FLAGS(6); }
SECTIONS {
 . = SIZEOF_HEADERS;
 .dynsym : { *(.dynsym) } :text
 .dynstr : { *(.dynstr) } :text
 .hash : { *(.hash) } :text
 .rela.dyn : { *(.rela.dyn) } :text
 .rela.plt : { *(.rela.plt) } :text
 .text : { *(.text*) } :text
 .rodata : { *(.rodata*) } :text
 . = ALIGN(4096);
 .dynamic : { *(.dynamic) } :data :dynamic
 .data : { *(.data*) } :data
 .got : { *(.got*) } :data
 .bss : { *(.bss*) *(COMMON) } :data
 /DISCARD/ : { *(.comment) *(.note*) }
}
