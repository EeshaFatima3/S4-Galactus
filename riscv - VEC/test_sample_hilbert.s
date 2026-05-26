.section .data
.align 4
buffer_A:       .space 1048576

.section .text
.globl _start

_start:
    li      sp, 0x81000000

    la      a0, test_image
    la      a1, buffer_A
    la      a2, hilbert_idx
    call    hilbert_scan_layer

    la      t0, buffer_A
    flw     fa0, 0(t0)
    flw     fa1, 4(t0)
    flw     fa2, 8(t0)
    flw     fa3, 12(t0)
    flw     fa4, 16(t0)
    flw     fa5, 20(t0)
    flw     fa6, 24(t0)
    flw     fa7, 28(t0)

    li      a7, 10
    ecall
