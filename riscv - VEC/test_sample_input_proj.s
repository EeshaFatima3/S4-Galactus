.section .data
.align 4
buffer_A:       .space 1048576
buffer_B:       .space 1048576

.section .text
.globl _start

_start:
    li      sp, 0x81000000

    la      a0, test_image
    la      a1, buffer_A
    la      a2, hilbert_idx
    call    hilbert_scan_layer

    la      a0, buffer_A
    la      a1, up_w
    la      a2, up_b
    la      a3, buffer_B
    call    input_projection

    la      t0, buffer_B
    flw     fa0, 0(t0)
    flw     fa1, 4(t0)
    flw     fa2, 8(t0)
    flw     fa3, 12(t0)

    li      a7, 10
    ecall
