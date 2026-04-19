.section .data
.align 2

gelu_test_input:
    .float -0.4884448051, 0.0439040959, 0.7398561239, -0.4469782114, -0.3048147559, 0.5110594034, -1.3302786350, -0.1775891632

.section .text
.globl _start
_start:
    la      t0, buf_a
    li      t1, 262144
    li      t2, 0
.zero_loop:
    bge     t2, t1, .zero_done
    sw      zero, 0(t0)
    addi    t0, t0, 4
    addi    t2, t2, 1
    j       .zero_loop
.zero_done:

    la      t0, gelu_test_input
    la      t1, buf_a
    li      t2, 0
    li      t3, 8
.copy_loop:
    bge     t2, t3, .copy_done
    flw     ft0, 0(t0)
    fsw     ft0, 0(t1)
    addi    t0, t0, 4
    addi    t1, t1, 4
    addi    t2, t2, 1
    j       .copy_loop
.copy_done:

    la      a0, buf_a
    call    gelu_inplace

    la      t0, buf_a
    flw     fa0, 0(t0)
    flw     fa1, 4(t0)
    flw     fa2, 8(t0)
    flw     fa3, 12(t0)
    flw     fa4, 16(t0)
    flw     fa5, 20(t0)
    flw     fa6, 24(t0)
    flw     fa7, 28(t0)

    unimp
