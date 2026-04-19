.section .text
.globl _start

_start:
    lui     s11, 0x40000

.L_poll_loop:
    lw      t0, 0(s11)
    li      t1, 1
    bne     t0, t1, .L_poll_loop

    addi    a0, s11, 0x1000
    call    model_forward

    sw      a0, 4(s11)

    la      t0, probs
    flw     fa0, 0(t0)
    flw     fa1, 4(t0)
    flw     fa2, 8(t0)
    flw     fa3, 12(t0)

    fsw     fa0, 8(s11)
    fsw     fa1, 12(s11)
    fsw     fa2, 16(s11)
    fsw     fa3, 20(s11)

    li      t2, 2
    sw      t2, 0(s11)

    j       .L_poll_loop
