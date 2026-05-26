.section .text
.globl softmax_forward
softmax_forward:
    addi    sp, sp, -32
    sw      ra,   0(sp)
    sw      s0,   4(sp)
    sw      s1,   8(sp)
    sw      s2,  12(sp)
    sw      s3,  16(sp)
    fsw     fs0, 20(sp)
    fsw     fs1, 24(sp)

    mv      s0, a2
    mv      s1, a0
    mv      s2, a1

    flw     ft0, 0(s1)
    li      t0, 1
.sm_max_loop:
    bge     t0, s0, .sm_max_done
    slli    t1, t0, 2
    add     t1, s1, t1
    flw     ft1, 0(t1)
    flt.s   t2, ft0, ft1
    beq     t2, zero, .sm_max_skip
    fmv.s   ft0, ft1
.sm_max_skip:
    addi    t0, t0, 1
    j       .sm_max_loop

.sm_max_done:
    fmv.s   fs0, ft0

    fmv.w.x fs1, zero
    li      s3, 0

.sm_exp_loop:
    bge     s3, s0, .sm_exp_done
    slli    t1, s3, 2
    add     t3, s1, t1
    flw     ft1, 0(t3)
    fsub.s  fa0, ft1, fs0
    call    exp_f
    slli    t1, s3, 2
    add     t3, s2, t1
    fsw     fa0, 0(t3)
    fadd.s  fs1, fs1, fa0
    addi    s3, s3, 1
    j       .sm_exp_loop

.sm_exp_done:
    li      s3, 0
.sm_div_loop:
    bge     s3, s0, .sm_div_done
    slli    t1, s3, 2
    add     t3, s2, t1
    flw     ft1, 0(t3)
    fdiv.s  ft1, ft1, fs1
    fsw     ft1, 0(t3)
    addi    s3, s3, 1
    j       .sm_div_loop

.sm_div_done:
    lw      ra,   0(sp)
    lw      s0,   4(sp)
    lw      s1,   8(sp)
    lw      s2,  12(sp)
    lw      s3,  16(sp)
    flw     fs0, 20(sp)
    flw     fs1, 24(sp)
    addi    sp, sp, 32
    ret
