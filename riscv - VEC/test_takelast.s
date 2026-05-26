
.section .text
.globl _start
_start:
    la      t0, buf_a
    li      t1, 0                # t = 0 (row)
    li      t2, 4096

.tl_fill_outer:
    bge     t1, t2, .tl_fill_done
    fcvt.s.w ft0, t1             # ft0 = (float)t
    li      t3, 0                # h = 0
    li      t4, 64

.tl_fill_inner:
    bge     t3, t4, .tl_fill_inner_done
    fsw     ft0, 0(t0)
    addi    t0, t0, 4
    addi    t3, t3, 1
    j       .tl_fill_inner

.tl_fill_inner_done:
    addi    t1, t1, 1
    j       .tl_fill_outer

.tl_fill_done:
    la      a0, buf_a
    la      a1, post_pool
    call    take_last

    la      t0, post_pool
    flw     fa0, 0(t0)           
    flw     fa1, 4(t0)           
    flw     fa2, 8(t0)           
    flw     fa3, 12(t0)          
    flw     fa4, 16(t0)
    flw     fa5, 20(t0)
    flw     fa6, 24(t0)
    flw     fa7, 28(t0)

    unimp
