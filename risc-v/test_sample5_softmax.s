.section .data
.align 2

sm_test_input:
    .float 3.1594207287
    .float -2.5765304565
    .float -2.4812905788
    .float 1.4903841019

.section .text
.globl _start
_start:
    la      t0, sm_test_input
    la      t1, logits
    flw     ft0, 0(t0)
    flw     ft1, 4(t0)
    flw     ft2, 8(t0)
    flw     ft3, 12(t0)
    fsw     ft0, 0(t1)
    fsw     ft1, 4(t1)
    fsw     ft2, 8(t1)
    fsw     ft3, 12(t1)

    la      a0, logits
    la      a1, probs
    li      a2, 4
    call    softmax_forward

    la      t0, probs
    flw     fa0, 0(t0)
    flw     fa1, 4(t0)
    flw     fa2, 8(t0)
    flw     fa3, 12(t0)

    unimp
