.section .text
.globl input_projection


input_projection:
    li      t0, 4096

.ip_timestep_loop:

    flw     ft0, 0(a0)
    mv      t1, a1
    mv      t2, a2
    li      t3, 64

.ip_vloop:
 
    vsetvli t4, t3, e32, m8, ta, ma
    
    vle32.v v8, (t1)
    vle32.v v16, (t2)
    vfmul.vf v8, v8, ft0
    vfadd.vv v8, v8, v16
    vse32.v v8, (a3)

    slli    t5, t4, 2
    add     t1, t1, t5
    add     t2, t2, t5
    add     a3, a3, t5
    sub     t3, t3, t4
    bnez    t3, .ip_vloop

    addi    a0, a0, 4
    addi    t0, t0, -1
    bnez    t0, .ip_timestep_loop

    ret
