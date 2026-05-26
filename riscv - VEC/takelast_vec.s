.section .text
.globl take_last_timestep


take_last_timestep:
  
    addi    t0, a2, -1      

    slli    t0, t0, 8       

    add     t1, a0, t0      

    vsetvli t2, a3, e32, m8, ta, ma

    vle32.v v8, (t1)      
    vse32.v v8, (a1)       

    ret
