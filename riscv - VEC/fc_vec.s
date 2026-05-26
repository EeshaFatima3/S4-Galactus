.section .text
.globl fc_layer

fc_layer:
   
    addi    sp, sp, -16
    sw      ra, 12(sp)

    vsetvli t0, a4, e32, m8, ta, ma

    vle32.v v0, (a0)

    mv      t1, a5             
    mv      t2, a1             
    mv      t3, a2             
    mv      t4, a3             

.fc_loop:
 
     vle32.v v8, (t2)           
    vfmul.vv v16, v8, v0       

    fmv.w.x ft2, zero          
    vfmv.s.f v24, ft2          
    vfredosum.vs v24, v16, v24 
    vfmv.f.s ft0, v24        


    flw     ft1, 0(t3)         
    fadd.s  ft0, ft0, ft1      

    fsw     ft0, 0(t4)

    addi    t2, t2, 256        
    addi    t3, t3, 4          
    addi    t4, t4, 4          

    addi    t1, t1, -1
    bgtz    t1, .fc_loop

   
    lw      ra, 12(sp)
    addi    sp, sp, 16
    ret
