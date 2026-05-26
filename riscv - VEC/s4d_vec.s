.section .text
.globl s4d_layer

s4d_layer:
    addi    sp, sp, -32
    sw      ra, 28(sp)
    sw      s0, 24(sp)
    sw      s1, 20(sp)
    sw      s2, 16(sp)
    sw      s3, 12(sp)

    li      t0, 64
    vsetvli t0, t0, e32, m4, ta, ma
    
    li      t4, 64              
    mv      s2, a0              
    mv      s3, a1             

.channel_loop:
 
    vle32.v v8, (a2)            
    vle32.v v12, (a3)          
    vle32.v v16, (a6)          
    vle32.v v20, (a7)          
    

    vmv.v.i v0, 0               
    vmv.v.i v4, 0

    mv      s0, s2
    mv      s1, s3
    li      t1, 4096            

.time_loop:
    flw     ft0, 0(s0)         


    vfmul.vv v24, v0, v8        
    vfmul.vv v28, v4, v12       
    vfsub.vv v24, v24, v28      
    vle32.v  v28, (a4)          
    vfmacc.vf v24, ft0, v28     

    
    vfmul.vv v28, v4, v8        
    vfmacc.vv v28, v0, v12      
    vmv.v.v  v0, v24            
    vle32.v  v24, (a5)          
    vfmacc.vf v28, ft0, v24     
    vmv.v.v  v4, v28            

   
    vfmul.vv v24, v0, v16       
    vfmul.vv v28, v4, v20       
    vfsub.vv v24, v24, v28      


    fmv.w.x ft2, zero           
    vfmv.s.f v28, ft2           
    vfredosum.vs v28, v24, v28  
    vfmv.f.s ft1, v28           
    fadd.s  ft1, ft1, ft1       
    fsw     ft1, 0(s1)          

 
    addi    s0, s0, 256
    addi    s1, s1, 256
    addi    t1, t1, -1
    bgtz    t1, .time_loop

 
    addi    a2, a2, 256         
    addi    a3, a3, 256         
    addi    a6, a6, 256         
    addi    a7, a7, 256       

    addi    s2, s2, 4           
    addi    s3, s3, 4           
    addi    t4, t4, -1
    bgtz    t4, .channel_loop

  
    lw      ra, 28(sp)
    lw      s0, 24(sp)
    lw      s1, 20(sp)
    lw      s2, 16(sp)
    lw      s3, 12(sp)
    addi    sp, sp, 32
    ret
