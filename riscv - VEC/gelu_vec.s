.section .data
.align 2
const_0_797885:  .float 0.797885
const_0_044715:  .float 0.044715
const_0_5:       .float 0.5
const_1_0:       .float 1.0
const_n0_333:    .float -0.333333
const_0_133:     .float 0.133333

.section .text
.globl gelu_layer

gelu_layer:

.gelu_loop:
    beqz    a2, .gelu_end
    
    vsetvli t0, a2, e32, m1, ta, ma
  

    vle32.v v8, (a0)

    vfmul.vv v10, v8, v8            
    vfmul.vv v12, v10, v8           
    
    la      t1, const_0_044715
    flw     ft0, 0(t1)
    vfmul.vf v14, v12, ft0          
    vfadd.vv v14, v8, v14           
    
    la      t1, const_0_797885
    flw     ft0, 0(t1)
    vfmul.vf v14, v14, ft0         

  
    vfmul.vv v16, v14, v14          
    vfmul.vv v18, v16, v14          
    vfmul.vv v20, v18, v16          

    la      t1, const_n0_333
    flw     ft0, 0(t1)
    vfmul.vf v22, v18, ft0          # -0.333 * y^3

    la      t1, const_0_133
    flw     ft0, 0(t1)
    vfmul.vf v24, v20, ft0          # 0.133 * y^5

    vfadd.vv v26, v14, v22          # y - 0.333*y^3
    vfadd.vv v26, v26, v24          


    la      t1, const_1_0
    flw     ft0, 0(t1)
    vfmin.vf v26, v26, ft0          
    fneg.s  ft1, ft0                
    vfmax.vf v26, v26, ft1          

    #GELU = 0.5 * x * (1 + tanh(y))
    vfadd.vf v26, v26, ft0          
    la      t1, const_0_5
    flw     ft0, 0(t1)
    vfmul.vf v26, v26, ft0          
    vfmul.vv v28, v8, v26           

    
    vse32.v v28, (a1)

    
    slli    t1, t0, 2
    add     a0, a0, t1
    add     a1, a1, t1
    sub     a2, a2, t0
    j       .gelu_loop

.gelu_end:
    ret
