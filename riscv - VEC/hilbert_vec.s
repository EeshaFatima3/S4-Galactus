.section .text
.globl hilbert_scan_layer

hilbert_scan_layer:
    li      t1, 4096

.hs_vloop:
    
    vsetvli t0, t1, e32, m8, ta, ma
    
    
    vle32.v v8, (a2)
    vsll.vi v8, v8, 2
    vluxei32.v v16, (a0), v8
    vse32.v v16, (a1)


    slli    t2, t0, 2
    add     a2, a2, t2
    add     a1, a1, t2
    sub     t1, t1, t0
    bnez    t1, .hs_vloop

    ret
