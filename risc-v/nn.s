.section .data

.globl post_hilbert
post_hilbert:   .space 16384

.globl buf_a
buf_a:          .space 1048576

.globl buf_b
buf_b:          .space 1048576

.globl s4_kernel
s4_kernel:      .space 1048576

.globl post_pool
post_pool:      .space 256

.globl logits
logits:         .space 16

.globl probs
probs:          .space 16

.section .text

.globl hilbert_scan_layer
hilbert_scan_layer:
    li      t0, 0
    li      t1, 4096
    la      t2, hilbert_idx
    mv      a2, a1

.hs_loop:
    bge     t0, t1, .hs_done

    lw      t3, 0(t2)
    slli    t4, t3, 2
    add     t5, a0, t4
    flw     ft0, 0(t5)
    fsw     ft0, 0(a2)

    addi    t2, t2, 4
    addi    a2, a2, 4
    addi    t0, t0, 1
    j       .hs_loop

.hs_done:
    ret

.globl input_projection
input_projection:
    li      t0, 0
    li      t1, 4096
    mv      a2, a1

.ip_outer:
    bge     t0, t1, .ip_done

    flw     ft0, 0(a0)

    la      t2, up_w
    la      t3, up_b
    li      t4, 0
    li      t5, 64

.ip_inner:
    bge     t4, t5, .ip_inner_done

    flw     ft1, 0(t2)
    flw     ft2, 0(t3)
    fmul.s  ft3, ft0, ft1
    fadd.s  ft3, ft3, ft2
    fsw     ft3, 0(a2)

    addi    t2, t2, 4
    addi    t3, t3, 4
    addi    a2, a2, 4
    addi    t4, t4, 1
    j       .ip_inner

.ip_inner_done:
    addi    a0, a0, 4
    addi    t0, t0, 1
    j       .ip_outer

.ip_done:
    ret

.globl s4d_layer
s4d_layer:
    addi    sp, sp, -100
    sw      ra,  0(sp)
    sw      s0,  4(sp)
    sw      s1,  8(sp)
    sw      s2,  12(sp)
    sw      s3,  16(sp)
    sw      s4,  20(sp)
    sw      s5,  24(sp)
    sw      s6,  28(sp)
    sw      s7,  32(sp)
    sw      s8,  36(sp)
    sw      s9,  40(sp)
    sw      s10, 44(sp)
    sw      s11, 48(sp)
    fsw     fs0, 52(sp)
    fsw     fs1, 56(sp)
    fsw     fs2, 60(sp)
    fsw     fs3, 64(sp)
    fsw     fs4, 68(sp)
    fsw     fs5, 72(sp)
    fsw     fs6, 76(sp)
    fsw     fs7, 80(sp)
    fsw     fs8, 84(sp)
    fsw     fs9, 88(sp)
    fsw     fs10,92(sp)
    fsw     fs11,96(sp)

    mv      s3, a0
    mv      s4, a1
    mv      s5, a2
    mv      s6, a3
    mv      s7, a4
    mv      s8, a5
    mv      s9, a6
    mv      s10,a7

    li      s0, 0

.s4_h_loop:
    li      t0, 64
    bge     s0, t0, .s4_h_done

    slli    t0, s0, 2
    add     t0, s3, t0
    flw     fa0, 0(t0)
    call    exp_f
    fmv.s   fs0, fa0

    li      s1, 0

.s4_n_loop:
    li      t0, 32
    bge     s1, t0, .s4_n_done

    li      t0, 32
    mul     t0, s0, t0
    add     t0, t0, s1
    slli    t0, t0, 2

    add     t1, s4, t0
    flw     fa0, 0(t1)
    call    exp_f
    fneg.s  fa0, fa0
    fmv.s   fs11, fa0

    add     t1, s5, t0
    flw     fs7, 0(t1)

    fmul.s  fs1, fs11, fs0
    fmul.s  fs2, fs7, fs0

    fmv.s   fa0, fs1
    call    exp_f
    fmv.s   fs5, fa0

    fmv.s   fa0, fs2
    call    cos_f
    fmv.s   fs6, fa0

    fmv.s   fa0, fs2
    call    sin_f
    fmv.s   fs7, fa0

    fmul.s  ft3, fs5, fs6
    fmul.s  ft4, fs5, fs7

    li      t1, 0x3F800000
    fmv.w.x ft5, t1
    fsub.s  ft3, ft3, ft5

    add     t1, s6, t0
    flw     ft6, 0(t1)
    add     t1, s7, t0
    flw     ft7, 0(t1)

    # FIX: denominator uses raw -exp(logAre) and raw Aim (no /dt)
    # fs11 = -exp(logAre) raw, still valid from earlier
    # reload raw Aim from memory (fs7 was overwritten by sin_f)
    add     t1, s5, t0
    flw     ft1, 0(t1)          # ft1 = raw Aim
    fmul.s  ft2, fs11, fs11     # exp(logAre)^2
    fmul.s  ft5, ft1, ft1       # Aim^2
    fadd.s  ft2, ft2, ft5       # denominator = exp(logAre)^2 + Aim^2
    # numerator uses fs1=a_r, fs2=a_i (dt-scaled, correct)
    fmv.s   ft0, fs1
    fmv.s   ft1, fs2

    fmul.s  fs8, ft6, ft3
    fmul.s  fs9, ft7, ft4
    fsub.s  fs8, fs8, fs9

    fmul.s  fs9, ft6, ft4
    fmul.s  fs10, ft7, ft3
    fadd.s  fs9, fs9, fs10

    fmul.s  ft3, fs8, ft0
    fmul.s  ft4, fs9, ft1
    fadd.s  ft3, ft3, ft4
    fdiv.s  fs3, ft3, ft2

    fmul.s  ft3, fs9, ft0
    fmul.s  ft4, fs8, ft1
    fsub.s  ft3, ft3, ft4
    fdiv.s  fs4, ft3, ft2

    li      t1, 0x3F800000
    fmv.w.x fs8, t1
    fmv.w.x fs9, t1
    fmv.w.x fs10, x0

    la      t2, s4_kernel
    li      t3, 4096
    mul     t3, s0, t3
    slli    t3, t3, 2
    add     t2, t2, t3

    li      s2, 0

.s4_t_loop:
    li      t3, 4096
    bge     s2, t3, .s4_t_done

    fmul.s  ft5, fs8, fs9
    fmul.s  ft6, fs8, fs10

    fmul.s  ft7, fs3, ft5
    fmul.s  ft0, fs4, ft6
    fsub.s  ft7, ft7, ft0

    flw     ft0, 0(t2)
    fadd.s  ft0, ft0, ft7
    fsw     ft0, 0(t2)

    fmul.s  fs8, fs8, fs5

    fmul.s  ft0, fs9, fs6
    fmul.s  ft1, fs10, fs7
    fsub.s  fs11, ft0, ft1

    fmul.s  ft0, fs9, fs7
    fmul.s  ft1, fs10, fs6
    fadd.s  fs10, ft0, ft1
    fmv.s   fs9, fs11

    addi    t2, t2, 4
    addi    s2, s2, 1
    j       .s4_t_loop

.s4_t_done:
    addi    s1, s1, 1
    j       .s4_n_loop

.s4_n_done:
    la      t2, s4_kernel
    li      t3, 4096
    mul     t3, s0, t3
    slli    t3, t3, 2
    add     t2, t2, t3

    li      t3, 4096
    li      t4, 0
    li      t5, 0x40000000
    fmv.w.x ft1, t5

.s4_scale_loop:
    bge     t4, t3, .s4_scale_done
    flw     ft0, 0(t2)
    fmul.s  ft0, ft0, ft1
    fsw     ft0, 0(t2)
    addi    t2, t2, 4
    addi    t4, t4, 1
    j       .s4_scale_loop

.s4_scale_done:
    addi    s0, s0, 1
    j       .s4_h_loop

.s4_h_done:
    li      s0, 0

.s4_conv_h:
    li      t0, 64
    bge     s0, t0, .s4_conv_done

    slli    t0, s0, 2
    add     t0, s8, t0
    flw     fs5, 0(t0)

    la      t2, s4_kernel
    li      t3, 4096
    mul     t3, s0, t3
    slli    t3, t3, 2
    add     t2, t2, t3

    li      s2, 0

.s4_conv_t:
    li      t0, 4096
    bge     s2, t0, .s4_conv_t_done

    fmv.w.x ft0, x0
    li      t3, 0
    mv      t4, t2

    li      t5, 64
    mul     t5, s2, t5
    add     t5, t5, s0
    slli    t5, t5, 2
    add     t5, s9, t5

.s4_inner_s:
    bgt     t3, s2, .s4_inner_done
    flw     ft1, 0(t4)
    flw     ft2, 0(t5)
    fmul.s  ft3, ft1, ft2
    fadd.s  ft0, ft0, ft3
    addi    t4, t4, 4
    addi    t5, t5, -256
    addi    t3, t3, 1
    j       .s4_inner_s

.s4_inner_done:
    li      t5, 64
    mul     t5, s2, t5
    add     t5, t5, s0
    slli    t5, t5, 2
    add     t5, s9, t5
    flw     ft1, 0(t5)
    fmul.s  ft2, fs5, ft1
    fadd.s  ft0, ft0, ft2

    li      t5, 64
    mul     t5, s2, t5
    add     t5, t5, s0
    slli    t5, t5, 2
    add     t5, s10, t5
    fsw     ft0, 0(t5)

    addi    s2, s2, 1
    j       .s4_conv_t

.s4_conv_t_done:
    addi    s0, s0, 1
    j       .s4_conv_h

.s4_conv_done:
    lw      ra,  0(sp)
    lw      s0,  4(sp)
    lw      s1,  8(sp)
    lw      s2,  12(sp)
    lw      s3,  16(sp)
    lw      s4,  20(sp)
    lw      s5,  24(sp)
    lw      s6,  28(sp)
    lw      s7,  32(sp)
    lw      s8,  36(sp)
    lw      s9,  40(sp)
    lw      s10, 44(sp)
    lw      s11, 48(sp)
    flw     fs0, 52(sp)
    flw     fs1, 56(sp)
    flw     fs2, 60(sp)
    flw     fs3, 64(sp)
    flw     fs4, 68(sp)
    flw     fs5, 72(sp)
    flw     fs6, 76(sp)
    flw     fs7, 80(sp)
    flw     fs8, 84(sp)
    flw     fs9, 88(sp)
    flw     fs10,92(sp)
    flw     fs11,96(sp)
    addi    sp, sp, 100
    ret

.section .data
.align 2
gelu_sqrt2pi:   .float 0.7978845608
gelu_coeff:     .float 0.044715
gelu_half:      .float 0.5
gelu_one:       .float 1.0

.section .text
.globl gelu_inplace
gelu_inplace:
    addi    sp, sp, -32
    sw      ra,   0(sp)
    sw      s0,   4(sp)
    sw      s1,   8(sp)
    fsw     fs0, 12(sp)
    fsw     fs1, 16(sp)
    fsw     fs2, 20(sp)
    fsw     fs3, 24(sp)
    fsw     fs4, 28(sp)

    mv      s0, a0
    li      s1, 0
    li      t1, 262144

    la      t2, gelu_sqrt2pi
    flw     fs0, 0(t2)
    la      t2, gelu_coeff
    flw     fs1, 0(t2)
    la      t2, gelu_half
    flw     fs2, 0(t2)
    la      t2, gelu_one
    flw     fs3, 0(t2)

.gelu_loop:
    bge     s1, t1, .gelu_done

    flw     ft0, 0(s0)

    fmul.s  ft1, ft0, ft0
    fmul.s  ft2, ft1, ft0

    fmul.s  ft3, fs1, ft2
    fadd.s  ft3, ft0, ft3
    fmul.s  ft3, fs0, ft3

    fmv.s   fs4, ft0

    fmv.s   fa0, ft3
    call    tanh_f

    li      t1, 262144

    fadd.s  ft4, fs3, fa0
    fmul.s  ft5, fs2, fs4
    fmul.s  ft5, ft5, ft4

    fsw     ft5, 0(s0)

    addi    s0, s0, 4
    addi    s1, s1, 1
    j       .gelu_loop

.gelu_done:
    lw      ra,   0(sp)
    lw      s0,   4(sp)
    lw      s1,   8(sp)
    flw     fs0, 12(sp)
    flw     fs1, 16(sp)
    flw     fs2, 20(sp)
    flw     fs3, 24(sp)
    flw     fs4, 28(sp)
    addi    sp, sp, 32
    ret

.globl take_last
take_last:
     li      t0, 4095
    li      t1, 64
    mul     t0, t0, t1          # 4095 * 64
    slli    t0, t0, 2           # * 4 bytes
    add     t0, a0, t0          # t0 = &buf[4095][0]
    
    li      t1, 0               # h = 0
    li      t2, 64
.tl_loop:
    bge     t1, t2, .tl_done
    flw     ft0, 0(t0)
    fsw     ft0, 0(a1)
    addi    t0, t0, 4
    addi    a1, a1, 4
    addi    t1, t1, 1
    j       .tl_loop
.tl_done:
    ret


.globl fc_forward
fc_forward:
    la      t2, fc_w
    la      t3, fc_b
    li      t0, 0
    li      t1, 4
    mv      a2, a1

.fc_c_loop:
    bge     t0, t1, .fc_done

    flw     ft0, 0(t3)
    addi    t3, t3, 4

    li      t4, 0
    li      t5, 64
    mv      t6, a0

.fc_h_loop:
    bge     t4, t5, .fc_h_done
    flw     ft1, 0(t2)
    flw     ft2, 0(t6)
    fmul.s  ft3, ft1, ft2
    fadd.s  ft0, ft0, ft3
    addi    t2, t2, 4
    addi    t6, t6, 4
    addi    t4, t4, 1
    j       .fc_h_loop

.fc_h_done:
    fsw     ft0, 0(a2)
    addi    a2, a2, 4
    addi    t0, t0, 1
    j       .fc_c_loop

.fc_done:
    ret

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

.globl model_forward
model_forward:
    addi    sp, sp, -4
    sw      ra, 0(sp)

    la      a1, post_hilbert
    call    hilbert_scan_layer

    la      a0, post_hilbert
    la      a1, buf_a
    call    input_projection

    la      a0, s4_0_log_dt
    la      a1, s4_0_logAre
    la      a2, s4_0_Aim
    la      a3, s4_0_Cre
    la      a4, s4_0_Cim
    la      a5, s4_0_D
    la      a6, buf_a
    la      a7, buf_b

    la      t0, s4_kernel
    li      t1, 262144
    li      t2, 0
    
.zero_k1:
    bge     t2, t1, .zero_k1_done
    sw      zero, 0(t0)
    addi    t0, t0, 4
    addi    t2, t2, 1
    j       .zero_k1
.zero_k1_done:
    call    s4d_layer

    la      a0, buf_b
    call    gelu_inplace

    la      a0, s4_1_log_dt
    la      a1, s4_1_logAre
    la      a2, s4_1_Aim
    la      a3, s4_1_Cre
    la      a4, s4_1_Cim
    la      a5, s4_1_D
    la      a6, buf_b
    la      a7, buf_a

    la      t0, s4_kernel
    li      t1, 262144
    li      t2, 0
.zero_k2:
    bge     t2, t1, .zero_k2_done
    sw      zero, 0(t0)
    addi    t0, t0, 4
    addi    t2, t2, 1
    j       .zero_k2
.zero_k2_done:
    call    s4d_layer

    la      a0, buf_a
    call    gelu_inplace

    la      a0, buf_a
    la      a1, post_pool
    call    take_last

    la      a0, post_pool
    la      a1, logits
    call    fc_forward

    la      a0, logits
    la      a1, probs
    li      a2, 4
    call    softmax_forward

    la      t0, probs
    flw     ft0, 0(t0)
    li      a0, 0
    li      t1, 1

.argmax_loop:
    li      t2, 4
    bge     t1, t2, .argmax_done
    slli    t3, t1, 2
    add     t3, t0, t3
    flw     ft1, 0(t3)
    fle.s   t4, ft1, ft0
    bne     t4, zero, .argmax_skip
    fmv.s   ft0, ft1
    mv      a0, t1
.argmax_skip:
    addi    t1, t1, 1
    j       .argmax_loop

.argmax_done:
    lw      ra, 0(sp)
    addi    sp, sp, 4
    ret