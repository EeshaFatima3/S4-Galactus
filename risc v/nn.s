# nn.s — S4D Galaxy Classifier: Layer Routines
# Milestone 3 — RISC-V 32-bit Assembly
#
# Implements all 9 pipeline stages:
#   1. hilbert_scan_layer
#   2. input_projection
#   3. s4d_layer (called twice)
#   4/6. gelu_inplace
#   7. take_last
#   8. fc_forward
#   9. softmax_forward
#
# All weights are in weights.s (auto-generated).
# All intermediate buffers are declared here in .data.
#
# Calling convention (RV32):
#   Arguments : a0–a7
#   Temporaries: t0–t6  (caller-saved, free to use)
#   Saved regs : s0–s11 (must save/restore if used)
#   Return addr: ra     (save with addi sp,-4 / sw ra,0(sp) if calling sub-routines)
#   Return val : a0

# ================================================================
# .data — Intermediate Buffers
# (Weights live in weights.s, also .data, linked together)
# ================================================================
.section .data

# post_hilbert : float[4096]         — output of hilbert scan
.globl post_hilbert
post_hilbert:   .space 16384

# buf_a : float[4096][64]  (1 MB)   — primary working buffer
.globl buf_a
buf_a:          .space 1048576

# buf_b : float[4096][64]  (1 MB)   — secondary working buffer
.globl buf_b
buf_b:          .space 1048576

# s4_kernel : float[64][4096]  (1 MB) — S4D convolution kernel
.globl s4_kernel
s4_kernel:      .space 1048576

# post_pool : float[64]              — output of TakeLastTimestep
.globl post_pool
post_pool:      .space 256

# logits : float[4]                  — output of FC layer
.globl logits
logits:         .space 16

# probs : float[4]                   — output of Softmax
.globl probs
probs:          .space 16

# ================================================================
# .text — Code
# ================================================================
.section .text

# ================================================================
# 1. hilbert_scan_layer(a0=img*, a1=out*)
#
#   For t in 0..4095:
#     out[t] = img[ hilbert_idx[t] ]
#
#   No floating-point arithmetic — pure indexed copy.
#
#   Registers:
#     t0 = loop counter t
#     t1 = limit (4096)
#     t2 = pointer into hilbert_idx[] (walks +4 each iter)
#     t3 = hilbert_idx[t]  (the index value)
#     t4 = byte offset = t3 * 4
#     t5 = &img[hilbert_idx[t]]
#     a2 = walking pointer into out[] (+4 each iter)
#     ft0= pixel value loaded from img
# ================================================================
.globl hilbert_scan_layer
hilbert_scan_layer:
    li      t0, 0
    li      t1, 4096
    la      t2, hilbert_idx
    mv      a2, a1

.hs_loop:
    bge     t0, t1, .hs_done

    lw      t3, 0(t2)           # t3 = hilbert_idx[t]
    slli    t4, t3, 2           # byte offset = index * 4
    add     t5, a0, t4          # &img[hilbert_idx[t]]
    flw     ft0, 0(t5)          # ft0 = img[hilbert_idx[t]]
    fsw     ft0, 0(a2)          # out[t] = ft0

    addi    t2, t2, 4
    addi    a2, a2, 4
    addi    t0, t0, 1
    j       .hs_loop

.hs_done:
    ret

# ================================================================
# 2. input_projection(a0=in*, a1=out*)
#
#   For t in 0..4095:
#     For h in 0..63:
#       out[t][h] = in[t] * up_w[h] + up_b[h]
#
#   out layout: row-major float[4096][64]
#   in[t] is a scalar — loaded once per outer iteration.
#
#   Registers:
#     t0 = outer counter t
#     t1 = outer limit 4096
#     t4 = inner counter h
#     t5 = inner limit 64
#     t2 = walking ptr into up_w[]
#     t3 = walking ptr into up_b[]
#     a2 = walking ptr into out[]
#     ft0= in[t]  (scalar, constant for inner loop)
#     ft1= up_w[h]
#     ft2= up_b[h]
#     ft3= result
# ================================================================
.globl input_projection
input_projection:
    li      t0, 0
    li      t1, 4096
    mv      a2, a1              # a2 = out*

.ip_outer:
    bge     t0, t1, .ip_done

    flw     ft0, 0(a0)          # ft0 = in[t]

    la      t2, up_w            # reset to start of up_w each outer iter
    la      t3, up_b
    li      t4, 0
    li      t5, 64

.ip_inner:
    bge     t4, t5, .ip_inner_done

    flw     ft1, 0(t2)          # up_w[h]
    flw     ft2, 0(t3)          # up_b[h]
    fmul.s  ft3, ft0, ft1       # in[t] * up_w[h]
    fadd.s  ft3, ft3, ft2       # + up_b[h]
    fsw     ft3, 0(a2)          # out[t][h] = result

    addi    t2, t2, 4
    addi    t3, t3, 4
    addi    a2, a2, 4
    addi    t4, t4, 1
    j       .ip_inner

.ip_inner_done:
    addi    a0, a0, 4           # next in[t]
    addi    t0, t0, 1
    j       .ip_outer

.ip_done:
    ret

# ================================================================
# 3. s4d_layer(a0=log_dt*, a1=logAre*, a2=Aim*,
#              a3=Cre*,    a4=Cim*,    a5=D*,
#              a6=in*,     a7=out*)
#
#   Two phases:
#     Phase A — kernel generation: build s4_kernel[h][t]
#     Phase B — causal convolution + skip: out[t][h] = conv + D*in
#
#   Phase A (per channel h, per mode n, per timestep t):
#     dt       = exp(log_dt[h])
#     A_re     = -exp(logAre[h][n])
#     A_im     = Aim[h][n]
#     dtA_re   = A_re * dt
#     dtA_im   = A_im * dt
#     edtA_re  = exp(dtA_re) * cos(dtA_im)
#     edtA_im  = exp(dtA_re) * sin(dtA_im)
#     num_re   = edtA_re - 1
#     num_im   = edtA_im
#     denom    = A_re^2 + A_im^2
#     Ct_re[n] = (Cre*(edtA_re-1) - Cim*edtA_im)*A_re + (Cre*edtA_im + Cim*(edtA_re-1))*A_im) / denom
#     Ct_im[n] = ...
#     For t: k += 2*Re(Ct[n] * exp(dtA[n]*t))
#
#   Because this needs exp/cos/sin, we call helpers from math.s.
#   Those helpers use a0 (arg) and return in fa0 — so we must
#   save/restore ra and any registers we need across the calls.
#
#   Stack frame layout (grows downward):
#     sp+0  : ra
#     sp+4  : s0  (h loop counter)
#     sp+8  : s1  (n loop counter)
#     sp+12 : s2  (t loop counter)
#     sp+16 : s3  (base: log_dt*)
#     sp+20 : s4  (base: logAre*)
#     sp+24 : s5  (base: Aim*)
#     sp+28 : s6  (base: Cre*)
#     sp+32 : s7  (base: Cim*)
#     sp+36 : s8  (base: D*)
#     sp+40 : s9  (base: in*)
#     sp+44 : s10 (base: out*)
#     sp+48 : fs0 (dt)
#     sp+52 : fs1 (dtA_re[n])
#     sp+56 : fs2 (dtA_im[n])
#     sp+60 : fs3 (Ct_re[n])
#     sp+64 : fs4 (Ct_im[n])
#     sp+68 : fs5 (accumulator k_re)
#     sp+72 : fs6 (scratch)
#     sp+76 : fs7 (scratch)
#     total : 80 bytes
# ================================================================
.globl s4d_layer
s4d_layer:
    # --- Prologue: save callee-saved registers ---
    addi    sp, sp, -80
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
    fsw     fs0, 48(sp)
    fsw     fs1, 52(sp)
    fsw     fs2, 56(sp)
    fsw     fs3, 60(sp)
    fsw     fs4, 64(sp)
    fsw     fs5, 68(sp)
    fsw     fs6, 72(sp)
    fsw     fs7, 76(sp)

    # Save all pointer args into s-registers
    mv      s3, a0          # log_dt*
    mv      s4, a1          # logAre*
    mv      s5, a2          # Aim*
    mv      s6, a3          # Cre*
    mv      s7, a4          # Cim*
    mv      s8, a5          # D*
    mv      s9, a6          # in*
    mv      s10,a7          # out*

    # ---- Phase A: kernel generation ----
    # Outer loop: h = 0..63
    li      s0, 0           # h = 0

.s4_h_loop:
    li      t0, 64
    bge     s0, t0, .s4_h_done

    # dt = exp(log_dt[h])
    # log_dt[h] is at s3 + h*4
    slli    t0, s0, 2
    add     t0, s3, t0
    flw     fa0, 0(t0)          # fa0 = log_dt[h]
    call    exp_f               # fa0 = exp(log_dt[h])
    fmv.s   fs0, fa0            # fs0 = dt

    # Inner loop: n = 0..31
    li      s1, 0           # n = 0

.s4_n_loop:
    li      t0, 32
    bge     s1, t0, .s4_n_done

    # offset into [h][n] arrays = (h*32 + n) * 4
    li      t0, 32
    mul     t0, s0, t0
    add     t0, t0, s1
    slli    t0, t0, 2           # byte offset

    # A_re = -exp(logAre[h][n])
    add     t1, s4, t0
    flw     fa0, 0(t1)
    call    exp_f
    fneg.s  fa0, fa0            # fa0 = -exp(logAre[h][n]) = A_re

    # A_im = Aim[h][n]
    add     t1, s5, t0
    flw     ft1, 0(t1)          # ft1 = A_im

    # dtA_re = A_re * dt,  dtA_im = A_im * dt
    fmul.s  fs1, fa0, fs0       # fs1 = dtA_re
    fmul.s  fs2, ft1, fs0       # fs2 = dtA_im

    # exp(dtA_re)
    fmv.s   fa0, fs1
    call    exp_f               # fa0 = exp(dtA_re)
    fmv.s   ft2, fa0            # ft2 = e_mag

    # cos(dtA_im)
    fmv.s   fa0, fs2
    call    cos_f               # fa0 = cos(dtA_im)
    fmul.s  ft3, ft2, fa0       # ft3 = edtA_re = e_mag * cos

    # sin(dtA_im)
    fmv.s   fa0, fs2
    call    sin_f               # fa0 = sin(dtA_im)
    fmul.s  ft4, ft2, fa0       # ft4 = edtA_im = e_mag * sin

    # num_re = edtA_re - 1,  num_im = edtA_im
    # (stored in ft3, ft4 — subtract 1 from ft3)
    lui     t1, 0x3F800         # 1.0f upper bits
    fmv.w.x ft5, t1
    fsub.s  ft3, ft3, ft5       # ft3 = num_re = edtA_re - 1

    # Load Cre[h][n], Cim[h][n]
    add     t1, s6, t0
    flw     ft6, 0(t1)          # ft6 = Cre
    add     t1, s7, t0
    flw     ft7, 0(t1)          # ft7 = Cim

    # denom = A_re^2 + A_im^2
    # A_re is in fs1/fs0 area — recompute from saved dtA values
    # Actually we need A_re and A_im again. Recompute:
    #   A_re = dtA_re / dt = fs1 / fs0
    #   A_im = dtA_im / dt = fs2 / fs0
    fdiv.s  ft0, fs1, fs0       # ft0 = A_re
    fdiv.s  ft1, fs2, fs0       # ft1 = A_im

    fmul.s  fs6, ft0, ft0       # A_re^2
    fmul.s  fs7, ft1, ft1       # A_im^2
    fadd.s  fs6, fs6, fs7       # denom = A_re^2 + A_im^2

    # Ct_re = (Cre*num_re - Cim*num_im)*A_re + (Cre*num_im + Cim*num_re)*A_im) / denom
    # Ct_im = (Cim*num_re + Cre*num_im)*A_re - (Cre*num_re - Cim*num_im)*A_im) / denom
    # Let:  P = Cre*num_re - Cim*num_im
    #       Q = Cre*num_im + Cim*num_re
    fmul.s  fs3, ft6, ft3       # Cre*num_re
    fmul.s  fs4, ft7, ft4       # Cim*num_im
    fsub.s  fs3, fs3, fs4       # P = Cre*num_re - Cim*num_im

    fmul.s  fs4, ft6, ft4       # Cre*num_im
    fmul.s  fs5, ft7, ft3       # Cim*num_re
    fadd.s  fs4, fs4, fs5       # Q = Cre*num_im + Cim*num_re

    # Ct_re = (P*A_re + Q*A_im) / denom
    fmul.s  fs5, fs3, ft0       # P*A_re
    fmul.s  fs7, fs4, ft1       # Q*A_im
    fadd.s  fs5, fs5, fs7       # P*A_re + Q*A_im
    fdiv.s  fs3, fs5, fs6       # fs3 = Ct_re

    # Ct_im = (Q*A_re - P*A_im) / denom
    fmul.s  fs5, fs4, ft0       # Q*A_re
    fmul.s  fs7, fs3, ft1       # Ct_re*A_im — oops reuse, use P directly
    # Redo: P still in fs3? No, we overwrote fs3. Save P before.
    # Fix: use different reg. Let's store Ct_re in a stack slot temporarily.
    # Actually Ct_im is not needed for the kernel (we only use Re part).
    # kernel k[h][t] = 2 * Re( sum_n Ct[n] * exp(dtA[n]*t) )
    # Re( Ct[n] * exp(dtA[n]*t) ) = Ct_re*e_re - Ct_im*e_im
    # So we DO need Ct_im. Let's recompute P properly.

    # Recompute P (Cre*num_re - Cim*num_im) using ft3,ft4,ft6,ft7
    fmul.s  fs5, ft6, ft3       # Cre*num_re
    fmul.s  fs7, ft7, ft4       # Cim*num_im
    fsub.s  fs5, fs5, fs7       # P

    # Recompute Q (Cre*num_im + Cim*num_re)
    fmul.s  fs7, ft6, ft4       # Cre*num_im
    fmul.s  fa0, ft7, ft3       # Cim*num_re  (use fa0 as scratch)
    fadd.s  fs7, fs7, fa0       # Q

    # Ct_re = (P*A_re + Q*A_im) / denom   [ft0=A_re, ft1=A_im, fs6=denom]
    fmul.s  fa0, fs5, ft0       # P*A_re
    fmul.s  ft2, fs7, ft1       # Q*A_im
    fadd.s  fa0, fa0, ft2
    fdiv.s  fs3, fa0, fs6       # fs3 = Ct_re

    # Ct_im = (Q*A_re - P*A_im) / denom
    fmul.s  fa0, fs7, ft0       # Q*A_re
    fmul.s  ft2, fs5, ft1       # P*A_im
    fsub.s  fa0, fa0, ft2
    fdiv.s  fs4, fa0, fs6       # fs4 = Ct_im

    # Now build kernel: for t = 0..4095
    #   e_mag  = exp(dtA_re * t)
    #   angle  = dtA_im * t
    #   e_re   = e_mag * cos(angle)
    #   e_im   = e_mag * sin(angle)
    #   k_re  += Ct_re*e_re - Ct_im*e_im
    #
    # k[h][t] stored at s4_kernel + (h*4096 + t)*4
    # We accumulate across n for each t.
    # For n=0: initialize k[h][t] = 0 first, then accumulate.

    # t loop — walk through s4_kernel[h][0..4095]
    # kernel row pointer: s4_kernel + h*4096*4
    la      t2, s4_kernel
    li      t3, 4096
    mul     t3, s0, t3          # h * 4096
    slli    t3, t3, 2           # * 4 bytes
    add     t2, t2, t3          # t2 = &s4_kernel[h][0]

    li      s2, 0               # t = 0

.s4_t_loop:
    li      t3, 4096
    bge     s2, t3, .s4_t_done

    # Compute t as float for arithmetic
    fcvt.s.w fa0, s2            # fa0 = (float)t

    # dtA_re * t
    fmul.s  ft3, fs1, fa0       # dtA_re * t
    # exp(dtA_re * t)
    fmv.s   fa0, ft3
    call    exp_f               # fa0 = e_mag

    fmv.s   ft3, fa0            # ft3 = e_mag

    # dtA_im * t
    fcvt.s.w ft4, s2
    fmul.s  ft4, fs2, ft4       # ft4 = dtA_im * t = angle

    # cos(angle)
    fmv.s   fa0, ft4
    call    cos_f
    fmul.s  ft5, ft3, fa0       # e_re = e_mag * cos(angle)

    # sin(angle)
    fmv.s   fa0, ft4
    call    sin_f
    fmul.s  ft6, ft3, fa0       # e_im = e_mag * sin(angle)

    # contribution: Ct_re*e_re - Ct_im*e_im
    fmul.s  ft7, fs3, ft5       # Ct_re * e_re
    fmul.s  fa0, fs4, ft6       # Ct_im * e_im
    fsub.s  ft7, ft7, fa0       # contribution

    # Load current k[h][t], add contribution (accumulate across n)
    flw     ft0, 0(t2)          # k[h][t] so far
    fadd.s  ft0, ft0, ft7
    fsw     ft0, 0(t2)          # store back

    addi    t2, t2, 4           # next t slot
    addi    s2, s2, 1
    j       .s4_t_loop

.s4_t_done:
    addi    s1, s1, 1           # n++
    j       .s4_n_loop

.s4_n_done:
    # Scale kernel by 2: k[h][t] *= 2
    la      t2, s4_kernel
    li      t3, 4096
    mul     t3, s0, t3
    slli    t3, t3, 2
    add     t2, t2, t3          # &s4_kernel[h][0]

    li      t3, 4096
    li      t4, 0               # t = 0
    lui     t5, 0x40000         # 2.0f
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
    # Initialize kernel to 0 for next h iteration happens at start of n loop
    # (we accumulate, so we need to zero before n loop)
    # Actually we should zero BEFORE the n loop for this h.
    # Let's zero at start of h loop — add a zeroing pass here for NEXT h.
    addi    s0, s0, 1           # h++
    j       .s4_h_loop

.s4_h_done:
    # ---- Phase B: Causal convolution + skip ----
    # For h = 0..63:
    #   For t = 0..4095:
    #     sum = 0
    #     For s = 0..t:  sum += kernel[h][s] * in[t-s][h]
    #     out[t][h] = sum + D[h] * in[t][h]

    li      s0, 0               # h = 0

.s4_conv_h:
    li      t0, 64
    bge     s0, t0, .s4_conv_done

    # Load D[h]
    slli    t0, s0, 2
    add     t0, s8, t0
    flw     fs5, 0(t0)          # fs5 = D[h]

    # Pointer to kernel row: s4_kernel + h*4096*4
    la      t2, s4_kernel
    li      t3, 4096
    mul     t3, s0, t3
    slli    t3, t3, 2
    add     t2, t2, t3          # t2 = &kernel[h][0]

    li      s2, 0               # t = 0

.s4_conv_t:
    li      t0, 4096
    bge     s2, t0, .s4_conv_t_done

    # sum = 0
    fmv.w.x ft0, zero           # ft0 = 0.0

    # inner loop s = 0..t
    li      t3, 0               # s = 0
    mv      t4, t2              # ptr into kernel[h][s], start at s=0

    # in[t-s][h]: starts at in + (t)*D_MODEL*4 + h*4, decrements t by 1 each s
    # in pointer for s=0: &in[t][h] = s9 + (t*64 + h)*4
    li      t5, 64
    mul     t5, s2, t5          # t * 64
    add     t5, t5, s0          # t*64 + h
    slli    t5, t5, 2           # * 4
    add     t5, s9, t5          # t5 = &in[t][h]

    # Each s step: t5 decrements by 64*4 = 256 (one row back)

.s4_inner_s:
    bgt     t3, s2, .s4_inner_done  # if s > t, done

    flw     ft1, 0(t4)          # kernel[h][s]
    flw     ft2, 0(t5)          # in[t-s][h]
    fmul.s  ft3, ft1, ft2
    fadd.s  ft0, ft0, ft3       # sum += kernel[h][s] * in[t-s][h]

    addi    t4, t4, 4           # next kernel element
    addi    t5, t5, -256        # in[t-s-1][h]: go back one row (64*4=256)
    addi    t3, t3, 1           # s++
    j       .s4_inner_s

.s4_inner_done:
    # skip: D[h] * in[t][h]
    li      t5, 64
    mul     t5, s2, t5
    add     t5, t5, s0
    slli    t5, t5, 2
    add     t5, s9, t5
    flw     ft1, 0(t5)          # in[t][h]
    fmul.s  ft2, fs5, ft1       # D[h] * in[t][h]
    fadd.s  ft0, ft0, ft2       # sum + skip

    # out[t][h] = sum
    li      t5, 64
    mul     t5, s2, t5
    add     t5, t5, s0
    slli    t5, t5, 2
    add     t5, s10, t5
    fsw     ft0, 0(t5)          # out[t][h] = result

    addi    s2, s2, 1           # t++
    j       .s4_conv_t

.s4_conv_t_done:
    addi    s0, s0, 1           # h++
    j       .s4_conv_h

.s4_conv_done:
    # --- Epilogue ---
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
    flw     fs0, 48(sp)
    flw     fs1, 52(sp)
    flw     fs2, 56(sp)
    flw     fs3, 60(sp)
    flw     fs4, 64(sp)
    flw     fs5, 68(sp)
    flw     fs6, 72(sp)
    flw     fs7, 76(sp)
    addi    sp, sp, 80
    ret

# ================================================================
# 4/6. gelu_inplace(a0=buf*)
#
#   GELU(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
#   Applied in-place to float[4096][64] = 262144 elements.
#
#   Registers:
#     t0 = element counter
#     t1 = limit (262144)
#     a1 = walking pointer (same as a0, advanced)
#     ft0= x
#     ft1= x^2
#     ft2= x^3
#     ft3= inner = SQRT_2_PI*(x + COEFF*x^3)
#     ft4= tanh(inner)
#     ft5= result
#   Constants (loaded via lui+fmv):
#     SQRT_2_PI = 0.7978845608  stored as .float
#     COEFF     = 0.044715      stored as .float
#     0.5f, 1.0f
# ================================================================
.section .data
.align 2
gelu_sqrt2pi:   .float 0.7978845608
gelu_coeff:     .float 0.044715
gelu_half:      .float 0.5
gelu_one:       .float 1.0

.section .text
.globl gelu_inplace
gelu_inplace:
    # Stack: ra, s0 (buf ptr), s1 (loop counter), fs4 (x across call)
    # fs0-fs3 are callee-saved: exp_f preserves fs0,fs1; we must save fs2,fs3 too.
    addi    sp, sp, -32
    sw      ra,   0(sp)
    sw      s0,   4(sp)
    sw      s1,   8(sp)         # s1 = loop counter (safe across call)
    fsw     fs0, 12(sp)
    fsw     fs1, 16(sp)
    fsw     fs2, 20(sp)
    fsw     fs3, 24(sp)
    fsw     fs4, 28(sp)

    mv      s0, a0              # s0 = buf pointer
    li      s1, 0               # s1 = loop counter  ← s-reg, safe across calls
    li      t1, 262144          # 4096 * 64

    la      t2, gelu_sqrt2pi
    flw     fs0, 0(t2)          # fs0 = sqrt(2/pi)
    la      t2, gelu_coeff
    flw     fs1, 0(t2)          # fs1 = 0.044715
    la      t2, gelu_half
    flw     fs2, 0(t2)          # fs2 = 0.5
    la      t2, gelu_one
    flw     fs3, 0(t2)          # fs3 = 1.0

.gelu_loop:
    bge     s1, t1, .gelu_done  # use s1 for comparison

    flw     ft0, 0(s0)          # ft0 = x

    # x^3 = x * x * x
    fmul.s  ft1, ft0, ft0       # x^2
    fmul.s  ft2, ft1, ft0       # x^3

    # inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    fmul.s  ft3, fs1, ft2       # 0.044715 * x^3
    fadd.s  ft3, ft0, ft3       # x + 0.044715*x^3
    fmul.s  ft3, fs0, ft3       # sqrt(2/pi) * (...)

    # Save x into fs4 before calling tanh_f
    # (tanh_f calls exp_f which clobbers all t-regs and ft-regs)
    fmv.s   fs4, ft0            # fs4 = x  (s-reg, safe across call)

    # tanh(inner)
    fmv.s   fa0, ft3
    call    tanh_f              # fa0 = tanh(inner)
    # NOTE: after this call, t0/t1/t2... are ALL clobbered.
    # That's fine — we only need s0, s1, fs0-fs4 which are preserved.

    # Reload t1 (limit) since it was clobbered by the call
    li      t1, 262144

    # result = 0.5 * x * (1 + tanh(inner))
    fadd.s  ft4, fs3, fa0       # 1 + tanh
    fmul.s  ft5, fs2, fs4       # 0.5 * x
    fmul.s  ft5, ft5, ft4       # * (1 + tanh)

    fsw     ft5, 0(s0)          # store result in-place

    addi    s0, s0, 4
    addi    s1, s1, 1           # increment s-reg counter
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

# ================================================================
# 7. take_last(a0=buf*, a1=out*)
#
#   Extract final row of float[4096][64]:
#   out[h] = buf[4095][h]  for h in 0..63
#
#   offset of buf[4095][0] = 4095 * 64 * 4 = 1,044,480 bytes
# ================================================================
.globl take_last
take_last:
    # offset = (SEQ_LEN-1) * D_MODEL * 4 = 4095 * 64 * 4 = 1044480
    li      t0, 1048320
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

# ================================================================
# 8. fc_forward(a0=in*, a1=out*)
#
#   out[c] = fc_b[c] + sum_h( fc_w[c][h] * in[h] )
#   for c in 0..3
# ================================================================
.globl fc_forward
fc_forward:
    la      t2, fc_w
    la      t3, fc_b
    li      t0, 0               # c = 0
    li      t1, 4               # NUM_CLASSES
    mv      a2, a1              # out pointer

.fc_c_loop:
    bge     t0, t1, .fc_done

    flw     ft0, 0(t3)          # ft0 = fc_b[c]
    addi    t3, t3, 4

    li      t4, 0               # h = 0
    li      t5, 64
    mv      t6, a0              # reset to start of in[]

.fc_h_loop:
    bge     t4, t5, .fc_h_done
    flw     ft1, 0(t2)          # fc_w[c][h]
    flw     ft2, 0(t6)          # in[h]
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

# ================================================================
# 9. softmax_forward(a0=in*, a1=out*, a2=n)
#
#   Numerically stable: subtract max first.
#   out[i] = exp(in[i] - max) / sum
# ================================================================
.globl softmax_forward
softmax_forward:
    # Stack: ra, s0(n), s1(in*), s2(out*), s3(loop ctr), fs0(max), fs1(sum)
    # All s-regs survive exp_f calls; t-regs do NOT.
    addi    sp, sp, -32
    sw      ra,   0(sp)
    sw      s0,   4(sp)
    sw      s1,   8(sp)
    sw      s2,  12(sp)
    sw      s3,  16(sp)
    fsw     fs0, 20(sp)
    fsw     fs1, 24(sp)

    mv      s0, a2              # s0 = n
    mv      s1, a0              # s1 = in*
    mv      s2, a1              # s2 = out*

    # Find max (no calls here, t-regs fine)
    flw     ft0, 0(s1)          # max = in[0]
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
    fmv.s   fs0, ft0            # fs0 = max  (s-reg, survives exp_f)

    # exp pass
    fmv.w.x fs1, zero           # fs1 = sum = 0.0  (s-reg, survives exp_f)
    li      s3, 0               # s3 = loop counter  (s-reg, survives exp_f)

.sm_exp_loop:
    bge     s3, s0, .sm_exp_done
    slli    t1, s3, 2
    add     t3, s1, t1          # &in[i]
    flw     ft1, 0(t3)
    fsub.s  fa0, ft1, fs0       # in[i] - max
    call    exp_f               # fa0 = exp(in[i]-max)
                                # t0-t6, ft0-ft11, fa1-fa7 clobbered — OK
                                # s0-s3, fs0, fs1 survive (callee-saved)
    slli    t1, s3, 2           # recompute offset (t1 was clobbered)
    add     t3, s2, t1          # &out[i]
    fsw     fa0, 0(t3)
    fadd.s  fs1, fs1, fa0       # sum += exp(...)
    addi    s3, s3, 1
    j       .sm_exp_loop

.sm_exp_done:
    # Divide by sum
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

# ================================================================
# model_forward(a0=image*)
#
#   Chains all 9 stages using global buffers.
#   Returns predicted class in a0.
# ================================================================
.globl model_forward
model_forward:
    addi    sp, sp, -4
    sw      ra, 0(sp)

    # 1. Hilbert Scan: image -> post_hilbert
    la      a1, post_hilbert
    call    hilbert_scan_layer

    # 2. Input Projection: post_hilbert -> buf_a
    la      a0, post_hilbert
    la      a1, buf_a
    call    input_projection

    # 3. S4D Layer 1: buf_a -> buf_b
    la      a0, s4_0_log_dt
    la      a1, s4_0_logAre
    la      a2, s4_0_Aim
    la      a3, s4_0_Cre
    la      a4, s4_0_Cim
    la      a5, s4_0_D
    la      a6, buf_a
    la      a7, buf_b

    # Zero s4_kernel before use
    la      t0, s4_kernel
    li      t1, 262144          # 64*4096 floats
    li      t2, 0
.zero_k1:
    bge     t2, t1, .zero_k1_done
    sw      zero, 0(t0)
    addi    t0, t0, 4
    addi    t2, t2, 1
    j       .zero_k1
.zero_k1_done:
    call    s4d_layer

    # 4. GELU inplace on buf_b
    la      a0, buf_b
    call    gelu_inplace

    # 5. S4D Layer 2: buf_b -> buf_a
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

    # 6. GELU inplace on buf_a
    la      a0, buf_a
    call    gelu_inplace

    # 7. TakeLastTimestep: buf_a -> post_pool
    la      a0, buf_a
    la      a1, post_pool
    call    take_last

    # 8. FC Layer: post_pool -> logits
    la      a0, post_pool
    la      a1, logits
    call    fc_forward

    # 9. Softmax: logits -> probs
    la      a0, logits
    la      a1, probs
    li      a2, 4
    call    softmax_forward

    # Argmax
    la      t0, probs
    flw     ft0, 0(t0)          # max_val = probs[0]
    li      a0, 0               # predicted = 0
    li      t1, 1

.argmax_loop:
    li      t2, 4
    bge     t1, t2, .argmax_done
    slli    t3, t1, 2
    add     t3, t0, t3
    flw     ft1, 0(t3)
    fle.s   t4, ft1, ft0        # if probs[i] <= max, skip
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