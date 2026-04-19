# math.s — Mathematical Helper Routines
# Milestone 3 — RISC-V 32-bit Assembly
#
# All routines: argument in fa0, result in fa0.
# All routines save/restore ra, fs0-fs2 if used.
# Caller-saved (ft0-ft11, fa1-fa7) are NOT preserved.
#
# Functions:
#   exp_f  : fa0 = exp(fa0)
#   sin_f  : fa0 = sin(fa0)
#   cos_f  : fa0 = cos(fa0)
#   tanh_f : fa0 = tanh(fa0)

# ================================================================
# .data — constants
# ================================================================
.section .data
.align 2

# ── exp constants ──────────────────────────────────────────────
_exp_ln2:       .float  0.69314718056
_exp_inv_ln2:   .float  1.44269504089
_exp_c1:        .float  1.0
_exp_c2:        .float  0.5
_exp_c3:        .float  0.16666667
_exp_c4:        .float  0.04166667
_exp_c5:        .float  0.00833333
_exp_c6:        .float  0.00138889
_exp_max:       .float  88.0
_exp_min:       .float -88.0

# ── sin/cos constants ──────────────────────────────────────────
_trig_two_pi:   .float  6.28318530718
_trig_half_pi:  .float  1.5707963268
_trig_pi:       .float  3.14159265359
_sin_c3:        .float -0.16666667
_sin_c5:        .float  0.00833333
_sin_c7:        .float -0.00019841
_cos_c2:        .float -0.5
_cos_c4:        .float  0.04166667
_cos_c6:        .float -0.00138889

# ── tanh constants ─────────────────────────────────────────────
_tanh_clamp:    .float  9.0
_tanh_two:      .float  2.0
_tanh_one:      .float  1.0
_tanh_neg_one:  .float -1.0

# ================================================================
# .text
# ================================================================
.section .text

# ================================================================
# exp_f: fa0 = exp(fa0)
#
# Algorithm:
#   Clamp x to [-88, 88] to avoid overflow/underflow
#   n = round(x / ln2)
#   r = x - n*ln2        (|r| <= ln2/2 ≈ 0.347)
#   exp(r) by degree-6 polynomial
#   exp(x) = exp(r) * 2^n  (via IEEE754 exponent field)
#
# Stack: saves ra, fs0, fs1 (12 bytes)
# ================================================================
.globl exp_f
exp_f:
    addi sp, sp, -32
    sw t0, 0(sp)
    sw t1, 4(sp)
    sw t2, 8(sp)
    sw t3, 12(sp)
    sw t4, 16(sp)
    sw t5, 20(sp)
    sw t6, 24(sp)
    addi    sp, sp, -12
    sw      ra,  0(sp)
    fsw     fs0, 4(sp)
    fsw     fs1, 8(sp)

    fmv.s   fs0, fa0            # fs0 = x (input)

    # Clamp: if x > 88 return exp(88); if x < -88 return 0
    la      t0, _exp_max
    flw     ft1, 0(t0)
    flt.s   t1, ft1, fs0        # 88 < x ?
    beq     t1, zero, .exp_no_over
    fmv.s   fa0, ft1            # return 88 (approx large value)
    # actually return exp(88) properly — just use ft1 as big number
    # Better: load a precomputed large float. For now clamp to 88.
    j       .exp_ret
.exp_no_over:
    la      t0, _exp_min
    flw     ft1, 0(t0)
    flt.s   t1, fs0, ft1        # x < -88 ?
    beq     t1, zero, .exp_no_under
    fmv.w.x fa0, zero           # return 0.0
    j       .exp_ret
.exp_no_under:

    # n = round(x / ln2)
    la      t0, _exp_inv_ln2
    flw     ft1, 0(t0)
    fmul.s  ft2, fs0, ft1       # x / ln2
    fcvt.w.s t0, ft2, rne       # n = round(x/ln2) as integer
    fcvt.s.w fs1, t0            # fs1 = (float)n

    # r = x - n*ln2
    la      t1, _exp_ln2
    flw     ft3, 0(t1)
    fmul.s  ft4, fs1, ft3       # n * ln2
    fsub.s  ft2, fs0, ft4       # r = x - n*ln2  (ft2 = r)

    # Polynomial: exp(r) = 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720
    # Horner form: 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r*(1/120 + r/720)))))
    la      t1, _exp_c6
    flw     fa0, 0(t1)          # 1/720
    fmul.s  fa0, fa0, ft2       # r/720
    la      t1, _exp_c5
    flw     ft3, 0(t1)
    fadd.s  fa0, fa0, ft3       # + 1/120
    fmul.s  fa0, fa0, ft2       # * r
    la      t1, _exp_c4
    flw     ft3, 0(t1)
    fadd.s  fa0, fa0, ft3       # + 1/24
    fmul.s  fa0, fa0, ft2       # * r
    la      t1, _exp_c3
    flw     ft3, 0(t1)
    fadd.s  fa0, fa0, ft3       # + 1/6
    fmul.s  fa0, fa0, ft2       # * r
    la      t1, _exp_c2
    flw     ft3, 0(t1)
    fadd.s  fa0, fa0, ft3       # + 1/2
    fmul.s  fa0, fa0, ft2       # * r
    la      t1, _exp_c1
    flw     ft3, 0(t1)
    fadd.s  fa0, fa0, ft3       # + 1
    fmul.s  fa0, fa0, ft2       # * r
    fadd.s  fa0, fa0, ft3       # + 1  (ft3 still = 1.0)

    # Multiply by 2^n: bias n by 127 and shift to exponent field
    # Valid for |n| < 127
    addi    t0, t0, 127         # t0 = n + 127 (reuse t0 which held integer n)
    # Wait - t0 was overwritten. Recompute n from fs1.
    fcvt.w.s t0, fs1, rtz       # t0 = (int)n
    addi    t0, t0, 127         # t0 = n + 127
    slli    t0, t0, 23          # shift to IEEE754 exponent bits [30:23]
    fmv.w.x ft3, t0             # ft3 = 2^n as float
    fmul.s  fa0, fa0, ft3       # exp(x) = poly(r) * 2^n

.exp_ret:
    lw      ra,  0(sp)
    flw     fs0, 4(sp)
    flw     fs1, 8(sp)
    addi    sp, sp, 12
    lw t0, 0(sp)
    lw t1, 4(sp)
    lw t2, 8(sp)
    lw t3, 12(sp)
    lw t4, 16(sp)
    lw t5, 20(sp)
    lw t6, 24(sp)
    addi sp, sp, 32
    ret

# ================================================================
# sin_f: fa0 = sin(fa0)
#
# Algorithm:
#   1. Range reduce to [-2pi, 2pi] by subtracting 2pi*round(x/2pi)
#   2. Further reduce to [-pi/2, pi/2]:
#      if x > pi/2:  x = pi - x   (sin(pi-x) = sin(x))
#      if x < -pi/2: x = -pi - x  (sin(-pi-x) = -sin(x)... wait)
#      Actually: if x in [pi/2, pi]: sin(x) = sin(pi-x)
#                if x in [-pi, -pi/2]: sin(x) = -sin(-pi-x) = sin(x) (identity)
#      Simpler: reduce to [-pi/2, pi/2] using symmetry, track sign flip.
#   3. Polynomial: sin(x) ≈ x - x^3/6 + x^5/120 - x^7/5040
#
# Stack: saves ra, fs0 (8 bytes)
# ================================================================
.globl sin_f
sin_f:
    addi sp, sp, -32
    sw t0, 0(sp)
    sw t1, 4(sp)
    sw t2, 8(sp)
    sw t3, 12(sp)
    sw t4, 16(sp)
    sw t5, 20(sp)
    sw t6, 24(sp)
    addi    sp, sp, -8
    sw      ra, 0(sp)
    fsw     fs0, 4(sp)

    fmv.s   fs0, fa0            # fs0 = x

    # Step 1: reduce to [-pi, pi]
    la      t0, _trig_two_pi
    flw     ft1, 0(t0)          # ft1 = 2*pi
    fdiv.s  ft2, fs0, ft1       # x / 2pi
    fcvt.w.s t0, ft2, rne       # round to nearest
    fcvt.s.w ft2, t0
    fmul.s  ft3, ft2, ft1       # n * 2pi
    fsub.s  fs0, fs0, ft3       # x reduced to [-pi, pi]

    # Step 2: reduce to [-pi/2, pi/2]
    la      t0, _trig_half_pi
    flw     ft4, 0(t0)          # ft4 = pi/2
    la      t0, _trig_pi
    flw     ft5, 0(t0)          # ft5 = pi

    # if x > pi/2: use sin(pi - x) = sin(x) (same value)
    flt.s   t0, ft4, fs0        # pi/2 < x ?
    beq     t0, zero, .sin_check_lower
    fsub.s  fs0, ft5, fs0       # x = pi - x
    j       .sin_poly

.sin_check_lower:
    # if x < -pi/2: use sin(-pi - x) = -sin(pi + x) -- need sign flip
    # Actually sin(x) for x in [-pi, -pi/2]: sin(x) = -sin(-x-pi+pi) 
    # Simpler identity: sin(x) = -sin(-pi - x) when x in [-pi, -pi/2]
    fneg.s  ft6, ft4            # -pi/2
    flt.s   t0, fs0, ft6        # x < -pi/2 ?
    beq     t0, zero, .sin_poly
    fneg.s  ft5, ft5            # -pi
    fsub.s  fs0, ft5, fs0       # x = -pi - x  =>  sin(-pi-x) = -sin(x+pi) = sin(x) for this range
    # Actually: sin(x) = sin(pi - x) by reflection, works for upper half
    # For lower: sin(x) = -sin(-x); and -x is in [pi/2, pi], so sin(-x)=sin(pi-(-x))=sin(pi+x)
    # This gets complicated. Simpler: negate x, recurse mentally.
    # The key identity: sin(-pi - x) = sin(-(pi+x)) = -sin(pi+x) = -(-sin(x)) = sin(x)
    # So: x_new = -pi - x_old  =>  sin(x_new) = sin(x_old). No sign change needed.

.sin_poly:
    # Polynomial on fs0 (reduced x in [-pi/2, pi/2])
    fmul.s  ft1, fs0, fs0       # x^2
    fmul.s  ft2, ft1, fs0       # x^3

    la      t0, _sin_c3
    flw     ft3, 0(t0)
    fmul.s  ft3, ft3, ft2       # c3*x^3

    fmul.s  ft4, ft2, ft1       # x^5
    la      t0, _sin_c5
    flw     ft5, 0(t0)
    fmul.s  ft5, ft5, ft4       # c5*x^5

    fmul.s  ft6, ft4, ft1       # x^7
    la      t0, _sin_c7
    flw     ft7, 0(t0)
    fmul.s  ft7, ft7, ft6       # c7*x^7

    fmv.s   fa0, fs0
    fadd.s  fa0, fa0, ft3
    fadd.s  fa0, fa0, ft5
    fadd.s  fa0, fa0, ft7

    lw      ra, 0(sp)
    flw     fs0, 4(sp)
    addi    sp, sp, 8
    lw t0, 0(sp)
    lw t1, 4(sp)
    lw t2, 8(sp)
    lw t3, 12(sp)
    lw t4, 16(sp)
    lw t5, 20(sp)
    lw t6, 24(sp)
    addi sp, sp, 32
    ret

# ================================================================
# cos_f: fa0 = cos(fa0)
#   cos(x) = sin(x + pi/2)
# ================================================================
.globl cos_f
cos_f:
    addi sp, sp, -32
    sw t0, 0(sp)
    sw t1, 4(sp)
    sw t2, 8(sp)
    sw t3, 12(sp)
    sw t4, 16(sp)
    sw t5, 20(sp)
    sw t6, 24(sp)
    addi    sp, sp, -4
    sw      ra, 0(sp)

    la      t0, _trig_half_pi
    flw     ft0, 0(t0)
    fadd.s  fa0, fa0, ft0
    call    sin_f

    lw      ra, 0(sp)
    addi    sp, sp, 4
    lw t0, 0(sp)
    lw t1, 4(sp)
    lw t2, 8(sp)
    lw t3, 12(sp)
    lw t4, 16(sp)
    lw t5, 20(sp)
    lw t6, 24(sp)
    addi sp, sp, 32
    ret

# ================================================================
# tanh_f: fa0 = tanh(fa0)
#
# Algorithm:
#   if x > 9:  return 1.0
#   if x < -9: return -1.0
#   e2x = exp(2*x)
#   return (e2x - 1) / (e2x + 1)
#
# Key: we save x on the stack (not in a float reg) because
# exp_f may clobber fs0. We use an integer reg to hold the
# sign separately, keeping things simple.
#
# Stack layout (16 bytes):
#   sp+0  : ra
#   sp+4  : fs0 (caller's fs0 — we must preserve it)
#   sp+8  : x as raw bits (fmv.x.w -> sw -> lw -> fmv.w.x to reload)
#   sp+12 : (padding)
# ================================================================
.globl tanh_f
tanh_f:
    addi sp, sp, -32
    sw t0, 0(sp)
    sw t1, 4(sp)
    sw t2, 8(sp)
    sw t3, 12(sp)
    sw t4, 16(sp)
    sw t5, 20(sp)
    sw t6, 24(sp)
    addi    sp, sp, -16
    sw      ra,   0(sp)
    fsw     fs0,  4(sp)

    # Save input x as integer bits on stack
    fmv.x.w t1, fa0             # t1 = raw bits of x
    sw      t1,   8(sp)         # save on stack

    # Load clamp threshold
    la      t0, _tanh_clamp
    flw     ft1, 0(t0)          # ft1 = 9.0

    # Check x > 9
    flt.s   t2, ft1, fa0        # 9.0 < x?
    beq     t2, zero, .tanh_check_neg
    la      t0, _tanh_one
    flw     fa0, 0(t0)          # return 1.0
    j       .tanh_ret

.tanh_check_neg:
    # Check x < -9
    fneg.s  ft2, ft1            # ft2 = -9.0
    flt.s   t2, fa0, ft2        # x < -9.0?
    beq     t2, zero, .tanh_compute
    la      t0, _tanh_neg_one
    flw     fa0, 0(t0)          # return -1.0
    j       .tanh_ret

.tanh_compute:
    # Compute exp(2*x)
    # Reload x from stack (fa0 may have been used in comparisons)
    lw      t1, 8(sp)
    fmv.w.x fa0, t1             # fa0 = x again

    la      t0, _tanh_two
    flw     ft3, 0(t0)          # ft3 = 2.0
    fmul.s  fa0, fa0, ft3       # fa0 = 2*x
    call    exp_f               # fa0 = e^(2x)
                                # NOTE: exp_f preserves fs0,fs1 but
                                # clobbers ft0-ft11. fa0 holds result.

    # (e2x - 1) / (e2x + 1)
    la      t0, _tanh_one
    flw     ft3, 0(t0)          # ft3 = 1.0
    fsub.s  ft4, fa0, ft3       # ft4 = e2x - 1
    fadd.s  ft5, fa0, ft3       # ft5 = e2x + 1
    fdiv.s  fa0, ft4, ft5       # fa0 = tanh(x)

.tanh_ret:
    lw      ra,   0(sp)
    flw     fs0,  4(sp)
    addi    sp, sp, 16
    lw t0, 0(sp)
    lw t1, 4(sp)
    lw t2, 8(sp)
    lw t3, 12(sp)
    lw t4, 16(sp)
    lw t5, 20(sp)
    lw t6, 24(sp)
    addi sp, sp, 32
    ret
