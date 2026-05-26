# =============================================================
# tests/test_softmax.s — Softmax Test
# =============================================================
#
# Strategy:
#   Input logits = [0.6658, -0.8044, 2.2583, -3.0594]
#   (These are the FC outputs for all-1.0 input)
#   Expected probs = [0.16208, 0.03726, 0.79676, 0.00391]
# Expected hex:
#   0x3e25f72e  0x3d189a7d  0x3f4bf87f  0x3b800705
#
# Build & run:
#   ./build.sh -a tests/test_softmax.s -l nn.s -l math.s -l weights.s
#   python3 tests/check_softmax.py
# =============================================================

.section .data
.align 2
sm_test_logits: .float  0.6658437252
                .float -0.8043879867
                .float  2.2583341599
                .float -3.0594279766

.section .text
.globl _start
_start:
    # Copy test logits into logits buffer
    la      t0, sm_test_logits
    la      t1, logits
    flw     ft0, 0(t0)
    flw     ft1, 4(t0)
    flw     ft2, 8(t0)
    flw     ft3, 12(t0)
    fsw     ft0, 0(t1)
    fsw     ft1, 4(t1)
    fsw     ft2, 8(t1)
    fsw     ft3, 12(t1)

    # Run softmax
    la      a0, logits
    la      a1, probs
    li      a2, 4
    call    softmax_forward

    # Check probs[0..3]
    la      t0, probs
    flw     fa0, 0(t0)     # expect: 0x3e25f72e  (0.16208)
    flw     fa1, 4(t0)     # expect: 0x3d189a7d  (0.03726)
    flw     fa2, 8(t0)     # expect: 0x3f4bf87f  (0.79676)
    flw     fa3, 12(t0)    # expect: 0x3b800705  (0.00391)

    unimp
