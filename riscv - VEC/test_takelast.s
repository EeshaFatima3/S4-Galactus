# =============================================================
# tests/test_takelast.s — TakeLastTimestep Test
# =============================================================
#
# Strategy:
#   Fill buf_a so that row t contains (float)t in every column.
#   So buf_a[4095][h] = 4095.0 for all h.
#   After take_last, post_pool[0..63] should all = 4095.0
#
# Expected hex: 0x45800000 = 4096.0 ... wait: 4095.0 = ?
#   4095.0 = 0x457FC000
#
# Build & run:
#   ./build.sh -a tests/test_takelast.s -l nn.s -l math.s -l weights.s
#   python3 tests/check_takelast.py
# =============================================================

.section .text
.globl _start
_start:
    # ── Fill buf_a: row t = value (float)t in all 64 columns ────
    la      t0, buf_a
    li      t1, 0                # t = 0 (row)
    li      t2, 4096

.tl_fill_outer:
    bge     t1, t2, .tl_fill_done
    fcvt.s.w ft0, t1             # ft0 = (float)t
    li      t3, 0                # h = 0
    li      t4, 64

.tl_fill_inner:
    bge     t3, t4, .tl_fill_inner_done
    fsw     ft0, 0(t0)
    addi    t0, t0, 4
    addi    t3, t3, 1
    j       .tl_fill_inner

.tl_fill_inner_done:
    addi    t1, t1, 1
    j       .tl_fill_outer

.tl_fill_done:
    # ── Run take_last ────────────────────────────────────────────
    la      a0, buf_a
    la      a1, post_pool
    call    take_last

    # ── Check: post_pool[0..7] should all = 4095.0 ──────────────
    # 4095.0 in IEEE 754 = 0x457fc000
    la      t0, post_pool
    flw     fa0, 0(t0)           # expect 0x457fc000
    flw     fa1, 4(t0)           # expect 0x457fc000
    flw     fa2, 8(t0)           # expect 0x457fc000
    flw     fa3, 12(t0)          # expect 0x457fc000
    flw     fa4, 16(t0)
    flw     fa5, 20(t0)
    flw     fa6, 24(t0)
    flw     fa7, 28(t0)

    unimp
