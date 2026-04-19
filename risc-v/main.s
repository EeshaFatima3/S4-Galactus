# main.s — S4D Galaxy Classifier Demo Program
# Milestone 3 — RISC-V 32-bit Assembly
#
# How to build and run:
#   python3 gen_image.py 0
#   ./build.sh -a main.s -l nn.s -l math.s -l weights.s -l image.s
#
# How to read the result:
#   The simulator prints "tohost: X" where X is the predicted class (1-4).
#   We write (class + 1) to tohost because tohost=1 means "exit(0)" normally,
#   so we use class+1 to distinguish: 1=class0, 2=class1, 3=class2, 4=class3.
#
#   Alternatively: run with -f log and grep for the probs fsw entries.
#
# Class mapping:
#   0 = smooth_round
#   1 = smooth_cigar
#   2 = edge_on_disk
#   3 = unbarred_spiral

.section .data
.align 2
# Store probs here so we can also read from log if needed
result_class: .word 0

.section .text
.globl _start
_start:
    # ── Run full forward pass ────────────────────────────────────
    la      a0, test_image
    call    model_forward
    # a0 = predicted class index (0-3)

    # ── Save predicted class to memory ──────────────────────────
    la      t0, result_class
    sw      a0, 0(t0)

    # ── Load probs into fa0-fa3 so they appear in log ───────────
    la      t1, probs
    flw     fa0, 0(t1)      # prob[0] = P(smooth_round)
    flw     fa1, 4(t1)      # prob[1] = P(smooth_cigar)
    flw     fa2, 8(t1)      # prob[2] = P(edge_on_disk)
    flw     fa3, 12(t1)     # prob[3] = P(unbarred_spiral)

    # ── Write (class + 1) to tohost so whisper reports it ───────
    # tohost address = 0xd0580000
    # Writing value N means: exit with code N-1
    # So predicted class 0 → write 1, class 1 → write 2, etc.
    addi    a0, a0, 1           # class + 1  (1..4)
    li      t0, 0xd0580000      # tohost address (may need lui+addi)
    sw      a0, 0(t0)

    unimp
