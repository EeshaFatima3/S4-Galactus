# main.s — The Final Demo
.section .text
.globl _start

_start:
    # 1. Initialize stack pointer
    li      sp, 0x81000000

    # 2. Load the test image pointer
    la      a0, test_image

    # 3. Run the full neural network pipeline
    call    model_forward

.halt:
    j       .halt