# main.s — Pure Bare-Metal Inference Pass
.section .text
.globl _start

_start:
    # Initialize the stack pointer 
    li      sp, 0x81000000

    la      a0, test_image

    # Execute the complete 10.1-Billion instruction S4D pipeline
    call    model_forward

    li      t0, 0xd0580000
    li      t1, 1
    sw      t1, 0(t0)

.halt:
    j       .halt
