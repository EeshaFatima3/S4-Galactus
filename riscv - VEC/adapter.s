.section .data
.align 4

.globl buf_a
.globl buf_b
.globl post_pool
.globl logits
.globl probs
.globl post_hilbert

buf_a:          .space 1048576
buf_b:          .space 1048576
post_hilbert:   .space 16384
post_pool:      .space 256
logits:         .space 16
probs:          .space 16

.section .text

.globl gelu_inplace
gelu_inplace:
    addi    sp, sp, -16
    sw      ra, 12(sp)
    mv      a1, a0             
    li      a2, 262144         
    call    gelu_layer
    lw      ra, 12(sp)
    addi    sp, sp, 16
    ret

.globl take_last
take_last:
    addi    sp, sp, -16
    sw      ra, 12(sp)

    li      a2, 4096          
    li      a3, 64            
    call    take_last_timestep
    lw      ra, 12(sp)
    addi    sp, sp, 16
    ret

.globl fc_forward
fc_forward:
    addi    sp, sp, -16
    sw      ra, 12(sp)
   
    mv      a3, a1            
    la      a1, fc_w          
    la      a2, fc_b          
    li      a4, 64            
    li      a5, 4             
    call    fc_layer
    lw      ra, 12(sp)
    addi    sp, sp, 16
    ret
