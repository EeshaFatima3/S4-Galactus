# main.s 



.section .text
.globl _start
_start:
    #  step 1: Run full forward pass
    # model_forward(a0 = image*)  ---]   returns predicted class in a0
    la      a0, test_image
    call    model_forward
   

    # step 2: Halt ,  result is in a0 
   
    
    la      t0, probs
    flw     fa0, 0(t0)      # prob[0] = smooth_round
    flw     fa1, 4(t0)      # prob[1] = smooth_cigar
    flw     fa2, 8(t0)      # prob[2] = edge_on_disk
    flw     fa3, 12(t0)     # prob[3] = unbarred_spiral

    # a0 still holds predicted class from model_forward
    unimp
