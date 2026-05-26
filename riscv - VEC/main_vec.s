.section .data
.align 4

# --- MEMORY BANKS ---


buffer_A:       .space 1048576   
buffer_B:       .space 1048576   
buffer_small:   .space 256       # For TakeLast (64 floats)
buffer_logits:  .space 16        # For FC (4 floats)
final_probs:    .space 16        # For Softmax (4 floats)



dummy_zeros:    
    .rept 64
    .float 0.0
    .endr

dummy_ones:     
    .rept 64
    .float 1.0
    .endr




.section .text
.globl _start

_start:
    li      sp, 0x81000000

    # 1. Hilbert Scan
    la      a0, test_image       # From sample0.s
    la      a1, buffer_A
    la      a2, hilbert_idx      # From weights.s
    call    hilbert_scan_layer

    # 2. Input Projection
    la      a0, buffer_A
    la      a1, up_w             # From weights.s
    la      a2, up_b             # From weights.s
    la      a3, buffer_B
    call    input_projection


# 3. S4D Layer 1
la      a0, buffer_B         # Input u
    la      a1, buffer_A         # Output y
    la      a2, s4_0_logAre      # A_re (From weights.s)
    la      a3, s4_0_Aim         # A_im (From weights.s)
    la      a4, dummy_ones       # B_re (Missing, so mathematically 1.0)
    la      a5, dummy_zeros      # B_im (Missing, so mathematically 0.0)
    la      a6, s4_0_Cre         # C_re (From weights.s)
    la      a7, s4_0_Cim         # C_im (From weights.s)
    call    s4d_layer


    # 4. GELU 1
    la      a0, buffer_A
    la      a1, buffer_A
    li      a2, 262144
    call    gelu_layer




# 5. S4D Layer 2
la      a0, buffer_A         # Input u
    la      a1, buffer_B         # Output y
    la      a2, s4_1_logAre      # A_re 
    la      a3, s4_1_Aim         # A_im
    la      a4, dummy_ones       # B_re 
    la      a5, dummy_zeros      # B_im 
    la      a6, s4_1_Cre         # C_re
    la      a7, s4_1_Cim         # C_im
    call    s4d_layer


    # 6. GELU 2
    la      a0, buffer_B
    la      a1, buffer_B
    li      a2, 262144
    call    gelu_layer

    # 7. Take Last
    la      a0, buffer_B
    la      a1, buffer_small
    li      a2, 4096
    li      a3, 64
    call    take_last_timestep

    # 8. FC Layer
    la      a0, buffer_small
    la      a1, fc_w             # From weights.s
    la      a2, fc_b             # From weights.s
    la      a3, buffer_logits
    li      a4, 64
    li      a5, 4
    call    fc_layer

    # 9. Softmax
    la      a0, buffer_logits
    la      a1, final_probs
    call    softmax_forward      # From our new extracted file

# --- TEAMMATE'S TRICK: Load final probabilities into fa0-fa3 ---
    la      t0, final_probs
    flw     fa0, 0(t0)
    flw     fa1, 4(t0)
    flw     fa2, 8(t0)
    flw     fa3, 12(t0)

# Trigger a clean exit
    li a7, 10           # 10 is the 'ecall' exit code for the simulator
    ecall               # Stop execution cleanly
