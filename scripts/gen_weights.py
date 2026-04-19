import sys
import os
import numpy as np

def generate_weights(bin_path, out_path="weights.s"):
    if not os.path.exists(bin_path):
        print(f"Error: Could not find {bin_path}")
        sys.exit(1)

    print(f"Reading weights from {bin_path}...")
    
    with open(bin_path, 'rb') as f:
        # 1. Hilbert Indices (int32)
        hilbert_idx = np.fromfile(f, dtype=np.int32, count=4096)

        # 2. Input Projection (weight, bias)
        up_weight = np.fromfile(f, dtype=np.float32, count=64)
        up_bias   = np.fromfile(f, dtype=np.float32, count=64)

        # 3. S4D Layer 1 Parameters
        s4_0_log_dt = np.fromfile(f, dtype=np.float32, count=64)
        s4_0_logAre = np.fromfile(f, dtype=np.float32, count=64 * 32)
        s4_0_Aim    = np.fromfile(f, dtype=np.float32, count=64 * 32)
        
        # --- THE FIX: De-interleaving the Layer 1 C Matrix ---
        s4_0_C_raw = np.fromfile(f, dtype=np.float32, count=64 * 32 * 2).reshape(64, 32, 2)
        s4_0_Cre   = s4_0_C_raw[:, :, 0].flatten()
        s4_0_Cim   = s4_0_C_raw[:, :, 1].flatten()
        
        s4_0_D      = np.fromfile(f, dtype=np.float32, count=64)

        # 4. S4D Layer 2 Parameters
        s4_1_log_dt = np.fromfile(f, dtype=np.float32, count=64)
        s4_1_logAre = np.fromfile(f, dtype=np.float32, count=64 * 32)
        s4_1_Aim    = np.fromfile(f, dtype=np.float32, count=64 * 32)
        
        # --- THE FIX: De-interleaving the Layer 2 C Matrix ---
        s4_1_C_raw = np.fromfile(f, dtype=np.float32, count=64 * 32 * 2).reshape(64, 32, 2)
        s4_1_Cre   = s4_1_C_raw[:, :, 0].flatten()
        s4_1_Cim   = s4_1_C_raw[:, :, 1].flatten()
        
        s4_1_D      = np.fromfile(f, dtype=np.float32, count=64)

        # 5. Fully Connected Layer
        fc_weight = np.fromfile(f, dtype=np.float32, count=4 * 64)
        fc_bias   = np.fromfile(f, dtype=np.float32, count=4)

    print(f"Writing fixed assembly weights to {out_path}...")
    
    with open(out_path, 'w') as out:
        out.write("# weights.s — auto-generated from model_weights.bin\n")
        out.write("# Includes numpy de-interleaving fix for the C matrices\n\n")
        out.write(".section .data\n\n")

        def write_label(name, arr, is_int=False):
            out.write(f".globl {name}\n")
            out.write(f"{name}:\n")
            directive = ".word" if is_int else ".float"
            for val in arr:
                if is_int:
                    out.write(f"    {directive} {int(val)}\n")
                else:
                    out.write(f"    {directive} {float(val):.10g}\n")
            out.write("\n")

        # Write everything out in the exact order the assembly expects
        write_label("hilbert_idx", hilbert_idx, is_int=True)
        write_label("up_weight", up_weight)
        write_label("up_bias", up_bias)
        write_label("s4_0_log_dt", s4_0_log_dt)
        write_label("s4_0_logAre", s4_0_logAre)
        write_label("s4_0_Aim", s4_0_Aim)
        write_label("s4_0_Cre", s4_0_Cre)
        write_label("s4_0_Cim", s4_0_Cim)
        write_label("s4_0_D", s4_0_D)
        write_label("s4_1_log_dt", s4_1_log_dt)
        write_label("s4_1_logAre", s4_1_logAre)
        write_label("s4_1_Aim", s4_1_Aim)
        write_label("s4_1_Cre", s4_1_Cre)
        write_label("s4_1_Cim", s4_1_Cim)
        write_label("s4_1_D", s4_1_D)
        write_label("fc_weight", fc_weight)
        write_label("fc_bias", fc_bias)

    print("Success! The weights.s file is ready.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 gen_weights.py <path_to_model_weights.bin>")
        sys.exit(1)
    generate_weights(sys.argv[1])
