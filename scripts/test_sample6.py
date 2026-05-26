import os
import re
import struct

CLASS_NAMES = ["Smooth Round", "Smooth Cigar", "Edge-on Disk", "Unbarred Spiral"]

print("[*] Compiling Vector Softmax Test for Sample 0...")

build_cmd = "./build.sh -a tests/test_sample6_softmax.s -l adapter.s -l milestone4/hilbert_vec.s -l milestone4/uproject_vec.s -l milestone4/gelu_vec.s -l milestone4/s4d_vec.s -l milestone4/takelast_vec.s -l milestone4/fc_vec.s -l math.s -l softmax.s -l weights.s"
os.system(build_cmd)

print("[*] Parsing Whisper Execution Log...")

log_path = "build/logs/test_sample6_softmax.txt"
if not os.path.exists(log_path):
    print("[-] Error: Log file not found. The compile likely failed.")
    exit(1)

with open(log_path, 'r') as f:
    log_data = f.read()


matches = re.findall(r'f\s+([0-9a-fA-F]{2})\s+([0-9a-fA-F]{8})', log_data)

results = {}

for reg, val_hex in matches:
    idx = int(reg, 16)

    if 10 <= idx <= 13:

        float_val = struct.unpack('>f', bytes.fromhex(val_hex))[0]
        results[f"fa{idx-10}"] = (val_hex, float_val)

if not results:
    print("[-] No floating point values were saved to fa0-fa3. The script crashed early.")
else:
    max_prob = -1.0
    predicted_class = -1
    
    print("\n=== FINAL SOFTMAX PROBABILITIES ===")
    for i in range(4):
        reg_name = f"fa{i}"
        if reg_name in results:
            val_hex, float_val = results[reg_name]
            print(f"  Class {i} ({CLASS_NAMES[i]}): {float_val:.10f}  (Raw Hex: {val_hex})")
            
            if float_val > max_prob:
                max_prob = float_val
                predicted_class = i
        else:
            print(f"  Class {i} ({CLASS_NAMES[i]}): [NOT FOUND IN LOG]")
            
    print(f"\n>> PREDICTED CLASS: {predicted_class} ({CLASS_NAMES[predicted_class]})")
