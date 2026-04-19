import json

# The exact probabilities extracted from your C-Model screenshots
c_model_data = {
    0: [0.1545, 0.3571, 0.2345, 0.2539],
    1: [0.1118, 0.3126, 0.3762, 0.1994],
    2: [0.0364, 0.0787, 0.0247, 0.8602],
    3: [0.0316, 0.3854, 0.2961, 0.2869],
    4: [0.0000, 0.2762, 0.7226, 0.0012],
    5: [0.0424, 0.3579, 0.4468, 0.1529],
    6: [0.0810, 0.1631, 0.1312, 0.6246],
    7: [0.0025, 0.4557, 0.4598, 0.0820],
    8: [0.0854, 0.2496, 0.2282, 0.4368],
    9: [0.1488, 0.2273, 0.2323, 0.3916]
}

def main():
    try:
        with open('validation_results.json', 'r') as f:
            riscv_data = json.load(f)
            
        print(f"\n{'Sample':<8} | {'Class':<7} | {'C-Model':<12} | {'RISC-V':<12} | {'Difference'}")
        print("=" * 75)
        
        for i in range(10):
            key = f"softmax_sample{i}"
            if key in riscv_data:
                # Get RISC-V array
                rv_probs = riscv_data[key].get("got", [0, 0, 0, 0])
                c_probs = c_model_data[i]
                
                # Print row-by-row comparisons
                for cls in range(4):
                    c_val = c_probs[cls]
                    rv_val = rv_probs[cls]
                    diff = abs(c_val - rv_val)
                    print(f"Sample {i:<2} | Class {cls} | {c_val:<12.6f} | {rv_val:<12.6f} | {diff:.8f}")
                
                # Calculate the predicted class (highest probability)
                c_pred = c_probs.index(max(c_probs))
                rv_pred = rv_probs.index(max(rv_probs))
                
                # Check if they match
                match_status = "PASS ✓" if c_pred == rv_pred else "FAIL ✗"
                
                # Print the summary line
                print(f"  -> Prediction: C-Model = Class {c_pred} | RISC-V = Class {rv_pred} | {match_status}")
            else:
                print(f"Sample {i:<2} | No RISC-V data found in JSON.")
            
            print("-" * 75)
            
    except FileNotFoundError:
        print("[!] Error: Could not find validation_results.json. Make sure it is in the same folder!")

if __name__ == "__main__":
    main()
