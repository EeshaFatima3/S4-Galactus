import os

for i in range(10):
    print(f"\n--- VALIDATING SAMPLE {i} ---")
    script_name = f"test_sample{i}.py"
    
    # Create the test script for this specific sample
    with open(script_name, "w") as f:
        f.write(open("test_my_math.py").read().replace("sample0", f"sample{i}"))
    
    # Run the validation
    os.system(f"python3 {script_name}")
