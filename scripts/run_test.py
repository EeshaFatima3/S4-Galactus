import os
import subprocess
import sys

# Directory where sample0_input.bin, sample0_s4d1.bin, etc. are located
SAMPLES_DIR = "."  
NUM_SAMPLES = 10   # Adjust this if you want to run on all your samples

def run_tests():
    print("==================================================")
    print(" S4D Numerical Validation Test Suite")
    print("==================================================")
    
    # 1. Compile the test program
    print("[*] Compiling test suite...")
    compile_cmd = ["gcc", "-O2", "-o", "test_suite", "test.c", "-lm"]
    if subprocess.run(compile_cmd).returncode != 0:
        print("[!] Compilation failed!")
        sys.exit(1)
        
    passed_samples = 0
    failed_samples = 0

    # 2. Run the C program for each sample ID
    for i in range(NUM_SAMPLES):
        # The C program returns exit code 0 if all layers PASS, 1 if any FAIL
        result = subprocess.run(["./test_suite", SAMPLES_DIR, str(i)], capture_output=True, text=True)
        
        # Print the output from the C program
        print(result.stdout)
        
        if result.returncode == 0:
            passed_samples += 1
        else:
            failed_samples += 1
            print(f"--> Sample {i} FAILED validation.")

    # 3. Aggregate Reporting
    print("==================================================")
    print(" Validation Summary")
    print("==================================================")
    print(f" Total Samples Tested : {NUM_SAMPLES}")
    print(f" Samples Passed       : {passed_samples}")
    print(f" Samples Failed       : {failed_samples}")
    print("==================================================")

    if failed_samples > 0:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED SUCCESSFULLY! Exit Code 0.")
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
