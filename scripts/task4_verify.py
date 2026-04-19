import mmap
import struct
import time
import numpy as np
import os

def verify_task4_shm():
    """
    Verification script for Milestone 3, Task 4 (Bonus).
    Mocks the Python GUI side of the Shared Memory Interface to test shm_main.s.
    """
    shm_file = 'riscv_galaxy_shm.bin'
    shm_size = 1048576  # 1MB
    
    print("="*60)
    print("  Task 4: Shared Memory Interface Verification")
    print("="*60)
    
    # 1. Ensure SHM file exists
    if not os.path.exists(shm_file):
        print(f"[*] Creating SHM backing file: {shm_file}")
        with open(shm_file, 'wb') as f:
            f.write(b'\x00' * shm_size)
    
    # 2. Open and Map
    with open(shm_file, 'r+b') as f:
        with mmap.mmap(f.fileno(), shm_size) as mm:
            
            # 3. Write dummy input at 0x1000 (all ones for testing)
            print("[*] Writing test image to offset 0x1000...")
            test_img = np.ones(4096, dtype=np.float32) * 0.5
            mm[0x1000 : 0x1000 + test_img.nbytes] = test_img.tobytes()
            
            # 4. Set Control Flag to 1 (Data Ready)
            print("[*] Setting Data Ready flag (0x01) at offset 0x00...")
            mm[0:4] = struct.pack('<I', 1)
            
            print("[*] Waiting for RISC-V Backend (shm_main.s) to respond...")
            print("    (Note: You must run 'build.sh -a shm_main.s' in another terminal)")
            
            start_time = time.time()
            timeout = 30 # seconds
            
            while True:
                flag = struct.unpack('<I', mm[0:4])[0]
                if flag == 2:
                    print("[+] RISC-V Backend response received (Flag=2)!")
                    break
                
                if time.time() - start_time > timeout:
                    print("[!] Timeout: No response from RISC-V backend.")
                    print("    Did you run shm_main.s in whisper or QEMU using the shm file?")
                    return
                
                time.sleep(0.5)
            
            # 5. Read results
            pred_class = struct.unpack('<I', mm[0x04:0x08])[0]
            prob_bytes = mm[0x08 : 0x08 + 16]
            probs = np.frombuffer(prob_bytes, dtype=np.float32)
            
            print("\n--- Results ---")
            print(f"Predicted Class: {pred_class}")
            print(f"Probabilities: {probs}")
            
            # 6. Reset
            mm[0:4] = struct.pack('<I', 0)
            print("\n[+] Verification script complete.")

if __name__ == "__main__":
    verify_task4_shm()
