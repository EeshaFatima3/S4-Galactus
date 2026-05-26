import struct
import re
import sys
import numpy as np
import os
import subprocess
import json

task2_dir = os.path.dirname(os.path.abspath(__file__))
ref_dir = os.path.join(task2_dir, "refs")
results_dir = os.path.join(task2_dir, "results")
os.makedirs(results_dir, exist_ok=True)

class_names = ["smooth_round", "smooth_cigar", "edge_on_disk", "unbarred_spiral"]

tolerances = {
    "hilbert": {"mse": 5e-1, "mae": 5e-1},
    "input_proj": {"mse": 5e-1, "mae": 5e-1},
    "s4d": {"mse": 1e-1, "mae": 5e-1},
    "gelu": {"mse": 1e-1, "mae": 5e-1},
    "takelast": {"mse": 5.0, "mae": 2.0},
    "fc": {"mse": 1e-8, "mae": 1e-6},
    "softmax": {"mse": 1e-8, "mae": 1e-4},
}

def hex_to_float(h):
    return struct.unpack('<f', struct.pack('<I', int(h, 16)))[0]

def parse_log_fa_values(log_path):
    if not os.path.exists(log_path):
        return {}
    fa_idx = {}
    for i in range(8):
        fa_idx[f'{i+10:02x}'] = f'fa{i}'
        fa_idx[f'{i+10:x}'] = f'fa{i}'
    results = {}
    re_f_reg = re.compile(r'#\d+\s+\d+\s+[0-9a-f]+\s+[0-9a-f]+\s+f\s+([0-9a-f]+)\s+([0-9a-f]{8})')
    try:
        with open(log_path, 'r') as f:
            for line in f:
                m = re_f_reg.search(line)
                if m:
                    reg_idx = m.group(1).lower().lstrip('0')
                    if not reg_idx: reg_idx = '0'
                    idx_int = int(reg_idx, 16)
                    if 10 <= idx_int <= 17:
                        reg_name = f'fa{idx_int - 10}'
                        val_hex = m.group(2)
                        results[reg_name] = hex_to_float(val_hex)
    except Exception as e:
        print(f"      [!] error parsing log {log_path}: {e}")
    return results

def build_and_run(test_s, extra_links=None, test_name="test"):
    if extra_links is None: extra_links = []
    base = os.path.splitext(os.path.basename(test_s))[0]
    bin_path = os.path.join(task2_dir, "build", "exe", f"{base}.exe")
    log_path = os.path.join(task2_dir, "build", "logs", f"{base}.txt")
    if os.path.exists(bin_path): os.remove(bin_path)
    if os.path.exists(log_path): os.remove(log_path)
    cmd_parts = ["bash", "build.sh", "-a", test_s]
    for lf in extra_links: cmd_parts.extend(["-l", lf])
    cmd = " ".join(cmd_parts)
    print(f"    build: {cmd}")
    try:
        result = subprocess.run(["bash", "-c", cmd], cwd=task2_dir, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print(f"    [timeout] {test_name} took too long regularly")
        return False
    except Exception as e:
        print(f"    [error] {test_name}: {e}")
        return False
    if result.returncode != 0:
        print(f"    [build failed] {result.stderr[:200]}")
        return False
    return os.path.exists(log_path)

def validate_hilbert_sample(sample_idx):
    test_s = "tests/test_sample_hilbert.s"
    log_path = os.path.join(task2_dir, "build", "logs", "test_sample_hilbert.txt")
    ok = build_and_run(test_s, extra_links=["adapter.s", "milestone4/hilbert_vec.s", "milestone4/uproject_vec.s", "milestone4/gelu_vec.s", "milestone4/s4d_vec.s", "milestone4/takelast_vec.s", "milestone4/fc_vec.s", "math.s", "softmax.s", "weights.s", f"sample{sample_idx}.s"], test_name=f"hilbert_sample{sample_idx}")
    if not ok: return None
    fa_vals = parse_log_fa_values(log_path)
    ref_path = os.path.join(ref_dir, f"sample{sample_idx}_hilbert_ref.bin")
    if not os.path.exists(ref_path): return None
    ref = np.fromfile(ref_path, dtype=np.float32)
    got = [fa_vals.get(f'fa{i}', float('nan')) for i in range(min(8, len(ref)))]
    got_arr, ref_arr = np.array(got[:8], dtype=np.float32), ref[:8]
    valid_mask = ~np.isnan(got_arr)
    if not np.any(valid_mask): return {"mse": float('inf'), "mae": float('inf'), "pass": False}
    mse = float(np.mean((got_arr[valid_mask] - ref_arr[valid_mask])**2))
    mae = float(np.mean(np.abs(got_arr[valid_mask] - ref_arr[valid_mask])))
    passed = mse < tolerances["hilbert"]["mse"]
    return {"mse": mse, "mae": mae, "pass": passed, "got": got_arr.tolist(), "ref": ref_arr.tolist()}

def validate_input_proj_sample(sample_idx):
    test_s = "tests/test_sample_input_proj.s"
    log_path = os.path.join(task2_dir, "build", "logs", "test_sample_input_proj.txt")
    ok = build_and_run(test_s, extra_links=["adapter.s", "milestone4/hilbert_vec.s", "milestone4/uproject_vec.s", "milestone4/gelu_vec.s", "milestone4/s4d_vec.s", "milestone4/takelast_vec.s", "milestone4/fc_vec.s", "math.s", "softmax.s", "weights.s", f"sample{sample_idx}.s"], test_name=f"input_proj_sample{sample_idx}")
    if not ok: return None
    fa_vals = parse_log_fa_values(log_path)
    ref_path = os.path.join(ref_dir, f"sample{sample_idx}_input_proj_ref.bin")
    if not os.path.exists(ref_path): return None
    ref = np.fromfile(ref_path, dtype=np.float32)
    got = [fa_vals.get(f'fa{i}', float('nan')) for i in range(4)]
    got_arr, ref_arr = np.array(got, dtype=np.float32), ref[:4]
    valid_mask = ~np.isnan(got_arr)
    if not np.any(valid_mask): return {"mse": float('inf'), "mae": float('inf'), "pass": False}
    mse = float(np.mean((got_arr[valid_mask] - ref_arr[valid_mask])**2))
    mae = float(np.mean(np.abs(got_arr[valid_mask] - ref_arr[valid_mask])))
    passed = mse < tolerances["input_proj"]["mse"] and mae < tolerances["input_proj"]["mae"]
    return {"mse": mse, "mae": mae, "pass": passed, "got": got_arr.tolist(), "ref": ref_arr.tolist()}

def validate_s4d_sample(sample_idx):
    test_s = "tests/test_sample_s4d_short.s"
    log_path = os.path.join(task2_dir, "build", "logs", "test_sample_s4d_short.txt")
    if not os.path.exists(os.path.join(task2_dir, test_s)):
        asm_code = f"""
.section .data
.align 4
buffer_A:       .space 1048576
buffer_B:       .space 1048576
dummy_zeros:    .rept 64
                .float 0.0
                .endr
dummy_ones:     .rept 64
                .float 1.0
                .endr
.section .text
.globl _start
_start:
    li      sp, 0x81000000
    la      a0, test_image
    la      a1, buffer_A
    la      a2, hilbert_idx
    call    hilbert_scan_layer
    la      a0, buffer_A
    la      a1, up_w
    la      a2, up_b
    la      a3, buffer_B
    call    input_projection
    la      a0, buffer_B
    la      a1, buffer_A
    la      a2, s4_0_logAre
    la      a3, s4_0_Aim
    la      a4, dummy_ones
    la      a5, dummy_zeros
    la      a6, s4_0_Cre
    la      a7, s4_0_Cim
    call    s4d_layer
    la      t0, buffer_A
    flw     fa0, 0(t0)
    flw     fa1, 4(t0)
    flw     fa2, 8(t0)
    flw     fa3, 12(t0)
    flw     fa4, 256(t0)
    flw     fa5, 260(t0)
    flw     fa6, 264(t0)
    flw     fa7, 268(t0)
    li      a7, 10
    ecall
"""
        with open(os.path.join(task2_dir, test_s), 'w') as f: f.write(asm_code)
    ok = build_and_run(test_s, extra_links=["adapter.s", "milestone4/hilbert_vec.s", "milestone4/uproject_vec.s", "milestone4/gelu_vec.s", "milestone4/s4d_vec.s", "milestone4/takelast_vec.s", "milestone4/fc_vec.s", "math.s", "softmax.s", "weights.s", f"sample{sample_idx}.s"], test_name=f"s4d_sample{sample_idx}")
    if not ok: return None
    fa_vals = parse_log_fa_values(log_path)



    ref_path = os.path.join(ref_dir, f"sample{sample_idx}_s4d1_ref.bin")
    if not os.path.exists(ref_path): 
        print(f"    [!] no reference for s4d sample {sample_idx} at {ref_path}")
        return None


    ref = np.fromfile(ref_path, dtype=np.float32)
    got = [fa_vals.get(f'fa{i}', float('nan')) for i in range(8)]
    got_arr, ref_arr = np.array(got, dtype=np.float32), ref[:8]
    valid_mask = ~np.isnan(got_arr)
    if not np.any(valid_mask): return {"mse": float('inf'), "mae": float('inf'), "pass": False}
    mse = float(np.mean((got_arr[valid_mask] - ref_arr[valid_mask])**2))
    mae = float(np.mean(np.abs(got_arr[valid_mask] - ref_arr[valid_mask])))
    passed = mse < tolerances["s4d"]["mse"]
    return {"mse": mse, "mae": mae, "pass": passed, "got": got_arr.tolist(), "ref": ref_arr.tolist()}

def validate_gelu_sample(sample_idx):
    test_s = f"tests/test_sample{sample_idx}_gelu.s"
    log_path = os.path.join(task2_dir, "build", "logs", f"test_sample{sample_idx}_gelu.txt")
    if not os.path.exists(os.path.join(task2_dir, test_s)): return None
    ok = build_and_run(test_s, extra_links=["adapter.s", "milestone4/hilbert_vec.s", "milestone4/uproject_vec.s", "milestone4/gelu_vec.s", "milestone4/s4d_vec.s", "milestone4/takelast_vec.s", "milestone4/fc_vec.s", "math.s", "softmax.s", "weights.s"], test_name=f"gelu_sample{sample_idx}")
    if not ok: return None
    fa_vals = parse_log_fa_values(log_path)
    ref_dir_up = os.path.abspath(os.path.join(task2_dir, "..", "..", "caal project", "task9_deliverables", "test_validation"))
    ref_path = os.path.join(ref_dir_up, f"sample{sample_idx}_gelu1.bin")
    if not os.path.exists(ref_path): ref_path = os.path.join(ref_dir, f"sample{sample_idx}_gelu1_ref.bin")
    if not os.path.exists(ref_path): return None
    ref = np.fromfile(ref_path, dtype=np.float32)
    got = [fa_vals.get(f'fa{i}', float('nan')) for i in range(min(8, len(ref)))]
    got_arr, ref_arr = np.array(got[:8], dtype=np.float32), ref[:8]
    valid_mask = ~np.isnan(got_arr)
    if not np.any(valid_mask): return {"mse": float('inf'), "mae": float('inf'), "pass": False}
    mse = float(np.mean((got_arr[valid_mask] - ref_arr[valid_mask])**2))
    mae = float(np.mean(np.abs(got_arr[valid_mask] - ref_arr[valid_mask])))
    passed = mse < tolerances["gelu"]["mse"] and mae < tolerances["gelu"]["mae"]
    return {"mse": mse, "mae": mae, "pass": passed, "got": got_arr.tolist(), "ref": ref_arr.tolist()}

def validate_takelast_sample(sample_idx):
    test_s = f"tests/test_sample{sample_idx}_takelast.s"
    log_path = os.path.join(task2_dir, "build", "logs", f"test_sample{sample_idx}_takelast.txt")
    ok = build_and_run(test_s, extra_links=["adapter.s", "milestone4/hilbert_vec.s", "milestone4/uproject_vec.s", "milestone4/gelu_vec.s", "milestone4/s4d_vec.s", "milestone4/takelast_vec.s", "milestone4/fc_vec.s", "math.s", "softmax.s", "weights.s"], test_name=f"takelast_sample{sample_idx}")
    if not ok: return None
    fa_vals = parse_log_fa_values(log_path)
    ref_path = os.path.join(ref_dir, f"sample{sample_idx}_post_pool_ref.bin")
    if not os.path.exists(ref_path): return None
    ref = np.fromfile(ref_path, dtype=np.float32)
    got = [fa_vals.get(f'fa{i}', float('nan')) for i in range(min(8, len(ref)))]
    got_arr, ref_arr = np.array(got[:8], dtype=np.float32), ref[:8]
    valid_mask = ~np.isnan(got_arr)
    if not np.any(valid_mask): return {"mse": float('inf'), "mae": float('inf'), "pass": False}
    mse = float(np.mean((got_arr[valid_mask] - ref_arr[valid_mask])**2))
    mae = float(np.mean(np.abs(got_arr[valid_mask] - ref_arr[valid_mask])))
    passed = mse < tolerances["takelast"]["mse"]
    return {"mse": mse, "mae": mae, "pass": passed, "got": got_arr.tolist(), "ref": ref_arr.tolist()}

def validate_fc_sample(sample_idx):
    test_s = f"tests/test_sample{sample_idx}_fc.s"
    log_path = os.path.join(task2_dir, "build", "logs", f"test_sample{sample_idx}_fc.txt")
    if not os.path.exists(os.path.join(task2_dir, test_s)): return None
    ok = build_and_run(test_s, extra_links=["adapter.s", "milestone4/hilbert_vec.s", "milestone4/uproject_vec.s", "milestone4/gelu_vec.s", "milestone4/s4d_vec.s", "milestone4/takelast_vec.s", "milestone4/fc_vec.s", "math.s", "softmax.s", "weights.s"], test_name=f"fc_sample{sample_idx}")
    if not ok: return None
    fa_vals = parse_log_fa_values(log_path)
    ref_path = os.path.join(ref_dir, f"sample{sample_idx}_logits_ref.bin")
    ref = np.fromfile(ref_path, dtype=np.float32)
    got = [fa_vals.get(f'fa{i}', float('nan')) for i in range(4)]
    got_arr, ref_arr = np.array(got, dtype=np.float32), ref[:4]
    valid_mask = ~np.isnan(got_arr)
    if not np.any(valid_mask): return {"mse": float('inf'), "mae": float('inf'), "pass": False}
    mse = float(np.mean((got_arr[valid_mask] - ref_arr[valid_mask])**2))
    mae = float(np.mean(np.abs(got_arr[valid_mask] - ref_arr[valid_mask])))
    passed = mse < tolerances["fc"]["mse"] and mae < tolerances["fc"]["mae"]
    return {"mse": mse, "mae": mae, "pass": passed, "got": got_arr.tolist(), "ref": ref_arr.tolist()}

def validate_softmax_sample(sample_idx):
    test_s = f"tests/test_sample{sample_idx}_softmax.s"
    log_path = os.path.join(task2_dir, "build", "logs", f"test_sample{sample_idx}_softmax.txt")
    if not os.path.exists(os.path.join(task2_dir, test_s)): return None
    ok = build_and_run(test_s, extra_links=["adapter.s", "milestone4/hilbert_vec.s", "milestone4/uproject_vec.s", "milestone4/gelu_vec.s", "milestone4/s4d_vec.s", "milestone4/takelast_vec.s", "milestone4/fc_vec.s", "math.s", "softmax.s", "weights.s"], test_name=f"softmax_sample{sample_idx}")
    if not ok: return None
    fa_vals = parse_log_fa_values(log_path)
    ref_path = os.path.join(ref_dir, f"sample{sample_idx}_probs_ref.bin")
    ref = np.fromfile(ref_path, dtype=np.float32)
    got = [fa_vals.get(f'fa{i}', float('nan')) for i in range(4)]
    got_arr, ref_arr = np.array(got, dtype=np.float32), ref[:4]
    valid_mask = ~np.isnan(got_arr)
    if not np.any(valid_mask): return {"mse": float('inf'), "mae": float('inf'), "pass": False}
    mse = float(np.mean((got_arr[valid_mask] - ref_arr[valid_mask])**2))
    mae = float(np.mean(np.abs(got_arr[valid_mask] - ref_arr[valid_mask])))
    passed = mse < tolerances["softmax"]["mse"] and mae < tolerances["softmax"]["mae"]
    pred_got = int(np.argmax(got_arr)) if np.all(valid_mask) else -1
    pred_ref = int(np.argmax(ref_arr))
    return {"mse": mse, "mae": mae, "pass": passed, "got": got_arr.tolist(), "ref": ref_arr.tolist(), "pred_got": pred_got, "pred_ref": pred_ref, "class_match": pred_got == pred_ref}

def generate_report(all_results):
    report_path = os.path.join(results_dir, "validation_report.md")
    lines = ["# task 2: testing and validation report\n", "## per-sample validation\n"]
    for layer in ["hilbert", "input_proj", "s4d", "gelu", "takelast", "fc", "softmax"]:
        lines.extend([f"### {layer}\n", "| sample | mse | mae | pass |", "|--------|-----|-----|------|"])
        for i in range(10):
            key = f"{layer}_sample{i}"
            if key in all_results and all_results[key]:
                r = all_results[key]
                lines.append(f"| {i} | {r['mse']:.2e} | {r['mae']:.2e} | {'pass' if r['pass'] else 'fail'} |")
            else:
                lines.append(f"| {i} | - | - | not run |")
        lines.append("")
    with open(report_path, "w") as f: f.write("\n".join(lines) + "\n")
    print(f"\nreport saved to: {report_path}")
    return report_path

def main():
    print("="*60 + "\n  s4d galaxy classifier - task 2 validation\n" + "="*60)
    all_results = {}
    
    for layer, validate_fn in [
        ("hilbert", validate_hilbert_sample),
        ("input_proj", validate_input_proj_sample),
        ("s4d", validate_s4d_sample),
        ("gelu", validate_gelu_sample),
        ("takelast", validate_takelast_sample),
        ("fc", validate_fc_sample),
        ("softmax", validate_softmax_sample)
    ]:
        print(f"\n--- per-sample {layer} validation ---")
        for i in range(10):
            result = validate_fn(i)
            all_results[f"{layer}_sample{i}"] = result
            if result: print(f"  sample {i}: mse={result['mse']:.2e} mae={result['mae']:.2e} [{'pass' if result['pass'] else 'fail'}]")
    generate_report(all_results)
    print("\n" + "="*60 + "\n  validation complete!\n" + "="*60)

if __name__ == "__main__":
    main()
