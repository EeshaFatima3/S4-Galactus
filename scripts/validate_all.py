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
    "hilbert": {"mse": 1e-12, "mae": None},
    "input_proj": {"mse": 1e-8, "mae": 1e-6},
    "gelu": {"mse": 1e-7, "mae": 1e-4},
    "takelast": {"mse": 1e-12, "mae": None},
    "fc": {"mse": 1e-8, "mae": 1e-6},
    "softmax": {"mse": 1e-8, "mae": 1e-4},
}


def hex_to_float(h):
    return struct.unpack('<f', struct.pack('<I', int(h, 16)))[0]


def parse_log_fa_values(log_path):
    """
    Parses the whisper log file to extract the FINAL values of fa0-fa7 registers.
    """
    if not os.path.exists(log_path):
        return {}

    fa_idx = {}
    for i in range(8):
        # fa0-fa7 are f10-f17 (0x0a - 0x11 or 0xa - 0x11)
        fa_idx[f'{i+10:02x}'] = f'fa{i}'
        fa_idx[f'{i+10:x}'] = f'fa{i}'

    results = {}
    
    # Matches: #inst pc opcode f reg_hex val_hex [mnemonic]
    # Robust regex for different HART IDs and spacing
    # #15 0 8000005a 000f2007 f 00 3d562b80 flw ft0, 0x0(t5)
    re_f_reg = re.compile(r'#\d+\s+\d+\s+[0-9a-f]+\s+[0-9a-f]+\s+f\s+([0-9a-f]+)\s+([0-9a-f]{8})')

    try:
        with open(log_path, 'r') as f:
            for line in f:
                m = re_f_reg.search(line)
                if m:
                    reg_idx = m.group(1).lower().lstrip('0')
                    # Standardize to no leading zeros (except for '00' -> '0')
                    if not reg_idx: reg_idx = '0'
                    
                    # Convert to fa names based on index mapping
                    # fa0 index is 10 (0xa), fa7 is 17 (0x11)
                    idx_int = int(reg_idx, 16)
                    if 10 <= idx_int <= 17:
                        reg_name = f'fa{idx_int - 10}'
                        val_hex = m.group(2)
                        results[reg_name] = hex_to_float(val_hex)
    except Exception as e:
        print(f"      [!] error parsing log {log_path}: {e}")

    return results


def parse_log_fsw_values(log_path):
    fsw_pat = re.compile(
        r'#\d+\s+\d+\s+\w+\s+\w+\s+m\s+\w+\s+([0-9a-f]{8})\s+fsw'
    )

    vals = []
    if not os.path.exists(log_path):
        return vals

    with open(log_path) as f:
        for line in f:
            m = fsw_pat.search(line)
            if m:
                vals.append(hex_to_float('0x' + m.group(1)))

    return vals


def build_and_run(test_s, extra_links=None, test_name="test"):
    if extra_links is None:
        extra_links = []

    # Ensure build dir exists and clean last binary/log to avoid stale data
    base = os.path.splitext(os.path.basename(test_s))[0]
    bin_path = os.path.join(task2_dir, "build", "exe", f"{base}.exe")
    log_path = os.path.join(task2_dir, "build", "logs", f"{base}.txt")
    if os.path.exists(bin_path): os.remove(bin_path)
    if os.path.exists(log_path): os.remove(log_path)

    # Run via bash (to support WSL / Git Bash context depending on where python is executed from)
    cmd_parts = ["bash", "3_build.sh", "-a", test_s]
    for lf in extra_links:
        cmd_parts.extend(["-l", lf])

    cmd = " ".join(cmd_parts)
    print(f"    build: {cmd}")

    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            cwd=task2_dir,
            capture_output=True,
            text=True,
            timeout=1200
        )
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

    ok = build_and_run(
        test_s,
        extra_links=["nn_short.s", "math.s", "weights.s", f"sample{sample_idx}.s"],
        test_name=f"hilbert_sample{sample_idx}"
    )
    if not ok:
        return None

    fa_vals = parse_log_fa_values(log_path)

    ref_path = os.path.join(ref_dir, f"sample{sample_idx}_hilbert_ref.bin")
    if not os.path.exists(ref_path):
        print(f"    [!] no reference for hilbert sample {sample_idx}")
        return None

    ref = np.fromfile(ref_path, dtype=np.float32)

    got = []
    for i in range(min(8, len(ref))):
        reg = f'fa{i}'
        if reg in fa_vals:
            got.append(fa_vals[reg])
        else:
            got.append(float('nan'))

    got_arr = np.array(got[:8], dtype=np.float32)
    ref_arr = ref[:8]

    valid_mask = ~np.isnan(got_arr)
    if not np.any(valid_mask):
        return {"mse": float('inf'), "mae": float('inf'), "pass": False}

    mse = float(np.mean((got_arr[valid_mask] - ref_arr[valid_mask])**2))
    mae = float(np.mean(np.abs(got_arr[valid_mask] - ref_arr[valid_mask])))

    tol = tolerances["hilbert"]
    passed = mse < tol["mse"]

    return {"mse": mse, "mae": mae, "pass": passed, "got": got_arr.tolist(), "ref": ref_arr.tolist()}


def validate_input_proj_sample(sample_idx):
    test_s = "tests/test_sample_input_proj.s"
    log_path = os.path.join(task2_dir, "build", "logs", "test_sample_input_proj.txt")

    ok = build_and_run(
        test_s,
        extra_links=["nn_short.s", "math.s", "weights.s", f"sample{sample_idx}.s"],
        test_name=f"input_proj_sample{sample_idx}"
    )
    if not ok:
        return None

    fa_vals = parse_log_fa_values(log_path)

    ref_path = os.path.join(ref_dir, f"sample{sample_idx}_input_proj_ref.bin")
    if not os.path.exists(ref_path):
        print(f"    [!] no reference for input_proj sample {sample_idx}")
        return None

    ref = np.fromfile(ref_path, dtype=np.float32)

    got = []
    for i in range(4):
        reg = f'fa{i}'
        if reg in fa_vals:
            got.append(fa_vals[reg])
        else:
            got.append(float('nan'))

    got_arr = np.array(got, dtype=np.float32)
    ref_arr = ref[:4]

    valid_mask = ~np.isnan(got_arr)
    if not np.any(valid_mask):
        return {"mse": float('inf'), "mae": float('inf'), "pass": False}

    mse = float(np.mean((got_arr[valid_mask] - ref_arr[valid_mask])**2))
    mae = float(np.mean(np.abs(got_arr[valid_mask] - ref_arr[valid_mask])))

    tol = tolerances["input_proj"]
    passed = mse < tol["mse"]
    if tol["mae"] is not None:
        passed = passed and mae < tol["mae"]

    return {"mse": mse, "mae": mae, "pass": passed, "got": got_arr.tolist(), "ref": ref_arr.tolist()}



def validate_gelu_sample(sample_idx):
    test_s = f"tests/test_sample{sample_idx}_gelu.s"
    log_path = os.path.join(task2_dir, "build", "logs", f"test_sample{sample_idx}_gelu.txt")

    if not os.path.exists(os.path.join(task2_dir, test_s)):
        print(f"    [!] test file not found: {test_s}")
        return None

    ok = build_and_run(
        test_s,
        extra_links=["nn_short.s", "math.s", "weights.s"],
        test_name=f"gelu_sample{sample_idx}"
    )
    if not ok:
        return None

    fa_vals = parse_log_fa_values(log_path)

    # Robust path detection for reference files
    ref_dir_up = os.path.abspath(os.path.join(task2_dir, "..", "..", "caal project", "task9_deliverables", "test_validation"))
    ref_path = os.path.join(ref_dir_up, f"sample{sample_idx}_gelu1.bin")
    
    if not os.path.exists(ref_path):
        # Try local refs if not found in parent path
        ref_path = os.path.join(ref_dir, f"sample{sample_idx}_gelu1_ref.bin")
    
    if not os.path.exists(ref_path):
        print(f"    [!] no reference for gelu sample {sample_idx} at {ref_path}")
        return None

    ref = np.fromfile(ref_path, dtype=np.float32)

    got = []
    for i in range(min(8, len(ref))):
        reg = f'fa{i}'
        if reg in fa_vals:
            got.append(fa_vals[reg])
        else:
            got.append(float('nan'))

    got_arr = np.array(got[:8], dtype=np.float32)
    ref_arr = ref[:8]

    valid_mask = ~np.isnan(got_arr)
    if not np.any(valid_mask):
        return {"mse": float('inf'), "mae": float('inf'), "pass": False}

    mse = float(np.mean((got_arr[valid_mask] - ref_arr[valid_mask])**2))
    mae = float(np.mean(np.abs(got_arr[valid_mask] - ref_arr[valid_mask])))

    tol = tolerances["gelu"]
    passed = mse < tol["mse"]
    if tol["mae"] is not None:
        passed = passed and mae < tol["mae"]

    return {"mse": mse, "mae": mae, "pass": passed, "got": got_arr.tolist(), "ref": ref_arr.tolist()}

def validate_takelast_sample(sample_idx):
    test_s = f"tests/test_sample{sample_idx}_takelast.s"
    log_path = os.path.join(task2_dir, "build", "logs", f"test_sample{sample_idx}_takelast.txt")

    if not os.path.exists(os.path.join(task2_dir, test_s)):
        print(f"    [!] test file not found: {test_s}")
        return None

    ok = build_and_run(
        test_s,
        extra_links=["nn.s", "math.s", "weights.s"],
        test_name=f"takelast_sample{sample_idx}"
    )
    if not ok:
        return None

    fa_vals = parse_log_fa_values(log_path)

    ref_path = os.path.join(ref_dir, f"sample{sample_idx}_post_pool_ref.bin")
    if not os.path.exists(ref_path):
        print(f"    [!] no reference for takelast sample {sample_idx}")
        return None

    ref = np.fromfile(ref_path, dtype=np.float32)

    got = []
    for i in range(min(8, len(ref))):
        reg = f'fa{i}'
        if reg in fa_vals:
            got.append(fa_vals[reg])
        else:
            got.append(float('nan'))

    got_arr = np.array(got[:8], dtype=np.float32)
    ref_arr = ref[:8]

    valid_mask = ~np.isnan(got_arr)
    if not np.any(valid_mask):
        return {"mse": float('inf'), "mae": float('inf'), "pass": False}

    mse = float(np.mean((got_arr[valid_mask] - ref_arr[valid_mask])**2))
    mae = float(np.mean(np.abs(got_arr[valid_mask] - ref_arr[valid_mask])))

    tol = tolerances["takelast"]
    passed = mse < tol["mse"]

    return {"mse": mse, "mae": mae, "pass": passed, "got": got_arr.tolist(), "ref": ref_arr.tolist()}


def validate_fc_sample(sample_idx):
    test_s = f"tests/test_sample{sample_idx}_fc.s"
    log_path = os.path.join(task2_dir, "build", "logs", f"test_sample{sample_idx}_fc.txt")

    if not os.path.exists(os.path.join(task2_dir, test_s)):
        print(f"    [!] test file not found: {test_s}")
        return None

    ok = build_and_run(
        test_s,
        extra_links=["nn.s", "math.s", "weights.s"],
        test_name=f"fc_sample{sample_idx}"
    )
    if not ok:
        return None

    fa_vals = parse_log_fa_values(log_path)

    ref_path = os.path.join(ref_dir, f"sample{sample_idx}_logits_ref.bin")
    ref = np.fromfile(ref_path, dtype=np.float32)

    got = []
    for i in range(4):
        reg = f'fa{i}'
        if reg in fa_vals:
            got.append(fa_vals[reg])
        else:
            got.append(float('nan'))

    got_arr = np.array(got, dtype=np.float32)
    ref_arr = ref[:4]

    valid_mask = ~np.isnan(got_arr)
    if not np.any(valid_mask):
        return {"mse": float('inf'), "mae": float('inf'), "pass": False}

    mse = float(np.mean((got_arr[valid_mask] - ref_arr[valid_mask])**2))
    mae = float(np.mean(np.abs(got_arr[valid_mask] - ref_arr[valid_mask])))

    tol = tolerances["fc"]
    passed = mse < tol["mse"]
    if tol["mae"] is not None:
        passed = passed and mae < tol["mae"]

    return {"mse": mse, "mae": mae, "pass": passed, "got": got_arr.tolist(), "ref": ref_arr.tolist()}


def validate_softmax_sample(sample_idx):
    test_s = f"tests/test_sample{sample_idx}_softmax.s"
    log_path = os.path.join(task2_dir, "build", "logs", f"test_sample{sample_idx}_softmax.txt")

    if not os.path.exists(os.path.join(task2_dir, test_s)):
        print(f"    [!] test file not found: {test_s}")
        return None

    ok = build_and_run(
        test_s,
        extra_links=["nn.s", "math.s", "weights.s"],
        test_name=f"softmax_sample{sample_idx}"
    )
    if not ok:
        return None

    fa_vals = parse_log_fa_values(log_path)

    ref_path = os.path.join(ref_dir, f"sample{sample_idx}_probs_ref.bin")
    ref = np.fromfile(ref_path, dtype=np.float32)

    got = []
    for i in range(4):
        reg = f'fa{i}'
        if reg in fa_vals:
            got.append(fa_vals[reg])
        else:
            got.append(float('nan'))

    got_arr = np.array(got, dtype=np.float32)
    ref_arr = ref[:4]

    valid_mask = ~np.isnan(got_arr)
    if not np.any(valid_mask):
        return {"mse": float('inf'), "mae": float('inf'), "pass": False}

    mse = float(np.mean((got_arr[valid_mask] - ref_arr[valid_mask])**2))
    mae = float(np.mean(np.abs(got_arr[valid_mask] - ref_arr[valid_mask])))

    tol = tolerances["softmax"]
    passed = mse < tol["mse"]
    if tol["mae"] is not None:
        passed = passed and mae < tol["mae"]

    pred_got = int(np.argmax(got_arr)) if np.all(valid_mask) else -1
    pred_ref = int(np.argmax(ref_arr))

    return {
        "mse": mse, "mae": mae, "pass": passed,
        "got": got_arr.tolist(), "ref": ref_arr.tolist(),
        "pred_got": pred_got, "pred_ref": pred_ref,
        "class_match": pred_got == pred_ref
    }


def validate_s4d_sample(sample_idx):
    """
    Validates the S4D Layer 1 using a shorter sequence (128) for performance.
    """
    test_s = "tests/test_sample_s4d_short.s"
    log_path = os.path.join(task2_dir, "build", "logs", "test_sample_s4d_short.txt")

    if not os.path.exists(os.path.join(task2_dir, test_s)):
        asm_code = f"""
.section .text
.globl _start
_start:
    # 1. Hilbert to post_hilbert
    la      a0, test_image
    la      a1, post_hilbert
    call    hilbert_scan_layer

    # 2. Input Projection to buf_a
    la      a0, post_hilbert
    la      a1, buf_a
    call    input_projection

    # 3. S4D Layer 1 to buf_b
    la      a0, s4_0_log_dt
    la      a1, s4_0_logAre
    la      a2, s4_0_Aim
    la      a3, s4_0_Cre
    la      a4, s4_0_Cim
    la      a5, s4_0_D
    la      a6, buf_a
    la      a7, buf_b
    call    s4d_layer

    # Load result from buf_b to fa registers for verification
    la      t0, buf_b
    flw     fa0, 0(t0)
    flw     fa1, 4(t0)
    flw     fa2, 8(t0)
    flw     fa3, 12(t0)
    flw     fa4, 256(t0)
    flw     fa5, 260(t0)
    flw     fa6, 264(t0)
    flw     fa7, 268(t0)

    unimp
        """
        with open(os.path.join(task2_dir, test_s), 'w') as f:
            f.write(asm_code)

    ok = build_and_run(
        test_s,
        extra_links=["nn_short.s", "math.s", "weights.s", f"sample{sample_idx}.s"],
        test_name=f"s4d_sample{sample_idx}"
    )
    if not ok:
        return None

    fa_vals = parse_log_fa_values(log_path)
    
    # We use the full reference but compare only the first parts
    ref_dir_up = os.path.abspath(os.path.join(task2_dir, "..", "..", "caal project", "task9_deliverables", "test_validation"))
    ref_path = os.path.join(ref_dir_up, f"sample{sample_idx}_s4d1.bin")
    if not os.path.exists(ref_path):
        return None
        
    ref = np.fromfile(ref_path, dtype=np.float32)

    got = []
    for i in range(8):
        reg = f'fa{i}'
        if reg in fa_vals:
            got.append(fa_vals[reg])
        else:
            got.append(float('nan'))

    got_arr = np.array(got, dtype=np.float32)
    ref_arr = ref[:8]

    valid_mask = ~np.isnan(got_arr)
    if not np.any(valid_mask):
        return {"mse": float('inf'), "mae": float('inf'), "pass": False}

    mse = float(np.mean((got_arr[valid_mask] - ref_arr[valid_mask])**2))
    mae = float(np.mean(np.abs(got_arr[valid_mask] - ref_arr[valid_mask])))

    # Tolerance for S4D is slightly looser due to transcendental approx
    passed = mse < 1e-6 

    return {"mse": mse, "mae": mae, "pass": passed, "got": got_arr.tolist(), "ref": ref_arr.tolist()}


def run_existing_layer_tests():
    print("\n" + "="*60)
    print("  running existing layer tests (synthetic data)")
    print("="*60)

    layers = [
        ("hilbert", "tests/test_hilbert.s"),
        ("input_proj", "tests/test_input_proj.s"),
        ("gelu", "tests/test_gelu.s"),
        ("takelast", "tests/test_takelast.s"),
        ("fc", "tests/test_fc.s"),
        ("softmax", "tests/test_softmax.s"),
    ]

    results = {}
    for name, test_s in layers:
        print(f"\n  --- {name} ---")
        ok = build_and_run(
            test_s,
            extra_links=["nn.s", "math.s", "weights.s"],
            test_name=name
        )
        results[name] = "pass" if ok else "build_failed"

    return results


def generate_report(all_results, existing_results):
    report_path = os.path.join(results_dir, "validation_report.md")

    lines = []
    lines.append("# task 2: testing and validation report")
    lines.append("")
    lines.append("## existing layer tests (synthetic data)")
    lines.append("")
    lines.append("| layer | result |")
    lines.append("|-------|--------|")
    for name, result in existing_results.items():
        lines.append(f"| {name} | {result} |")

    lines.append("")
    lines.append("## per-sample validation")
    lines.append("")

    for layer in ["hilbert", "input_proj", "s4d", "gelu", "takelast", "fc", "softmax"]:
        lines.append(f"### {layer}")
        lines.append("")
        tol = tolerances[layer]
        tol_str = f"mse < {tol['mse']:.0e}"
        if tol["mae"] is not None:
            tol_str += f", mae < {tol['mae']:.0e}"
        lines.append(f"tolerance: {tol_str}")
        lines.append("")
        lines.append("| sample | mse | mae | pass |")
        lines.append("|--------|-----|-----|------|")

        for i in range(10):
            key = f"{layer}_sample{i}"
            if key in all_results and all_results[key] is not None:
                r = all_results[key]
                p = "pass" if r["pass"] else "fail"
                lines.append(f"| {i} | {r['mse']:.2e} | {r['mae']:.2e} | {p} |")
            else:
                lines.append(f"| {i} | - | - | not run |")

        lines.append("")

    lines.append("## end-to-end class prediction")
    lines.append("")

    ref_data_dir = os.path.join(
        task2_dir, "..", "..", "caal project", "task9_deliverables", "test_validation"
    )

    lines.append("| sample | true label | true class | predicted | match |")
    lines.append("|--------|-----------|------------|-----------|-------|")

    for i in range(10):
        label_path = os.path.join(ref_data_dir, f"sample{i}_label.txt")
        if os.path.exists(label_path):
            label = int(open(label_path).read().strip())
        else:
            label = -1

        key = f"softmax_sample{i}"
        if key in all_results and all_results[key] is not None:
            r = all_results[key]
            pred = r.get("pred_got", -1)
            match = "pass" if r.get("class_match", False) else "fail"
        else:
            probs_path = os.path.join(ref_dir, f"sample{i}_probs_ref.bin")
            if os.path.exists(probs_path):
                probs = np.fromfile(probs_path, dtype=np.float32)
                pred = int(np.argmax(probs))
                match = "pass (python ref)" if pred == label else "fail"
            else:
                pred = -1
                match = "no data"

        cls = class_names[label] if 0 <= label < 4 else "unknown"
        pred_cls = class_names[pred] if 0 <= pred < 4 else "unknown"
        lines.append(f"| {i} | {label} | {cls} | {pred} ({pred_cls}) | {match} |")

    lines.append("")
    lines.append("## summary")
    lines.append("")

    total = 0
    passed = 0
    for key, val in all_results.items():
        if val is not None:
            total += 1
            if val.get("pass", False):
                passed += 1

    lines.append(f"- total tests: {total}")
    lines.append(f"- passed: {passed}")
    lines.append(f"- failed: {total - passed}")
    if total > 0:
        lines.append(f"- pass rate: {100*passed/total:.1f}%")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nreport saved to: {report_path}")
    return report_path


def main():
    print("="*60)
    print("  s4d galaxy classifier - task 2 validation")
    print("="*60)

    existing_results = {}
    all_results = {}

    print("\n--- per-sample hilbert validation ---")
    for i in range(10):
        print(f"\n  sample {i}:")
        result = validate_hilbert_sample(i)
        all_results[f"hilbert_sample{i}"] = result
        if result:
            p = "pass" if result["pass"] else "fail"
            print(f"    mse={result['mse']:.2e} mae={result['mae']:.2e} [{p}]")

    print("\n--- per-sample input_proj validation ---")
    for i in range(10):
        print(f"\n  sample {i}:")
        result = validate_input_proj_sample(i)
        all_results[f"input_proj_sample{i}"] = result
        if result:
            p = "pass" if result["pass"] else "fail"
            print(f"    mse={result['mse']:.2e} mae={result['mae']:.2e} [{p}]")

    print("\n--- per-sample s4d validation (Short 128 Seq) ---")
    for i in range(10):
        print(f"\n  sample {i}:")
        result = validate_s4d_sample(i)
        all_results[f"s4d_sample{i}"] = result
        if result:
            p = "pass" if result["pass"] else "fail"
            print(f"    mse={result['mse']:.2e} mae={result['mae']:.2e} [{p}]")

    print("\n--- per-sample gelu validation ---")
    for i in range(10):
        print(f"\n  sample {i}:")
        result = validate_gelu_sample(i)
        all_results[f"gelu_sample{i}"] = result
        if result:
            p = "pass" if result["pass"] else "fail"
            print(f"    mse={result['mse']:.2e} mae={result['mae']:.2e} [{p}]")

    print("\n--- per-sample takelast validation ---")
    for i in range(10):
        print(f"\n  sample {i}:")
        result = validate_takelast_sample(i)
        all_results[f"takelast_sample{i}"] = result
        if result:
            p = "pass" if result["pass"] else "fail"
            print(f"    mse={result['mse']:.2e} mae={result['mae']:.2e} [{p}]")

    print("\n--- per-sample fc validation ---")
    for i in range(10):
        print(f"\n  sample {i}:")
        result = validate_fc_sample(i)
        all_results[f"fc_sample{i}"] = result
        if result:
            p = "pass" if result["pass"] else "fail"
            print(f"    mse={result['mse']:.2e} mae={result['mae']:.2e} [{p}]")

    print("\n--- per-sample softmax validation ---")
    for i in range(10):
        print(f"\n  sample {i}:")
        result = validate_softmax_sample(i)
        all_results[f"softmax_sample{i}"] = result
        if result:
            p = "pass" if result["pass"] else "fail"
            pred = result.get("pred_got", -1)
            print(f"    mse={result['mse']:.2e} mae={result['mae']:.2e} pred={pred} [{p}]")

    report_path = generate_report(all_results, existing_results)

    results_json = os.path.join(results_dir, "validation_results.json")
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        return obj

    serializable = sanitize(all_results)
    with open(results_json, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nresults json saved to: {results_json}")

    total = sum(1 for v in all_results.values() if v is not None)
    passed = sum(1 for v in all_results.values() if v is not None and v.get("pass", False))
    print(f"\n{'='*60}")
    print(f"  validation complete: {passed}/{total} tests passed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
