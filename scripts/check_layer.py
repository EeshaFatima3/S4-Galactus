"""
check_layer.py  —  Universal layer checker for VeeR-ISS logs
=============================================================
Replaces all individual check_*.py files.

VeeR-ISS log format:
  Scalar load:  #N 0 ADDR ENC r RR    HEXVAL   instruction
  Float  load:  #N 0 ADDR ENC f RR    HEXVAL   flw faX, ...
  Float  store: #N 0 ADDR ENC m ADDR  HEXVAL   fsw ...

  Where RR is a 2-hex-digit register index:
    fa0=0a  fa1=0b  fa2=0c  fa3=0d  fa4=0e  fa5=0f  fa6=10  fa7=11

Usage:
    python3 tests/check_layer.py <layer>

    layer = hilbert | input_proj | gelu | takelast | fc | softmax
"""

import struct, re, sys, numpy as np, os

# ── Register index → name mapping ────────────────────────────────
# VeeR-ISS uses hex register indices in the log
# Float regs: f0=ft0 f1=ft1 ... f10=fa0 f11=fa1 ... f18=fs0 ...
# fa0-fa7 = architectural float arg regs = indices 0x0a..0x11
FA_IDX = {f'0{i+10:x}': f'fa{i}' for i in range(8)}   # 0a→fa0 .. 0h→fa7
# Also handle without leading zero if needed
FA_IDX.update({f'{i+10:x}': f'fa{i}' for i in range(8)})

def hex_to_float(h):
    return struct.unpack('<f', struct.pack('<I', int(h, 16)))[0]

def parse_log(log_path):
    """
    Returns:
      flw_fa  : list of (reg_name, value) for every flw into fa0-fa7
      fsw_vals: list of float values from every fsw instruction
    """
    # Pattern for float loads:
    # #N 0 ADDR ENC f RR HEXVAL  flw ...
    flw_pat = re.compile(
        r'#\d+\s+\d+\s+\w+\s+\w+\s+f\s+([0-9a-f]{2})\s+([0-9a-f]{8})\s+(?:c\.)?flw'
    )
    # Pattern for float stores:
    # #N 0 ADDR ENC m MEMADDR HEXVAL  fsw ...
    fsw_pat = re.compile(
        r'#\d+\s+\d+\s+\w+\s+\w+\s+m\s+\w+\s+([0-9a-f]{8})\s+fsw'
    )

    flw_fa   = []   # (reg_name, value) only for fa0..fa7
    fsw_vals = []   # all stored float values in order

    with open(log_path) as f:
        for line in f:
            # float loads
            m = flw_pat.search(line)
            if m:
                reg_idx = m.group(1)
                hexval  = m.group(2)
                val     = hex_to_float('0x' + hexval)
                if reg_idx in FA_IDX:
                    flw_fa.append((FA_IDX[reg_idx], val))

            # float stores
            m = fsw_pat.search(line)
            if m:
                fsw_vals.append(hex_to_float('0x' + m.group(1)))

    return flw_fa, fsw_vals

def last_fa_values(flw_fa):
    """Get the LAST value loaded into each fa register."""
    result = {}
    for reg, val in flw_fa:
        result[reg] = val
    return result

# ════════════════════════════════════════════════════════════════
# Layer checkers
# ════════════════════════════════════════════════════════════════

def check_hilbert():
    log  = 'build/logs/test_hilbert.txt'
    ref  = np.fromfile('testdata/ref_hilbert.bin', dtype=np.float32)
    flw_fa, fsw_vals = parse_log(log)

    print("=" * 58)
    print("Hilbert Scan — Spot Check (fa0..fa7 = out[0..7])")
    print("=" * 58)
    fa_vals = last_fa_values(flw_fa)
    all_pass = True
    for i in range(8):
        reg = f'fa{i}'
        got = fa_vals.get(reg)
        exp = float(ref[i])
        if got is None:
            print(f"  [FAIL] out[{i}] — {reg} not found in log"); all_pass=False; continue
        err = abs(got - exp)
        ok  = err < 1e-6        # hilbert is indexing — should be near-exact
        print(f"  [{'PASS' if ok else 'FAIL'}] out[{i}]  got={got:.8f}  exp={exp:.8f}  err={err:.2e}")
        if not ok: all_pass = False

    print()
    # Full array: ALL fsw entries are from hilbert (4096 stores)
    got_arr = np.array(fsw_vals[4096:8192], dtype=np.float32)
    ref_arr = ref[:len(got_arr)]
    mse = float(np.mean((got_arr-ref_arr)**2))
    mae = float(np.mean(np.abs(got_arr-ref_arr)))
    ok  = mse < 1e-12
    print(f"  Full array: {len(got_arr)} elements")
    print(f"  MSE={mse:.2e} (threshold 1e-12)  {'OK' if ok else 'FAIL'}")
    print(f"  MAE={mae:.2e}")
    if not ok: all_pass = False

    print()
    print(">>> HILBERT: " + ("ALL PASSED ✓" if all_pass else "FAILED ✗"))
    return all_pass

def check_input_proj():
    log  = 'build/logs/test_input_proj.txt'
    ref  = np.fromfile('testdata/ref_input_proj.bin', dtype=np.float32)
    flw_fa, fsw_vals = parse_log(log)

    print("=" * 58)
    print("Input Projection — Spot Check (buf_a[0][0..3])")
    print("=" * 58)
    # Expected: buf_a[0][h] = post_hilbert[0]*up_w[h] + up_b[h]
    # post_hilbert[0] = 0.0, so buf_a[0][h] = up_b[h]
    fa_vals = last_fa_values(flw_fa)
    ref_spot = ref[:4]
    all_pass = True
    for i in range(4):
        reg = f'fa{i}'
        got = fa_vals.get(reg)
        exp = float(ref_spot[i])
        if got is None:
            print(f"  [FAIL] buf_a[0][{i}] — not found"); all_pass=False; continue
        err = abs(got - exp)
        ok  = err < 1e-6
        print(f"  [{'PASS' if ok else 'FAIL'}] buf_a[0][{i}]  got={got:.8f}  exp={exp:.8f}  err={err:.2e}")
        if not ok: all_pass = False

    print()
    # Full array: first 4096 fsw = hilbert, next 262144 = projection
    proj_vals = fsw_vals[4096+4096:4096+4096+262144]
    print(f"  Total fsw in log : {len(fsw_vals)}")
    print(f"  Projection stores: {len(proj_vals)} (need 262144)")
    if len(proj_vals) < 262144:
        print(f"  [!] Not enough stores — got {len(proj_vals)}")
        all_pass = False
    else:
        got_arr = np.array(proj_vals[:262144], dtype=np.float32)
        ref_arr = ref[:262144]
        mse = float(np.mean((got_arr-ref_arr)**2))
        mae = float(np.mean(np.abs(got_arr-ref_arr)))
        ok_mse = mse < 1e-8
        ok_mae = mae < 1e-6
        print(f"  MSE={mse:.2e} (threshold 1e-08)  {'OK' if ok_mse else 'FAIL'}")
        print(f"  MAE={mae:.2e} (threshold 1e-06)  {'OK' if ok_mae else 'FAIL'}")
        if not (ok_mse and ok_mae): all_pass = False

    print()
    print(">>> INPUT PROJECTION: " + ("ALL PASSED ✓" if all_pass else "FAILED ✗"))
    return all_pass

def check_gelu():
    log = 'build/logs/test_gelu.txt'
    flw_fa, fsw_vals = parse_log(log)

    INPUTS   = [0.0,  1.0,      -1.0,       2.0,       0.5      ]
    EXPECTED = [0.0,  0.841192, -0.158808,  1.954598,  0.345714 ]

    print("=" * 60)
    print("GELU — 5 Known Inputs (fa0=0.0, fa1=1.0, fa2=-1.0, ...)")
    print("=" * 60)
    fa_vals  = last_fa_values(flw_fa)
    all_pass = True
    got_list = []
    for i in range(5):
        reg = f'fa{i}'
        got = fa_vals.get(reg)
        exp = EXPECTED[i]
        if got is None:
            print(f"  [FAIL] fa{i} (input={INPUTS[i]}) — not found in log")
            all_pass = False; got_list.append(0.0); continue
        got_list.append(got)
        err = abs(got - exp)
        ok  = err < 1e-4
        print(f"  [{'PASS' if ok else 'FAIL'}] input={INPUTS[i]:6.2f}  got={got:.6f}  exp={exp:.6f}  err={err:.2e}")
        if not ok: all_pass = False

    got_arr = np.array(got_list); exp_arr = np.array(EXPECTED)
    mse = float(np.mean((got_arr-exp_arr)**2))
    mae = float(np.mean(np.abs(got_arr-exp_arr)))
    print(f"\n  MSE={mse:.2e} (threshold 1e-07)  {'OK' if mse<1e-7 else 'FAIL'}")
    print(f"  MAE={mae:.2e} (threshold 1e-04)  {'OK' if mae<1e-4 else 'FAIL'}")
    if mse >= 1e-7 or mae >= 1e-4: all_pass = False

    print()
    print(">>> GELU: " + ("ALL PASSED ✓" if all_pass else "FAILED ✗"))
    return all_pass

def check_takelast():
    log = 'build/logs/test_takelast.txt'
    flw_fa, _ = parse_log(log)

    EXPECTED = 4095.0
    print("=" * 50)
    print("TakeLastTimestep — post_pool[0..7] = 4095.0")
    print("=" * 50)
    fa_vals  = last_fa_values(flw_fa)
    all_pass = True
    for i in range(8):
        reg = f'fa{i}'
        got = fa_vals.get(reg)
        if got is None:
            print(f"  [FAIL] post_pool[{i}] — not found"); all_pass=False; continue
        err = abs(got - EXPECTED)
        ok  = err < 1e-3
        print(f"  [{'PASS' if ok else 'FAIL'}] post_pool[{i}]  got={got:.1f}  exp={EXPECTED:.1f}  err={err:.2e}")
        if not ok: all_pass = False

    print()
    print(">>> TAKELAST: " + ("ALL PASSED ✓" if all_pass else "FAILED ✗"))
    return all_pass

def check_fc():
    log = 'build/logs/test_fc.txt'
    flw_fa, _ = parse_log(log)

    EXPECTED = [0.6658437252, -0.8043879867, 2.2583341599, -3.0594279766]
    print("=" * 60)
    print("FC Layer — logits[0..3] (input=all 1.0)")
    print("=" * 60)
    fa_vals  = last_fa_values(flw_fa)
    all_pass = True
    got_list = []
    for i in range(4):
        reg = f'fa{i}'
        got = fa_vals.get(reg)
        exp = EXPECTED[i]
        if got is None:
            print(f"  [FAIL] logits[{i}] — not found"); all_pass=False; got_list.append(0.0); continue
        got_list.append(got)
        err = abs(got - exp)
        ok  = err < 1e-4
        print(f"  [{'PASS' if ok else 'FAIL'}] logits[{i}]  got={got:.7f}  exp={exp:.7f}  err={err:.2e}")
        if not ok: all_pass = False

    got_arr = np.array(got_list); exp_arr = np.array(EXPECTED)
    mse = float(np.mean((got_arr-exp_arr)**2))
    mae = float(np.mean(np.abs(got_arr-exp_arr)))
    print(f"\n  MSE={mse:.2e} (threshold 1e-08)  {'OK' if mse<1e-8 else 'FAIL'}")
    print(f"  MAE={mae:.2e} (threshold 1e-06)  {'OK' if mae<1e-6 else 'FAIL'}")
    if mse >= 1e-8 or mae >= 1e-6: all_pass = False

    print()
    print(">>> FC LAYER: " + ("ALL PASSED ✓" if all_pass else "FAILED ✗"))
    return all_pass

def check_softmax():
    log = 'build/logs/test_softmax.txt'
    flw_fa, _ = parse_log(log)

    EXPECTED = [0.16207573, 0.03725671, 0.79676050, 0.00390709]
    print("=" * 60)
    print("Softmax — probs[0..3]")
    print("=" * 60)
    fa_vals  = last_fa_values(flw_fa)
    all_pass = True
    got_list = []
    prob_sum = 0.0
    for i in range(4):
        reg = f'fa{i}'
        got = fa_vals.get(reg)
        exp = EXPECTED[i]
        if got is None:
            print(f"  [FAIL] probs[{i}] — not found"); all_pass=False; got_list.append(0.0); continue
        got_list.append(got)
        prob_sum += got
        err = abs(got - exp)
        ok  = err < 1e-4
        print(f"  [{'PASS' if ok else 'FAIL'}] probs[{i}]  got={got:.7f}  exp={exp:.7f}  err={err:.2e}")
        if not ok: all_pass = False

    got_arr = np.array(got_list); exp_arr = np.array(EXPECTED)
    mse = float(np.mean((got_arr-exp_arr)**2))
    mae = float(np.mean(np.abs(got_arr-exp_arr)))
    print(f"\n  Sum of probs : {prob_sum:.8f}  (should be ~1.0)")
    print(f"  MSE={mse:.2e} (threshold 1e-08)  {'OK' if mse<1e-8 else 'FAIL'}")
    print(f"  MAE={mae:.2e} (threshold 1e-04)  {'OK' if mae<1e-4 else 'FAIL'}")
    if abs(prob_sum-1.0) > 1e-4: print("  [FAIL] probs don't sum to 1"); all_pass=False
    if mse >= 1e-8 or mae >= 1e-4: all_pass = False

    print()
    print(">>> SOFTMAX: " + ("ALL PASSED ✓" if all_pass else "FAILED ✗"))
    return all_pass

# ════════════════════════════════════════════════════════════════
LAYERS = {
    'hilbert':    check_hilbert,
    'input_proj': check_input_proj,
    'gelu':       check_gelu,
    'takelast':   check_takelast,
    'fc':         check_fc,
    'softmax':    check_softmax,
}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in LAYERS:
        print(f"Usage: python3 tests/check_layer.py <layer>")
        print(f"Layers: {', '.join(LAYERS)}")
        sys.exit(1)
    ok = LAYERS[sys.argv[1]]()
    sys.exit(0 if ok else 1)
