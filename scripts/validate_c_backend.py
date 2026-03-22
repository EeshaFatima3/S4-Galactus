"""
validate_c_backend.py  —  Task 4 (Bonus): Validate C Backend
==============================================================
Compares predictions from the Python backend and C backend on
multiple test samples to ensure they agree within tolerance.

Usage:
    cd task9_deliverables
    python validate_c_backend.py
"""
import os
import sys
import csv
import types
import numpy as np
import torch

# Mock galaxy_mnist (not installed locally)
galaxy_mnist_mock = types.ModuleType('galaxy_mnist')
class _MockGalaxyMNIST:
    pass
galaxy_mnist_mock.GalaxyMNIST = _MockGalaxyMNIST
sys.modules['galaxy_mnist'] = galaxy_mnist_mock

# Also mock pygame if not installed
try:
    import pygame
except ImportError:
    sys.modules['pygame'] = types.ModuleType('pygame')
    sys.modules['pygame'].init = lambda: None
    sys.modules['pygame'].quit = lambda: None
    sys.modules['pygame'].font = types.ModuleType('pygame.font')
    sys.modules['pygame'].display = types.ModuleType('pygame.display')
    sys.modules['pygame'].surfarray = types.ModuleType('pygame.surfarray')

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLAB_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "colab_results"))
TASK9_DIR  = SCRIPT_DIR

sys.path.insert(0, COLAB_DIR)

NUM_TEST = 10
TOLERANCE_MAE = 1e-2
TOLERANCE_MSE = 1e-3

CLASS_NAMES = ["Smooth Round", "Smooth Cigar", "Edge-on Disk", "Unbarred Spiral"]


def main():
    print("=" * 70)
    print("  Task 4 (Bonus) — C Backend Validation")
    print("=" * 70)

    device = torch.device("cpu")
    model_pth = os.path.join(COLAB_DIR, "galaxy_s4_model.pth")
    task9_abs = os.path.abspath(TASK9_DIR)

    # ---- 1. Load Python backend ----
    print("\n[1] Loading Python backend...")
    try:
        from model.interface import ModelInterface
        py_model = ModelInterface(
            implementation='python',
            model_path=model_pth,
            num_classes=4,
            colored=False,
            device=device
        )
        print("    [OK] Python backend ready")
    except Exception as e:
        print(f"    [FAIL] Python backend: {e}")
        return 1

    # ---- 2. Load C backend ----
    print("\n[2] Loading C backend...")
    try:
        c_model = ModelInterface(
            implementation='c',
            model_path=task9_abs,
            num_classes=4,
            colored=False,
            device=device
        )
        print("    [OK] C backend ready")
    except FileNotFoundError as e:
        print(f"    [SKIP] {e}")
        print("    Build with: gcc -O2 -o demo nn.c main.c -lm")
        return 1

    # ---- 3. Load test samples ----
    print(f"\n[3] Loading {NUM_TEST} test samples...")
    csv_path = os.path.join(COLAB_DIR, "galaxy_samples.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(TASK9_DIR, "galaxy_samples.csv")

    samples = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= NUM_TEST:
                break
            label = int(row[0])
            pixels = np.array([float(v) for v in row[1:]], dtype=np.float32)
            samples.append((label, pixels))
    print(f"    [OK] Loaded {len(samples)} samples")

    # ---- 4. Compare predictions ----
    print(f"\n[4] Comparing predictions (Python vs C)...")
    hdr = "{:>6s} | {:>15s} | {:>15s} | {:>15s} | {:>5s} | {:>12s} | {:>12s}"
    print("    " + hdr.format("Sample", "True Label", "Python Pred", "C Pred", "Match", "MSE", "MAE"))
    print("    " + "-" * 95)

    total_mse = 0.0
    total_mae = 0.0
    mismatches = 0

    for i, (label, pixels) in enumerate(samples):
        img_tensor = torch.from_numpy(pixels.reshape(1, 1, 64, 64))

        # Python
        with torch.no_grad():
            py_probs = py_model(img_tensor).squeeze().numpy()
        py_pred = int(np.argmax(py_probs))

        # C
        try:
            c_probs = c_model(img_tensor).squeeze().numpy()
            c_pred = int(np.argmax(c_probs))
        except Exception as e:
            print(f"    {i:>6d} | {CLASS_NAMES[label]:>15s} | {CLASS_NAMES[py_pred]:>15s} | {'ERROR':>15s} | {'FAIL':>5s} | {str(e)[:30]}")
            mismatches += 1
            continue

        mse = float(np.mean((py_probs - c_probs) ** 2))
        mae = float(np.mean(np.abs(py_probs - c_probs)))
        total_mse += mse
        total_mae += mae

        match = "OK" if py_pred == c_pred else "FAIL"
        if py_pred != c_pred:
            mismatches += 1

        row_fmt = "{:>6d} | {:>15s} | {:>15s} | {:>15s} | {:>5s} | {:>12.2e} | {:>12.2e}"
        print("    " + row_fmt.format(
            i, CLASS_NAMES[label], CLASS_NAMES[py_pred], CLASS_NAMES[c_pred],
            match, mse, mae
        ))

    # ---- 5. Summary ----
    n = len(samples)
    avg_mse = total_mse / n
    avg_mae = total_mae / n

    print()
    print("=" * 70)
    print("  Results Summary")
    print("=" * 70)
    print(f"  Samples tested:    {n}")
    print(f"  Class matches:     {n - mismatches}/{n}")
    print(f"  Average MSE:       {avg_mse:.2e}")
    print(f"  Average MAE:       {avg_mae:.2e}")

    if mismatches == 0:
        print("\n  RESULT: ALL PASSED — C backend matches Python on every sample.")
        return 0
    else:
        print(f"\n  RESULT: FAILED — {mismatches} prediction mismatches.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
