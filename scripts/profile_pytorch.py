"""
profile_pytorch.py  —  PyTorch Baseline Timing for Task 3.3
===========================================================
Runs the same inference 100 times through the PyTorch model and records
per-layer and total timing. Outputs to pytorch_timing.csv for comparison
with C benchmark results.

Usage:
    cd task9_deliverables
    python profile_pytorch.py
"""
import os, sys, time, csv
import numpy as np
import torch
import torch.nn as nn

# --- Model setup (same as other scripts) ---
MODEL_DIR = os.path.abspath("../colab_results/model")
sys.path.insert(0, MODEL_DIR)
from hilbert import HilbertScan
from tlts   import TakeLastTimestep
from s4d    import S4D

class GalaxyClassifierS4D(nn.Module):
    def __init__(self):
        super().__init__()
        self.hilbert_scan = HilbertScan()
        self.uproject  = nn.Linear(1, 64)
        self.s4_1      = S4D(d_model=64, d_state=64, transposed=False)
        self.act1      = nn.GELU(approximate='tanh')
        self.s4_2      = S4D(d_model=64, d_state=64, transposed=False)
        self.act2      = nn.GELU(approximate='tanh')
        self.take_last = TakeLastTimestep()
        self.fc        = nn.Linear(64, 4)
        self.softmax   = nn.Softmax(dim=-1)

NUM_RUNS = 100
STAGE_NAMES = [
    "Hilbert Scan",
    "Input Projection",
    "S4D Layer 1",
    "GELU 1",
    "S4D Layer 2",
    "GELU 2",
    "TakeLast+FC+Softmax"
]

def main():
    print("=" * 60)
    print("PyTorch Baseline Profiling")
    print("=" * 60)

    # Load model
    model = GalaxyClassifierS4D()
    model.load_state_dict(
        torch.load("galaxy_s4_model.pth", map_location="cpu", weights_only=True)
    )
    model.eval()
    print("[OK] Model loaded")

    # Load first sample image from CSV
    with open("galaxy_samples.csv", "r") as f:
        row = next(csv.reader(f))
    pixels = np.array([float(v) for v in row[1:]], dtype=np.float32)
    img = torch.from_numpy(pixels.reshape(1, 1, 64, 64))
    print(f"[OK] Loaded image ({pixels.shape[0]} pixels)")

    # Warm-up run
    with torch.no_grad():
        _ = model.softmax(model.fc(model.take_last(
            model.act2(model.s4_2(model.act1(model.s4_1(
                model.uproject(model.hilbert_scan(img)))[0]))[0]))))

    # Profiled runs
    print(f"\nRunning {NUM_RUNS} profiled iterations...")
    layer_times = {name: [] for name in STAGE_NAMES}
    total_times = []

    with torch.no_grad():
        for r in range(NUM_RUNS):
            t_total_start = time.perf_counter()

            # Stage 1: Hilbert Scan
            t0 = time.perf_counter()
            x = model.hilbert_scan(img)
            layer_times["Hilbert Scan"].append(time.perf_counter() - t0)

            # Stage 2: Input Projection
            t0 = time.perf_counter()
            x = model.uproject(x)
            layer_times["Input Projection"].append(time.perf_counter() - t0)

            # Stage 3: S4D Layer 1
            t0 = time.perf_counter()
            x, _ = model.s4_1(x)
            layer_times["S4D Layer 1"].append(time.perf_counter() - t0)

            # Stage 4: GELU 1
            t0 = time.perf_counter()
            x = model.act1(x)
            layer_times["GELU 1"].append(time.perf_counter() - t0)

            # Stage 5: S4D Layer 2
            t0 = time.perf_counter()
            x, _ = model.s4_2(x)
            layer_times["S4D Layer 2"].append(time.perf_counter() - t0)

            # Stage 6: GELU 2
            t0 = time.perf_counter()
            x = model.act2(x)
            layer_times["GELU 2"].append(time.perf_counter() - t0)

            # Stage 7: TakeLast + FC + Softmax
            t0 = time.perf_counter()
            x = model.take_last(x)
            logits = model.fc(x)
            probs = model.softmax(logits)
            layer_times["TakeLast+FC+Softmax"].append(time.perf_counter() - t0)

            total_times.append(time.perf_counter() - t_total_start)

    # Print results
    total_avg = np.mean(total_times) * 1000.0
    print(f"\n{'Layer':<25s} | {'Avg (ms)':>12s} | {'StdDev (ms)':>12s} | {'% Total':>10s}")
    print("-" * 65)

    for name in STAGE_NAMES:
        arr = np.array(layer_times[name]) * 1000.0
        pct = np.mean(layer_times[name]) / np.mean(total_times) * 100.0
        print(f"{name:<25s} | {np.mean(arr):12.4f} | {np.std(arr):12.6f} | {pct:9.2f}%")

    print("-" * 65)
    print(f"{'TOTAL':<25s} | {total_avg:12.4f} | {np.std(np.array(total_times)*1000):12.6f} | {'100.00%':>10s}")
    print(f"\nThroughput: {1000.0 / total_avg:.4f} images/sec")

    # Write CSV
    out_path = "pytorch_timing.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Layer", "Avg_ms", "StdDev_ms", "Percent"])
        for name in STAGE_NAMES:
            arr = np.array(layer_times[name]) * 1000.0
            pct = np.mean(layer_times[name]) / np.mean(total_times) * 100.0
            w.writerow([name, f"{np.mean(arr):.6f}", f"{np.std(arr):.6f}", f"{pct:.4f}"])
        w.writerow(["Total", f"{total_avg:.6f}", "0", "100.0"])

    print(f"\n[OK] Results written to {out_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
