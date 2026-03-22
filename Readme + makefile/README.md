# S4D Galaxy Classifier — Milestone 2: C Implementation & Numerical Validation

A bare-metal C implementation of the S4D (Structured State Space Diagonal) galaxy classifier, implementing all inference operations from scratch with no external ML dependencies. Built for deployment on RISC-V edge processors.

---

## Repository Structure

```
.
├── nn.c                      # Core inference pipeline (all layer implementations)
├── nn.h                      # Model constants, structs, and function declarations
├── main.c                    # Demo program — runs inference on a single image
├── validate.c                # Batch validation against PyTorch reference (CSV input)
├── test.c                    # Layer-by-layer numerical validation (binary reference files)
├── benchmark.c               # Performance benchmarking across 100 iterations
├── Makefile                  # Build system
├── model_weights.bin         # Binary weight file (from Milestone 1)
├── run_test.py               # Python script: compiles test.c and runs it on all samples
├── analyze_performance.py    # Task 3.3: generates all 5 performance charts
├── profile_pytorch.py        # PyTorch baseline timing for C vs Python comparison
├── model/
│   └── interface.py          # Unified model interface (Python / C / RISC-V backends)
└── launch_gui.py             # GUI launcher with backend selection (Task 4 Bonus)
```

---

## Requirements

**C Compiler (C11 support)**
```bash
gcc --version   # >= 7.0
# or
clang --version # >= 6.0
```

**Build system**
```bash
make --version
```

**Python 3.11+** (for test data generation, benchmarking, and optional GUI)
```bash
python3 --version
pip install numpy torch matplotlib
```

---

## Build

**Build all targets (default, -O2):**
```bash
make
```

This produces three executables: `main`, `validate`, and `benchmark`.

**Build individual targets:**
```bash
make main        # inference demo only
make validate    # batch CSV validation only
make benchmark   # performance benchmarking only
```

**Build at specific optimization levels:**
```bash
make O0          # no optimization
make O1          # basic optimizations
make O3          # aggressive optimizations
make Ofast       # O3 + fast math (may sacrifice IEEE compliance)
```

**Build benchmark at all optimization levels:**
```bash
make benchmark_O0
make benchmark_O1
make benchmark_O3
make benchmark_Ofast
```

**Generate assembly dumps (for Task 3.3 assembly analysis):**
```bash
make asm         # produces nn_O0.s, nn_O2.s, nn_O3.s
```

**Build with address sanitizer (for debugging):**
```bash
make sanitize
./main_asan model_weights.bin sample.bin
```

**Clean all build artifacts:**
```bash
make clean
```

---

## Usage

### Demo Program — Single Image Inference (`main`)

Accepts a binary image file (4096 floats for grayscale) and prints class probabilities.

```bash
./main <input_image.bin>
```

**Example:**
```bash
./main sample_0.bin
```

**Expected output:**
```
Loaded model weights (21124 parameters)
Running inference on: sample_0.bin

Class Probabilities:
Class 0 (Smooth Round):     0.0234
Class 1 (Smooth Cigar):     0.8543
Class 2 (Edge-on Disk):     0.1012
Class 3 (Unbarred Spiral):  0.0211

Predicted Class: 1 (Smooth Cigar)
```

> **Note:** The model weights file `model_weights.bin` must be present in the working directory.

---

### Batch Validation — CSV Input (`validate`)

Runs inference on all samples in a CSV file and reports per-sample predictions and overall accuracy.

```bash
./validate <weights_file> <csv_file>
```

**Example:**
```bash
./validate model_weights.bin galaxy_samples.csv
```

**Expected output format:**
```
=== S4D Galaxy Classifier Validation ===

Loading weights from model_weights.bin... OK
Loading samples from galaxy_samples.csv... OK (100 samples found)

Idx  True       Pred       Probabilities [Smooth Disk Edge Irreg]
------------------------------------------------------------
0    1          1          [ 0.023  0.854  0.101  0.021] PASS
1    0          0          [ 0.912  0.031  0.044  0.012] PASS
...

=========================================
VALIDATION SUMMARY
=========================================
Total samples: 100
Correct predictions: 100
Accuracy: 100.00%

SUCCESS: TASK 3.1.7 PASSED (100% accuracy)
```

---

### Layer-by-Layer Numerical Validation (`test.c` + `run_test.py`)

Validates each layer's C output against binary reference files exported from PyTorch.

**Step 1 — Generate reference data from PyTorch** (run once):
```bash
python3 generate_test_data.py
```
This creates files like `sample0_input.bin`, `sample0_input_proj.bin`, `sample0_s4d1.bin`, etc.

**Step 2 — Run validation across all samples:**
```bash
python3 run_test.py
```

**Example output:**
```
==================================================
 S4D Numerical Validation Test Suite
==================================================
[*] Compiling test suite...

UProject        | MSE:  1.23e-10 | MAE:  4.56e-08 | MaxErr:  9.12e-07 | [PASS]
S4D_1           | MSE:  3.45e-09 | MAE:  2.34e-05 | MaxErr:  8.90e-04 | [PASS]
GELU_1          | MSE:  1.67e-09 | MAE:  1.23e-05 | MaxErr:  4.56e-04 | [PASS]
S4D_2           | MSE:  4.12e-09 | MAE:  3.45e-05 | MaxErr:  9.78e-04 | [PASS]
GELU_2          | MSE:  2.34e-09 | MAE:  1.78e-05 | MaxErr:  5.67e-04 | [PASS]
Logits (FC)     | MSE:  8.90e-11 | MAE:  3.12e-08 | MaxErr:  7.45e-08 | [PASS]
100% Agreement  | C Pred: 1 | PyTorch Pred: 1 |           [PASS]
```

**Validation tolerances per layer:**

| Layer | MSE Target | MAE Target |
|---|---|---|
| Hilbert Scan / TakeLastTimestep | < 1e-12 | exact match |
| Linear (UProject, FC) | < 1e-8 | < 1e-6 |
| S4D Layers | < 1e-7 | < 1e-4 |
| GELU | < 1e-7 | < 1e-4 |
| Softmax | < 1e-8 | < 1e-4 |

---

### Performance Benchmarking (`benchmark`)

Measures per-layer and total inference time over 100 iterations.

```bash
./benchmark [model_weights.bin] [sample_input.bin] [output.csv]
```

**Example:**
```bash
./benchmark model_weights.bin sample_0.bin benchmark_results_O2.csv
```

**Run at all optimization levels for comparison:**
```bash
make benchmark_O0 benchmark_O1 benchmark_O3 benchmark_Ofast
./benchmark_O0    model_weights.bin sample_0.bin benchmark_results_O0.csv
./benchmark_O1    model_weights.bin sample_0.bin benchmark_results_O1.csv
./benchmark_O3    model_weights.bin sample_0.bin benchmark_results_O3.csv
./benchmark_Ofast model_weights.bin sample_0.bin benchmark_results_Ofast.csv
```

**Generate all 5 performance charts** (requires `benchmark_results_*.csv` files):
```bash
python3 profile_pytorch.py        # generates pytorch_timing.csv
python3 analyze_performance.py    # generates all charts as .png files
```

Charts produced:
- `timing_breakdown.png` — per-layer inference time (pie chart)
- `optimization_comparison.png` — inference time vs. optimization level
- `instruction_count.png` — instruction count vs. optimization level
- `memory_footprint.png` — memory footprint breakdown
- `c_vs_python.png` — C vs. PyTorch timing comparison

---

## Model Architecture

The forward pass chains 9 stages using only static memory buffers:

```
Input (64×64 grayscale)
  │
  ▼
[1] Hilbert Scan        (64×64) → (4096,)
  │
  ▼
[2] Input Projection    (4096, 1) → (4096, 64)     [Linear: Y = XWᵀ + b]
  │
  ▼
[3] S4D Layer 1         (4096, 64) → (4096, 64)    [SSM convolution]
  │
  ▼
[4] GELU 1              in-place tanh approximation
  │
  ▼
[5] S4D Layer 2         (4096, 64) → (4096, 64)
  │
  ▼
[6] GELU 2              in-place
  │
  ▼
[7] TakeLastTimestep    (4096, 64) → (64)
  │
  ▼
[8] FC Layer            (64) → (4)                 [Linear]
  │
  ▼
[9] Softmax             (4) → (4) probabilities
```

**Output classes:** Smooth Round · Smooth Cigar · Edge-on Disk · Unbarred Spiral

---

## Weight File Format (`model_weights.bin`)

All parameters are stored as 32-bit floats in sequential binary format (little-endian):

| Offset (bytes) | Size (bytes) | Parameter | Type |
|---|---|---|---|
| 0 | 16384 | Hilbert indices (4096) | float32 → cast to int |
| 16384 | 256 | UProject weight (64×1) | float32 |
| 16640 | 256 | UProject bias (64) | float32 |
| 16896 | 256 | S4D L1 log_delta (64) | float32 |
| 17152 | 8192 | S4D L1 log_A_real (64×32) | float32 |
| 25344 | 8192 | S4D L1 A_imag (64×32) | float32 |
| 33536 | 16384 | S4D L1 C (64×32×2 interleaved) | float32 |
| 49920 | 256 | S4D L1 D_skip (64) | float32 |
| *(repeat for S4D L2)* | | | |
| ... | 1024 | FC weight (4×64) | float32 |
| ... | 16 | FC bias (4) | float32 |

Total: **21,124 parameters**

---

## Task 4 (Bonus): C Backend via Python Interface

The `model/interface.py` supports switching between backends at runtime.

**Launch GUI with Python backend (default):**
```bash
python3 launch_gui.py
python3 launch_gui.py --backend python
```

**Launch GUI with C backend:**
```bash
# First build the demo executable
make main

python3 launch_gui.py --backend c
```

**Validate C backend matches Python backend:**
```bash
python3 validate_c_backend.py
```

---

## Common Issues

**`cannot open model_weights.bin`** — Run all executables from the directory containing `model_weights.bin`, or provide the full path.

**`Expected 4096 floats`** — Ensure input `.bin` files contain exactly 4096 float32 values (grayscale). RGB images require 4096×3 = 12288 floats.

**Compilation error: `timespec_get` not found** — Add `-std=c11` (already in Makefile). On older GCC, try `-D_POSIX_C_SOURCE=200809L`.

**`run_test.py` exits with code 1** — One or more layers failed numerical validation. Check that `generate_test_data.py` was run first and reference `.bin` files exist in the current directory.

---

## Academic Integrity

All code is original work. External references consulted are cited in the project report (`m2_<teamname>.pdf`). Do not copy or redistribute without permission.
