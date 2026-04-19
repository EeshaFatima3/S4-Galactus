# S4 Galaxy Classification — RISC-V Scalar Implementation

This repository contains two complementary components for the S4-based galaxy morphology classification project:

1. **Python / PyTorch base** — data loaders, model scaffolding, S4D reference implementation, and a training notebook.
2. **RISC-V scalar inference engine** — a hand-written RISC-V 32-bit assembly implementation of the trained neural network, compiled and simulated on the VeeR EH1 core via Whisper ISS.

**Requirements:** Python 3.11+, PyTorch 2.0+, CUDA (optional), `riscv32-unknown-elf` toolchain, Whisper ISS

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Python — Installation & Usage](#python--installation--usage)
  - [Model Modules](#model-modules)
  - [Training](#training)
  - [Interactive Visualization Tool](#interactive-visualization-tool)
  - [Implementation Tasks](#implementation-tasks)
  - [Fixed Constraints](#fixed-constraints)
  - [Dependencies](#dependencies)
- [RISC-V — Build System](#risc-v--build-system)
  - [Prerequisites](#prerequisites)
  - [Build Script Usage](#build-script-usage)
  - [Makefile Usage](#makefile-usage)
  - [Math Library](#math-library-maths)
  - [Neural Network Engine](#neural-network-engine)
  - [Build Output Layout](#build-output-layout)
  - [Benchmarking Report](#benchmarking-report)
  - [Configuration](#configuration)
- [Support & References](#support--references)

---

## Repository Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── main.py                        # Interactive visualization tool
├── train.ipynb                    # Training notebook
│
├── model/                         # Python model implementations
│   ├── __init__.py
│   ├── gclassifier.py             # Galaxy classifier (TODO: forward pass)
│   ├── s4d.py                     # S4D reference implementation (fully implemented)
│   ├── hilbert.py                 # Hilbert curve (TODO: _d2xy method)
│   ├── tlts.py                    # TakeLastTimestep (TODO: forward)
│   ├── interface.py               # Unified model interface (M3/M4)
│   ├── functions.py               # Utility functions
│   └── gui.py                     # GUI components
│
└── risc-v/                        # RISC-V scalar inference engine
    ├── build.sh                   # Main bash build script
    ├── Makefile                   # Make build system
    ├── veer/
    │   ├── link.ld                # Linker script for VeeR EH1
    │   └── whisper.json           # Whisper simulator configuration
    ├── math.s                     # Math helper routines (exp, sin, cos, tanh)
    ├── nn.s                       # Neural network control flow / inference pipeline
    ├── weights.s                  # Pre-trained model weights (conv2d_w, fc_w, fc_b)
    ├── main.s                     # Entry point / bare-metal main
    ├── sample0.s                  # Test input sample 0
    ├── sample1.s                  # Test input sample 1
    ├── sample2.s                  # Test input sample 2
    ├── sample3.s                  # Test input sample 3
    ├── sample4.s                  # Test input sample 4
    ├── sample5.s                  # Test input sample 5
    ├── sample6.s                  # Test input sample 6
    ├── sample7.s                  # Test input sample 7
    ├── sample8.s                  # Test input sample 8
    ├── sample9.s                  # Test input sample 9
    ├── validation_results.json    # Model validation output / accuracy results
    └── build/                     # Generated output directory (auto-created)
        ├── exe/                   # Linked ELF executables
        ├── hex/                   # Verilog hex files for simulation
        ├── dis/                   # Disassembly listings
        ├── asm/                   # Compiled/linked assembly dumps
        ├── obj/                   # Compiled object files
        └── logs/                  # Whisper execution and instruction profiling logs
```

---

## Python — Installation & Usage

```bash
cd space-state-model
pip install -r requirements.txt
```

### Model Modules

**`model/gclassifier.py`** — Galaxy classifier architecture
- `GalaxyClassifierS4D` — Main model combining Hilbert scanning, S4 layers, and classification head
- TODO: Complete `forward()` method

**`model/s4d.py`** — Diagonal S4 layer
- Fully implemented reference implementation
- Study for S4 architecture patterns, FFT convolution, diagonal parameterization

**`model/hilbert.py`** — Hilbert curve utilities
- `HilbertScan` — Converts 2D images to 1D sequences
- TODO: Complete `_d2xy()` method

**`model/tlts.py`** — Sequence pooling
- `TakeLastTimestep` — Extracts final timestep for classification
- TODO: Implement extraction logic

**`model/functions.py`** — Helper utilities including matrix operations and discretization methods

### Training

Launch the interactive training notebook with step-by-step explanations:

```bash
jupyter notebook train.ipynb
```

The notebook includes data loading and preprocessing, model initialization, a training loop with validation, logging and visualization, along with TODO markers for required implementations.

### Interactive Visualization Tool

Launch the interactive galaxy explorer GUI:

```bash
python main.py --python -m galaxy_s4_model.pth
```

Full usage:

```
usage: main.py [-h] (--python | --riscv) [--model-path MODEL_PATH] [--colored] [--data-dir DATA_DIR]

options:
  -h, --help                        Show this help message and exit
  --python, -p                      Use Python model implementation
  --riscv                           Use RISC-V model implementation
  --model-path MODEL_PATH, -m       Path to trained model file (default: galaxy_s4_model.pth)
  --colored, -c                     Use colored (RGB) images instead of grayscale
  --data-dir DATA_DIR               Root directory for dataset (default: ./data)

Examples:
  main.py --python -m galaxy_model.pth
  main.py -p -m galaxy_model.pth --colored
  main.py --riscv
```

**Controls:**
- `LEFT Arrow` — Previous sample
- `RIGHT Arrow` — Next sample
- `R` — Random sample
- `Q` — Quit

### Implementation Tasks

1. **`model/gclassifier.py`** — Complete `GalaxyClassifierS4D.forward()`
   - Connect Hilbert scanning → linear projection → S4 blocks → GELU activation → final timestep extraction → classification head
   - Handle tensor shapes: `(B, C, 64, 64)` → `(B, 4)`

2. **`model/hilbert.py`** — Implement `_d2xy()`
   - Convert 1D distance along Hilbert curve to 2D `(x, y)` coordinates
   - Recursive algorithm for arbitrary power-of-two grid sizes

3. **`model/tlts.py`** — Implement `TakeLastTimestep.forward()`
   - Extract final timestep from sequence tensor
   - Shape: `(B, L, D)` → `(B, D)`

4. **`train.ipynb`** — Fill TODO sections
   - Training loop implementation
   - Validation/test evaluation
   - Visualization functions

### Fixed Constraints

Do not modify these values (required for multi-milestone compatibility):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | `64` | Hidden dimension |
| `d_state` | `64` | State space dimension |
| `image_size` | `64` | Image resolution |
| `num_classes` | `4` | Galaxy morphology classes |

### Dependencies

Key packages: `torch`, `numpy`, `matplotlib`, `pygame`, `einops`, `galaxy_mnist`

---

## RISC-V — Build System

A build toolchain for compiling, linking, and executing the neural network inference engine on the VeeR EH1 core via the Whisper ISS.

### Prerequisites

| Tool | Purpose |
|------|---------|
| `riscv32-unknown-elf-gcc` | Cross-compiler for RV32IMFC target |
| `riscv32-unknown-elf-objcopy` | Binary conversion (ELF → Verilog hex) |
| `riscv32-unknown-elf-objdump` | Disassembly and data inspection |
| `whisper` | VeeR EH1 instruction set simulator |

Make sure all of the above are available on your `PATH` before building.

### Build Script Usage

```bash
./build.sh [options] <file> [<file> ...]
```

| Flag | Description |
|------|-------------|
| `-a` | Compile and execute one or more assembly (`.s`) files |
| `-c` | Clean all generated files (removes the `build/` directory) |
| `-e` | Execute the last compiled binary using Whisper |
| `-g [opt_flag]` | Compile a C (`.c`) file to assembly/object; optionally pass `-O2` or `-O3` |
| `-h` | Show help message |
| `-l <file>` | Link an additional assembly file (used with `-a`) |

**Compile and run the full inference pipeline:**
```bash
./build.sh -a main.s -l nn.s -l math.s -l weights.s
```

**Compile and run with a specific sample:**
```bash
./build.sh -a main.s -l sample0.s -l nn.s -l math.s -l weights.s
```

**Execute a previously compiled binary:**
```bash
./build.sh -e main.s
```

**Clean all build artifacts:**
```bash
./build.sh -c
```

### Makefile Usage

A `Makefile` is also provided and mirrors the `build.sh` behaviour:

| Command | Description |
|---------|-------------|
| `make` | Compile + run (default) |
| `make run EXTRA=sample0.s` | Compile and run with a specific sample injected |
| `make run-all` | Build and run all 10 samples separately, saving individual logs |
| `make bench` | Print static and dynamic instruction counts |
| `make dis` | Dump disassembly to stdout |
| `make clean` | Remove all build artifacts |
| `make help` | Show all available targets |

### Math Library (`math.s`)

Hand-optimized floating-point routines for RISC-V 32-bit. ABI convention:

- **Argument / return value:** `fa0`
- **Preserved registers:** `ra`, `fs0`–`fs2`
- **Caller-saved** (`ft0`–`ft11`, `fa1`–`fa7`) are **not** preserved

| Function | Description |
|----------|-------------|
| `exp_f` | `exp(x)` — clamped to `[-88, 88]`, degree-6 polynomial via range reduction |
| `sin_f` | `sin(x)` — Taylor series approximation |
| `cos_f` | `cos(x)` — Taylor series approximation |
| `tanh_f` | `tanh(x)` — clamped to `[-9, 9]` |

**`exp_f` algorithm:** clamp x → compute `n = round(x / ln2)` → reduce `r = x − n·ln2` → evaluate degree-6 polynomial → scale by `2ⁿ` via IEEE 754 exponent field.

### Neural Network Engine

The inference pipeline is a full **S4D (Diagonal State Space) model** implemented entirely in RISC-V scalar assembly, with pre-trained weights embedded as `.float` constants in the `.data` section of `weights.s`.

| Symbol | Description |
|--------|-------------|
| `up_w` / `up_b` | Input projection weights and biases |
| `s4_0_log_dt` / `s4_1_log_dt` | S4D layer timescale parameters |
| `fc_w` / `fc_b` | Fully connected layer weights and biases → 4-class output |

The inference flow processes 4,096-pixel galaxy images through 9 stages:

```
Hilbert Scan → Input Projection (up_w) → S4D Layer 1 → GELU → S4D Layer 2 → GELU → Take Last Timestep → FC (fc_w + fc_b) → Softmax
```

Math routines from `math.s` (`exp_f`, `sin_f`, `cos_f`, `tanh_f`) are used during S4D kernel generation and GELU activations.

### Build Output Layout

| Path | Contents |
|------|----------|
| `build/exe/<n>.exe` | Linked ELF binary |
| `build/hex/<n>.hex` | Verilog hex file for Whisper simulation |
| `build/dis/<n>.dis` | Full disassembly with source interleaving |
| `build/dis/<n>.data` | Raw dump of the `.data` section |
| `build/asm/<n>.s` | Disassembly using canonical instruction names |
| `build/obj/<n>.o` | Compiled object file (C compilation only) |
| `build/logs/<n>.txt` | Whisper ISS execution and profiling log |

### Benchmarking Report

Instruction profiling is generated automatically by Whisper via `--profileinst`. Logs are saved to `build/logs/<n>.txt` for each run.

#### Instruction Profiling

The `build.sh` script executes the model via Whisper. Ensure the `--profileinst` flag is used to enable instruction-level profiling for Task 3. Logs are saved to `build/logs/<filename>.txt`.

**Step 1 — Run profiling:**
```bash
./build.sh -a shm_main.s -l nn.s -l math.s -l weights.s
```

**Step 2 — Generate stats:**
```bash
python count_stats.py build/logs/shm_main.txt
```

This produces the **Dynamic Instruction Count** and family breakdown (R-type, I-type, F-type, etc.) needed for your LaTeX report.

#### Static Analysis

Static instruction counts (by family) can be extracted using the disassembly:

```bash
# Total instruction count
grep -v "^;" build/asm/shm_main.s | grep -c "\w"
```

#### Dynamic Analysis

Total executed instructions and family distributions (R, I, S, B, U, J, F) are recorded in the Whisper log files. Use `count_stats.py` to parse these logs for your Milestone 3 report:

```bash
python count_stats.py build/logs/shm_main.txt
```

Per-sample logs are available under `build/logs/` after running `make run-all`.

**Validation results** (accuracy across all 10 test samples) are recorded in `validation_results.json`.

### Configuration

The following variables at the top of `build.sh` / `Makefile` can be adjusted:

| Variable | Default | Description |
|----------|---------|-------------|
| `GCC_PREFIX` | `riscv32-unknown-elf` | Cross-compiler toolchain prefix |
| `ABI` | `-march=rv32imfc -mabi=ilp32f` | Target ISA and ABI flags (scalar only) |
| `LINK` | `veer/link.ld` | Path to the linker script |
| `WHISPER_CFG` | `veer/whisper.json` | Path to Whisper configuration |
| `BUILD_DIR` | `build` | Root directory for all build outputs |

Whisper is invoked with start address `0x80000000`, tohost address `0xd0580000`, and instruction profiling enabled. The linker uses `-nostdlib` (matching `build.sh`) to strip all standard libraries for bare-metal execution.

---

## Support & References

**Technical Questions:** s.taha.29208@khi.iba.edu.pk

**References:**
- Gu et al. (2022) — "Efficiently Modeling Long Sequences with Structured State Spaces" (ICLR)
- Gu et al. (2022) — "On the Parameterization and Initialization of Diagonal State Space Models" (NeurIPS)
