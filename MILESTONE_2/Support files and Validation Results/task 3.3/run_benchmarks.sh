#!/bin/bash
# ===========================================================
# run_benchmarks.sh  —  Task 3.3 Automation Script
# ===========================================================
# Builds all optimization levels, runs benchmarks, generates
# assembly files, counts instructions, and collects results
# into CSVs for analyze_performance.py to visualize.
#
# Usage:
#   chmod +x run_benchmarks.sh
#   ./run_benchmarks.sh
# ===========================================================

set -e

WEIGHTS="model_weights.bin"
SAMPLE="sample_bins/sample_0.bin"

echo "=========================================================="
echo "  S4D Benchmark Suite — Task 3.3"
echo "=========================================================="

# --- Step 1: Build all optimization levels ---
echo ""
echo "[Step 1] Building at all optimization levels..."
make all_opts
echo "[OK] Built: bench_O0, bench_O1, bench_O2, bench_O3, bench_Ofast"

# --- Step 2: Run each benchmark ---
echo ""
echo "[Step 2] Running benchmarks (100 iterations each)..."

OPT_LEVELS=("O0" "O1" "O2" "O3" "Ofast")

for opt in "${OPT_LEVELS[@]}"; do
    echo ""
    echo "--- Running bench_${opt} ---"
    ./bench_${opt} ${WEIGHTS} ${SAMPLE} benchmark_results_${opt}.csv
done

echo ""
echo "[OK] All benchmarks complete."

# --- Step 3: Generate assembly dumps ---
echo ""
echo "[Step 3] Generating assembly dumps..."
make assembly
echo "[OK] Generated: nn_O0.s, nn_O2.s, nn_O3.s"

# --- Step 4: Assembly instruction counts ---
echo ""
echo "[Step 4] Counting assembly instructions..."
echo "Opt_Level,Total_Instructions" > asm_instruction_counts.csv
for sfile in nn_O0.s nn_O2.s nn_O3.s; do
    opt=$(echo $sfile | sed 's/nn_//;s/\.s//')
    count=$(grep -cE '^\s+[a-z]' $sfile 2>/dev/null || echo 0)
    echo "${opt},${count}" >> asm_instruction_counts.csv
    echo "  ${sfile}: ${count} instructions (static)"
done
echo "[OK] Static instruction counts written to asm_instruction_counts.csv"

# --- Step 5: Runtime instruction counts (if perf available) ---
echo ""
echo "[Step 5] Runtime instruction counts (perf stat)..."
if command -v perf &> /dev/null; then
    echo "Opt_Level,Instructions_Executed" > perf_instruction_counts.csv
    for opt in "${OPT_LEVELS[@]}"; do
        count=$(perf stat -e instructions ./bench_${opt} ${WEIGHTS} ${SAMPLE} /dev/null 2>&1 | \
                grep instructions | awk '{print $1}' | tr -d ',')
        echo "${opt},${count}" >> perf_instruction_counts.csv
        echo "  bench_${opt}: ${count} instructions executed"
    done
    echo "[OK] perf results written to perf_instruction_counts.csv"
else
    echo "[SKIP] perf not available. Using valgrind if available..."
    if command -v valgrind &> /dev/null; then
        echo "Opt_Level,Instructions_Executed" > perf_instruction_counts.csv
        for opt in "${OPT_LEVELS[@]}"; do
            count=$(valgrind --tool=callgrind --callgrind-out-file=/dev/null \
                    ./bench_${opt} ${WEIGHTS} ${SAMPLE} /dev/null 2>&1 | \
                    grep "refs:" | head -1 | awk '{print $4}' | tr -d ',')
            echo "${opt},${count}" >> perf_instruction_counts.csv
            echo "  bench_${opt}: ${count} instruction refs"
        done
        echo "[OK] valgrind results written to perf_instruction_counts.csv"
    else
        echo "[SKIP] Neither perf nor valgrind available."
        echo "[INFO] You can install perf or valgrind, or run manually:"
        echo "       perf stat -e instructions ./bench_O2 ${WEIGHTS} ${SAMPLE} /dev/null"
    fi
fi

# --- Step 6: Run PyTorch baseline ---
echo ""
echo "[Step 6] Running PyTorch baseline profiling..."
python3 profile_pytorch.py
echo "[OK] pytorch_timing.csv generated"

# --- Step 7: Generate charts ---
echo ""
echo "[Step 7] Generating visualization charts..."
python3 analyze_performance.py
echo "[OK] Charts generated"

echo ""
echo "=========================================================="
echo "  ALL DONE — Files generated:"
echo "    benchmark_results_O0.csv ... benchmark_results_Ofast.csv"
echo "    asm_instruction_counts.csv"
echo "    perf_instruction_counts.csv (if perf/valgrind available)"
echo "    pytorch_timing.csv"
echo "    Charts: timing_breakdown.png, optimization_comparison.png,"
echo "           instruction_count.png, memory_footprint.png,"
echo "           c_vs_python.png"
echo "=========================================================="
