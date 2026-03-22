"""
analyze_performance.py  -  Task 3.3 Visualization & Analysis
=============================================================
Reads benchmark CSV files and generates the 5 mandatory charts:

  1. timing_breakdown.png       - Per-layer timing breakdown (pie chart)
  2. optimization_comparison.png - Inference time vs. optimization level
  3. instruction_count.png       - Instruction count vs. optimization level
  4. memory_footprint.png        - Memory footprint breakdown
  5. c_vs_python.png             - C vs. Python timing comparison

Also prints a comprehensive summary table to stdout.

Usage:
    python analyze_performance.py

Prerequisites:
    - benchmark_results_O0.csv through benchmark_results_Ofast.csv
    - pytorch_timing.csv
    - asm_instruction_counts.csv (optional)
    - perf_instruction_counts.csv (optional)
"""
import os
import csv
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not found. Charts will be skipped.")
    print("         Install with: pip install matplotlib")

# ================================================================
# Helpers
# ================================================================
def read_benchmark_csv(path):
    """Read a benchmark_results CSV into a dict of layer->values."""
    data = {}
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["Layer"]] = {
                "avg_ms":    float(row["Avg_ms"]),
                "stddev_ms": float(row["StdDev_ms"]),
                "percent":   float(row["Percent"]),
            }
    return data


# ================================================================
# Chart 1: Per-Layer Timing Breakdown (Pie Chart)
# ================================================================
def chart_timing_breakdown(data, filename="timing_breakdown.png"):
    if not HAS_MPL or data is None:
        return
    layers = [k for k in data if k != "Total"]
    times  = [data[k]["avg_ms"] for k in layers]
    pcts   = [data[k]["percent"] for k in layers]

    colors = ['#2196F3', '#4CAF50', '#FF5722', '#FFC107',
              '#E91E63', '#9C27B0', '#00BCD4']

    fig, ax = plt.subplots(figsize=(12, 7))

    # Only label slices >= 3% to avoid overlap on tiny slices
    def autopct_fn(pct):
        return f'{pct:.1f}%' if pct >= 3.0 else ''

    wedges, texts, autotexts = ax.pie(
        times, autopct=autopct_fn,
        colors=colors[:len(layers)], startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(linewidth=0.8, edgecolor='white'),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
        at.set_color('white')

    # Use legend instead of inline labels - avoids all overlap
    legend_labels = [
        f"{name}  -  {times[i]:.2f} ms  ({pcts[i]:.2f}%)"
        for i, name in enumerate(layers)
    ]
    ax.legend(wedges, legend_labels, title="Layers", title_fontsize=11,
              fontsize=10, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, framealpha=0.9)

    ax.set_title("Per-Layer Inference Time Breakdown (-O2)",
                 fontsize=14, fontweight='bold', pad=15)

    total_ms = data["Total"]["avg_ms"] if "Total" in data else sum(times)
    fig.text(0.5, 0.01,
             f"Total inference: {total_ms:.2f} ms  |  Throughput: {(1000.0/total_ms):.2f} img/s",
             ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {filename}")


# ================================================================
# Chart 2: Inference Time vs. Optimization Level
# ================================================================
def chart_optimization_comparison(opt_data, filename="optimization_comparison.png"):
    if not HAS_MPL:
        return
    levels = []
    times  = []
    for opt in ["O0", "O1", "O2", "O3", "Ofast"]:
        if opt_data.get(opt) and "Total" in opt_data[opt]:
            levels.append(f"-{opt}")
            times.append(opt_data[opt]["Total"]["avg_ms"])

    if not levels:
        print(f"  [SKIP] {filename} - no optimization data found")
        return

    speedups = [times[0] / t for t in times]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(levels, times, color='#2196F3', alpha=0.8, label='Inference Time')
    ax1.set_xlabel("Compiler Optimization Level", fontsize=12)
    ax1.set_ylabel("Inference Time (ms)", fontsize=12, color='#2196F3')
    ax1.tick_params(axis='y', labelcolor='#2196F3')

    # Add time labels on bars
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{t:.1f}ms', ha='center', va='bottom', fontsize=10)

    # Speedup line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(levels, speedups, 'o-', color='#FF5722', linewidth=2, markersize=8, label='Speedup')
    ax2.set_ylabel("Speedup (vs -O0)", fontsize=12, color='#FF5722')
    ax2.tick_params(axis='y', labelcolor='#FF5722')

    for i, s in enumerate(speedups):
        ax2.annotate(f'{s:.2f}x', (levels[i], s), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9, color='#FF5722')

    ax1.set_title("Inference Time vs. Compiler Optimization Level", fontsize=14, fontweight='bold')
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {filename}")


# ================================================================
# Chart 3: Instruction Count vs. Optimization Level
# ================================================================
def chart_instruction_count(filename="instruction_count.png"):
    if not HAS_MPL:
        return

    # Try runtime instruction counts first (perf/valgrind), then static
    data = {}
    source = "static"

    if os.path.exists("perf_instruction_counts.csv"):
        source = "runtime (perf/valgrind)"
        with open("perf_instruction_counts.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[row["Opt_Level"]] = int(row["Instructions_Executed"])

    if not data and os.path.exists("asm_instruction_counts.csv"):
        with open("asm_instruction_counts.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[row["Opt_Level"]] = int(row["Total_Instructions"])

    if not data:
        print(f"  [SKIP] {filename} - no instruction count data found")
        return

    levels = list(data.keys())
    counts = list(data.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(levels, counts, color='#9C27B0', alpha=0.8)
    ax.set_xlabel("Optimization Level", fontsize=12)
    ax.set_ylabel(f"Instruction Count ({source})", fontsize=12)
    ax.set_title("Instruction Count vs. Optimization Level", fontsize=14, fontweight='bold')

    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{c:,}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {filename}")


# ================================================================
# Chart 4: Memory Footprint Breakdown
# ================================================================
def chart_memory_footprint(filename="memory_footprint.png"):
    if not HAS_MPL:
        return

    # Static calculation based on nn.h constants
    SEQ_LEN = 4096
    D_MODEL = 64
    N_HALF  = 32
    NUM_CLS = 4
    C_IN    = 1

    components = {
        "Hilbert Indices":     SEQ_LEN * 4,                           # int[4096]
        "UProject W+B":        (D_MODEL * C_IN + D_MODEL) * 4,       # float[64+64]
        "S4D L1 Params":       (D_MODEL + D_MODEL*N_HALF*2 + D_MODEL*N_HALF*2 + D_MODEL) * 4,
        "S4D L2 Params":       (D_MODEL + D_MODEL*N_HALF*2 + D_MODEL*N_HALF*2 + D_MODEL) * 4,
        "FC W+B":              (NUM_CLS * D_MODEL + NUM_CLS) * 4,    # float[4*64+4]
        "Post-Hilbert Buf":    SEQ_LEN * C_IN * 4,                   # float[4096]
        "Buffer A":            SEQ_LEN * D_MODEL * 4,                # float[4096][64]
        "Buffer B":            SEQ_LEN * D_MODEL * 4,                # float[4096][64]
        "S4D Kernel Buf":      D_MODEL * SEQ_LEN * 4,                # float[64][4096]
        "Post-Pool Buf":       D_MODEL * 4,                          # float[64]
    }

    names    = list(components.keys())
    sizes_kb = [v / 1024.0 for v in components.values()]
    total_kb = sum(sizes_kb)
    total_mb = total_kb / 1024.0

    # Group by category
    param_names = names[:5]
    buf_names   = names[5:]
    param_sizes = sizes_kb[:5]
    buf_sizes   = sizes_kb[5:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bar chart of all components
    colors_p = ['#2196F3'] * len(param_names)
    colors_b = ['#FF5722'] * len(buf_names)
    all_colors = colors_p + colors_b

    bars = ax1.barh(names[::-1], sizes_kb[::-1], color=all_colors[::-1], alpha=0.85)
    ax1.set_xlabel("Size (KB)", fontsize=12)
    ax1.set_title("Memory Footprint Breakdown", fontsize=14, fontweight='bold')
    for bar, sz in zip(bars, sizes_kb[::-1]):
        ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{sz:.1f} KB', va='center', fontsize=9)

    # Right: pie chart (params vs buffers)
    param_total = sum(param_sizes)
    buf_total   = sum(buf_sizes)
    ax2.pie([param_total, buf_total],
            labels=[f'Parameters\n({param_total:.1f} KB)',
                    f'Buffers\n({buf_total:.1f} KB)'],
            autopct='%1.1f%%',
            colors=['#2196F3', '#FF5722'],
            startangle=90, textprops={'fontsize': 11})
    ax2.set_title(f"Total: {total_mb:.2f} MB", fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {filename}")

    # Print text summary
    print("\n  Memory Footprint Summary:")
    print(f"  {'Component':<25s} | {'Size':>10s}")
    print(f"  {'-'*25}-+-{'-'*10}")
    for n, s in zip(names, sizes_kb):
        print(f"  {n:<25s} | {s:>8.1f} KB")
    print(f"  {'-'*25}-+-{'-'*10}")
    print(f"  {'TOTAL':<25s} | {total_mb:>7.2f} MB")


# ================================================================
# Chart 5: C vs. Python Timing Comparison
# ================================================================
def chart_c_vs_python(c_data, py_path="pytorch_timing.csv", filename="c_vs_python.png"):
    if not HAS_MPL or c_data is None:
        return

    py_data = read_benchmark_csv(py_path)
    if py_data is None:
        print(f"  [SKIP] {filename} - pytorch_timing.csv not found")
        return

    layers = [k for k in c_data if k != "Total"]
    c_times  = [c_data[k]["avg_ms"] for k in layers]
    py_times = [py_data[k]["avg_ms"] for k in layers if k in py_data]

    if len(py_times) != len(layers):
        print(f"  [SKIP] {filename} - layer mismatch between C and Python")
        return

    c_total  = c_data["Total"]["avg_ms"]  if "Total" in c_data  else sum(c_times)
    py_total = py_data["Total"]["avg_ms"] if "Total" in py_data else sum(py_times)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: per-layer comparison
    x = np.arange(len(layers))
    w = 0.35
    ax1.bar(x - w/2, c_times,  w, label=f'C (-O2): {c_total:.2f}ms',  color='#2196F3', alpha=0.85)
    ax1.bar(x + w/2, py_times, w, label=f'Python: {py_total:.2f}ms', color='#FF5722', alpha=0.85)
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Time (ms)", fontsize=12)
    ax1.set_title("Per-Layer: C vs. Python", fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=10)

    # Right: total comparison
    bars = ax2.bar(["C (-O2)", "Python/PyTorch"], [c_total, py_total],
                   color=['#2196F3', '#FF5722'], alpha=0.85)
    ax2.set_ylabel("Total Inference Time (ms)", fontsize=12)
    ax2.set_title("Total Inference Time Comparison", fontsize=14, fontweight='bold')
    for bar, t in zip(bars, [c_total, py_total]):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{t:.2f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

    speedup = py_total / c_total if c_total > 0 else 0
    ax2.text(0.5, 0.85, f'Speedup: {speedup:.2f}x',
             transform=ax2.transAxes, ha='center', fontsize=13,
             fontweight='bold', color='green' if speedup > 1 else 'red')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {filename}")


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 60)
    print("Task 3.3 - Performance Analysis & Visualization")
    print("=" * 60)

    # Load benchmark data for each optimization level
    opt_data = {}
    for opt in ["O0", "O1", "O2", "O3", "Ofast"]:
        path = f"benchmark_results_{opt}.csv"
        d = read_benchmark_csv(path)
        if d:
            opt_data[opt] = d
            print(f"  Loaded: {path}")
        else:
            print(f"  [SKIP] {path} not found")

    # If no per-opt data, try default benchmark_results.csv as O2
    if not opt_data:
        d = read_benchmark_csv("benchmark_results.csv")
        if d:
            opt_data["O2"] = d
            print("  Loaded: benchmark_results.csv (as -O2)")

    # Use O2 as the primary dataset
    primary = opt_data.get("O2") or (list(opt_data.values())[0] if opt_data else None)

    # Generate all 5 charts
    print("\nGenerating charts:")
    chart_timing_breakdown(primary)
    chart_optimization_comparison(opt_data)
    chart_instruction_count()
    chart_memory_footprint()
    chart_c_vs_python(primary)

    # Print optimization comparison table
    if opt_data:
        print("\n" + "=" * 60)
        print("Optimization Level Comparison")
        print("=" * 60)
        print(f"{'Level':<8s} | {'Total (ms)':>12s} | {'Speedup':>10s} | {'Throughput':>12s}")
        print("-" * 50)
        base_time = None
        for opt in ["O0", "O1", "O2", "O3", "Ofast"]:
            if opt in opt_data and "Total" in opt_data[opt]:
                t = opt_data[opt]["Total"]["avg_ms"]
                if base_time is None:
                    base_time = t
                speedup = base_time / t if t > 0 else 0
                throughput = 1000.0 / t if t > 0 else 0
                print(f"-{opt:<7s} | {t:12.2f} | {speedup:9.2f}x | {throughput:10.2f} img/s")

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
