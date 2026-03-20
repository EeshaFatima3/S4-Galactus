import torch
import time
import matplotlib.pyplot as plt
import pandas as pd
from s4_recurrent import S4Recurrent
from s4_convolutional import S4Convolutional


def run_benchmarks():
    # --- SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 16
    d_state = 64
    batch_size = 2
    seq_lengths = [64, 256, 1024, 4096]
   
    # Initialize models
    model_rec = S4Recurrent(d_model=d_model, d_state=d_state).to(device)
    model_conv = S4Convolutional(d_model=d_model, d_state=d_state).to(device)
   
    # CRITICAL: Copy weights to ensure numerical equivalence
    model_conv.load_state_dict(model_rec.state_dict())
    model_rec.eval()
    model_conv.eval()


    # --- PART 1: NUMERICAL VALIDATION (Task 5.3, Step 1-6) ---
    print("\n--- Numerical Validation ---")
    u_val = torch.randn(batch_size, 100, d_model).to(device)
    with torch.no_grad():
        out_rec = model_rec(u_val)
        out_conv = model_conv(u_val)
   
    max_diff = torch.max(torch.abs(out_rec - out_conv)).item()
    print(f"Max Difference: {max_diff:.2e}")
    if max_diff < 1e-5:
        print("✅ Numerical Equivalence Confirmed!")
    else:
        print("❌ Models Diverge. Check your discretization logic.")


    # --- PART 2: BENCHMARKING (Timing) ---
    print("\n--- Performance Benchmarking ---")
    results = []


    for L in seq_lengths:
        u = torch.randn(batch_size, L, d_model).to(device)
       
        # Warmup (important for accurate timing)
        for _ in range(5):
            _ = model_rec(u)
            _ = model_conv(u)
       
        # Time Recurrent
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model_rec(u)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_rec = (time.perf_counter() - start) * 1000 # Convert to ms
       
        # Time Convolutional
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model_conv(u)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_conv = (time.perf_counter() - start) * 1000 # Convert to ms
       
        results.append({"Length": L, "Recurrent (ms)": t_rec, "Convolutional (ms)": t_conv})
        print(f"L={L:4d} | Recurrent: {t_rec:7.2f}ms | Conv: {t_conv:7.2f}ms")


    # --- PART 3: GENERATE TABLE & GRAPH ---
    # Create a Table (Pandas makes this look clean)
    df = pd.DataFrame(results)
    print("\n--- Benchmark Table ---")
    print(df.to_string(index=False))
   
    # Save the Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["Length"], df["Recurrent (ms)"], marker='o', label='Recurrent S4')
    plt.plot(df["Length"], df["Convolutional (ms)"], marker='s', label='Convolutional S4')
    plt.xscale('log', base=2) # Using log scale because lengths double/quadruple
    plt.yscale('log')
    plt.xlabel('Sequence Length (L)')
    plt.ylabel('Time (ms)')
    plt.title('Execution Time: Recurrent vs. Convolutional S4')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('s4_performance_plot.png')
    print("\nGraph saved as 's4_performance_plot.png'. Upload this to your report!")


if __name__ == "__main__":
    run_benchmarks()


