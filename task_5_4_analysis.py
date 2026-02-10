import torch
import torch.nn as nn
import time
import math
from model.s4d import S4D

class S4D_FFT(S4D):
    """Subclass to perform FFT-based convolution for benchmarking."""
    def forward(self, u):
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)
        dt = torch.exp(self.log_dt) 
        C = torch.view_as_complex(self.C) 
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag 
        dtA = A * dt.unsqueeze(-1)  
        K_exp = torch.exp(dtA.unsqueeze(-1) * torch.arange(L, device=u.device)) 
        C_tilde = C * (torch.exp(dtA) - 1.) / A
        k = 2 * torch.einsum('hn, hnl -> hl', C_tilde, K_exp).real 
        
        # FFT Convolution
        k_f = torch.fft.rfft(k, n=2*L) 
        u_f = torch.fft.rfft(u, n=2*L) 
        y = torch.fft.irfft(u_f * k_f, n=2*L)[..., :L] 
        
        y = y + u * self.D.unsqueeze(-1)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_analysis(h=64, n=64, lengths=[64, 256, 1024, 4096]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Task 5.4 Analysis (H={h}, N={n}) on {device}")
    
    # 1. Parameter Analysis
    s4d_conv = S4D(d_model=h, d_state=n)
    s4d_params = count_parameters(s4d_conv)
    full_s4_params = h * (n*n + n + n + 1 + 1)
    
    print("\n--- [Task 5.4.1] Parameter Count Analysis ---")
    print(f"Full S4 Layer: {full_s4_params:,}")
    print(f"S4D Layer:      {s4d_params:,}")
    print(f"Reduction:      {full_s4_params / s4d_params:.2f}x")

    # 2. Forward Pass Trace
    print("\n--- [Task 5.4.2] Forward Pass Trace (L=64) ---")
    print("1. Input Shape: (Batch, H, L) = (1, 64, 64)")
    print("2. Materialize complex A: A = -exp(log_A_real) + i * A_imag")
    print("3. Discretize: dtA = A * dt")
    print("4. Kernel K(t) = C * exp(dtA * t) * B (Diagonal case)")
    print("5. Convolution: y = conv1d(u, K, groups=H) for causal linear recurrence")

    # 3. Execution Time Benchmarking
    print("\n--- [Task 5.4.3] Benchmarking: FFT vs Direct Convolution ---")
    print(f"{'Length':<10} | {'FFT (ms)':<15} | {'Direct (ms)':<15}")
    print("-" * 45)
    
    s4d_fft = S4D_FFT(d_model=h, d_state=n).to(device)
    s4d_conv = s4d_conv.to(device)
    
    for L in lengths:
        u = torch.randn(1, h, L).to(device)
        
        # Benchmark FFT
        for _ in range(5): _ = s4d_fft(u) # Warmup
        if device == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50): _ = s4d_fft(u)
        if device == 'cuda': torch.cuda.synchronize()
        fft_time = (time.time() - t0) * 1000 / 50
        
        # Benchmark Direct
        for _ in range(5): _ = s4d_conv(u) # Warmup
        if device == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50): _ = s4d_conv(u)
        if device == 'cuda': torch.cuda.synchronize()
        conv_time = (time.time() - t0) * 1000 / 50
        
        print(f"{L:<10} | {fft_time:<15.4f} | {conv_time:<15.4f}")

if __name__ == "__main__":
    run_analysis()
