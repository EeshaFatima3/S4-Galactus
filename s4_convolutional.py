import torch
import torch.nn as nn
import torch.nn.functional as F

class S4Convolutional(nn.Module):
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Initialize with identical names and shapes as Recurrent
        self.A = nn.Parameter(-torch.rand(d_state))
        self.B = nn.Parameter(torch.randn(d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.zeros(d_model))

        dt_init = torch.empty(1).uniform_(dt_min, dt_max)
        self.log_dt = nn.Parameter(torch.log(dt_init))

    def compute_kernel(self, L):
        dt = torch.exp(self.log_dt)
        A_bar = torch.exp(dt * self.A) # (N)
        B_bar = (A_bar - 1.0) / self.A * self.B # (N)

        # Build Kernel: K_i = sum(C * (A^i * B))
        # This kernel will have shape (H, L)
        kernel_elements = []
        curr_state = B_bar # Initial: A^0 * B_bar
        
        for _ in range(L):
            # C is (H, N), curr_state is (N). Result is (H)
            # This replicates the summation over N in the recurrent version
            k_i = torch.sum(self.C * curr_state, dim=-1)
            kernel_elements.append(k_i)
            curr_state = A_bar * curr_state
            
        return torch.stack(kernel_elements, dim=1) # (H, L)

    def forward(self, u):
        B, L, H = u.shape
        K = self.compute_kernel(L) # (H, L)

        u_t = u.transpose(1, 2) # (B, H, L)
        weight = K.view(H, 1, -1) # (H, 1, L)
        
        # Parallel Convolution: groups=H ensures channel independence
        y = F.conv1d(u_t, weight, padding=L-1, groups=H)
        y = y[:, :, :L]
        
        # Skip connection: D is (H), broadcast across batch B and length L
        y = y + (self.D.view(1, H, 1) * u_t)
        
        return y.transpose(1, 2)