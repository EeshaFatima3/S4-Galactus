import torch
import torch.nn as nn

class S4Recurrent(nn.Module):
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model   # H
        self.d_state = d_state   # N

        # A and B are vectors of size N. 
        # C is (H, N) - each channel H has its own projection of the N states.
        self.A = nn.Parameter(-torch.rand(d_state)) 
        self.B = nn.Parameter(torch.randn(d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.zeros(d_model))

        dt_init = torch.empty(1).uniform_(dt_min, dt_max)
        self.log_dt = nn.Parameter(torch.log(dt_init))

    def forward(self, u):
        B, L, H = u.shape
        device = u.device

        # 1. Discretization
        dt = torch.exp(self.log_dt)
        A_bar = torch.exp(dt * self.A) # (N)
        B_bar = (A_bar - 1.0) / self.A * self.B # (N)

        # 2. State: (Batch, Channels H, State N)
        x = torch.zeros(B, H, self.d_state, device=device)
        outputs = []

        # 3. Step-by-Step Loop
        for k in range(L):
            u_k = u[:, k, :] # (B, H)
            
            # Update state: Each channel H gets the same B_bar * u_k contribution
            # x: (B, H, N), B_bar: (N), u_k: (B, H)
            x = A_bar * x + B_bar * u_k.unsqueeze(-1)
            
            # Output: y = sum(C * x) along N-dimension + D * u
            # C: (H, N), x: (B, H, N) -> elementwise mul then sum over N
            y_k = torch.sum(self.C * x, dim=-1) + (u_k * self.D)
            outputs.append(y_k)

        return torch.stack(outputs, dim=1)